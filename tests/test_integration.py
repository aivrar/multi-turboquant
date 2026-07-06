# SPDX-License-Identifier: MIT
"""End-to-end integration tests for the full KV cache pipeline.

Tests the complete flow: encode -> paged cache write -> cache read -> decode -> attention.
Covers all methods, TriAttention composability, and combined modes.
Simulates what actually happens inside a vLLM forward pass.
"""

import pytest
import torch

from multi_turboquant import CacheConfig, CacheMethod, get_preset
from multi_turboquant.kernels.triton.attention_backend import (
    is_multi_tq_kv_cache,
    get_method_family,
    get_packed_dim,
    triattention_score_and_mask,
)
from multi_turboquant.kernels.triton.iso_encode_kernel import (
    iso_encode_vectorized,
    iso_decode_vectorized,
    _pack_vectorized,
    _unpack_vectorized,
)
from multi_turboquant.kernels.triton.planar_encode_kernel import (
    planar_encode_vectorized,
    planar_decode_vectorized,
)
from multi_turboquant.kernels.triton.multi_tq_attention import (
    MultiTQKVCacheManager,
    create_kv_cache_manager,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def model_params():
    """Typical 7B model parameters."""
    return {
        "head_dim": 128,
        "num_kv_heads": 8,
        "num_heads": 32,  # GQA: 32 query heads, 8 KV heads
        "num_layers": 32,
        "block_size": 16,
    }


@pytest.fixture
def kv_tensors(model_params):
    """Random KV tensors simulating a decode step."""
    torch.manual_seed(42)
    return {
        "key": torch.randn(4, model_params["num_kv_heads"], model_params["head_dim"]),
        "value": torch.randn(4, model_params["num_kv_heads"], model_params["head_dim"]),
        "query": torch.randn(4, model_params["num_heads"], model_params["head_dim"]),
    }


@pytest.fixture
def paged_cache(model_params):
    """Empty paged KV cache."""
    num_blocks = 16
    block_size = model_params["block_size"]
    num_kv_heads = model_params["num_kv_heads"]
    return {
        "num_blocks": num_blocks,
        "block_size": block_size,
    }


# ─── Vectorized encode/decode tests ────────────────────────────────────────────

class TestVectorizedEncode:
    """Test the vectorized (no-Python-loop) encode/decode kernels."""

    def test_iso_vectorized_roundtrip(self):
        x = torch.randn(8, 4, 128)
        for bits in (3, 4):
            packed = iso_encode_vectorized(x, bits)
            decoded = iso_decode_vectorized(packed, 128, bits, torch.float32)
            assert decoded.shape == x.shape
            cos = torch.nn.functional.cosine_similarity(
                x.reshape(-1, 128), decoded.reshape(-1, 128), dim=-1,
            ).mean()
            assert cos > 0.95, f"IsoQuant {bits}bit cosine too low: {cos}"

    def test_planar_vectorized_roundtrip(self):
        x = torch.randn(8, 4, 128)
        for bits in (3, 4):
            packed = planar_encode_vectorized(x, bits)
            decoded = planar_decode_vectorized(packed, 128, bits, torch.float32)
            assert decoded.shape == x.shape
            cos = torch.nn.functional.cosine_similarity(
                x.reshape(-1, 128), decoded.reshape(-1, 128), dim=-1,
            ).mean()
            assert cos > 0.95, f"PlanarQuant {bits}bit cosine too low: {cos}"

    def test_vectorized_matches_reference(self):
        """Vectorized kernels should match the reference implementations."""
        from multi_turboquant.methods.isoquant import iso_encode, iso_decode
        from multi_turboquant.methods.planarquant import planar_encode, planar_decode

        x = torch.randn(4, 2, 128)

        # IsoQuant
        ref_packed = iso_encode(x, 3)
        vec_packed = iso_encode_vectorized(x, 3)
        ref_decoded = iso_decode(ref_packed, 128, 3, torch.float32)
        vec_decoded = iso_decode_vectorized(vec_packed, 128, 3, torch.float32)
        # They won't be bit-identical (different packing order) but quality should match
        ref_mse = (x - ref_decoded).square().mean()
        vec_mse = (x - vec_decoded).square().mean()
        assert abs(ref_mse - vec_mse) / max(ref_mse, 1e-8) < 0.1, (
            f"IsoQuant vectorized MSE drift: ref={ref_mse:.6f} vec={vec_mse:.6f}"
        )

        # PlanarQuant
        ref_packed = planar_encode(x, 3)
        vec_packed = planar_encode_vectorized(x, 3)
        ref_decoded = planar_decode(ref_packed, 128, 3, torch.float32)
        vec_decoded = planar_decode_vectorized(vec_packed, 128, 3, torch.float32)
        ref_mse = (x - ref_decoded).square().mean()
        vec_mse = (x - vec_decoded).square().mean()
        assert abs(ref_mse - vec_mse) / max(ref_mse, 1e-8) < 0.1

    def test_pack_unpack_roundtrip(self):
        """Vectorized pack/unpack should be lossless."""
        for bits in (1, 2, 3, 4, 5):
            max_val = (1 << bits) - 1
            indices = torch.randint(0, max_val + 1, (8, 4, 128), dtype=torch.uint8)
            packed = _pack_vectorized(indices, bits)
            unpacked = _unpack_vectorized(packed, 128, bits)
            assert torch.equal(indices, unpacked), f"Pack/unpack mismatch at {bits} bits"


# ─── Paged KV cache write/read tests ───────────────────────────────────────────

class TestPagedCacheOps:
    """Test writing to and reading from a paged KV cache."""

    @pytest.mark.parametrize("dtype,bits", [
        ("isoquant3", 3), ("isoquant4", 4),
        ("planarquant3", 3), ("planarquant4", 4),
    ])
    def test_write_read_paged_cache(self, dtype, bits, model_params):
        """Write tokens to paged cache, read them back, verify reconstruction."""
        head_dim = model_params["head_dim"]
        num_kv_heads = model_params["num_kv_heads"]
        block_size = model_params["block_size"]
        packed_dim = get_packed_dim(dtype, head_dim)

        # Create paged cache
        num_blocks = 4
        k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, packed_dim, dtype=torch.uint8)
        v_cache = torch.zeros_like(k_cache)

        # Create manager
        mgr = MultiTQKVCacheManager(
            kv_cache_dtype=dtype,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            num_heads=model_params["num_heads"],
            scale=1.0 / (head_dim ** 0.5),
        )

        # Write 8 tokens to slots 0-7
        key = torch.randn(8, num_kv_heads, head_dim)
        value = torch.randn(8, num_kv_heads, head_dim)
        slot_mapping = torch.arange(8, dtype=torch.int64)

        mgr.write_kv_cache(key, value, k_cache, v_cache, slot_mapping)

        # Verify something was written
        assert k_cache.sum() != 0, "K cache should have non-zero data"
        assert v_cache.sum() != 0, "V cache should have non-zero data"

        # Read back and decode
        block_table = torch.tensor([[0]], dtype=torch.int64)  # 1 sequence, block 0
        seq_lens = torch.tensor([8], dtype=torch.int32)
        query = torch.randn(1, model_params["num_heads"], head_dim)
        output = torch.zeros_like(query)

        mgr.decode_attention(query, k_cache, v_cache, block_table, seq_lens, output)

        # Output should be non-zero (attention computed)
        assert output.abs().sum() > 0, "Attention output should be non-zero"

    def test_slot_mapping_negative_skips(self, model_params):
        """Negative slot mappings should be skipped (padding tokens)."""
        head_dim = model_params["head_dim"]
        num_kv_heads = model_params["num_kv_heads"]
        block_size = model_params["block_size"]
        packed_dim = get_packed_dim("isoquant3", head_dim)

        k_cache = torch.zeros(2, block_size, num_kv_heads, packed_dim, dtype=torch.uint8)
        v_cache = torch.zeros_like(k_cache)

        mgr = create_kv_cache_manager("isoquant3", head_dim, num_kv_heads, 32, 0.1)

        key = torch.randn(4, num_kv_heads, head_dim)
        value = torch.randn(4, num_kv_heads, head_dim)
        # Slots: 0, -1 (skip), 2, -1 (skip)
        slot_mapping = torch.tensor([0, -1, 2, -1], dtype=torch.int64)

        mgr.write_kv_cache(key, value, k_cache, v_cache, slot_mapping)

        # Only slots 0 and 2 should have data
        assert k_cache[0, 0].sum() != 0  # slot 0
        assert k_cache[0, 1].sum() == 0  # slot 1 skipped
        assert k_cache[0, 2].sum() != 0  # slot 2


# ─── TriAttention integration tests ────────────────────────────────────────────

class TestTriAttentionIntegration:
    """Test TriAttention scoring, eviction, and composition with quantization."""

    def test_score_real_keys(self):
        """Score realistic key vectors (not random — simulate attention patterns)."""
        # Create keys with some tokens being more "important" (higher energy)
        torch.manual_seed(42)
        keys = torch.randn(64, 8, 128)
        # Make tokens 10, 20, 30 high-energy (should score higher)
        keys[10] *= 5.0
        keys[20] *= 5.0
        keys[30] *= 5.0

        mask = triattention_score_and_mask(keys, budget=16, window=4)

        assert mask.sum() == 16
        assert mask[-4:].all()  # window preserved
        # High-energy tokens should be more likely to be kept
        # (not guaranteed due to frequency scoring, but likely)

    def test_eviction_before_quantized_write(self, model_params):
        """TriAttention mask should filter tokens before KV cache write."""
        head_dim = model_params["head_dim"]
        num_kv_heads = model_params["num_kv_heads"]
        block_size = model_params["block_size"]
        packed_dim = get_packed_dim("isoquant3", head_dim)

        k_cache = torch.zeros(4, block_size, num_kv_heads, packed_dim, dtype=torch.uint8)
        v_cache = torch.zeros_like(k_cache)

        # Create manager with TriAttention enabled
        import os
        old_env = os.environ.get("TRIATTENTION_ENABLED", "")
        os.environ["TRIATTENTION_ENABLED"] = "1"

        mgr = MultiTQKVCacheManager(
            kv_cache_dtype="isoquant3",
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            num_heads=model_params["num_heads"],
            scale=1.0 / (head_dim ** 0.5),
        )
        mgr.triattention_enabled = True
        mgr.triattention_budget = 8
        mgr.triattention_window = 2

        os.environ["TRIATTENTION_ENABLED"] = old_env

        # 16 tokens in, but only 8 should survive eviction
        key = torch.randn(16, num_kv_heads, head_dim)
        value = torch.randn(16, num_kv_heads, head_dim)
        pre_rope_key = key.clone()  # pre-RoPE for scoring
        slot_mapping = torch.arange(16, dtype=torch.int64)

        mgr.write_kv_cache(
            key, value, k_cache, v_cache, slot_mapping,
            pre_rope_key=pre_rope_key,
        )

        # Only 8 slots should have been written (budget=8)
        written_slots = 0
        for block_idx in range(k_cache.shape[0]):
            for slot in range(k_cache.shape[1]):
                if k_cache[block_idx, slot].sum() != 0:
                    written_slots += 1

        assert written_slots <= 8, f"Expected <=8 written slots, got {written_slots}"

    def test_combined_triattention_plus_quantization(self):
        """Test the full combined mode: evict tokens then compress survivors."""
        torch.manual_seed(42)
        # Simulate a 64-token sequence
        keys = torch.randn(64, 8, 128)
        values = torch.randn(64, 8, 128)

        # Step 1: TriAttention scoring
        mask = triattention_score_and_mask(keys, budget=16, window=4)
        assert mask.sum() == 16

        # Step 2: Filter
        kept_keys = keys[mask]
        kept_values = values[mask]
        assert kept_keys.shape[0] == 16

        # Step 3: Quantize survivors with IsoQuant
        k_packed = iso_encode_vectorized(kept_keys, 3)
        v_packed = iso_encode_vectorized(kept_values, 3)

        # Step 4: Decode
        k_decoded = iso_decode_vectorized(k_packed, 128, 3, torch.float32)
        v_decoded = iso_decode_vectorized(v_packed, 128, 3, torch.float32)

        assert k_decoded.shape == (16, 8, 128)
        assert v_decoded.shape == (16, 8, 128)

        # Quality check on survivors
        cos = torch.nn.functional.cosine_similarity(
            kept_keys.reshape(-1, 128), k_decoded.reshape(-1, 128), dim=-1,
        ).mean()
        assert cos > 0.95

        # Memory savings calculation
        original_bytes = 64 * 8 * 128 * 2  # 64 tokens, fp16
        compressed_bytes = k_packed.numel()  # 16 tokens, packed
        ratio = original_bytes / compressed_bytes
        assert ratio > 10, f"Combined ratio should be >10x, got {ratio:.1f}x"


# ─── Full forward pass simulation ──────────────────────────────────────────────

class TestFullForwardPass:
    """Simulate a complete vLLM-style forward pass with paged attention."""

    @pytest.mark.parametrize("method", [
        "isoquant3", "isoquant4", "planarquant3", "planarquant4",
    ])
    def test_prefill_then_decode(self, method, model_params):
        """Simulate prefill (write KV) then decode (read KV + attention)."""
        head_dim = model_params["head_dim"]
        num_kv_heads = model_params["num_kv_heads"]
        num_heads = model_params["num_heads"]
        block_size = model_params["block_size"]
        packed_dim = get_packed_dim(method, head_dim)
        scale = 1.0 / (head_dim ** 0.5)

        # Paged cache
        num_blocks = 8
        k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, packed_dim, dtype=torch.uint8)
        v_cache = torch.zeros_like(k_cache)

        mgr = MultiTQKVCacheManager(
            kv_cache_dtype=method,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale=scale,
        )

        # Phase 1: Prefill — write 32 tokens to cache
        torch.manual_seed(42)
        prefill_key = torch.randn(32, num_kv_heads, head_dim)
        prefill_value = torch.randn(32, num_kv_heads, head_dim)
        slot_mapping = torch.arange(32, dtype=torch.int64)

        mgr.write_kv_cache(prefill_key, prefill_value, k_cache, v_cache, slot_mapping)

        # Phase 2: Decode — 1 new token, attend to all 32 cached tokens
        decode_query = torch.randn(1, num_heads, head_dim)
        block_table = torch.arange(2, dtype=torch.int64).unsqueeze(0)  # 2 blocks = 32 tokens
        seq_lens = torch.tensor([32], dtype=torch.int32)
        output = torch.zeros(1, num_heads, head_dim)

        mgr.decode_attention(decode_query, k_cache, v_cache, block_table, seq_lens, output)

        # Output should be reasonable (non-zero, finite)
        assert output.abs().sum() > 0, "Decode output should be non-zero"
        assert torch.isfinite(output).all(), "Decode output should be finite"

    def test_multi_sequence_decode(self, model_params):
        """Test decode with multiple sequences of different lengths."""
        head_dim = model_params["head_dim"]
        num_kv_heads = model_params["num_kv_heads"]
        num_heads = model_params["num_heads"]
        block_size = model_params["block_size"]
        method = "isoquant3"
        packed_dim = get_packed_dim(method, head_dim)
        scale = 1.0 / (head_dim ** 0.5)

        num_blocks = 16
        k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, packed_dim, dtype=torch.uint8)
        v_cache = torch.zeros_like(k_cache)

        mgr = MultiTQKVCacheManager(
            kv_cache_dtype=method,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale=scale,
        )

        # Write 3 sequences of lengths 16, 32, 8
        torch.manual_seed(42)
        total_tokens = 16 + 32 + 8
        all_keys = torch.randn(total_tokens, num_kv_heads, head_dim)
        all_values = torch.randn(total_tokens, num_kv_heads, head_dim)

        # Seq 0: blocks 0 (16 tokens)
        # Seq 1: blocks 1-2 (32 tokens)
        # Seq 2: block 3 (8 tokens)
        slot_mapping = torch.arange(total_tokens, dtype=torch.int64)
        mgr.write_kv_cache(all_keys, all_values, k_cache, v_cache, slot_mapping)

        # Decode: 3 queries (1 per sequence)
        decode_query = torch.randn(3, num_heads, head_dim)
        block_table = torch.zeros(3, 4, dtype=torch.int64)
        block_table[0, 0] = 0          # seq 0: 1 block
        block_table[1, 0:2] = torch.tensor([1, 2])  # seq 1: 2 blocks
        block_table[2, 0] = 3          # seq 2: 1 block
        seq_lens = torch.tensor([16, 32, 8], dtype=torch.int32)
        output = torch.zeros(3, num_heads, head_dim)

        mgr.decode_attention(decode_query, k_cache, v_cache, block_table, seq_lens, output)

        # All sequences should produce output
        for b in range(3):
            assert output[b].abs().sum() > 0, f"Sequence {b} output should be non-zero"
            assert torch.isfinite(output[b]).all(), f"Sequence {b} output should be finite"


# ─── Method dispatch tests ──────────────────────────────────────────────────────

class TestMethodDispatch:
    """Test that the dispatch layer routes correctly."""

    def test_dispatch_detection(self):
        for dtype in ("turboquant25", "turboquant35"):
            assert is_multi_tq_kv_cache(dtype)
            assert get_method_family(dtype) == "turboquant"
        for dtype in ("isoquant3", "isoquant4"):
            assert is_multi_tq_kv_cache(dtype)
            assert get_method_family(dtype) == "isoquant"
        for dtype in ("planarquant3", "planarquant4"):
            assert is_multi_tq_kv_cache(dtype)
            assert get_method_family(dtype) == "planarquant"
        assert not is_multi_tq_kv_cache("auto")
        assert not is_multi_tq_kv_cache("fp8_e4m3")

    def test_create_kv_cache_manager(self):
        mgr = create_kv_cache_manager("isoquant3", 128, 8, 32, 0.1)
        assert mgr is not None
        assert mgr.family == "isoquant"
        assert mgr.bits == 3

        mgr = create_kv_cache_manager("planarquant4", 128, 8, 32, 0.1)
        assert mgr is not None
        assert mgr.family == "planarquant"
        assert mgr.bits == 4

        # TurboQuant returns None (handled by existing infrastructure)
        mgr = create_kv_cache_manager("turboquant35", 128, 8, 32, 0.1)
        assert mgr is None

    def test_packed_dim_all_methods(self):
        for dtype in ("turboquant25", "turboquant35", "isoquant3", "isoquant4",
                       "planarquant3", "planarquant4"):
            pd = get_packed_dim(dtype, 128)
            assert pd > 0
            assert pd < 128 * 2  # Must be smaller than fp16


# ─── Preset integration tests ──────────────────────────────────────────────────

class TestPresetIntegration:
    """Test that presets produce valid configs for the full pipeline."""

    @pytest.mark.parametrize("preset_name", [
        "k_only_iso", "k_only_planar", "balanced", "speed",
        "max_compression", "quality", "extreme", "long_context",
        "no_calibration_symmetric", "no_calibration_quality",
        "godzilla_kvarn4", "godzilla_kvarn2_max", "godzilla_kvarn8_quality",
    ])
    def test_preset_config_valid(self, preset_name):
        config = get_preset(preset_name)
        warnings = config.validate()
        # Warnings are OK (head_dim specific), errors would throw
        assert config.k_method is not None
        assert config.v_method is not None

        # Estimate bytes should be positive
        bytes_4k = config.estimate_kv_bytes(4096)
        assert bytes_4k > 0

    def test_extreme_preset_has_triattention(self):
        config = get_preset("extreme")
        assert config.triattention_enabled
        assert config.triattention_budget > 0
        assert config.k_method == CacheMethod.TURBO3_TCQ
        assert config.v_method == CacheMethod.TURBO3_TCQ


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
