# SPDX-License-Identifier: MIT
"""Tests for all KV cache compression methods.

Tests encode/decode roundtrip, quality, packing correctness,
and method registration for all supported methods.
"""

import pytest
import torch

from multi_turboquant import (
    CacheConfig,
    CacheMethod,
    compress,
    decompress,
    get_method,
    list_methods,
    registered_methods,
    get_preset,
    list_presets,
    recommend_preset,
)
from multi_turboquant.methods.base import CompressedKV
from multi_turboquant.config import METHOD_BITS, CALIBRATION_FREE


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def random_kv():
    """Random KV tensor [seq_len, num_heads, head_dim]."""
    torch.manual_seed(42)
    return torch.randn(64, 8, 128, dtype=torch.float32)


@pytest.fixture
def small_kv():
    """Small KV tensor for quick tests."""
    torch.manual_seed(42)
    return torch.randn(4, 2, 128, dtype=torch.float32)


# ─── Registry tests ────────────────────────────────────────────────────────────

class TestRegistry:
    def test_all_methods_registered(self):
        methods = registered_methods()
        # At minimum, the core methods should be registered
        assert CacheMethod.TURBO3 in methods
        assert CacheMethod.TURBO3_TCQ in methods
        assert CacheMethod.ISO3 in methods
        assert CacheMethod.PLANAR3 in methods
        assert CacheMethod.TRIATTENTION in methods

    def test_list_methods_returns_info(self):
        infos = list_methods()
        assert len(infos) > 0
        for info in infos:
            assert info.bits > 0
            assert info.transform_name
            assert info.description

    def test_get_method_returns_instance(self):
        method = get_method(CacheMethod.TURBO3)
        assert method is not None
        info = method.info()
        assert info.method == CacheMethod.TURBO3

    def test_unknown_method_raises(self):
        with pytest.raises(KeyError):
            get_method(CacheMethod.FP16)  # FP16 is baseline, not registered as compression


# ─── TurboQuant tests ──────────────────────────────────────────────────────────

class TestTurboQuant:
    @pytest.mark.parametrize("method", [
        CacheMethod.TURBO2, CacheMethod.TURBO3, CacheMethod.TURBO4,
    ])
    def test_roundtrip(self, small_kv, method):
        m = get_method(method)
        compressed = m.encode(small_kv)
        decoded = m.decode(compressed, dtype=torch.float32)
        assert decoded.shape == small_kv.shape
        # Lossy compression — check reasonable reconstruction
        cos_sim = torch.nn.functional.cosine_similarity(
            small_kv.reshape(-1, 128), decoded.reshape(-1, 128), dim=-1,
        ).mean()
        assert cos_sim > 0.5, f"Cosine similarity too low: {cos_sim}"

    def test_packed_dim(self):
        m = get_method(CacheMethod.TURBO3)
        pd = m.packed_dim(128)
        assert pd > 0
        assert pd < 128 * 2  # Should be smaller than fp16

    def test_compression_ratio(self):
        m = get_method(CacheMethod.TURBO3)
        ratio = m.compression_ratio()
        assert 4.0 < ratio < 6.0  # ~5x for turbo3


# ─── TCQ tests ──────────────────────────────────────────────────────────────────

class TestTCQ:
    @pytest.mark.parametrize("method", [
        CacheMethod.TURBO2_TCQ, CacheMethod.TURBO3_TCQ,
    ])
    def test_roundtrip(self, small_kv, method):
        m = get_method(method)
        compressed = m.encode(small_kv)
        decoded = m.decode(compressed, dtype=torch.float32)
        assert decoded.shape == small_kv.shape

    def test_tcq_better_quality_than_base(self, small_kv):
        """TCQ should have equal or better quality than base TurboQuant."""
        base = get_method(CacheMethod.TURBO3)
        tcq = get_method(CacheMethod.TURBO3_TCQ)

        base_compressed = base.encode(small_kv)
        tcq_compressed = tcq.encode(small_kv)

        base_decoded = base.decode(base_compressed, dtype=torch.float32)
        tcq_decoded = tcq.decode(tcq_compressed, dtype=torch.float32)

        base_mse = (small_kv - base_decoded).square().mean()
        tcq_mse = (small_kv - tcq_decoded).square().mean()

        # TCQ should be at least as good (lower MSE)
        # Allow some tolerance for edge cases
        assert tcq_mse <= base_mse * 1.1, (
            f"TCQ MSE ({tcq_mse:.6f}) should be <= base MSE ({base_mse:.6f})"
        )


# ─── IsoQuant tests ─────────────────────────────────────────────────────────────

class TestIsoQuant:
    @pytest.mark.parametrize("method", [CacheMethod.ISO3, CacheMethod.ISO4])
    def test_roundtrip(self, small_kv, method):
        m = get_method(method)
        compressed = m.encode(small_kv)
        decoded = m.decode(compressed, dtype=torch.float32)
        assert decoded.shape == small_kv.shape

    def test_no_calibration_needed(self):
        assert CacheMethod.ISO3 in CALIBRATION_FREE
        assert CacheMethod.ISO4 in CALIBRATION_FREE

    def test_quaternion_orthogonality(self):
        """The quaternion rotation should be orthogonal (R^T R = I)."""
        from multi_turboquant.methods.isoquant import _rotation_matrix_4d
        R = _rotation_matrix_4d("cpu", None)
        identity = torch.eye(4)
        product = R.T @ R
        assert torch.allclose(product, identity, atol=1e-6)


# ─── PlanarQuant tests ──────────────────────────────────────────────────────────

class TestPlanarQuant:
    @pytest.mark.parametrize("method", [CacheMethod.PLANAR3, CacheMethod.PLANAR4])
    def test_roundtrip(self, small_kv, method):
        m = get_method(method)
        compressed = m.encode(small_kv)
        decoded = m.decode(compressed, dtype=torch.float32)
        assert decoded.shape == small_kv.shape

    def test_givens_invertibility(self):
        """Givens rotation should be perfectly invertible."""
        from multi_turboquant.methods.planarquant import (
            _apply_givens_rotation,
        )
        x = torch.randn(1, 4, 128)
        rotated = _apply_givens_rotation(x)
        recovered = _apply_givens_rotation(rotated, inverse=True)
        assert torch.allclose(x, recovered, atol=1e-5)

    def test_metal_support(self):
        m = get_method(CacheMethod.PLANAR3)
        assert m.supports_metal()


# ─── TriAttention tests ─────────────────────────────────────────────────────────

class TestTriAttention:
    def test_frequency_scoring(self, small_kv):
        from multi_turboquant.methods.triattention import compute_frequency_scores
        scores = compute_frequency_scores(small_kv)
        assert scores.shape == (small_kv.shape[0], small_kv.shape[1])
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_token_selection(self):
        from multi_turboquant.methods.triattention import select_tokens
        scores = torch.rand(32, 8)  # 32 tokens, 8 heads
        mask = select_tokens(scores, budget=16, window=4)
        assert mask.shape == (32,)
        assert mask.sum() <= 16

    def test_under_budget_keeps_all(self):
        from multi_turboquant.methods.triattention import select_tokens
        scores = torch.rand(8, 4)
        mask = select_tokens(scores, budget=32, window=4)
        assert mask.all()  # All tokens kept when under budget


# ─── Config tests ───────────────────────────────────────────────────────────────

class TestConfig:
    def test_symmetric_config(self):
        config = CacheConfig(
            k_method=CacheMethod.TURBO3,
            v_method=CacheMethod.TURBO3,
        )
        assert config.is_symmetric
        assert not config.is_k_only

    def test_k_only_config(self):
        config = CacheConfig(
            k_method=CacheMethod.ISO3,
            v_method=CacheMethod.FP16,
        )
        assert not config.is_symmetric
        assert config.is_k_only

    def test_compression_ratio(self):
        config = CacheConfig(
            k_method=CacheMethod.TURBO3,
            v_method=CacheMethod.TURBO3,
        )
        assert config.k_compression > 4.0
        assert config.v_compression > 4.0

    def test_estimate_bytes(self):
        config = CacheConfig(
            k_method=CacheMethod.FP16,
            v_method=CacheMethod.FP16,
            head_dim=128,
            num_kv_heads=8,
            num_layers=32,
        )
        bytes_4k = config.estimate_kv_bytes(4096)
        assert bytes_4k > 0

        config_compressed = CacheConfig(
            k_method=CacheMethod.TURBO3,
            v_method=CacheMethod.TURBO3,
            head_dim=128,
            num_kv_heads=8,
            num_layers=32,
        )
        bytes_compressed = config_compressed.estimate_kv_bytes(4096)
        assert bytes_compressed < bytes_4k

    def test_validate_warns_unsupported_head_dim(self):
        config = CacheConfig(
            k_method=CacheMethod.ISO3,
            v_method=CacheMethod.FP16,
            head_dim=256,
        )
        warnings = config.validate()
        assert len(warnings) > 0


# ─── Preset tests ───────────────────────────────────────────────────────────────

class TestPresets:
    def test_all_presets_exist(self):
        presets = list_presets()
        assert "balanced" in presets
        assert "speed" in presets
        assert "k_only_iso" in presets
        assert "extreme" in presets

    def test_get_preset(self):
        config = get_preset("balanced")
        assert config.k_method == CacheMethod.TURBO3_TCQ
        assert config.v_method == CacheMethod.TURBO3_TCQ

    def test_recommend_preset(self):
        name = recommend_preset(
            vram_gb=24.0,
            model_size_b=7.0,
            context_length=4096,
            has_calibration=True,
        )
        assert name in list_presets()

    def test_recommend_extreme_for_tight_vram(self):
        name = recommend_preset(
            vram_gb=12.0,
            model_size_b=70.0,
            context_length=32768,
        )
        assert name in ("extreme", "max_compression")


# ─── Integration tests ─────────────────────────────────────────────────────────

class TestIntegration:
    def test_llamacpp_args(self):
        from multi_turboquant.integration import get_llamacpp_args
        config = CacheConfig(
            k_method=CacheMethod.ISO3,
            v_method=CacheMethod.FP16,
        )
        args = get_llamacpp_args(config)
        assert "--cache-type-k" in args
        assert "iso3" in args
        assert "--cache-type-v" in args
        assert "f16" in args

    def test_llamacpp_command(self):
        from multi_turboquant.integration import get_llamacpp_command
        config = CacheConfig(
            k_method=CacheMethod.TURBO3,
            v_method=CacheMethod.TURBO3,
        )
        cmd = get_llamacpp_command(
            config,
            model_path="/opt/models/test.gguf",
            port=8080,
        )
        assert cmd[0] == "llama-server"
        assert "/opt/models/test.gguf" in cmd
        assert "8080" in cmd

    def test_bridge_adapter(self):
        from multi_turboquant.integration import BridgeAdapter
        config = CacheConfig(
            k_method=CacheMethod.TURBO3_TCQ,
            v_method=CacheMethod.TURBO3_TCQ,
        )
        adapter = BridgeAdapter(config)
        status = adapter.get_status()
        assert status["k_method"] == "turbo3_tcq"
        assert status["is_symmetric"]
        assert status["preset"] == "balanced"

    def test_bridge_ui_options(self):
        from multi_turboquant.integration import BridgeAdapter
        config = CacheConfig()
        adapter = BridgeAdapter(config)
        options = adapter.get_ui_options()
        assert len(options) > 0
        assert all("value" in o for o in options)


# ─── Compress/decompress API tests ──────────────────────────────────────────────

class TestAPI:
    def test_compress_decompress(self, small_kv):
        config = CacheConfig(
            k_method=CacheMethod.ISO3,
            v_method=CacheMethod.FP16,
        )
        compressed = compress(small_kv, config, which="k")
        assert isinstance(compressed, CompressedKV)
        decoded = decompress(compressed)
        assert decoded.shape == small_kv.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
