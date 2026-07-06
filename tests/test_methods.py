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
from multi_turboquant.config import (
    CALIBRATION_FREE,
    METHOD_BITS,
    METHOD_FAMILIES,
    SUPPORTED_HEAD_DIMS,
    MethodFamily,
)


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
        assert CacheMethod.ROTOR3 in methods
        assert CacheMethod.ROTOR4 in methods
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


# ─── RotorQuant tests ──────────────────────────────────────────────────────────

class TestRotorQuant:
    @pytest.mark.parametrize("method", [CacheMethod.ROTOR3, CacheMethod.ROTOR4])
    def test_roundtrip(self, small_kv, method):
        m = get_method(method)
        compressed = m.encode(small_kv)
        decoded = m.decode(compressed, dtype=torch.float32)
        assert decoded.shape == small_kv.shape

    def test_roundtrip_quality_rotor3(self, random_kv):
        m = get_method(CacheMethod.ROTOR3)
        compressed = m.encode(random_kv)
        decoded = m.decode(compressed, dtype=torch.float32)
        cos_sim = torch.nn.functional.cosine_similarity(
            random_kv.reshape(-1, 128), decoded.reshape(-1, 128), dim=-1,
        ).mean()
        assert cos_sim > 0.97, f"rotor3 cosine similarity too low: {cos_sim}"

    def test_roundtrip_quality_rotor4(self, random_kv):
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            m = get_method(CacheMethod.ROTOR4)
        compressed = m.encode(random_kv)
        decoded = m.decode(compressed, dtype=torch.float32)
        cos_sim = torch.nn.functional.cosine_similarity(
            random_kv.reshape(-1, 128), decoded.reshape(-1, 128), dim=-1,
        ).mean()
        assert cos_sim > 0.99, f"rotor4 cosine similarity too low: {cos_sim}"

    def test_rotor4_marked_experimental(self):
        """MethodInfo.experimental must be True for rotor4, False for rotor3."""
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            rotor3 = get_method(CacheMethod.ROTOR3)
            rotor4 = get_method(CacheMethod.ROTOR4)
        assert rotor3.info().experimental is False
        assert rotor4.info().experimental is True

    def test_rotor4_emits_experimental_warning(self):
        """Fresh warning state: constructing rotor4 must emit a UserWarning."""
        from multi_turboquant.methods import rotorquant as _rq
        _rq._EXPERIMENTAL_WARNED.discard(CacheMethod.ROTOR4)
        with pytest.warns(UserWarning, match="experimental"):
            get_method(CacheMethod.ROTOR4)

    def test_no_calibration_needed(self):
        assert CacheMethod.ROTOR3 in CALIBRATION_FREE
        assert CacheMethod.ROTOR4 in CALIBRATION_FREE

    def test_rotation_orthogonality(self):
        """The 3x3 SO(3) rotation should be orthogonal (R^T R = I)."""
        from multi_turboquant.methods.rotorquant import _rotation_matrix_3d
        R = _rotation_matrix_3d("cpu", None)
        identity = torch.eye(3)
        product = R.T @ R
        assert torch.allclose(product, identity, atol=1e-6)

    def test_rotation_proper_rotation(self):
        """det(R) should be +1 for a proper SO(3) rotation (not a reflection)."""
        from multi_turboquant.methods.rotorquant import _rotation_matrix_3d
        R = _rotation_matrix_3d("cpu", None)
        assert torch.allclose(torch.linalg.det(R), torch.tensor(1.0), atol=1e-6)

    def test_sandwich_invertibility(self):
        """Rotor sandwich then inverse should recover the input exactly."""
        from multi_turboquant.methods.rotorquant import _apply_rotor_sandwich
        x = torch.randn(1, 4, 129)  # must be multiple of 3
        rotated = _apply_rotor_sandwich(x)
        recovered = _apply_rotor_sandwich(rotated, inverse=True)
        assert torch.allclose(x, recovered, atol=1e-5)

    def test_padding_head_dim_128(self, small_kv):
        """head_dim=128 is not divisible by 3 — verify padding round-trips cleanly."""
        m = get_method(CacheMethod.ROTOR3)
        compressed = m.encode(small_kv)
        decoded = m.decode(compressed, dtype=torch.float32)
        # Output head_dim must match input (padding was stripped)
        assert decoded.shape == small_kv.shape
        assert decoded.shape[-1] == 128

    def test_padding_head_dim_64(self):
        """head_dim=64 also needs padding (64 % 3 == 1 -> pad to 66)."""
        torch.manual_seed(7)
        x = torch.randn(4, 2, 64, dtype=torch.float32)
        m = get_method(CacheMethod.ROTOR3)
        compressed = m.encode(x)
        decoded = m.decode(compressed, dtype=torch.float32)
        assert decoded.shape == x.shape

    def test_experimental_warning_rotor4_only(self):
        """rotor3 must NOT emit the experimental warning."""
        # Fresh warning state: call rotor3 and assert no UserWarning
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            get_method(CacheMethod.ROTOR3)  # should not raise


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

    def test_kvarn_metadata(self):
        kvarn_methods = [
            CacheMethod.KVARN2, CacheMethod.KVARN3, CacheMethod.KVARN4,
            CacheMethod.KVARN5, CacheMethod.KVARN6, CacheMethod.KVARN8,
        ]
        for method in kvarn_methods:
            assert METHOD_FAMILIES[method] == MethodFamily.KVARN
            assert method in CALIBRATION_FREE
            assert METHOD_BITS[method] == float(method.value.removeprefix("kvarn"))
        assert SUPPORTED_HEAD_DIMS[MethodFamily.KVARN] == (128, 256, 384, 512)

        config = CacheConfig(
            k_method=CacheMethod.KVARN4,
            v_method=CacheMethod.KVARN4,
            head_dim=64,
        )
        warnings = config.validate()
        assert any("head_dim=64" in warning for warning in warnings)


# ─── Preset tests ───────────────────────────────────────────────────────────────

class TestPresets:
    def test_all_presets_exist(self):
        presets = list_presets()
        assert "balanced" in presets
        assert "speed" in presets
        assert "k_only_iso" in presets
        assert "extreme" in presets
        assert "godzilla_kvarn4" in presets
        assert "godzilla_kvarn2_max" in presets
        assert "godzilla_kvarn8_quality" in presets

    def test_get_preset(self):
        config = get_preset("balanced")
        assert config.k_method == CacheMethod.TURBO3_TCQ
        assert config.v_method == CacheMethod.TURBO3_TCQ

    def test_godzilla_kvarn_presets(self):
        config = get_preset("godzilla_kvarn4")
        assert config.k_method == CacheMethod.KVARN4
        assert config.v_method == CacheMethod.KVARN4

        config = get_preset("godzilla_kvarn2_max")
        assert config.k_method == CacheMethod.KVARN2
        assert config.v_method == CacheMethod.KVARN2

        config = get_preset("godzilla_kvarn8_quality")
        assert config.k_method == CacheMethod.KVARN8
        assert config.v_method == CacheMethod.KVARN8

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

    def test_llamacpp_context_extension_args(self):
        from multi_turboquant.integration import (
            LlamaCppContextExtensionConfig,
            get_llamacpp_args,
            get_llamacpp_command,
        )

        config = CacheConfig(
            k_method=CacheMethod.ISO3,
            v_method=CacheMethod.FP16,
        )
        context_extension = LlamaCppContextExtensionConfig(
            rope_scale=8,
            yarn_orig_ctx=4096,
        )

        args = get_llamacpp_args(
            config,
            context_size=32768,
            context_extension=context_extension,
        )

        assert args[args.index("-c") + 1] == "32768"
        assert args[args.index("--rope-scaling") + 1] == "yarn"
        assert args[args.index("--rope-scale") + 1] == "8"
        assert args[args.index("--yarn-orig-ctx") + 1] == "4096"

        cmd = get_llamacpp_command(
            config,
            model_path="/opt/models/test.gguf",
            context_size=32768,
            context_extension=context_extension,
        )
        assert cmd[cmd.index("--rope-scaling") + 1] == "yarn"
        assert cmd[cmd.index("--host") + 1] == "0.0.0.0"

    def test_llamacpp_rejects_invalid_context_extension_args(self):
        from multi_turboquant.integration import (
            LlamaCppContextExtensionConfig,
            get_llamacpp_args,
        )

        config = CacheConfig()

        with pytest.raises(ValueError, match="rope_scale or rope_freq_scale"):
            get_llamacpp_args(
                config,
                context_extension=LlamaCppContextExtensionConfig(
                    rope_scale=8,
                    rope_freq_scale=0.125,
                ),
            )
        with pytest.raises(ValueError, match="YaRN options require"):
            get_llamacpp_args(
                config,
                context_extension=LlamaCppContextExtensionConfig(
                    rope_scaling="linear",
                    yarn_orig_ctx=4096,
                ),
            )
        with pytest.raises(ValueError, match="extra_args cannot contain blanks"):
            get_llamacpp_args(
                config,
                context_extension=LlamaCppContextExtensionConfig(
                    extra_args=("  ",),
                ),
            )

    def test_llamacpp_patched_triattention_args(self):
        from multi_turboquant.integration import get_llamacpp_args
        config = CacheConfig(
            k_method=CacheMethod.TURBO3,
            v_method=CacheMethod.TURBO3,
            triattention_enabled=True,
            use_custom_triattention_llamacpp=True,
            triattention_stats_path="model.triattention",
            triattention_budget=2048,
            triattention_window=256,
            triattention_log=True,
        )
        args = get_llamacpp_args(config)
        assert "--cache-type-k" in args
        assert "turbo3" in args
        assert "--triattention-stats" in args
        assert "model.triattention" in args
        assert "--triattention-budget" in args
        assert "2048" in args
        assert "--triattention-window" in args
        assert "256" in args
        assert "--triattention-log" in args

    def test_llamacpp_rejects_triattention_as_cache_type(self):
        from multi_turboquant.integration import get_llamacpp_args
        config = CacheConfig(
            k_method=CacheMethod.TRIATTENTION,
            v_method=CacheMethod.FP16,
            triattention_enabled=True,
            use_custom_triattention_llamacpp=True,
            triattention_stats_path="model.triattention",
        )
        with pytest.raises(ValueError, match="token eviction"):
            get_llamacpp_args(config)

    def test_llamacpp_godzilla_kvarn_args(self):
        from multi_turboquant.integration import LlamaCppProfile, get_llamacpp_args

        config = CacheConfig(
            k_method=CacheMethod.KVARN4,
            v_method=CacheMethod.KVARN4,
        )
        args = get_llamacpp_args(config, fork_profile=LlamaCppProfile.GODZILLA)
        assert args == [
            "--cache-type-k", "kvarn4",
            "--cache-type-v", "kvarn4",
            "-fa", "on",
        ]

    def test_llamacpp_rejects_kvarn_without_godzilla_profile(self):
        from multi_turboquant.integration import get_llamacpp_args

        config = CacheConfig(
            k_method=CacheMethod.KVARN4,
            v_method=CacheMethod.KVARN4,
        )
        with pytest.raises(ValueError, match="fork_profile='godzilla'"):
            get_llamacpp_args(config)

    def test_llamacpp_rejects_mixed_kvarn_config(self):
        from multi_turboquant.integration import get_llamacpp_args

        config = CacheConfig(
            k_method=CacheMethod.KVARN4,
            v_method=CacheMethod.FP16,
        )
        with pytest.raises(ValueError, match="both K and V"):
            get_llamacpp_args(config, fork_profile="godzilla")

    def test_llamacpp_rejects_kvarn_with_triattention(self):
        from multi_turboquant.integration import get_llamacpp_args

        config = CacheConfig(
            k_method=CacheMethod.KVARN4,
            v_method=CacheMethod.KVARN4,
            triattention_enabled=True,
        )
        with pytest.raises(ValueError, match="KVX-2"):
            get_llamacpp_args(config, fork_profile="godzilla")

    def test_llamacpp_godzilla_speculative_args(self):
        from multi_turboquant.integration import (
            LlamaCppSpeculativeConfig,
            get_llamacpp_args,
        )

        config = CacheConfig(k_method=CacheMethod.ISO3, v_method=CacheMethod.FP16)
        speculative = LlamaCppSpeculativeConfig(
            spec_type="dflash",
            draft_model="draft.gguf",
            draft_gpu_layers="all",
            draft_n_max=16,
            branch_budget=0,
            dflash_cross_ctx=512,
        )
        args = get_llamacpp_args(
            config,
            fork_profile="godzilla",
            speculative=speculative,
        )
        assert "--spec-type" in args
        assert "dflash" in args
        assert "--spec-draft-model" in args
        assert "draft.gguf" in args
        assert "--spec-draft-n-max" in args
        assert "16" in args
        assert "--spec-branch-budget" in args
        assert "0" in args
        assert "--spec-dflash-cross-ctx" in args
        assert "512" in args

    def test_llamacpp_rejects_invalid_speculative_args(self):
        from multi_turboquant.integration import (
            LlamaCppSpeculativeConfig,
            get_llamacpp_args,
        )

        config = CacheConfig(k_method=CacheMethod.ISO3, v_method=CacheMethod.FP16)
        with pytest.raises(ValueError, match="DFlash requires"):
            get_llamacpp_args(
                config,
                fork_profile="godzilla",
                speculative=LlamaCppSpeculativeConfig(spec_type="dflash"),
            )
        with pytest.raises(ValueError, match="draft cache type cannot use KVarN"):
            get_llamacpp_args(
                config,
                fork_profile="godzilla",
                speculative=LlamaCppSpeculativeConfig(
                    spec_type="dflash",
                    draft_model="draft.gguf",
                    draft_cache_type_k=CacheMethod.KVARN4,
                ),
            )

    def test_compatibility_godzilla_kvarn(self):
        from multi_turboquant.compatibility import check_config, get_available_methods
        from multi_turboquant.hardware import GPU, PlatformInfo

        cuda = PlatformInfo(
            os="linux",
            arch="x86_64",
            gpus=[GPU(0, "cuda", 24576, vendor="nvidia", compute="cuda")],
            cuda_available=True,
        )
        rocm = PlatformInfo(
            os="linux",
            arch="x86_64",
            gpus=[GPU(0, "amd", 24576, vendor="amd", compute="rocm")],
            rocm_available=True,
        )
        config = CacheConfig(
            k_method=CacheMethod.KVARN4,
            v_method=CacheMethod.KVARN4,
        )

        assert any("Godzilla" in i.message for i in check_config(config, cuda))
        assert check_config(config, cuda, fork_profile="godzilla") == []
        assert CacheMethod.KVARN4 not in get_available_methods(cuda)
        assert CacheMethod.KVARN4 in get_available_methods(
            cuda,
            fork_profile="godzilla",
        )
        assert any(
            "requires CUDA" in i.message
            for i in check_config(config, rocm, fork_profile="godzilla")
        )

    def test_llamacpp_cuda_weight_share_wrapper(self):
        from multi_turboquant.integration import (
            CudaWeightShareConfig,
            get_llamacpp_command,
        )
        config = CacheConfig()
        cmd = get_llamacpp_command(
            config,
            model_path="/opt/models/test.gguf",
            cuda_weight_share=CudaWeightShareConfig(
                enabled=True,
                library_path="/opt/cuda-llm-weight-share.so",
                model_size_bytes=123456,
                model_size_tolerance=1024,
                ipc_name="/cuda_vram_ipc_test",
            ),
        )
        assert cmd[0] == "env"
        assert "LD_PRELOAD=/opt/cuda-llm-weight-share.so" in cmd
        assert "MODEL_SIZE=123456" in cmd
        assert "MODEL_SIZE_TOLERANCE=1024" in cmd
        assert "CUDA_VRAM_IPC_NAME=/cuda_vram_ipc_test" in cmd
        assert "llama-server" in cmd

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
