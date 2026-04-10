# SPDX-License-Identifier: MIT
"""Multi-TurboQuant — Unified KV cache compression for LLM inference.

Combines TurboQuant (WHT), TCQ, IsoQuant (quaternion), PlanarQuant (Givens),
and TriAttention (token eviction) under a single API.

Quick start:
    from multi_turboquant import CacheConfig, CacheMethod, compress, decompress

    # K-only IsoQuant (beats FP16 speed, no calibration needed)
    config = CacheConfig(k_method=CacheMethod.ISO3, v_method=CacheMethod.FP16)

    # Symmetric TurboQuant (5x compression, fastest on Ampere)
    config = CacheConfig(k_method=CacheMethod.TURBO3, v_method=CacheMethod.TURBO3)

    # Use a named preset
    from multi_turboquant import get_preset
    config = get_preset("balanced")  # turbo3_tcq symmetric

    # Encode/decode KV vectors
    compressed = compress(kv_tensor, config, layer_idx=0)
    reconstructed = decompress(compressed, config)
"""

__version__ = "0.1.0"

from .config import CacheConfig, CacheMethod, MethodFamily, METHOD_BITS, METHOD_COMPRESSION
from .registry import get_method, list_methods, registered_methods, is_registered
from .presets import get_preset, list_presets, recommend_preset, PRESETS
from .planner import plan_agents, plan_scenarios, PlanResult

# Import methods to trigger registration
from . import methods as _methods  # noqa: F401

from .methods.base import CompressionMethod, CompressedKV, MethodInfo


def compress(
    x,  # torch.Tensor [..., num_heads, head_dim]
    config: CacheConfig,
    *,
    layer_idx: int = 0,
    head_indices=None,  # torch.Tensor | None
    which: str = "k",   # "k" or "v"
):
    """Compress KV vectors using the method specified in config.

    Args:
        x: Input tensor.
        config: Cache configuration.
        layer_idx: Layer index for per-layer calibration.
        head_indices: Optional outlier mask indices.
        which: "k" for key cache, "v" for value cache.

    Returns:
        CompressedKV from the appropriate method.
    """
    method_enum = config.k_method if which == "k" else config.v_method

    # Thread config-level kwargs into method construction
    method_kwargs = {}
    if config.turboquant_metadata_path:
        method_kwargs["metadata_path"] = config.turboquant_metadata_path
    if hasattr(config, "head_dim") and config.head_dim:
        method_kwargs["head_dim"] = config.head_dim

    method = get_method(method_enum, **method_kwargs)
    return method.encode(x, head_indices=head_indices, layer_idx=layer_idx)


def decompress(compressed: CompressedKV, config: CacheConfig = None, *, dtype=None):
    """Decompress KV vectors.

    Args:
        compressed: CompressedKV from compress().
        config: Optional config (method is stored in CompressedKV).
        dtype: Output dtype (default: torch.float16).

    Returns:
        Reconstructed tensor.
    """
    import torch
    if dtype is None:
        dtype = torch.float16
    method = get_method(compressed.method)
    return method.decode(compressed, dtype=dtype)


__all__ = [
    # Core API
    "compress",
    "decompress",
    "CacheConfig",
    "CacheMethod",
    "MethodFamily",
    # Registry
    "get_method",
    "list_methods",
    "registered_methods",
    "is_registered",
    # Presets
    "get_preset",
    "list_presets",
    "recommend_preset",
    "PRESETS",
    # Planner
    "plan_agents",
    "plan_scenarios",
    "PlanResult",
    # Types
    "CompressionMethod",
    "CompressedKV",
    "MethodInfo",
    # Constants
    "METHOD_BITS",
    "METHOD_COMPRESSION",
    # Version
    "__version__",
]
