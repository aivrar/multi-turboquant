# SPDX-License-Identifier: MIT
"""Monkeypatch vLLM to support all Multi-TurboQuant compression methods.

Replaces the shell-script-based approach (setup_vllm.sh) with a Python
runtime monkeypatch that can inject any method.

Usage:
    from multi_turboquant.integration import patch_vllm
    from multi_turboquant import CacheConfig, CacheMethod

    config = CacheConfig(k_method=CacheMethod.TURBO3_TCQ, v_method=CacheMethod.TURBO3_TCQ)
    patch_vllm(config)

    # Now start vLLM normally — it will use TurboQuant KV cache
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from ..config import CacheConfig, CacheMethod, MethodFamily, METHOD_FAMILIES

_PATCHED = False
_PATCH_CONFIG: CacheConfig | None = None


def patch_vllm(config: CacheConfig, *, force: bool = False) -> bool:
    """Monkeypatch vLLM to support the configured compression method.

    This must be called BEFORE importing vLLM's attention modules.

    Args:
        config: Cache configuration to apply.
        force: Re-apply even if already patched.

    Returns:
        True if patching was successful.
    """
    global _PATCHED, _PATCH_CONFIG

    if _PATCHED and not force:
        return True

    _check_rotor_unsupported(config)

    try:
        import vllm
    except ImportError:
        raise ImportError(
            "vLLM is not installed. Install with: pip install vllm"
        )

    # Determine which method families need patching
    families = {METHOD_FAMILIES[config.k_method], METHOD_FAMILIES[config.v_method]}

    if MethodFamily.BASELINE in families and len(families) == 1:
        return True  # No patching needed for baseline

    # --- Patch CacheDType enum ---
    _patch_cache_dtype_enum(config)

    # --- Patch attention backend ---
    if MethodFamily.TURBOQUANT in families or MethodFamily.TCQ in families:
        _patch_turboquant_attention(config)

    if MethodFamily.ISOQUANT in families:
        _patch_isoquant_attention(config)

    if MethodFamily.PLANARQUANT in families:
        _patch_planarquant_attention(config)

    # --- Install unified Multi-TQ attention backend ---
    if families - {MethodFamily.BASELINE}:
        _install_multi_tq_backend(config)

    # --- Patch TriAttention (if enabled) ---
    if config.triattention_enabled:
        _patch_triattention(config)
        # Set environment variables for the attention backend
        import os
        os.environ["TRIATTENTION_ENABLED"] = "1"
        os.environ["TRIATTENTION_BUDGET"] = str(config.triattention_budget)
        os.environ["TRIATTENTION_WINDOW"] = str(config.triattention_window)
        if config.triattention_stats_path:
            os.environ["TRIATTENTION_STATS_PATH"] = config.triattention_stats_path

    _PATCHED = True
    _PATCH_CONFIG = config
    return True


def is_vllm_patched() -> bool:
    """Check if vLLM has been patched."""
    return _PATCHED


def get_patch_config() -> CacheConfig | None:
    """Return the config used for patching, if any."""
    return _PATCH_CONFIG


def _patch_cache_dtype_enum(config: CacheConfig) -> None:
    """Add custom cache dtype entries to vLLM's CacheDType enum."""
    try:
        # vLLM v1 attention
        from vllm.v1.attention.backends import utils as attn_utils
        if hasattr(attn_utils, "CacheDType"):
            dtype_enum = attn_utils.CacheDType
            # Add entries for each configured method
            _add_dtype_entries(dtype_enum, config)
    except (ImportError, AttributeError):
        pass


def _add_dtype_entries(dtype_enum: Any, config: CacheConfig) -> None:
    """Dynamically add dtype entries to vLLM's enum."""
    method_to_dtype = {
        CacheMethod.TURBO2: "turboquant25",
        CacheMethod.TURBO3: "turboquant35",
        CacheMethod.TURBO4: "turboquant35",
        CacheMethod.TURBO2_TCQ: "turboquant25",
        CacheMethod.TURBO3_TCQ: "turboquant35",
        CacheMethod.ISO3: "isoquant3",
        CacheMethod.ISO4: "isoquant4",
        CacheMethod.PLANAR3: "planarquant3",
        CacheMethod.PLANAR4: "planarquant4",
    }
    for method in [config.k_method, config.v_method]:
        dtype_name = method_to_dtype.get(method)
        if dtype_name and not hasattr(dtype_enum, dtype_name):
            try:
                dtype_enum._value2member_map_[dtype_name] = dtype_name
            except (AttributeError, TypeError):
                pass


def _install_multi_tq_backend(config: CacheConfig) -> None:
    """Install the unified Multi-TQ attention backend into vLLM.

    This makes the MultiTQKVCacheManager and dispatch functions
    available to vLLM's attention infrastructure.
    """
    try:
        from ..kernels.triton.attention_backend import is_multi_tq_kv_cache
        from ..kernels.triton.multi_tq_attention import (
            MultiTQKVCacheManager,
            create_kv_cache_manager,
        )

        # Register in vLLM's ops namespace so the attention backend can find it
        ops_module_path = "vllm.v1.attention.ops"
        ops_module = sys.modules.get(ops_module_path)
        if ops_module is not None:
            setattr(ops_module, "multi_tq_attention",
                    importlib.import_module("multi_turboquant.kernels.triton.multi_tq_attention"))
            setattr(ops_module, "multi_tq_backend",
                    importlib.import_module("multi_turboquant.kernels.triton.attention_backend"))
            import logging
            logging.getLogger(__name__).info(
                "Multi-TurboQuant unified backend installed into vLLM ops"
            )
        else:
            import logging
            logging.getLogger(__name__).info(
                "Multi-TurboQuant backend available (vLLM ops module not yet loaded)"
            )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "Failed to install Multi-TQ backend: %s", e
        )


def _patch_turboquant_attention(config: CacheConfig) -> None:
    """Inject TurboQuant Triton kernels into vLLM's attention path."""
    try:
        # Attempt to inject into vLLM's ops module
        ops_module_path = "vllm.v1.attention.ops"
        if ops_module_path not in sys.modules:
            try:
                importlib.import_module(ops_module_path)
            except ImportError:
                return

        ops_module = sys.modules.get(ops_module_path)
        if ops_module is None:
            return

        # Inject our kv_cache module
        from ..kernels.triton import turboquant_decode, turboquant_update
        setattr(ops_module, "turboquant_kv_cache",
                importlib.import_module("multi_turboquant.methods.turboquant"))
        setattr(ops_module, "turboquant_decode", turboquant_decode)
        setattr(ops_module, "turboquant_update", turboquant_update)

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "TurboQuant vLLM patch failed (non-fatal): %s. "
            "vLLM may have a different internal structure than expected.", e
        )


def _patch_isoquant_attention(config: CacheConfig) -> None:
    """Inject IsoQuant kernels into vLLM's attention path."""
    try:
        ops_module_path = "vllm.v1.attention.ops"
        ops_module = sys.modules.get(ops_module_path)
        if ops_module is None:
            return

        from ..kernels.triton import isoquant_ops
        setattr(ops_module, "isoquant_ops", isoquant_ops)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "IsoQuant vLLM patch failed (non-fatal): %s", e
        )


def _patch_planarquant_attention(config: CacheConfig) -> None:
    """Inject PlanarQuant kernels into vLLM's attention path."""
    try:
        ops_module_path = "vllm.v1.attention.ops"
        ops_module = sys.modules.get(ops_module_path)
        if ops_module is None:
            return

        from ..kernels.triton import planarquant_ops
        setattr(ops_module, "planarquant_ops", planarquant_ops)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "PlanarQuant vLLM patch failed (non-fatal): %s", e
        )


def _patch_triattention(config: CacheConfig) -> None:
    """Inject TriAttention token eviction into vLLM's scheduler.

    EXPERIMENTAL: The actual scheduler/block-manager hook is not yet
    implemented. This currently only makes the scoring module importable
    so that downstream code can use it manually.
    """
    try:
        from ..kernels.triton import triattention_evict  # noqa: F401
        import logging
        logging.getLogger(__name__).info(
            "TriAttention module loaded but scheduler integration is "
            "EXPERIMENTAL — manual eviction only for now"
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "TriAttention module failed to load: %s", e
        )


def _check_rotor_unsupported(config: CacheConfig) -> None:
    """Raise a clear error if rotor methods are requested for vLLM.

    Rotor methods are not yet supported by vLLM's cache-dtype enum or
    attention backend. Users should use the Python API directly, or
    swap to iso3/iso4/planar3/planar4 for inference-backend use.
    """
    for label, method in [("K", config.k_method), ("V", config.v_method)]:
        if METHOD_FAMILIES[method] == MethodFamily.ROTORQUANT:
            raise NotImplementedError(
                f"{label} method {method.value} is not supported by vLLM yet — "
                f"cache-type registration pending. Use the Python API directly "
                f"(multi_turboquant.compress / decompress), or fall back to "
                f"iso3/iso4/planar3/planar4."
            )


def get_vllm_kv_cache_dtype(config: CacheConfig) -> str:
    """Get the vLLM --kv-cache-dtype flag value for a config.

    Args:
        config: Cache configuration.

    Returns:
        String for vLLM's --kv-cache-dtype argument.
    """
    _check_rotor_unsupported(config)
    # For symmetric configs, return the dtype string
    method_to_dtype = {
        CacheMethod.TURBO2: "turboquant25",
        CacheMethod.TURBO3: "turboquant35",
        CacheMethod.TURBO4: "turboquant35",
        CacheMethod.TURBO2_TCQ: "turboquant25",
        CacheMethod.TURBO3_TCQ: "turboquant35",
        CacheMethod.ISO3: "isoquant3",
        CacheMethod.ISO4: "isoquant4",
        CacheMethod.PLANAR3: "planarquant3",
        CacheMethod.PLANAR4: "planarquant4",
        CacheMethod.FP16: "auto",
        CacheMethod.Q8_0: "fp8_e4m3",
    }

    if config.is_symmetric:
        return method_to_dtype.get(config.k_method, "auto")

    # Asymmetric: vLLM doesn't natively support this,
    # would need custom attention backend
    return method_to_dtype.get(config.k_method, "auto")
