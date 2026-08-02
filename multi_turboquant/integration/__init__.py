# SPDX-License-Identifier: MIT
"""Framework integration connectors.

Modules:
    vllm_patch     — Monkeypatch vLLM for all compression methods
    llamacpp_args  — Generate llama.cpp CLI flags
    bridge_adapter — Drop-in adapter for Llama_TQ bridge
"""

from .vllm_patch import patch_vllm, is_vllm_patched
from .llamacpp_args import (
    LlamaCppContextExtensionConfig,
    LlamaCppProfile,
    LlamaCppSpeculativeConfig,
    get_llamacpp_args,
    get_llamacpp_command,
    normalize_llamacpp_profile,
)
from .llamacpp_scan import (
    LlamaCppCapabilities,
    parse_llamacpp_help,
    scan_llamacpp_binary,
)
from .weight_share import (
    CudaWeightShareConfig,
    get_cuda_weight_share_env,
    wrap_cuda_weight_share_command,
)
from .lmcache import (
    LMCacheIntegrationConfig,
    LMCacheLaunchPlan,
    LMCacheMode,
    build_lmcache_launch_plan,
)
from .bridge_adapter import BridgeAdapter
from .godzilla_workspace import (
    GodzillaCalibrationPlan,
    GodzillaIssue,
    inspect_godzilla_checkout,
    plan_godzilla_triattention,
    run_godzilla_triattention,
)
from .godzilla_gigatoken import (
    GodzillaGigatokenPlan,
    RuntimeIssue,
    build_godzilla_gigatoken,
    inspect_godzilla_gigatoken,
    plan_godzilla_gigatoken,
    prepare_godzilla_gigatoken,
    verify_godzilla_gigatoken,
)

__all__ = [
    "patch_vllm",
    "is_vllm_patched",
    "get_llamacpp_args",
    "get_llamacpp_command",
    "LlamaCppContextExtensionConfig",
    "LlamaCppProfile",
    "LlamaCppSpeculativeConfig",
    "normalize_llamacpp_profile",
    "LlamaCppCapabilities",
    "parse_llamacpp_help",
    "scan_llamacpp_binary",
    "CudaWeightShareConfig",
    "get_cuda_weight_share_env",
    "wrap_cuda_weight_share_command",
    "LMCacheIntegrationConfig",
    "LMCacheLaunchPlan",
    "LMCacheMode",
    "build_lmcache_launch_plan",
    "BridgeAdapter",
    "GodzillaCalibrationPlan",
    "GodzillaIssue",
    "inspect_godzilla_checkout",
    "plan_godzilla_triattention",
    "run_godzilla_triattention",
    "GodzillaGigatokenPlan",
    "RuntimeIssue",
    "build_godzilla_gigatoken",
    "inspect_godzilla_gigatoken",
    "plan_godzilla_gigatoken",
    "prepare_godzilla_gigatoken",
    "verify_godzilla_gigatoken",
]
