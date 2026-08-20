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
    CUDA_WEIGHT_SHARE_COMMIT,
    CUDA_WEIGHT_SHARE_URL,
    CudaWeightShareBuildPlan,
    CudaWeightShareConfig,
    build_cuda_weight_share,
    get_cuda_weight_share_env,
    inspect_cuda_weight_share_source,
    plan_cuda_weight_share_build,
    validate_cuda_weight_share_library,
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
    collect_godzilla_calibration_diagnostics,
    inspect_godzilla_checkout,
    plan_godzilla_triattention,
    run_godzilla_triattention,
)
from .godzilla_gigatoken import (
    DEFAULT_GODZILLA_PROFILE,
    GODZILLA_SOURCE_PROFILES,
    GodzillaGigatokenPlan,
    GodzillaSourceProfile,
    RuntimeIssue,
    build_godzilla_gigatoken,
    inspect_godzilla_gigatoken,
    plan_godzilla_gigatoken,
    prepare_godzilla_gigatoken,
    verify_godzilla_gigatoken,
    get_godzilla_source_profile,
)
from .godzilla_composition import (
    COMPOSITION_PROFILE,
    GodzillaComposition,
    GodzillaCompositionPlan,
    build_godzilla_composition,
    inspect_godzilla_composition,
    plan_godzilla_composition,
    prepare_godzilla_composition,
    validate_godzilla_composition,
    verify_godzilla_composition,
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
    "CudaWeightShareBuildPlan",
    "CUDA_WEIGHT_SHARE_URL",
    "CUDA_WEIGHT_SHARE_COMMIT",
    "inspect_cuda_weight_share_source",
    "plan_cuda_weight_share_build",
    "build_cuda_weight_share",
    "validate_cuda_weight_share_library",
    "get_cuda_weight_share_env",
    "wrap_cuda_weight_share_command",
    "LMCacheIntegrationConfig",
    "LMCacheLaunchPlan",
    "LMCacheMode",
    "build_lmcache_launch_plan",
    "BridgeAdapter",
    "GodzillaCalibrationPlan",
    "GodzillaIssue",
    "collect_godzilla_calibration_diagnostics",
    "inspect_godzilla_checkout",
    "plan_godzilla_triattention",
    "run_godzilla_triattention",
    "GodzillaGigatokenPlan",
    "GodzillaSourceProfile",
    "GODZILLA_SOURCE_PROFILES",
    "DEFAULT_GODZILLA_PROFILE",
    "RuntimeIssue",
    "build_godzilla_gigatoken",
    "inspect_godzilla_gigatoken",
    "plan_godzilla_gigatoken",
    "prepare_godzilla_gigatoken",
    "verify_godzilla_gigatoken",
    "get_godzilla_source_profile",
    "COMPOSITION_PROFILE",
    "GodzillaComposition",
    "GodzillaCompositionPlan",
    "build_godzilla_composition",
    "inspect_godzilla_composition",
    "plan_godzilla_composition",
    "prepare_godzilla_composition",
    "validate_godzilla_composition",
    "verify_godzilla_composition",
]
