# SPDX-License-Identifier: MIT
"""Framework integration connectors.

Modules:
    vllm_patch     — Monkeypatch vLLM for all compression methods
    llamacpp_args  — Generate llama.cpp CLI flags
    bridge_adapter — Drop-in adapter for Llama_TQ bridge
"""

from .vllm_patch import patch_vllm, is_vllm_patched
from .llamacpp_args import (
    LlamaCppProfile,
    LlamaCppSpeculativeConfig,
    get_llamacpp_args,
    get_llamacpp_command,
    normalize_llamacpp_profile,
)
from .weight_share import (
    CudaWeightShareConfig,
    get_cuda_weight_share_env,
    wrap_cuda_weight_share_command,
)
from .bridge_adapter import BridgeAdapter

__all__ = [
    "patch_vllm",
    "is_vllm_patched",
    "get_llamacpp_args",
    "get_llamacpp_command",
    "LlamaCppProfile",
    "LlamaCppSpeculativeConfig",
    "normalize_llamacpp_profile",
    "CudaWeightShareConfig",
    "get_cuda_weight_share_env",
    "wrap_cuda_weight_share_command",
    "BridgeAdapter",
]
