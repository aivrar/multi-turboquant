# SPDX-License-Identifier: MIT
"""Framework integration connectors.

Modules:
    vllm_patch     — Monkeypatch vLLM for all compression methods
    llamacpp_args  — Generate llama.cpp CLI flags
    bridge_adapter — Drop-in adapter for Llama_TQ bridge
"""

from .vllm_patch import patch_vllm, is_vllm_patched
from .llamacpp_args import get_llamacpp_args, get_llamacpp_command
from .bridge_adapter import BridgeAdapter

__all__ = [
    "patch_vllm",
    "is_vllm_patched",
    "get_llamacpp_args",
    "get_llamacpp_command",
    "BridgeAdapter",
]
