# SPDX-License-Identifier: MIT
"""Launch helpers for CUDA LLM weight sharing.

This module integrates with LD_PRELOAD-style CUDA weight-sharing helpers such as
pontostroy/cuda-llm-weight-share. It only prepares environment variables and a
wrapped command line; it does not vendor or reimplement the preload library.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CudaWeightShareConfig:
    """Configuration for wrapping a llama.cpp launch with CUDA weight sharing."""

    enabled: bool = False
    library_path: str = "./cuda-llm-weight-share.so"
    model_size_bytes: int | None = None
    model_size_tolerance: int = 0
    ipc_name: str = "/cuda_vram_ipc_auto"
    shm_wait_sec: int | None = None
    suppress_master_free: bool = False
    trace_callers: bool = False
    trace_depth: int | None = None
    trace_normal_allocs: bool = False

    def validate(self) -> list[str]:
        """Return configuration warnings."""
        warnings: list[str] = []
        if self.enabled and not self.library_path:
            warnings.append("CUDA weight sharing requires a preload library path")
        if self.model_size_bytes is not None and self.model_size_bytes < 0:
            warnings.append("MODEL_SIZE must be >= 0")
        if self.model_size_tolerance < 0:
            warnings.append("MODEL_SIZE_TOLERANCE must be >= 0")
        if self.shm_wait_sec is not None and self.shm_wait_sec < 0:
            warnings.append("CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC must be >= 0")
        if self.trace_depth is not None and self.trace_depth <= 0:
            warnings.append("CUDA_VRAM_IPC_TRACE_DEPTH must be > 0")
        return warnings


def get_cuda_weight_share_env(config: CudaWeightShareConfig) -> dict[str, str]:
    """Return environment variables for cuda-llm-weight-share."""
    if not config.enabled:
        return {}

    warnings = config.validate()
    if warnings:
        raise ValueError("; ".join(warnings))

    env = {
        "LD_PRELOAD": config.library_path,
        "MODEL_SIZE": str(config.model_size_bytes or 0),
        "MODEL_SIZE_TOLERANCE": str(config.model_size_tolerance),
        "CUDA_VRAM_IPC_NAME": config.ipc_name,
    }
    if config.shm_wait_sec is not None:
        env["CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC"] = str(config.shm_wait_sec)
    if config.suppress_master_free:
        env["CUDA_VRAM_IPC_SUPPRESS_MASTER_FREE"] = "1"
    if config.trace_callers:
        env["CUDA_VRAM_IPC_TRACE_CALLERS"] = "1"
    if config.trace_depth is not None:
        env["CUDA_VRAM_IPC_TRACE_DEPTH"] = str(config.trace_depth)
    if config.trace_normal_allocs:
        env["CUDA_VRAM_IPC_TRACE_NORMAL_ALLOCS"] = "1"
    return env


def wrap_cuda_weight_share_command(
    command: list[str],
    config: CudaWeightShareConfig,
) -> list[str]:
    """Prefix a command with env assignments for CUDA weight sharing."""
    env = get_cuda_weight_share_env(config)
    if not env:
        return command
    return ["env", *(f"{key}={value}" for key, value in env.items()), *command]
