# SPDX-License-Identifier: MIT
"""Backend-specific kernel implementations.

Subpackages:
    triton/  — Triton kernels for vLLM integration (Linux + CUDA only)
    cuda/    — Compiled CUDA kernels for llama.cpp
    metal/   — Apple Silicon kernels (PlanarQuant, IsoQuant)
"""
