# SPDX-License-Identifier: MIT
"""Triton GPU kernels for KV cache compression.

These kernels run on NVIDIA GPUs via Triton and are used for vLLM integration.
Triton is Linux-only — these modules will fail to import on Windows.

Available kernels:
    turboquant_decode    — TurboQuant/TCQ fused decode attention
    turboquant_update    — TurboQuant/TCQ fused quantize-and-store
    isoquant_ops         — IsoQuant quaternion rotation encode/decode
    planarquant_ops      — PlanarQuant Givens rotation encode/decode
    triattention_evict   — TriAttention token scoring and eviction
"""

import sys

if sys.platform == "win32":
    import warnings
    warnings.warn(
        "Triton kernels are Linux-only. "
        "Pure PyTorch fallbacks will be used on Windows.",
        RuntimeWarning,
        stacklevel=2,
    )
