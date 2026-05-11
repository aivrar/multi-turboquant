# SPDX-License-Identifier: MIT
"""Apple Silicon Metal kernels for PlanarQuant and IsoQuant (ForgeAttention).

Fused attention kernels via ``mlx.fast.metal_kernel``:

    ``fused_attention`` — fused QK / SV / flash decode / sparse SV
    ``calibration``     — per-head attention budget calibration

Community-contributed by `@user-23xyz <https://github.com/user-23xyz>`_
in PR #1; sibling project at
`user-23xyz/forgeattention <https://github.com/user-23xyz/forgeattention>`_.
The maintainer does not have Apple Silicon hardware, so this path is
community-maintained — please tag ``@user-23xyz`` on Metal/MLX-specific issues.

A parallel C++ port for llama.cpp lives in ``johndpope/llama-cpp-turboquant``
(compile with ``-DGGML_METAL=ON``).
"""
import warnings

warnings.warn(
    "Loaded multi_turboquant.kernels.metal — experimental, "
    "community-maintained Apple Silicon path (ForgeAttention by "
    "@user-23xyz). Verified on M4 Mini; not independently tested by "
    "the maintainer. Please tag @user-23xyz on Metal/MLX-specific issues.",
    UserWarning,
    stacklevel=2,
)
