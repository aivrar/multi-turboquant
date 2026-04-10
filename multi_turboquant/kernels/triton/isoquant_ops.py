# SPDX-License-Identifier: MIT
# Derived from scrya-com/rotorquant Triton research kernels
"""Triton kernels for IsoQuant quaternion rotation encode/decode.

These are adapted from the research implementations in
scrya-com/rotorquant/turboquant/triton_isoquant.py.

The quaternion 4D rotation is simple enough to fuse entirely into
the attention kernel, avoiding intermediate materialization.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from ...methods.isoquant import (
    QUAT_A, QUAT_B, QUAT_C, QUAT_D,
    ISO_GROUP_SIZE,
)


if HAS_TRITON:

    @triton.jit
    def _iso_rotate_4d(
        x0, x1, x2, x3,
        a: tl.constexpr, b: tl.constexpr,
        c: tl.constexpr, d: tl.constexpr,
        inverse: tl.constexpr,
    ):
        """Apply 4D isoclinic rotation in-register.

        For forward: R @ [x0, x1, x2, x3]^T
        For inverse: R^T @ [x0, x1, x2, x3]^T
        """
        if inverse:
            # R^T: transpose the rotation matrix
            y0 = a * x0 + b * x1 + c * x2 + d * x3
            y1 = -b * x0 + a * x1 + d * x2 - c * x3
            y2 = -c * x0 - d * x1 + a * x2 + b * x3
            y3 = -d * x0 + c * x1 - b * x2 + a * x3
        else:
            y0 = a * x0 - b * x1 - c * x2 - d * x3
            y1 = b * x0 + a * x1 - d * x2 + c * x3
            y2 = c * x0 + d * x1 + a * x2 - b * x3
            y3 = d * x0 - c * x1 + b * x2 + a * x3
        return y0, y1, y2, y3


    @triton.jit
    def _iso_encode_kernel(
        x_ptr, out_ptr,
        seq_len, num_heads, head_dim: tl.constexpr,
        bits: tl.constexpr,
        BLOCK_SEQ: tl.constexpr,
    ):
        """Fused IsoQuant encode: rotate + quantize + pack."""
        pid_head = tl.program_id(0)
        pid_seq = tl.program_id(1)

        seq_offsets = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
        mask = seq_offsets < seq_len

        # Load head vector
        base = seq_offsets * num_heads * head_dim + pid_head * head_dim
        x = tl.load(x_ptr + base[:, None] + tl.arange(0, head_dim)[None, :],
                     mask=mask[:, None], other=0.0)

        # Compute norm
        norm = tl.sqrt(tl.sum(x * x, axis=1) + 1e-12)

        # Normalize
        x_unit = x / norm[:, None]

        # Apply quaternion rotation to groups of 4
        num_groups = head_dim // 4
        for g in tl.static_range(num_groups):
            idx = g * 4
            x0 = x_unit[:, idx]
            x1 = x_unit[:, idx + 1]
            x2 = x_unit[:, idx + 2]
            x3 = x_unit[:, idx + 3]
            y0, y1, y2, y3 = _iso_rotate_4d(
                x0, x1, x2, x3,
                QUAT_A, QUAT_B, QUAT_C, QUAT_D,
                inverse=False,
            )
            x_unit = tl.where(
                tl.arange(0, head_dim)[None, :] == idx, y0[:, None], x_unit
            )
            x_unit = tl.where(
                tl.arange(0, head_dim)[None, :] == idx + 1, y1[:, None], x_unit
            )
            x_unit = tl.where(
                tl.arange(0, head_dim)[None, :] == idx + 2, y2[:, None], x_unit
            )
            x_unit = tl.where(
                tl.arange(0, head_dim)[None, :] == idx + 3, y3[:, None], x_unit
            )

        # Uniform quantize (per-vector min/max)
        levels = (1 << bits) - 1
        x_min = tl.min(x_unit, axis=1)
        x_max = tl.max(x_unit, axis=1)
        scale = (x_max - x_min) / levels
        scale = tl.where(scale < 1e-12, 1e-12, scale)
        indices = ((x_unit - x_min[:, None]) / scale[:, None] + 0.5).to(tl.uint8)

        # Store packed output (simplified — full packing in separate pass)
        tl.store(out_ptr + base[:, None] + tl.arange(0, head_dim)[None, :],
                 indices, mask=mask[:, None])


def iso_encode_triton(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Triton-accelerated IsoQuant encode. Falls back to PyTorch."""
    if not HAS_TRITON or not x.is_cuda:
        from ...methods.isoquant import iso_encode
        return iso_encode(x, bits)

    # For now, use PyTorch reference — Triton kernel is a template
    from ...methods.isoquant import iso_encode
    return iso_encode(x, bits)


def iso_decode_triton(
    packed: torch.Tensor, head_dim: int, bits: int, dtype: torch.dtype,
) -> torch.Tensor:
    """Triton-accelerated IsoQuant decode. Falls back to PyTorch."""
    if not HAS_TRITON or not packed.is_cuda:
        from ...methods.isoquant import iso_decode
        return iso_decode(packed, head_dim, bits, dtype)

    from ...methods.isoquant import iso_decode
    return iso_decode(packed, head_dim, bits, dtype)
