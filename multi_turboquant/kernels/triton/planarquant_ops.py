# SPDX-License-Identifier: MIT
# Derived from scrya-com/rotorquant Triton research kernels
"""Triton kernels for PlanarQuant Givens rotation encode/decode.

Adapted from scrya-com/rotorquant/turboquant/triton_planarquant.py.

The 2D Givens rotation is the simplest of all methods — just a
cos/sin multiply per pair of dimensions. Very efficient to fuse.
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

from ...methods.planarquant import GOLDEN_ANGLE


if HAS_TRITON:

    @triton.jit
    def _givens_rotate_pair(x0, x1, cos_t, sin_t, inverse: tl.constexpr):
        """Apply 2D Givens rotation to a pair of values."""
        if inverse:
            y0 = cos_t * x0 + sin_t * x1
            y1 = -sin_t * x0 + cos_t * x1
        else:
            y0 = cos_t * x0 - sin_t * x1
            y1 = sin_t * x0 + cos_t * x1
        return y0, y1


    @triton.jit
    def _planar_encode_kernel(
        x_ptr, out_ptr,
        seq_len, num_heads, head_dim: tl.constexpr,
        bits: tl.constexpr,
        BLOCK_SEQ: tl.constexpr,
    ):
        """Fused PlanarQuant encode: Givens rotate + quantize + pack."""
        pid_head = tl.program_id(0)
        pid_seq = tl.program_id(1)

        seq_offsets = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
        mask = seq_offsets < seq_len

        base = seq_offsets * num_heads * head_dim + pid_head * head_dim
        x = tl.load(x_ptr + base[:, None] + tl.arange(0, head_dim)[None, :],
                     mask=mask[:, None], other=0.0)

        norm = tl.sqrt(tl.sum(x * x, axis=1) + 1e-12)
        x_unit = x / norm[:, None]

        # Apply Givens rotations to dimension pairs
        num_pairs = head_dim // 2
        for p in tl.static_range(num_pairs):
            angle = GOLDEN_ANGLE * p
            cos_t = tl.math.cos(angle)
            sin_t = tl.math.sin(angle)
            idx = p * 2
            x0 = x_unit[:, idx]
            x1 = x_unit[:, idx + 1]
            y0, y1 = _givens_rotate_pair(x0, x1, cos_t, sin_t, inverse=False)
            x_unit = tl.where(
                tl.arange(0, head_dim)[None, :] == idx, y0[:, None], x_unit
            )
            x_unit = tl.where(
                tl.arange(0, head_dim)[None, :] == idx + 1, y1[:, None], x_unit
            )

        # Uniform quantize
        levels = (1 << bits) - 1
        x_min = tl.min(x_unit, axis=1)
        x_max = tl.max(x_unit, axis=1)
        scale = (x_max - x_min) / levels
        scale = tl.where(scale < 1e-12, 1e-12, scale)
        indices = ((x_unit - x_min[:, None]) / scale[:, None] + 0.5).to(tl.uint8)

        tl.store(out_ptr + base[:, None] + tl.arange(0, head_dim)[None, :],
                 indices, mask=mask[:, None])


def planar_encode_triton(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Triton-accelerated PlanarQuant encode."""
    if not HAS_TRITON or not x.is_cuda:
        from ...methods.planarquant import planar_encode
        return planar_encode(x, bits)
    from ...methods.planarquant import planar_encode
    return planar_encode(x, bits)


def planar_decode_triton(
    packed: torch.Tensor, head_dim: int, bits: int, dtype: torch.dtype,
) -> torch.Tensor:
    """Triton-accelerated PlanarQuant decode."""
    if not HAS_TRITON or not packed.is_cuda:
        from ...methods.planarquant import planar_decode
        return planar_decode(packed, head_dim, bits, dtype)
    from ...methods.planarquant import planar_decode
    return planar_decode(packed, head_dim, bits, dtype)
