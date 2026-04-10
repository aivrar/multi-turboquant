# SPDX-License-Identifier: MIT
"""Vectorized PlanarQuant encode/decode — replaces bit-serial Python loops.

The Givens 2D rotation is a cos/sin multiply per pair of dimensions.
Fully vectorized with torch ops — the rotation applies to all pairs
simultaneously via element-wise multiply on even/odd slices.
"""

from __future__ import annotations

import math
from functools import cache

import torch

from ...methods.planarquant import GOLDEN_ANGLE, PLANAR_GROUP_SIZE, NORM_BYTES


@cache
def _givens_cos_sin(
    device_type: str, device_index: int | None, num_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device_type, device_index)
    pair_indices = torch.arange(num_pairs, dtype=torch.float32, device=device)
    angles = GOLDEN_ANGLE * pair_indices
    return torch.cos(angles), torch.sin(angles)


def planar_encode_vectorized(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Fully vectorized PlanarQuant encode — no Python loops.

    Args:
        x: Input [*, num_heads, head_dim] float tensor.
        bits: Quantization bits (3 or 4).

    Returns:
        Packed bytes [*, num_heads, packed_dim].
    """
    orig_shape = x.shape
    head_dim = orig_shape[-1]
    assert head_dim % PLANAR_GROUP_SIZE == 0
    num_pairs = head_dim // 2

    x_fp32 = x.to(torch.float32)
    norms = x_fp32.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit = x_fp32 / norms

    # Givens rotation — fully vectorized via even/odd slicing
    cos_t, sin_t = _givens_cos_sin(x.device.type, x.device.index, num_pairs)
    x0 = unit[..., 0::2]  # even indices
    x1 = unit[..., 1::2]  # odd indices
    y0 = cos_t * x0 - sin_t * x1
    y1 = sin_t * x0 + cos_t * x1

    # Interleave back
    rotated = torch.stack([y0, y1], dim=-1).reshape(orig_shape)

    # Uniform quantize
    levels = (1 << bits) - 1
    x_min = rotated.amin(dim=-1, keepdim=True)
    x_max = rotated.amax(dim=-1, keepdim=True)
    scale = (x_max - x_min).clamp_min(1e-12) / levels
    zero = x_min
    indices = ((rotated - zero) / scale).round().clamp(0, levels).to(torch.uint8)

    # Vectorized bit packing (reuse iso kernel's implementation)
    from .iso_encode_kernel import _pack_vectorized, _scalar_to_bytes
    quant_packed = _pack_vectorized(indices, bits)
    flat_shape = indices.shape[:-1]
    norm_bytes = _scalar_to_bytes(norms.squeeze(-1), flat_shape)
    scale_bytes = _scalar_to_bytes(scale.squeeze(-1), flat_shape)
    zero_bytes = _scalar_to_bytes(zero.squeeze(-1), flat_shape)

    return torch.cat([quant_packed, scale_bytes, zero_bytes, norm_bytes], dim=-1)


def planar_decode_vectorized(
    packed: torch.Tensor, head_dim: int, bits: int, dtype: torch.dtype,
) -> torch.Tensor:
    """Fully vectorized PlanarQuant decode."""
    from .iso_encode_kernel import _unpack_vectorized, _bytes_to_scalar

    quant_bytes = (head_dim * bits + 7) // 8
    quant_packed = packed[..., :quant_bytes]
    cursor = quant_bytes

    scale = _bytes_to_scalar(packed[..., cursor:cursor + NORM_BYTES], packed.shape[:-1])
    cursor += NORM_BYTES
    zero = _bytes_to_scalar(packed[..., cursor:cursor + NORM_BYTES], packed.shape[:-1])
    cursor += NORM_BYTES
    norms = _bytes_to_scalar(packed[..., cursor:cursor + NORM_BYTES], packed.shape[:-1])

    indices = _unpack_vectorized(quant_packed, head_dim, bits)
    rotated_hat = indices.to(torch.float32) * scale + zero

    # Inverse Givens rotation
    num_pairs = head_dim // 2
    cos_t, sin_t = _givens_cos_sin(packed.device.type, packed.device.index, num_pairs)
    y0 = rotated_hat[..., 0::2]
    y1 = rotated_hat[..., 1::2]
    x0 = cos_t * y0 + sin_t * y1   # inverse: negate sin
    x1 = -sin_t * y0 + cos_t * y1

    unit_hat = torch.stack([x0, x1], dim=-1).reshape(*packed.shape[:-1], head_dim)
    return (unit_hat * norms).to(dtype)
