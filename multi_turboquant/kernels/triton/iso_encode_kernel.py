# SPDX-License-Identifier: MIT
"""Vectorized IsoQuant encode/decode — replaces bit-serial Python loops.

The quaternion 4D rotation is a simple 4x4 matmul per group of 4 dims.
This implementation is fully vectorized with torch ops, no Python loops
in the hot path. On CUDA this runs at near-Triton speed via torch.matmul.

When Triton is available, the fused kernel in isoquant_ops.py can be used
instead for single-kernel-launch performance.
"""

from __future__ import annotations

from functools import cache

import torch

from ...methods.isoquant import (
    QUAT_A, QUAT_B, QUAT_C, QUAT_D,
    ISO_GROUP_SIZE, NORM_BYTES,
)


@cache
def _rotation_matrix(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    a, b, c, d = QUAT_A, QUAT_B, QUAT_C, QUAT_D
    return torch.tensor([
        [a, -b, -c, -d],
        [b,  a, -d,  c],
        [c,  d,  a, -b],
        [d, -c,  b,  a],
    ], dtype=torch.float32, device=device)


@cache
def _inverse_rotation(device_type: str, device_index: int | None) -> torch.Tensor:
    return _rotation_matrix(device_type, device_index).T.contiguous()


def iso_encode_vectorized(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Fully vectorized IsoQuant encode — no Python loops.

    Args:
        x: Input [*, num_heads, head_dim] float tensor.
        bits: Quantization bits (3 or 4).

    Returns:
        Packed bytes [*, num_heads, packed_dim].
    """
    orig_shape = x.shape
    head_dim = orig_shape[-1]
    assert head_dim % ISO_GROUP_SIZE == 0

    x_fp32 = x.to(torch.float32)

    # Vector norms
    norms = x_fp32.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit = x_fp32 / norms

    # Quaternion rotation — reshape to groups of 4, matmul
    R = _rotation_matrix(x.device.type, x.device.index)
    grouped = unit.reshape(*orig_shape[:-1], head_dim // 4, 4)
    rotated = torch.matmul(grouped, R.T)  # [..., num_groups, 4]
    rotated = rotated.reshape(orig_shape)

    # Uniform quantize — fully vectorized
    levels = (1 << bits) - 1
    x_min = rotated.amin(dim=-1, keepdim=True)
    x_max = rotated.amax(dim=-1, keepdim=True)
    scale = (x_max - x_min).clamp_min(1e-12) / levels
    zero = x_min
    indices = ((rotated - zero) / scale).round().clamp(0, levels).to(torch.uint8)

    # Vectorized bit packing
    quant_packed = _pack_vectorized(indices, bits)

    # Pack norms/scale/zero as fp16 bytes
    flat_shape = indices.shape[:-1]  # [*, num_heads]
    norm_bytes = _scalar_to_bytes(norms.squeeze(-1), flat_shape)
    scale_bytes = _scalar_to_bytes(scale.squeeze(-1), flat_shape)
    zero_bytes = _scalar_to_bytes(zero.squeeze(-1), flat_shape)

    return torch.cat([quant_packed, scale_bytes, zero_bytes, norm_bytes], dim=-1)


def iso_decode_vectorized(
    packed: torch.Tensor, head_dim: int, bits: int, dtype: torch.dtype,
) -> torch.Tensor:
    """Fully vectorized IsoQuant decode."""
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

    # Inverse quaternion rotation
    R_inv = _inverse_rotation(packed.device.type, packed.device.index)
    grouped = rotated_hat.reshape(*packed.shape[:-1], head_dim // 4, 4)
    unit_hat = torch.matmul(grouped, R_inv.T).reshape(*packed.shape[:-1], head_dim)

    return (unit_hat * norms).to(dtype)


def _scalar_to_bytes(scalar: torch.Tensor, flat_shape: tuple) -> torch.Tensor:
    """Pack a per-head scalar to 2 bytes (fp16)."""
    flat = scalar.reshape(-1).to(torch.float16).contiguous().view(torch.uint8)
    return flat.reshape(*flat_shape, NORM_BYTES)


def _bytes_to_scalar(raw: torch.Tensor, flat_shape: tuple) -> torch.Tensor:
    """Unpack 2 bytes to per-head scalar (fp16 -> fp32)."""
    flat = raw.contiguous().reshape(-1, NORM_BYTES).view(torch.float16).to(torch.float32)
    return flat.reshape(*flat_shape, 1)


def _pack_vectorized(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Vectorized bit packing using shift-and-or on whole tensors.

    Much faster than the bit-serial Python loop in the base implementation.
    """
    head_dim = indices.shape[-1]
    total_bits = head_dim * bits
    packed_bytes = (total_bits + 7) // 8
    batch_shape = indices.shape[:-1]

    # Expand each index into its individual bits
    indices_i32 = indices.to(torch.int32)
    bit_shifts = torch.arange(bits, device=indices.device, dtype=torch.int32)
    # [..., head_dim, bits]
    individual_bits = ((indices_i32.unsqueeze(-1) >> bit_shifts) & 1)
    # Flatten to [..., head_dim * bits]
    all_bits = individual_bits.reshape(*batch_shape, total_bits)

    # Assign each bit to its byte and position within that byte
    bit_positions = torch.arange(total_bits, device=indices.device, dtype=torch.int64)
    byte_indices = bit_positions // 8
    within_byte = (bit_positions % 8).to(torch.int32)

    # Shift bits to their position within bytes
    shifted_bits = (all_bits << within_byte).to(torch.int32)

    # Scatter-add into bytes
    packed = torch.zeros(*batch_shape, packed_bytes, dtype=torch.int32, device=indices.device)
    packed.scatter_add_(-1, byte_indices.expand_as(shifted_bits), shifted_bits)

    return packed.to(torch.uint8)


def _unpack_vectorized(packed: torch.Tensor, head_dim: int, bits: int) -> torch.Tensor:
    """Vectorized bit unpacking — inverse of _pack_vectorized."""
    total_bits = head_dim * bits
    batch_shape = packed.shape[:-1]

    bit_positions = torch.arange(total_bits, device=packed.device, dtype=torch.int64)
    byte_indices = bit_positions // 8
    within_byte = (bit_positions % 8).to(torch.int32)

    # Gather the bytes each bit lives in
    packed_i32 = packed.to(torch.int32)
    gathered = packed_i32.gather(-1, byte_indices.expand(*batch_shape, total_bits))

    # Extract each bit
    all_bits = ((gathered >> within_byte) & 1)

    # Reshape to [..., head_dim, bits] and reconstruct values
    all_bits = all_bits.reshape(*batch_shape, head_dim, bits)
    bit_values = torch.arange(bits, device=packed.device, dtype=torch.int32)
    values = (all_bits << bit_values).sum(dim=-1)

    return values.to(torch.uint8)
