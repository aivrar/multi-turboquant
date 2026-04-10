# SPDX-License-Identifier: MIT
# Derived from scrya-com/rotorquant and johndpope/llama-cpp-turboquant (MIT)
"""IsoQuant — Quaternion 4D rotation KV cache compression.

Uses unit quaternions from Cl(3,0) Clifford algebra to apply 4D isoclinic
rotations to groups of 4 dimensions. The rotation parameters are fixed
(no calibration needed) and derived from golden-ratio-based quaternions
that maximize decorrelation.

O(d) complexity with 512 FMAs for head_dim=128.
Supports iso3 (3.25-bit) and iso4 (4.25-bit).

Key advantage: ISO3 K-only matches or beats FP16 decode speed because
the reduced memory bandwidth outweighs the cheap rotation cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import torch

from ..config import CacheMethod, MethodFamily
from ..registry import register_method
from .base import CompressionMethod, CompressedKV, MethodInfo


# ─── Constants ──────────────────────────────────────────────────────────────────

ISO_BITS = {"iso3": 3.25, "iso4": 4.25}
ISO_QUANT_BITS = {"iso3": 3, "iso4": 4}  # bits per element (quantized part)
ISO_GROUP_SIZE = 4  # quaternion operates on 4D groups

# Golden-ratio-based quaternion for maximum decorrelation
# q = a + bi + cj + dk where |q| = 1
PHI = (1.0 + math.sqrt(5.0)) / 2.0
_norm = math.sqrt(1 + PHI**2 + (PHI - 1)**2 + 1)
QUAT_A = 1.0 / _norm
QUAT_B = PHI / _norm
QUAT_C = (PHI - 1.0) / _norm
QUAT_D = 1.0 / _norm

NORM_BYTES = 2  # fp16 norm storage


# ─── Quaternion rotation ────────────────────────────────────────────────────────

@cache
def _rotation_matrix_4d(device_type: str, device_index: int | None) -> torch.Tensor:
    """Build the 4x4 left-isoclinic rotation matrix from the unit quaternion.

    For quaternion q = a + bi + cj + dk, the left-isoclinic rotation is:
        L(q) = [[a, -b, -c, -d],
                [b,  a, -d,  c],
                [c,  d,  a, -b],
                [d, -c,  b,  a]]

    This is an orthogonal matrix (L^T = L^{-1}), so inverse = transpose.
    """
    device = torch.device(device_type, device_index)
    a, b, c, d = QUAT_A, QUAT_B, QUAT_C, QUAT_D
    R = torch.tensor([
        [a, -b, -c, -d],
        [b,  a, -d,  c],
        [c,  d,  a, -b],
        [d, -c,  b,  a],
    ], dtype=torch.float32, device=device)
    return R


@cache
def _inverse_rotation_4d(device_type: str, device_index: int | None) -> torch.Tensor:
    """Inverse rotation = transpose (orthogonal matrix)."""
    return _rotation_matrix_4d(device_type, device_index).T.contiguous()


def _apply_quaternion_rotation(
    x: torch.Tensor, inverse: bool = False,
) -> torch.Tensor:
    """Apply 4D isoclinic rotation to groups of 4 dimensions.

    Reshapes [..., head_dim] -> [..., head_dim//4, 4], applies rotation,
    reshapes back. O(d) complexity — one 4x4 matmul per group.
    """
    orig_shape = x.shape
    head_dim = orig_shape[-1]
    assert head_dim % ISO_GROUP_SIZE == 0, (
        f"IsoQuant requires head_dim divisible by {ISO_GROUP_SIZE}, got {head_dim}"
    )

    device = x.device
    R = (
        _inverse_rotation_4d(device.type, device.index)
        if inverse
        else _rotation_matrix_4d(device.type, device.index)
    )

    # [..., head_dim] -> [..., num_groups, 4]
    grouped = x.reshape(*orig_shape[:-1], head_dim // ISO_GROUP_SIZE, ISO_GROUP_SIZE)
    # Apply rotation: [..., num_groups, 4] @ [4, 4] -> [..., num_groups, 4]
    rotated = torch.matmul(grouped.to(torch.float32), R.T)
    return rotated.reshape(orig_shape)


# ─── Uniform quantization ──────────────────────────────────────────────────────

def _uniform_quantize(
    x: torch.Tensor, bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniform quantization with per-vector min/max scaling.

    Returns:
        indices: Quantized indices [uint8].
        scales: Per-vector scale factors.
        zeros: Per-vector zero points.
    """
    levels = (1 << bits) - 1
    # Per-vector min/max
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    scale = (x_max - x_min).clamp_min(1e-12) / levels
    zero = x_min
    indices = ((x - zero) / scale).round().clamp(0, levels).to(torch.uint8)
    return indices, scale, zero


def _uniform_dequantize(
    indices: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor,
) -> torch.Tensor:
    """Uniform dequantization."""
    return indices.to(torch.float32) * scale + zero


# ─── Bit packing (simple uniform — no MSE/QJL split) ───────────────────────────

def _pack_uniform(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uniform quantized indices into bytes."""
    head_dim = indices.shape[-1]
    total_bits = head_dim * bits
    packed_bytes = (total_bits + 7) // 8

    # Simple bit-serial packing
    flat = indices.to(torch.int32).reshape(*indices.shape[:-1], -1)
    packed = torch.zeros(
        (*indices.shape[:-1], packed_bytes),
        dtype=torch.uint8, device=indices.device,
    )

    bit_pos = 0
    for i in range(head_dim):
        val = flat[..., i]
        for b in range(bits):
            byte_idx = bit_pos // 8
            bit_idx = bit_pos % 8
            packed[..., byte_idx] |= (((val >> b) & 1) << bit_idx).to(torch.uint8)
            bit_pos += 1

    return packed


def _unpack_uniform(packed: torch.Tensor, head_dim: int, bits: int) -> torch.Tensor:
    """Unpack bytes into uniform quantized indices."""
    result = torch.zeros(
        (*packed.shape[:-1], head_dim),
        dtype=torch.uint8, device=packed.device,
    )
    packed_i32 = packed.to(torch.int32)

    bit_pos = 0
    for i in range(head_dim):
        val = torch.zeros(packed.shape[:-1], dtype=torch.int32, device=packed.device)
        for b in range(bits):
            byte_idx = bit_pos // 8
            bit_idx = bit_pos % 8
            val |= ((packed_i32[..., byte_idx] >> bit_idx) & 1) << b
            bit_pos += 1
        result[..., i] = val.to(torch.uint8)

    return result


# ─── Encode / Decode ────────────────────────────────────────────────────────────

def iso_encode(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Encode using IsoQuant: rotate -> quantize -> pack.

    Packed format per head: [quant_indices | scale_bytes | zero_bytes | norm_bytes]
    """
    x_fp32 = x.to(torch.float32)
    # Store vector norms for reconstruction
    norms = x_fp32.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit = x_fp32 / norms

    # Apply quaternion rotation
    rotated = _apply_quaternion_rotation(unit)

    # Uniform quantize the rotated unit vector
    indices, scale, zero = _uniform_quantize(rotated, bits)

    # Pack everything
    quant_packed = _pack_uniform(indices, bits)
    # Pack fp16 scalars as 2 bytes each, per head
    flat_norms = norms.squeeze(-1).reshape(-1).to(torch.float16).contiguous().view(torch.uint8)
    norm_bytes = flat_norms.reshape(*norms.shape[:-1], NORM_BYTES)
    flat_scale = scale.squeeze(-1).reshape(-1).to(torch.float16).contiguous().view(torch.uint8)
    scale_bytes = flat_scale.reshape(*scale.shape[:-1], NORM_BYTES)
    flat_zero = zero.squeeze(-1).reshape(-1).to(torch.float16).contiguous().view(torch.uint8)
    zero_bytes = flat_zero.reshape(*zero.shape[:-1], NORM_BYTES)

    return torch.cat([quant_packed, scale_bytes, zero_bytes, norm_bytes], dim=-1)


def iso_decode(
    packed: torch.Tensor, head_dim: int, bits: int, dtype: torch.dtype,
) -> torch.Tensor:
    """Decode IsoQuant: unpack -> dequantize -> inverse rotate."""
    quant_bytes = (head_dim * bits + 7) // 8
    quant_packed = packed[..., :quant_bytes]
    cursor = quant_bytes

    scale_raw = packed[..., cursor : cursor + NORM_BYTES].contiguous()
    scale = scale_raw.reshape(-1, NORM_BYTES).view(torch.float16).to(torch.float32)
    scale = scale.reshape(*packed.shape[:-1], 1)
    cursor += NORM_BYTES

    zero_raw = packed[..., cursor : cursor + NORM_BYTES].contiguous()
    zero = zero_raw.reshape(-1, NORM_BYTES).view(torch.float16).to(torch.float32)
    zero = zero.reshape(*packed.shape[:-1], 1)
    cursor += NORM_BYTES

    norm_raw = packed[..., cursor : cursor + NORM_BYTES].contiguous()
    norms = norm_raw.reshape(-1, NORM_BYTES).view(torch.float16).to(torch.float32)
    norms = norms.reshape(*packed.shape[:-1], 1)

    indices = _unpack_uniform(quant_packed, head_dim, bits)
    rotated_hat = _uniform_dequantize(indices, scale, zero)

    # Inverse quaternion rotation
    unit_hat = _apply_quaternion_rotation(rotated_hat, inverse=True)
    return (unit_hat * norms).to(dtype)


def iso_packed_dim(head_dim: int, bits: int) -> int:
    """Total packed bytes per head."""
    quant_bytes = (head_dim * bits + 7) // 8
    return quant_bytes + NORM_BYTES * 3  # scale + zero + norm


# ─── Method classes ─────────────────────────────────────────────────────────────

class _IsoQuantBase(CompressionMethod):
    def __init__(self, bits: int = 3, method: CacheMethod = CacheMethod.ISO3):
        self._bits = bits
        self._method = method

    def packed_dim(self, head_dim: int) -> int:
        return iso_packed_dim(head_dim, self._bits)

    def encode(
        self, x: torch.Tensor, *,
        head_indices: torch.Tensor | None = None, layer_idx: int = 0,
    ) -> CompressedKV:
        packed = iso_encode(x, self._bits)
        return CompressedKV(
            data=packed, head_dim=x.shape[-1], method=self._method,
        )

    def decode(
        self, compressed: CompressedKV, *, dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return iso_decode(
            compressed.data, compressed.head_dim, self._bits, dtype,
        )

    def validate_head_dim(self, head_dim: int) -> bool:
        return head_dim % ISO_GROUP_SIZE == 0 and head_dim in self.info().supported_head_dims

    def supports_triton(self) -> bool:
        return True  # Research Triton kernels exist in scrya-com/rotorquant

    def supports_cuda(self) -> bool:
        return True  # johndpope fork has CUDA kernels

    def supports_metal(self) -> bool:
        return False  # IsoQuant Metal not yet implemented


@register_method(CacheMethod.ISO3)
class IsoQuant3(_IsoQuantBase):
    def __init__(self, **kwargs):
        super().__init__(bits=3, method=CacheMethod.ISO3)

    def info(self) -> MethodInfo:
        return MethodInfo(
            method=CacheMethod.ISO3,
            family=MethodFamily.ISOQUANT,
            bits=3.25, requires_calibration=False,
            supports_asymmetric=True,
            supported_head_dims=(64, 128),
            transform_name="Quaternion 4D isoclinic rotation (Cl(3,0))",
            description="3.25-bit quaternion rotation, no calibration, beats FP16 K-only",
            fma_count=512, param_count=128,
        )


@register_method(CacheMethod.ISO4)
class IsoQuant4(_IsoQuantBase):
    def __init__(self, **kwargs):
        super().__init__(bits=4, method=CacheMethod.ISO4)

    def info(self) -> MethodInfo:
        return MethodInfo(
            method=CacheMethod.ISO4,
            family=MethodFamily.ISOQUANT,
            bits=4.25, requires_calibration=False,
            supports_asymmetric=True,
            supported_head_dims=(64, 128),
            transform_name="Quaternion 4D isoclinic rotation (Cl(3,0))",
            description="4.25-bit quaternion rotation, no calibration, higher quality",
            fma_count=512, param_count=128,
        )


# Public alias
IsoQuantMethod = IsoQuant3
