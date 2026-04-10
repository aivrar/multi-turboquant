# SPDX-License-Identifier: MIT
# Derived from scrya-com/rotorquant and johndpope/llama-cpp-turboquant (MIT)
"""PlanarQuant — Givens 2D rotation KV cache compression.

The simplest rotation-based method: applies 2D Givens rotations to pairs
of dimensions. Each rotation is parameterized by a single angle theta,
chosen to maximize decorrelation between the two dimensions.

O(d) complexity with 256 FMAs for head_dim=128 (half of IsoQuant).
No calibration needed — rotation angles are derived from dimension indices.
Supports planar3 (3.25-bit) and planar4 (4.25-bit).
Has Metal (Apple Silicon) support in johndpope's llama.cpp fork.
"""

from __future__ import annotations

import math
from functools import cache

import torch

from ..config import CacheMethod, MethodFamily
from ..registry import register_method
from .base import CompressionMethod, CompressedKV, MethodInfo


# ─── Constants ──────────────────────────────────────────────────────────────────

PLANAR_BITS = {"planar3": 3.25, "planar4": 4.25}
PLANAR_QUANT_BITS = {"planar3": 3, "planar4": 4}
PLANAR_GROUP_SIZE = 2  # Givens rotation operates on pairs
NORM_BYTES = 2

# Golden angle for maximum decorrelation
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


# ─── Givens rotation ───────────────────────────────────────────────────────────

@cache
def _givens_angles(
    device_type: str, device_index: int | None, num_pairs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate Givens rotation angles for each dimension pair.

    Uses golden-angle spacing to ensure no two pairs have the same
    rotation, maximizing decorrelation across the full head dimension.

    Returns:
        (cos_theta, sin_theta) each of shape [num_pairs].
    """
    device = torch.device(device_type, device_index)
    pair_indices = torch.arange(num_pairs, dtype=torch.float32, device=device)
    # Golden-angle-spaced rotations
    angles = GOLDEN_ANGLE * pair_indices
    return torch.cos(angles), torch.sin(angles)


def _apply_givens_rotation(
    x: torch.Tensor, inverse: bool = False,
) -> torch.Tensor:
    """Apply 2D Givens rotations to pairs of dimensions.

    For each pair (x_2i, x_2i+1), applies:
        [cos θ  -sin θ] [x_2i  ]
        [sin θ   cos θ] [x_2i+1]

    For inverse, transpose the matrix (negate sin θ).
    """
    orig_shape = x.shape
    head_dim = orig_shape[-1]
    assert head_dim % PLANAR_GROUP_SIZE == 0, (
        f"PlanarQuant requires even head_dim, got {head_dim}"
    )
    num_pairs = head_dim // PLANAR_GROUP_SIZE
    device = x.device

    cos_t, sin_t = _givens_angles(device.type, device.index, num_pairs)

    if inverse:
        sin_t = -sin_t

    # [..., head_dim] -> [..., num_pairs, 2]
    paired = x.reshape(*orig_shape[:-1], num_pairs, 2).to(torch.float32)
    x0 = paired[..., 0]  # [..., num_pairs]
    x1 = paired[..., 1]  # [..., num_pairs]

    # Apply rotation
    y0 = cos_t * x0 - sin_t * x1
    y1 = sin_t * x0 + cos_t * x1

    result = torch.stack([y0, y1], dim=-1)  # [..., num_pairs, 2]
    return result.reshape(orig_shape)


# ─── Uniform quantization (shared with IsoQuant) ───────────────────────────────

def _uniform_quantize(
    x: torch.Tensor, bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    levels = (1 << bits) - 1
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    scale = (x_max - x_min).clamp_min(1e-12) / levels
    zero = x_min
    indices = ((x - zero) / scale).round().clamp(0, levels).to(torch.uint8)
    return indices, scale, zero


def _uniform_dequantize(
    indices: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor,
) -> torch.Tensor:
    return indices.to(torch.float32) * scale + zero


def _pack_uniform(indices: torch.Tensor, bits: int) -> torch.Tensor:
    head_dim = indices.shape[-1]
    packed_bytes = (head_dim * bits + 7) // 8
    flat = indices.to(torch.int32)
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
    result = torch.zeros(
        (*packed.shape[:-1], head_dim), dtype=torch.uint8, device=packed.device,
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

def planar_encode(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Encode using PlanarQuant: Givens rotate -> quantize -> pack."""
    x_fp32 = x.to(torch.float32)
    norms = x_fp32.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit = x_fp32 / norms

    rotated = _apply_givens_rotation(unit)
    indices, scale, zero = _uniform_quantize(rotated, bits)

    quant_packed = _pack_uniform(indices, bits)
    flat_norms = norms.squeeze(-1).reshape(-1).to(torch.float16).contiguous().view(torch.uint8)
    norm_bytes = flat_norms.reshape(*norms.shape[:-1], NORM_BYTES)
    flat_scale = scale.squeeze(-1).reshape(-1).to(torch.float16).contiguous().view(torch.uint8)
    scale_bytes = flat_scale.reshape(*scale.shape[:-1], NORM_BYTES)
    flat_zero = zero.squeeze(-1).reshape(-1).to(torch.float16).contiguous().view(torch.uint8)
    zero_bytes = flat_zero.reshape(*zero.shape[:-1], NORM_BYTES)

    return torch.cat([quant_packed, scale_bytes, zero_bytes, norm_bytes], dim=-1)


def planar_decode(
    packed: torch.Tensor, head_dim: int, bits: int, dtype: torch.dtype,
) -> torch.Tensor:
    """Decode PlanarQuant: unpack -> dequantize -> inverse Givens."""
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
    unit_hat = _apply_givens_rotation(rotated_hat, inverse=True)
    return (unit_hat * norms).to(dtype)


def planar_packed_dim(head_dim: int, bits: int) -> int:
    quant_bytes = (head_dim * bits + 7) // 8
    return quant_bytes + NORM_BYTES * 3


# ─── Method classes ─────────────────────────────────────────────────────────────

class _PlanarQuantBase(CompressionMethod):
    def __init__(self, bits: int = 3, method: CacheMethod = CacheMethod.PLANAR3):
        self._bits = bits
        self._method = method

    def packed_dim(self, head_dim: int) -> int:
        return planar_packed_dim(head_dim, self._bits)

    def encode(
        self, x: torch.Tensor, *,
        head_indices: torch.Tensor | None = None, layer_idx: int = 0,
    ) -> CompressedKV:
        packed = planar_encode(x, self._bits)
        return CompressedKV(
            data=packed, head_dim=x.shape[-1], method=self._method,
        )

    def decode(
        self, compressed: CompressedKV, *, dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return planar_decode(
            compressed.data, compressed.head_dim, self._bits, dtype,
        )

    def validate_head_dim(self, head_dim: int) -> bool:
        return head_dim % PLANAR_GROUP_SIZE == 0

    def supports_triton(self) -> bool:
        return True

    def supports_cuda(self) -> bool:
        return True

    def supports_metal(self) -> bool:
        return True  # johndpope fork includes Metal kernels


@register_method(CacheMethod.PLANAR3)
class PlanarQuant3(_PlanarQuantBase):
    def __init__(self, **kwargs):
        super().__init__(bits=3, method=CacheMethod.PLANAR3)

    def info(self) -> MethodInfo:
        return MethodInfo(
            method=CacheMethod.PLANAR3,
            family=MethodFamily.PLANARQUANT,
            bits=3.25, requires_calibration=False,
            supports_asymmetric=True,
            supported_head_dims=(64, 128),
            transform_name="Givens 2D rotation (golden-angle spaced)",
            description="3.25-bit Givens rotation, simplest transform, Metal support",
            fma_count=256, param_count=128,
        )


@register_method(CacheMethod.PLANAR4)
class PlanarQuant4(_PlanarQuantBase):
    def __init__(self, **kwargs):
        super().__init__(bits=4, method=CacheMethod.PLANAR4)

    def info(self) -> MethodInfo:
        return MethodInfo(
            method=CacheMethod.PLANAR4,
            family=MethodFamily.PLANARQUANT,
            bits=4.25, requires_calibration=False,
            supports_asymmetric=True,
            supported_head_dims=(64, 128),
            transform_name="Givens 2D rotation (golden-angle spaced)",
            description="4.25-bit Givens rotation, higher quality, Metal support",
            fma_count=256, param_count=128,
        )


# Public alias
PlanarQuantMethod = PlanarQuant3
