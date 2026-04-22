# SPDX-License-Identifier: MIT
# Derived from scrya-com/rotorquant (MIT)
"""RotorQuant — SO(3) sandwich-product rotation KV cache compression.

Implements the Clifford algebra Cl(3,0) rotor sandwich variant from
scrya-com/rotorquant. A unit rotor R parameterizes an SO(3) rotation
via the double-sided sandwich product R·v·R̃ applied to each group of
3 dimensions.

Reference Python implementation: the sandwich product of a pure vector
in Cl(3,0) is mathematically equivalent to a fixed 3x3 SO(3) rotation
matrix, which we precompute once from a golden-angle rotor about the
(1,1,1)/√3 axis (maximizes decorrelation across three dims). Upstream's
8-component multivector form fuses into a single Triton kernel; we
defer that to a future kernel port and run the 3x3 variant here.

Head-dim padding: 64 and 128 are not divisible by 3, so we pad with
zeros to the next multiple of 3 (66 and 129) before rotation, and
truncate on decode. Storage cost: 1-2 extra uint8 quant indices per
head, negligible vs the compression ratio.

rotor3 (3.25-bit) is stable. rotor4 (4.25-bit) is gated experimental —
upstream's 4-bit path has known dispatch crashes; our pure-torch impl
round-trips cleanly but the method is flagged until deeper validation.
"""

from __future__ import annotations

import math
import warnings
from functools import cache

import torch

from ..config import CacheMethod, MethodFamily
from ..registry import register_method
from .base import CompressionMethod, CompressedKV, MethodInfo


# ─── Constants ──────────────────────────────────────────────────────────────────

ROTOR_BITS = {"rotor3": 3.25, "rotor4": 4.25}
ROTOR_QUANT_BITS = {"rotor3": 3, "rotor4": 4}
ROTOR_GROUP_SIZE = 3  # Cl(3,0) rotor operates on 3-dim vector groups

NORM_BYTES = 2  # fp16 scalar storage

# Golden angle ≈ 137.5° — maximizes decorrelation between successive rotations
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


# ─── SO(3) rotation matrix from golden-angle rotor ──────────────────────────────

@cache
def _rotation_matrix_3d(device_type: str, device_index: int | None) -> torch.Tensor:
    """Build the 3x3 SO(3) rotation matrix from the golden-angle rotor.

    The rotor R = cos(θ/2) + sin(θ/2)·B rotates a pure vector v in Cl(3,0)
    via R·v·R̃, which on pure vectors is equivalent to an SO(3) matrix
    rotation. We choose axis n = (1,1,1)/√3 (symmetric across three dims)
    and θ = golden angle.

    Rodrigues' formula: R = I + sin(θ)·K + (1 - cos(θ))·K²
    where K is the skew-symmetric cross-product matrix of n.

    The matrix is orthogonal, so inverse = transpose.
    """
    device = torch.device(device_type, device_index)
    theta = GOLDEN_ANGLE
    c, s = math.cos(theta), math.sin(theta)

    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    # Skew-symmetric matrix K of n = (1,1,1)/√3
    K = torch.tensor([
        [0.0,        -inv_sqrt3,  inv_sqrt3],
        [inv_sqrt3,   0.0,       -inv_sqrt3],
        [-inv_sqrt3,  inv_sqrt3,  0.0],
    ], dtype=torch.float32, device=device)
    I = torch.eye(3, dtype=torch.float32, device=device)
    K2 = K @ K
    R = I + s * K + (1.0 - c) * K2
    return R.contiguous()


@cache
def _inverse_rotation_3d(device_type: str, device_index: int | None) -> torch.Tensor:
    """Inverse rotation = transpose (orthogonal matrix)."""
    return _rotation_matrix_3d(device_type, device_index).T.contiguous()


def _padded_head_dim(head_dim: int) -> int:
    """Smallest multiple of 3 ≥ head_dim."""
    return ((head_dim + ROTOR_GROUP_SIZE - 1) // ROTOR_GROUP_SIZE) * ROTOR_GROUP_SIZE


def _apply_rotor_sandwich(
    x: torch.Tensor, inverse: bool = False,
) -> torch.Tensor:
    """Apply SO(3) rotor-sandwich rotation to groups of 3 dimensions.

    Input x is assumed to already be padded to a multiple of 3 along the
    last axis. Reshapes [..., head_dim_padded] -> [..., num_groups, 3],
    applies R (or R^T for inverse), reshapes back.
    """
    orig_shape = x.shape
    head_dim = orig_shape[-1]
    assert head_dim % ROTOR_GROUP_SIZE == 0, (
        f"RotorQuant requires head_dim divisible by {ROTOR_GROUP_SIZE}, got {head_dim}"
    )

    device = x.device
    R = (
        _inverse_rotation_3d(device.type, device.index)
        if inverse
        else _rotation_matrix_3d(device.type, device.index)
    )

    grouped = x.reshape(*orig_shape[:-1], head_dim // ROTOR_GROUP_SIZE, ROTOR_GROUP_SIZE)
    # Rotate: [..., g, 3] @ [3, 3] -> [..., g, 3]
    rotated = torch.matmul(grouped.to(torch.float32), R.T)
    return rotated.reshape(orig_shape)


# ─── Uniform quantization (same pattern as isoquant) ────────────────────────────

def _uniform_quantize(
    x: torch.Tensor, bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniform quantization with per-vector min/max scaling."""
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


# ─── Bit packing ────────────────────────────────────────────────────────────────

def _pack_uniform(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uniform quantized indices into bytes."""
    head_dim = indices.shape[-1]
    total_bits = head_dim * bits
    packed_bytes = (total_bits + 7) // 8

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

def rotor_encode(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Encode via RotorQuant: pad → normalize → rotor-sandwich → quantize → pack.

    Packed format per head: [quant_indices | scale | zero | norm] (fp16 scalars).
    """
    x_fp32 = x.to(torch.float32)
    head_dim = x_fp32.shape[-1]
    padded_dim = _padded_head_dim(head_dim)

    if padded_dim != head_dim:
        pad_amount = padded_dim - head_dim
        x_fp32 = torch.nn.functional.pad(x_fp32, (0, pad_amount), value=0.0)

    norms = x_fp32.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit = x_fp32 / norms

    rotated = _apply_rotor_sandwich(unit)

    indices, scale, zero = _uniform_quantize(rotated, bits)
    quant_packed = _pack_uniform(indices, bits)

    flat_norms = norms.squeeze(-1).reshape(-1).to(torch.float16).contiguous().view(torch.uint8)
    norm_bytes = flat_norms.reshape(*norms.shape[:-1], NORM_BYTES)
    flat_scale = scale.squeeze(-1).reshape(-1).to(torch.float16).contiguous().view(torch.uint8)
    scale_bytes = flat_scale.reshape(*scale.shape[:-1], NORM_BYTES)
    flat_zero = zero.squeeze(-1).reshape(-1).to(torch.float16).contiguous().view(torch.uint8)
    zero_bytes = flat_zero.reshape(*zero.shape[:-1], NORM_BYTES)

    return torch.cat([quant_packed, scale_bytes, zero_bytes, norm_bytes], dim=-1)


def rotor_decode(
    packed: torch.Tensor, head_dim: int, bits: int, dtype: torch.dtype,
) -> torch.Tensor:
    """Decode RotorQuant: unpack → dequantize → inverse rotor → unpad."""
    padded_dim = _padded_head_dim(head_dim)
    quant_bytes = (padded_dim * bits + 7) // 8
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

    indices = _unpack_uniform(quant_packed, padded_dim, bits)
    rotated_hat = _uniform_dequantize(indices, scale, zero)
    unit_hat = _apply_rotor_sandwich(rotated_hat, inverse=True)
    full = (unit_hat * norms)
    if padded_dim != head_dim:
        full = full[..., :head_dim]
    return full.to(dtype)


def rotor_packed_dim(head_dim: int, bits: int) -> int:
    """Total packed bytes per head (uses padded head_dim for quant region)."""
    padded_dim = _padded_head_dim(head_dim)
    quant_bytes = (padded_dim * bits + 7) // 8
    return quant_bytes + NORM_BYTES * 3  # scale + zero + norm


# ─── Method classes ─────────────────────────────────────────────────────────────

_EXPERIMENTAL_WARNED: set[CacheMethod] = set()


class _RotorQuantBase(CompressionMethod):
    def __init__(
        self, bits: int = 3, method: CacheMethod = CacheMethod.ROTOR3,
        experimental: bool = False,
    ):
        self._bits = bits
        self._method = method
        self._experimental = experimental
        if experimental and method not in _EXPERIMENTAL_WARNED:
            warnings.warn(
                f"{method.value} is experimental — upstream's 4-bit rotor path has "
                "known dispatch crashes and quality regressions vs iso4. "
                "Use for research/testing only.",
                UserWarning,
                stacklevel=2,
            )
            _EXPERIMENTAL_WARNED.add(method)

    def packed_dim(self, head_dim: int) -> int:
        return rotor_packed_dim(head_dim, self._bits)

    def encode(
        self, x: torch.Tensor, *,
        head_indices: torch.Tensor | None = None, layer_idx: int = 0,
    ) -> CompressedKV:
        packed = rotor_encode(x, self._bits)
        return CompressedKV(
            data=packed, head_dim=x.shape[-1], method=self._method,
        )

    def decode(
        self, compressed: CompressedKV, *, dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return rotor_decode(
            compressed.data, compressed.head_dim, self._bits, dtype,
        )

    def validate_head_dim(self, head_dim: int) -> bool:
        return head_dim in self.info().supported_head_dims

    def supports_triton(self) -> bool:
        return False  # Triton kernel port deferred — pure torch path only.

    def supports_cuda(self) -> bool:
        return False  # No dedicated CUDA kernel; runs via torch ops on cuda.

    def supports_metal(self) -> bool:
        return False


@register_method(CacheMethod.ROTOR3)
class RotorQuant3(_RotorQuantBase):
    def __init__(self, **kwargs):
        super().__init__(bits=3, method=CacheMethod.ROTOR3)

    def info(self) -> MethodInfo:
        return MethodInfo(
            method=CacheMethod.ROTOR3,
            family=MethodFamily.ROTORQUANT,
            bits=3.25, requires_calibration=False,
            supports_asymmetric=True,
            supported_head_dims=(64, 128),
            transform_name="Cl(3,0) rotor sandwich (SO(3) rotation)",
            description="3.25-bit rotor sandwich, no calibration, groups of 3 dims (padded)",
            fma_count=128, param_count=9,  # 3x3 rotation matrix
        )


@register_method(CacheMethod.ROTOR4)
class RotorQuant4(_RotorQuantBase):
    def __init__(self, **kwargs):
        super().__init__(bits=4, method=CacheMethod.ROTOR4, experimental=True)

    def info(self) -> MethodInfo:
        return MethodInfo(
            method=CacheMethod.ROTOR4,
            family=MethodFamily.ROTORQUANT,
            bits=4.25, requires_calibration=False,
            supports_asymmetric=True,
            supported_head_dims=(64, 128),
            transform_name="Cl(3,0) rotor sandwich (SO(3) rotation)",
            description="4.25-bit rotor sandwich — experimental, quality regressions vs iso4",
            fma_count=128, param_count=9,
            experimental=True,
        )


# Public alias
RotorQuantMethod = RotorQuant3
