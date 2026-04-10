# SPDX-License-Identifier: MIT
# Derived from TheTom/llama-cpp-turboquant and spiritbuun/buun-llama-cpp (MIT/Apache-2.0)
"""TurboQuant — Walsh-Hadamard Transform KV cache compression.

Uses 128-dimensional WHT butterfly networks for O(d log d) transform,
with mixed-precision MSE + 1-bit QJL residual encoding.

Supports turbo2 (2.25-bit, ~7x), turbo3 (3.25-bit, ~5x), turbo4 (4.25-bit, ~3.8x).
Requires per-model calibration (turboquant_kv.json).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch

from ..config import CacheMethod, MethodFamily
from ..registry import register_method
from .base import CompressionMethod, CompressedKV, MethodInfo


# ─── Constants ──────────────────────────────────────────────────────────────────

OUTLIER_RATIOS = {"turbo2": 0.25, "turbo3": 0.50, "turbo4": 0.50}
GROUP_BITS = {"turbo2": (3, 2), "turbo3": (4, 3), "turbo4": (5, 4)}
BITS_PER_ELEMENT = {"turbo2": 2.25, "turbo3": 3.25, "turbo4": 4.25}
GROUP_ALIGNMENT = 16
SEED = 20250428
QJL_SEED_OFFSET = 10_000
QJL_SCALE = math.sqrt(math.pi / 2.0)
CODEBOOK_GRID_POINTS = 32768
CODEBOOK_EPS = 1e-6
VECTOR_NORM_BYTES = 2
RESIDUAL_NORM_BYTES = 2


# ─── Layout ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GroupLayout:
    dim: int
    bits: int
    mse_bits: int
    mse_payload_bytes: int
    qjl_payload_bytes: int
    qjl_offset: int
    vector_norm_offset: int
    residual_norm_offset: int
    packed_bytes: int


@dataclass(frozen=True)
class TurboLayout:
    groups: tuple[GroupLayout, GroupLayout]
    packed_dim: int


def get_outlier_count(head_dim: int, method: str) -> int:
    ratio = OUTLIER_RATIOS[method]
    aligned = int(round(head_dim * ratio / GROUP_ALIGNMENT) * GROUP_ALIGNMENT)
    if aligned <= 0 or aligned >= head_dim:
        raise ValueError(f"Unsupported head_dim {head_dim} for {method}")
    return aligned


def get_group_dims(head_dim: int, method: str) -> tuple[int, int]:
    outlier_count = get_outlier_count(head_dim, method)
    return outlier_count, head_dim - outlier_count


@cache
def compute_layout(method: str, head_dim: int) -> TurboLayout:
    group_dims = get_group_dims(head_dim, method)
    bits = GROUP_BITS[method]
    groups: list[GroupLayout] = []
    cursor = 0
    for dim, b in zip(group_dims, bits):
        mse_bits = b - 1
        mse_payload_bytes = (dim * mse_bits + 7) // 8
        qjl_payload_bytes = (dim + 7) // 8
        qjl_offset = cursor + mse_payload_bytes
        vector_norm_offset = qjl_offset + qjl_payload_bytes
        residual_norm_offset = vector_norm_offset + VECTOR_NORM_BYTES
        packed_bytes = (
            mse_payload_bytes + qjl_payload_bytes
            + VECTOR_NORM_BYTES + RESIDUAL_NORM_BYTES
        )
        groups.append(GroupLayout(
            dim=dim, bits=b, mse_bits=mse_bits,
            mse_payload_bytes=mse_payload_bytes,
            qjl_payload_bytes=qjl_payload_bytes,
            qjl_offset=qjl_offset,
            vector_norm_offset=vector_norm_offset,
            residual_norm_offset=residual_norm_offset,
            packed_bytes=packed_bytes,
        ))
        cursor += packed_bytes
    return TurboLayout(groups=tuple(groups), packed_dim=cursor)


# ─── Transforms ─────────────────────────────────────────────────────────────────

@cache
def _hadamard_block_sizes(dim: int) -> tuple[int, ...]:
    sizes: list[int] = []
    remaining = dim
    while remaining > 0:
        block = 1 << (remaining.bit_length() - 1)
        sizes.append(block)
        remaining -= block
    return tuple(sizes)


def _fwht_pow2(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform for power-of-2 dimensions."""
    orig_shape = x.shape
    size = orig_shape[-1]
    out = x.reshape(-1, size)
    block = 1
    while block < size:
        out = out.reshape(out.shape[0], -1, block * 2)
        left = out[..., :block]
        right = out[..., block : 2 * block]
        out = torch.cat((left + right, left - right), dim=-1)
        out = out.reshape(-1, size)
        block *= 2
    return out.reshape(orig_shape)


@cache
def _structured_signs(
    device_type: str, device_index: int | None, dim: int, seed: int,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    gen = torch.Generator(device=device.type)
    gen.manual_seed(seed)
    rng_device = device if device.type == "cuda" else device
    bits = torch.randint(0, 2, (dim,), generator=gen, device=rng_device)
    return torch.where(
        bits > 0,
        torch.ones(dim, dtype=torch.float32, device=device),
        -torch.ones(dim, dtype=torch.float32, device=device),
    )


def _apply_block_hadamard(
    x: torch.Tensor, signs: torch.Tensor,
    *, normalized: bool, inverse: bool,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    cursor = 0
    for block_size in _hadamard_block_sizes(x.shape[-1]):
        block = x[..., cursor : cursor + block_size]
        block_signs = signs[cursor : cursor + block_size]
        if inverse:
            block = _fwht_pow2(block)
            block = block * block_signs
        else:
            block = block * block_signs
            block = _fwht_pow2(block)
        if normalized:
            block = block / math.sqrt(block_size)
        outputs.append(block)
        cursor += block_size
    return torch.cat(outputs, dim=-1)


def mse_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return _apply_block_hadamard(x, signs, normalized=True, inverse=False)


def mse_inverse_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return _apply_block_hadamard(x, signs, normalized=True, inverse=True)


def qjl_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return _apply_block_hadamard(x, signs, normalized=False, inverse=False)


def qjl_inverse_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return _apply_block_hadamard(x, signs, normalized=False, inverse=True)


def get_rotation(device: torch.device, dim: int, seed_offset: int = 0) -> torch.Tensor:
    return _structured_signs(device.type, device.index, dim, SEED + seed_offset + dim)


def get_qjl_matrix(device: torch.device, dim: int, seed_offset: int = 0) -> torch.Tensor:
    return _structured_signs(
        device.type, device.index, dim,
        SEED + QJL_SEED_OFFSET + seed_offset + dim,
    )


# ─── Codebook ───────────────────────────────────────────────────────────────────

def _beta_pdf(x: torch.Tensor, dim: int) -> torch.Tensor:
    exponent = 0.5 * (dim - 3)
    log_norm = (
        torch.lgamma(torch.tensor(dim / 2.0, dtype=torch.float64))
        - 0.5 * math.log(math.pi)
        - torch.lgamma(torch.tensor((dim - 1) / 2.0, dtype=torch.float64))
    )
    base = (1.0 - x.square()).clamp_min(CODEBOOK_EPS)
    return torch.exp(log_norm + exponent * torch.log(base))


@cache
def _dimension_aware_codebook(dim: int, bits: int) -> torch.Tensor:
    """Lloyd's algorithm with beta-distribution weighting."""
    levels = 1 << bits
    grid = torch.linspace(
        -1.0 + CODEBOOK_EPS, 1.0 - CODEBOOK_EPS,
        CODEBOOK_GRID_POINTS, dtype=torch.float64,
    )
    weights = _beta_pdf(grid, dim)
    centroids = torch.linspace(
        -1.0 + 1.0 / (levels + 1), 1.0 - 1.0 / (levels + 1),
        levels, dtype=torch.float64,
    )
    for _ in range(200):
        bounds = torch.empty(levels + 1, dtype=torch.float64)
        bounds[0] = -1.0
        bounds[-1] = 1.0
        bounds[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        assignments = torch.bucketize(grid, bounds[1:-1])
        masses = torch.zeros(levels, dtype=torch.float64)
        sums = torch.zeros(levels, dtype=torch.float64)
        masses.scatter_add_(0, assignments, weights)
        sums.scatter_add_(0, assignments, weights * grid)
        new_centroids = sums / masses.clamp_min(1e-18)
        if torch.max(torch.abs(new_centroids - centroids)) < 1e-10:
            break
        centroids = new_centroids
    return centroids.to(torch.float32)


def get_centroids(device: torch.device, dim: int, bits: int) -> torch.Tensor:
    return _dimension_aware_codebook(dim, bits).to(device=device)


# ─── Bit packing ────────────────────────────────────────────────────────────────

@cache
def _bit_layout(
    device_type: str, device_index: int | None, head_size: int, bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(device_type, device_index)
    bit_offsets = torch.arange(bits, dtype=torch.int64, device=device)
    bit_offsets = bit_offsets.expand(head_size, bits).reshape(-1)
    flat_positions = torch.arange(head_size * bits, dtype=torch.int64, device=device)
    return bit_offsets, flat_positions // 8, flat_positions % 8


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    head_size = indices.shape[-1]
    packed_bytes = (head_size * bits + 7) // 8
    if packed_bytes == 0:
        return torch.empty(
            (*indices.shape[:-1], 0), dtype=torch.uint8, device=indices.device,
        )
    bit_offsets, byte_index, bit_index = _bit_layout(
        indices.device.type, indices.device.index, head_size, bits,
    )
    bit_offsets = bit_offsets.reshape(*((1,) * (indices.ndim - 1)), head_size, bits)
    bit_index = bit_index.reshape(*((1,) * (indices.ndim - 1)), head_size, bits)
    contrib = (
        ((indices.unsqueeze(-1).to(torch.int32) >> bit_offsets) & 1) << bit_index
    ).to(torch.int32)
    contrib = contrib.reshape(*indices.shape[:-1], head_size * bits)
    packed = torch.zeros(
        (*indices.shape[:-1], packed_bytes), dtype=torch.int32, device=indices.device,
    )
    scatter_index = byte_index.reshape(
        *((1,) * (indices.ndim - 1)), head_size * bits,
    ).expand_as(contrib)
    packed.scatter_add_(-1, scatter_index, contrib)
    return packed.to(torch.uint8)


def unpack_indices(packed: torch.Tensor, head_size: int, bits: int) -> torch.Tensor:
    if bits == 0:
        return torch.zeros(
            (*packed.shape[:-1], head_size), dtype=torch.uint8, device=packed.device,
        )
    bit_offsets, byte_index, bit_index = _bit_layout(
        packed.device.type, packed.device.index, head_size, bits,
    )
    gather_index = byte_index.reshape(
        *((1,) * (packed.ndim - 1)), head_size * bits,
    ).expand(*packed.shape[:-1], head_size * bits)
    selected = packed.to(torch.int32).gather(-1, gather_index)
    bits_view = ((selected >> bit_index) & 1).reshape(
        *packed.shape[:-1], head_size, bits
    )
    bit_offsets = bit_offsets.reshape(*((1,) * (packed.ndim - 1)), head_size, bits)
    return (bits_view << bit_offsets).sum(dim=-1).to(torch.uint8)


# ─── Norm packing ───────────────────────────────────────────────────────────────

def _norms_to_bytes(norms: torch.Tensor, byte_width: int) -> torch.Tensor:
    norm_half = norms.to(torch.float16).contiguous()
    return norm_half.reshape(-1).view(torch.uint8).reshape(*norm_half.shape, byte_width)


def _bytes_to_norms(norm_bytes: torch.Tensor, byte_width: int) -> torch.Tensor:
    raw = norm_bytes.contiguous().reshape(-1, byte_width).view(torch.float16)
    return raw.reshape(*norm_bytes.shape[:-1], 1).to(torch.float32)


# ─── Group scatter/gather ───────────────────────────────────────────────────────

def _gather_group(x: torch.Tensor, group_indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        x, dim=-1,
        index=group_indices.unsqueeze(0).expand(x.shape[0], -1, -1),
    )


def _scatter_output(
    head_size: int, dtype: torch.dtype,
    group_outputs: tuple[torch.Tensor, torch.Tensor],
    group_indices: tuple[torch.Tensor, torch.Tensor],
    num_heads: int,
) -> torch.Tensor:
    kv_head_map = torch.arange(num_heads, device=group_outputs[0].device, dtype=torch.int64)
    kv_head_map = kv_head_map % group_indices[0].shape[0]
    output = torch.zeros(
        (*group_outputs[0].shape[:-1], head_size),
        dtype=torch.float32, device=group_outputs[0].device,
    )
    for group_output, indices in zip(group_outputs, group_indices):
        per_query = indices.index_select(0, kv_head_map)
        output.scatter_add_(
            -1,
            per_query.unsqueeze(0).expand(group_output.shape[0], -1, -1),
            group_output,
        )
    return output.to(dtype=dtype)


# ─── Outlier mask computation ───────────────────────────────────────────────────

def build_outlier_masks(
    x: torch.Tensor, method: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute outlier (high-variance) and regular dimension indices."""
    outlier_count, _ = get_group_dims(x.shape[-1], method)
    x_fp32 = x.to(torch.float32)
    score = x_fp32.reshape(-1, x_fp32.shape[-2], x_fp32.shape[-1]).square().mean(dim=0)
    outlier_idx = torch.topk(score, k=outlier_count, dim=-1).indices
    outlier_idx = torch.sort(outlier_idx, dim=-1).values
    all_idx = torch.arange(x.shape[-1], device=x.device, dtype=torch.int64)
    all_idx = all_idx.unsqueeze(0).expand(score.shape[0], -1)
    regular_mask = torch.ones_like(all_idx, dtype=torch.bool)
    regular_mask.scatter_(1, outlier_idx, False)
    regular_idx = all_idx[regular_mask].reshape(score.shape[0], -1)
    return outlier_idx, regular_idx


# ─── Encode / Decode ────────────────────────────────────────────────────────────

def quantize_vectors(
    x: torch.Tensor,
    method: str,
    rotations: tuple[torch.Tensor, torch.Tensor],
    qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    centroids: dict[int, torch.Tensor],
    group_indices: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Quantize KV vectors using TurboQuant WHT encoding."""
    layout = compute_layout(method, x.shape[-1])
    groups = tuple(
        _gather_group(x.to(torch.float32), indices)
        for indices in group_indices
    )
    packed_groups: list[torch.Tensor] = []

    for group_x, gl, rotation, qjl_matrix in zip(
        groups, layout.groups, rotations, qjl_matrices,
    ):
        vector_norms = group_x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        unit = group_x / vector_norms
        rotated = mse_transform(unit, rotation)

        if gl.mse_bits > 0:
            mse_centroids = centroids[gl.mse_bits]
            mse_indices = torch.abs(rotated.unsqueeze(-1) - mse_centroids).argmin(dim=-1)
            indices_u8 = mse_indices.to(torch.uint8)
            rotated_hat = mse_centroids[mse_indices.long()]
        else:
            indices_u8 = torch.zeros_like(rotated, dtype=torch.uint8)
            rotated_hat = torch.zeros_like(rotated)

        mse_hat = mse_inverse_transform(rotated_hat, rotation)
        residual = unit - mse_hat
        residual_norms = residual.norm(dim=-1, keepdim=True)
        qjl_bits = (qjl_transform(residual, qjl_matrix) >= 0).to(torch.uint8)

        packed_groups.append(torch.cat((
            pack_indices(indices_u8, gl.mse_bits),
            pack_indices(qjl_bits, 1),
            _norms_to_bytes(vector_norms.squeeze(-1), VECTOR_NORM_BYTES),
            _norms_to_bytes(residual_norms.squeeze(-1), RESIDUAL_NORM_BYTES),
        ), dim=-1))

    return torch.cat(packed_groups, dim=-1)


def dequantize_vectors(
    packed: torch.Tensor,
    method: str,
    head_dim: int,
    rotations: tuple[torch.Tensor, torch.Tensor],
    qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    centroids: dict[int, torch.Tensor],
    group_indices: tuple[torch.Tensor, torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize TurboQuant packed vectors back to original space."""
    layout = compute_layout(method, head_dim)
    group_outputs: list[torch.Tensor] = []
    cursor = 0

    for gl, rotation, qjl_matrix in zip(
        layout.groups, rotations, qjl_matrices,
    ):
        group_packed = packed[..., cursor : cursor + gl.packed_bytes]
        cursor += gl.packed_bytes
        gc = 0

        mse_indices = unpack_indices(
            group_packed[..., gc : gc + gl.mse_payload_bytes], gl.dim, gl.mse_bits,
        )
        gc += gl.mse_payload_bytes

        qjl_bits = unpack_indices(
            group_packed[..., gc : gc + gl.qjl_payload_bytes], gl.dim, 1,
        )
        gc += gl.qjl_payload_bytes

        vector_norms = _bytes_to_norms(
            group_packed[..., gc : gc + VECTOR_NORM_BYTES], VECTOR_NORM_BYTES,
        )
        gc += VECTOR_NORM_BYTES

        residual_norms = _bytes_to_norms(
            group_packed[..., gc : gc + RESIDUAL_NORM_BYTES], RESIDUAL_NORM_BYTES,
        )

        rotated_hat = torch.zeros(
            (*group_packed.shape[:-1], gl.dim),
            dtype=torch.float32, device=packed.device,
        )
        if gl.mse_bits > 0:
            rotated_hat = centroids[gl.mse_bits][mse_indices.long()]

        mse_hat = mse_inverse_transform(rotated_hat, rotation)
        qjl_signs = qjl_bits.to(torch.float32).mul_(2.0).sub_(1.0)
        qjl_hat = qjl_inverse_transform(qjl_signs, qjl_matrix) * (
            QJL_SCALE / gl.dim
        )
        group_outputs.append((mse_hat + qjl_hat * residual_norms) * vector_norms)

    return _scatter_output(
        head_dim, dtype, (group_outputs[0], group_outputs[1]),
        group_indices, packed.shape[-2],
    )


# ─── Method class ───────────────────────────────────────────────────────────────

class _TurboQuantBase(CompressionMethod):
    """Base class for TurboQuant variants (turbo2/3/4)."""

    def __init__(
        self,
        method: str = "turbo3",
        group_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
        head_dim: int = 128,
    ):
        self._method = method
        self._group_indices = group_indices
        self._head_dim = head_dim

    def _get_transforms(
        self, device: torch.device,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        dict[int, torch.Tensor],
    ]:
        dims = get_group_dims(self._head_dim, self._method)
        rotations = tuple(get_rotation(device, d) for d in dims)
        qjl_matrices = tuple(get_qjl_matrix(device, d) for d in dims)
        bits = GROUP_BITS[self._method]
        centroids_dict = {}
        for d, b in zip(dims, bits):
            mse_bits = b - 1
            if mse_bits > 0:
                centroids_dict[mse_bits] = get_centroids(device, d, mse_bits)
        return rotations, qjl_matrices, centroids_dict

    def packed_dim(self, head_dim: int) -> int:
        return compute_layout(self._method, head_dim).packed_dim

    def encode(
        self, x: torch.Tensor, *,
        head_indices: torch.Tensor | None = None, layer_idx: int = 0,
    ) -> CompressedKV:
        device = x.device
        rotations, qjl_matrices, centroids = self._get_transforms(device)
        if self._group_indices is not None:
            gi = (
                self._group_indices[0].to(device),
                self._group_indices[1].to(device),
            )
        else:
            gi = build_outlier_masks(x, self._method)
        packed = quantize_vectors(
            x, self._method, rotations, qjl_matrices, centroids, gi,
        )
        return CompressedKV(
            data=packed, head_dim=x.shape[-1],
            method=CacheMethod(self._method),
            metadata={"group_indices": gi},
        )

    def decode(
        self, compressed: CompressedKV, *, dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        device = compressed.data.device
        rotations, qjl_matrices, centroids = self._get_transforms(device)
        gi = compressed.metadata["group_indices"]
        return dequantize_vectors(
            compressed.data, compressed.method.value, compressed.head_dim,
            rotations, qjl_matrices, centroids, gi, dtype,
        )

    def supports_triton(self) -> bool:
        return True

    def supports_cuda(self) -> bool:
        return True


@register_method(CacheMethod.TURBO2)
class TurboQuant2(_TurboQuantBase):
    def __init__(self, **kwargs):
        super().__init__(method="turbo2", **kwargs)

    def info(self) -> MethodInfo:
        return MethodInfo(
            method=CacheMethod.TURBO2,
            family=MethodFamily.TURBOQUANT,
            bits=2.25, requires_calibration=True,
            supports_asymmetric=True,
            supported_head_dims=(64, 128, 192, 256),
            transform_name="Walsh-Hadamard (128-d butterfly)",
            description="2.25-bit WHT compression, ~7x savings, highest compression",
            fma_count=16384, param_count=16384,
        )


@register_method(CacheMethod.TURBO3)
class TurboQuant3(_TurboQuantBase):
    def __init__(self, **kwargs):
        super().__init__(method="turbo3", **kwargs)

    def info(self) -> MethodInfo:
        return MethodInfo(
            method=CacheMethod.TURBO3,
            family=MethodFamily.TURBOQUANT,
            bits=3.25, requires_calibration=True,
            supports_asymmetric=True,
            supported_head_dims=(64, 128, 192, 256),
            transform_name="Walsh-Hadamard (128-d butterfly)",
            description="3.25-bit WHT compression, ~5x savings, fastest symmetric on Ampere",
            fma_count=16384, param_count=16384,
        )


@register_method(CacheMethod.TURBO4)
class TurboQuant4(_TurboQuantBase):
    def __init__(self, **kwargs):
        super().__init__(method="turbo4", **kwargs)

    def info(self) -> MethodInfo:
        return MethodInfo(
            method=CacheMethod.TURBO4,
            family=MethodFamily.TURBOQUANT,
            bits=4.25, requires_calibration=True,
            supports_asymmetric=True,
            supported_head_dims=(64, 128, 192, 256),
            transform_name="Walsh-Hadamard (128-d butterfly)",
            description="4.25-bit WHT compression, ~3.8x savings, near-lossless quality",
            fma_count=16384, param_count=16384,
        )


# Public alias
TurboQuantMethod = TurboQuant3
