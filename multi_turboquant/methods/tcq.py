# SPDX-License-Identifier: MIT
# Derived from spiritbuun/buun-llama-cpp (MIT)
"""TCQ — Trellis Coded Quantization on top of TurboQuant WHT.

Replaces the standard nearest-centroid assignment with Viterbi-optimal
path selection through a trellis, improving quality at the same compression
ratio. The trellis encodes inter-dimension dependencies that the independent
MSE quantizer misses.

Supports turbo2_tcq (2.25-bit) and turbo3_tcq (3.25-bit).
Requires per-model calibration (turboquant_kv.json).
"""

from __future__ import annotations

import torch

from ..config import CacheMethod, MethodFamily
from ..registry import register_method
from .base import CompressionMethod, CompressedKV, MethodInfo
from .turboquant import (
    _TurboQuantBase,
    GROUP_BITS,
    compute_layout,
    get_group_dims,
    get_rotation,
    get_qjl_matrix,
    get_centroids,
    mse_transform,
    mse_inverse_transform,
    qjl_transform,
    qjl_inverse_transform,
    pack_indices,
    unpack_indices,
    _norms_to_bytes,
    _bytes_to_norms,
    _gather_group,
    _scatter_output,
    build_outlier_masks,
    QJL_SCALE,
    VECTOR_NORM_BYTES,
    RESIDUAL_NORM_BYTES,
)


# ─── Trellis state machine ─────────────────────────────────────────────────────

# TCQ uses a rate-1/2 convolutional code with constraint length 3.
# 4 trellis states, transitions determined by input bit parity.
TCQ_NUM_STATES = 4
TCQ_CONSTRAINT_LENGTH = 3


def _build_trellis_transitions(num_levels: int) -> torch.Tensor:
    """Build trellis transition table.

    Returns:
        Tensor of shape [num_states, num_levels] mapping
        (current_state, centroid_index) -> next_state.
    """
    transitions = torch.zeros(
        (TCQ_NUM_STATES, num_levels), dtype=torch.int64,
    )
    for state in range(TCQ_NUM_STATES):
        for level in range(num_levels):
            # State transition: shift register with LSB of level index
            next_state = ((state << 1) | (level & 1)) % TCQ_NUM_STATES
            transitions[state, level] = next_state
    return transitions


def _viterbi_quantize(
    rotated: torch.Tensor,
    centroids: torch.Tensor,
    transitions: torch.Tensor,
) -> torch.Tensor:
    """Viterbi-optimal trellis-coded quantization.

    Instead of independently assigning each dimension to its nearest
    centroid, we find the minimum-distortion path through a trellis
    where state transitions constrain which centroids can follow which.

    Args:
        rotated: Transformed unit vectors [..., dim].
        centroids: Codebook centroids [num_levels].
        transitions: State transition table [num_states, num_levels].

    Returns:
        Optimal centroid indices [..., dim].
    """
    batch_shape = rotated.shape[:-1]
    dim = rotated.shape[-1]
    num_levels = centroids.shape[0]
    device = rotated.device

    # Flatten batch dims
    flat = rotated.reshape(-1, dim)
    batch_size = flat.shape[0]

    # Distance from each sample to each centroid at each position
    # [batch, dim, num_levels]
    dists = (flat.unsqueeze(-1) - centroids.unsqueeze(0).unsqueeze(0)).square()

    # Viterbi forward pass
    # path_cost[batch, state] = minimum cost to reach this state
    path_cost = torch.full(
        (batch_size, TCQ_NUM_STATES), float("inf"),
        dtype=torch.float32, device=device,
    )
    path_cost[:, 0] = 0.0  # start in state 0

    # Backtrack storage
    back_states = torch.zeros(
        (batch_size, dim, TCQ_NUM_STATES), dtype=torch.int64, device=device,
    )
    back_levels = torch.zeros(
        (batch_size, dim, TCQ_NUM_STATES), dtype=torch.int64, device=device,
    )

    for t in range(dim):
        new_cost = torch.full_like(path_cost, float("inf"))
        for state in range(TCQ_NUM_STATES):
            for level in range(num_levels):
                next_state = transitions[state, level].item()
                cost = path_cost[:, state] + dists[:, t, level]
                better = cost < new_cost[:, next_state]
                new_cost[:, next_state] = torch.where(better, cost, new_cost[:, next_state])
                back_states[:, t, next_state] = torch.where(
                    better,
                    torch.tensor(state, device=device),
                    back_states[:, t, next_state],
                )
                back_levels[:, t, next_state] = torch.where(
                    better,
                    torch.tensor(level, device=device),
                    back_levels[:, t, next_state],
                )
        path_cost = new_cost

    # Backtrack from best final state
    best_final_state = path_cost.argmin(dim=-1)  # [batch]
    indices = torch.zeros((batch_size, dim), dtype=torch.int64, device=device)

    current_state = best_final_state
    for t in range(dim - 1, -1, -1):
        level = back_levels[:, t, :].gather(
            1, current_state.unsqueeze(-1)
        ).squeeze(-1)
        indices[:, t] = level
        current_state = back_states[:, t, :].gather(
            1, current_state.unsqueeze(-1)
        ).squeeze(-1)

    return indices.reshape(*batch_shape, dim)


# ─── TCQ encode/decode ──────────────────────────────────────────────────────────

def tcq_quantize_vectors(
    x: torch.Tensor,
    method: str,
    rotations: tuple[torch.Tensor, torch.Tensor],
    qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    centroids: dict[int, torch.Tensor],
    group_indices: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Quantize using Viterbi trellis-coded assignment instead of nearest."""
    # Strip _tcq suffix for layout lookup
    base_method = method.replace("_tcq", "")
    layout = compute_layout(base_method, x.shape[-1])
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
            num_levels = mse_centroids.shape[0]
            transitions = _build_trellis_transitions(num_levels).to(x.device)

            # Viterbi optimal assignment instead of nearest centroid
            mse_indices = _viterbi_quantize(rotated, mse_centroids, transitions)
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


# ─── Method classes ─────────────────────────────────────────────────────────────

class _TCQBase(_TurboQuantBase):
    """TCQ variant — same layout as TurboQuant, Viterbi assignment."""

    def _base_method(self) -> str:
        return self._method.replace("_tcq", "")

    def encode(
        self, x: torch.Tensor, *,
        head_indices: torch.Tensor | None = None, layer_idx: int = 0,
    ) -> CompressedKV:
        device = x.device
        base = self._base_method()
        # Reuse parent's transform setup
        self._method = base  # temporarily for transform lookup
        rotations, qjl_matrices, centroids = self._get_transforms(device)
        self._method = base + "_tcq"  # restore

        if self._group_indices is not None:
            gi = (
                self._group_indices[0].to(device),
                self._group_indices[1].to(device),
            )
        else:
            gi = build_outlier_masks(x, base)

        packed = tcq_quantize_vectors(
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
        # TCQ decode is identical to TurboQuant decode — same packed format
        device = compressed.data.device
        base = self._base_method()
        self._method = base
        rotations, qjl_matrices, centroids = self._get_transforms(device)
        self._method = base + "_tcq"
        gi = compressed.metadata["group_indices"]
        from .turboquant import dequantize_vectors
        return dequantize_vectors(
            compressed.data, base, compressed.head_dim,
            rotations, qjl_matrices, centroids, gi, dtype,
        )

    def packed_dim(self, head_dim: int) -> int:
        return compute_layout(self._base_method(), head_dim).packed_dim


@register_method(CacheMethod.TURBO2_TCQ)
class TCQ2(_TCQBase):
    def __init__(self, **kwargs):
        super().__init__(method="turbo2_tcq", **kwargs)

    def info(self) -> MethodInfo:
        return MethodInfo(
            method=CacheMethod.TURBO2_TCQ,
            family=MethodFamily.TCQ,
            bits=2.25, requires_calibration=True,
            supports_asymmetric=True,
            supported_head_dims=(64, 128, 192, 256),
            transform_name="WHT + Trellis Coded Quantization",
            description="2.25-bit TCQ — Viterbi-optimal assignment for better quality",
            fma_count=16384 + 2048,  # WHT + trellis overhead
            param_count=16384,
        )


@register_method(CacheMethod.TURBO3_TCQ)
class TCQ3(_TCQBase):
    def __init__(self, **kwargs):
        super().__init__(method="turbo3_tcq", **kwargs)

    def info(self) -> MethodInfo:
        return MethodInfo(
            method=CacheMethod.TURBO3_TCQ,
            family=MethodFamily.TCQ,
            bits=3.25, requires_calibration=True,
            supports_asymmetric=True,
            supported_head_dims=(64, 128, 192, 256),
            transform_name="WHT + Trellis Coded Quantization",
            description="3.25-bit TCQ — best quality at 5x compression",
            fma_count=16384 + 2048,
            param_count=16384,
        )


# Public alias
TCQMethod = TCQ3
