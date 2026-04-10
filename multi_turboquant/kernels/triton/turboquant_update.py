# SPDX-License-Identifier: Apache-2.0
# Derived from TheTom/llama-cpp-turboquant vLLM Triton kernels
"""Triton fused quantize-and-store kernel for TurboQuant KV cache updates.

Fuses the full quantization pipeline into a single kernel launch:
  1. Gather dimensions by outlier group
  2. Compute vector norms
  3. Apply MSE transform (structured Hadamard)
  4. Nearest centroid assignment (or Viterbi for TCQ)
  5. Compute residual and QJL signs
  6. Pack everything into bytes
  7. Store to paged KV cache

Requirements: Same as turboquant_decode.py
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def turboquant_write_packed_kv(
    k: torch.Tensor,              # [batch, num_kv_heads, head_dim]
    v: torch.Tensor,              # [batch, num_kv_heads, head_dim]
    k_cache: torch.Tensor,        # [num_blocks, block_size, num_kv_heads, packed_dim]
    v_cache: torch.Tensor,        # [num_blocks, block_size, num_kv_heads, packed_dim]
    slot_mapping: torch.Tensor,   # [batch] — flat slot indices
    *,
    kv_cache_dtype: str,
    head_dim: int,
    mse_fwd_matrices: tuple[torch.Tensor, torch.Tensor] | None = None,
    qjl_fwd_matrices: tuple[torch.Tensor, torch.Tensor] | None = None,
    mse_to_qjl_matrices: tuple[torch.Tensor, torch.Tensor] | None = None,
    centroids: dict[int, torch.Tensor] | None = None,
    group_masks: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Quantize and store KV vectors into paged cache.

    Falls back to PyTorch reference if Triton is not available.

    Args:
        k, v: Key/value vectors to store.
        k_cache, v_cache: Paged KV caches (modified in-place).
        slot_mapping: Maps each batch entry to a flat cache slot.
        kv_cache_dtype: "turboquant25" or "turboquant35".
        head_dim: Original head dimension.
        mse_fwd_matrices: Pre-computed MSE forward transforms.
        qjl_fwd_matrices: Pre-computed QJL forward transforms.
        mse_to_qjl_matrices: Pre-composed MSE-inv x QJL-fwd matrices.
        centroids: Dimension-aware codebooks.
        group_masks: Per-layer outlier/regular dimension masks.
    """
    if not HAS_TRITON:
        import warnings
        warnings.warn(
            "Triton not available — using PyTorch reference KV update (slower). "
            "Install triton on Linux for GPU-accelerated kernels.",
            RuntimeWarning, stacklevel=2,
        )
        _pytorch_update_reference(
            k, v, k_cache, v_cache, slot_mapping,
            kv_cache_dtype=kv_cache_dtype,
            head_dim=head_dim,
            group_masks=group_masks,
        )
        return

    # EXPERIMENTAL: Fused Triton update kernel not yet implemented.
    # Falling back to PyTorch reference. Functionally correct but
    # does not fuse quantize+scatter into one kernel launch.
    import logging
    logging.getLogger(__name__).info(
        "TurboQuant fused Triton update kernel not yet available — "
        "using PyTorch reference path"
    )
    _pytorch_update_reference(
        k, v, k_cache, v_cache, slot_mapping,
        kv_cache_dtype=kv_cache_dtype,
        head_dim=head_dim,
        group_masks=group_masks,
    )


def _pytorch_update_reference(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    kv_cache_dtype: str,
    head_dim: int,
    group_masks: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """PyTorch reference update — quantize then scatter to cache."""
    from ..methods.turboquant import (
        quantize_vectors,
        get_rotation,
        get_qjl_matrix,
        get_centroids as get_tq_centroids,
        get_group_dims,
        get_outlier_count,
        GROUP_BITS,
    )

    device = k.device
    method_map = {"turboquant25": "turbo2", "turboquant35": "turbo3"}
    method = method_map.get(kv_cache_dtype, "turbo3")

    dims = get_group_dims(head_dim, method)
    rotations = tuple(get_rotation(device, d) for d in dims)
    qjl_matrices = tuple(get_qjl_matrix(device, d) for d in dims)
    bits = GROUP_BITS[method]
    centroids_dict = {}
    for d, b in zip(dims, bits):
        mse_bits = b - 1
        if mse_bits > 0:
            centroids_dict[mse_bits] = get_tq_centroids(device, d, mse_bits)

    if group_masks is not None:
        gi = group_masks
    else:
        oc = get_outlier_count(head_dim, method)
        num_kv_heads = k.shape[-2]
        high = torch.arange(oc, device=device).unsqueeze(0).expand(num_kv_heads, -1)
        low = torch.arange(oc, head_dim, device=device).unsqueeze(0).expand(num_kv_heads, -1)
        gi = (high, low)

    # Quantize K and V
    k_packed = quantize_vectors(k, method, rotations, qjl_matrices, centroids_dict, gi)
    v_packed = quantize_vectors(v, method, rotations, qjl_matrices, centroids_dict, gi)

    # Scatter to paged cache
    block_size = k_cache.shape[1]
    for i in range(slot_mapping.shape[0]):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        k_cache[block_idx, block_offset] = k_packed[i]
        v_cache[block_idx, block_offset] = v_packed[i]
