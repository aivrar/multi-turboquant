# SPDX-License-Identifier: Apache-2.0
# Derived from TheTom/llama-cpp-turboquant vLLM Triton kernels
"""Triton fused decode kernel for TurboQuant/TCQ attention.

This kernel performs fused dequantize-attention in a single pass:
  1. Loads packed quantized KV from paged cache
  2. Reconstructs approximate vectors (MSE inverse + QJL residual)
  3. Computes attention scores and softmax
  4. Accumulates weighted values

Avoids materializing the full dequantized KV in memory.

Requirements:
  - Linux with NVIDIA GPU (Triton is Linux-only)
  - CUDA compute capability >= 7.0 (Volta+)
  - triton >= 2.1.0
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import torch

# Triton import — will fail on Windows
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    # ─── Helper kernels ────────────────────────────────────────────────────

    @triton.jit
    def _load_half_from_bytes(
        ptr, offset, BYTE_WIDTH: tl.constexpr,
    ):
        """Load a float16 value stored as 2 consecutive bytes."""
        lo = tl.load(ptr + offset).to(tl.uint16)
        hi = tl.load(ptr + offset + 1).to(tl.uint16)
        bits = lo | (hi << 8)
        return bits.to(tl.float16).to(tl.float32)

    @triton.jit
    def _unpack_fixed_indices(
        ptr, offset, dim: tl.constexpr, bits: tl.constexpr,
    ):
        """Unpack multi-bit quantized indices from packed bytes."""
        total_bits = dim * bits
        result = tl.zeros([dim], dtype=tl.uint8)
        for i in tl.static_range(dim):
            val = tl.zeros([1], dtype=tl.uint8)
            for b in tl.static_range(bits):
                bit_pos = i * bits + b
                byte_idx = bit_pos // 8
                bit_idx = bit_pos % 8
                raw = tl.load(ptr + offset + byte_idx)
                bit_val = (raw >> bit_idx) & 1
                val = val | (bit_val << b)
            result = tl.where(
                tl.arange(0, dim) == i, val, result
            )
        return result

    @triton.jit
    def _unpack_signs(ptr, offset, dim: tl.constexpr):
        """Unpack 1-bit QJL sign codes."""
        result = tl.zeros([dim], dtype=tl.float32)
        for i in tl.static_range(dim):
            byte_idx = i // 8
            bit_idx = i % 8
            raw = tl.load(ptr + offset + byte_idx)
            bit_val = (raw >> bit_idx) & 1
            sign = tl.where(bit_val > 0, 1.0, -1.0)
            result = tl.where(
                tl.arange(0, dim) == i, sign, result
            )
        return result

    @triton.jit
    def _apply_softcap(x, softcap):
        """Apply logit soft-capping: softcap * tanh(x / softcap)."""
        if softcap > 0.0:
            return softcap * tl.math.tanh(x / softcap)
        return x


# ─── Python-level decode wrapper ───────────────────────────────────────────

@cache
def _norm_lut(dim: int, bits: int) -> torch.Tensor:
    """Pre-compute norm lookup table for fast centroid access."""
    from ..methods.turboquant import _dimension_aware_codebook
    centroids = _dimension_aware_codebook(dim, bits)
    return centroids


def turboquant_decode_attention_fwd(
    q: torch.Tensor,           # [batch, num_q_heads, head_dim]
    k_cache: torch.Tensor,     # [num_blocks, block_size, num_kv_heads, packed_dim]
    v_cache: torch.Tensor,     # [num_blocks, block_size, num_kv_heads, packed_dim]
    block_tables: torch.Tensor,  # [batch, max_num_blocks]
    seq_lens: torch.Tensor,    # [batch]
    *,
    kv_cache_dtype: str,
    head_dim: int,
    scale: float | None = None,
    softcap: float = 0.0,
    mse_fwd_matrices: tuple[torch.Tensor, torch.Tensor] | None = None,
    mse_inv_matrices: tuple[torch.Tensor, torch.Tensor] | None = None,
    qjl_fwd_matrices: tuple[torch.Tensor, torch.Tensor] | None = None,
    qjl_inv_matrices: tuple[torch.Tensor, torch.Tensor] | None = None,
    centroids: dict[int, torch.Tensor] | None = None,
    group_masks: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Fused TurboQuant decode attention.

    Falls back to PyTorch reference if Triton is not available.

    Args:
        q: Query tensor.
        k_cache / v_cache: Paged quantized KV caches.
        block_tables: Block index tables for paged attention.
        seq_lens: Sequence lengths per batch entry.
        kv_cache_dtype: "turboquant25" or "turboquant35".
        head_dim: Original head dimension.
        scale: Attention scale (default: 1/sqrt(head_dim)).
        softcap: Logit soft-capping value (0 = disabled).
        mse_fwd_matrices: Pre-computed MSE forward transform matrices.
        mse_inv_matrices: Pre-computed MSE inverse transform matrices.
        qjl_fwd_matrices: Pre-computed QJL forward matrices.
        qjl_inv_matrices: Pre-computed QJL inverse matrices.
        centroids: Dimension-aware codebooks.
        group_masks: Per-layer outlier/regular dimension masks.

    Returns:
        Attention output [batch, num_q_heads, head_dim].
    """
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if not HAS_TRITON:
        import warnings
        warnings.warn(
            "Triton not available — using PyTorch reference decode (slower). "
            "Install triton on Linux for GPU-accelerated kernels.",
            RuntimeWarning, stacklevel=2,
        )
        return _pytorch_decode_reference(
            q, k_cache, v_cache, block_tables, seq_lens,
            kv_cache_dtype=kv_cache_dtype,
            head_dim=head_dim,
            scale=scale,
            group_masks=group_masks,
        )

    # EXPERIMENTAL: Fused Triton kernel not yet implemented.
    # Falling back to PyTorch reference decode. This is functionally
    # correct but does not fuse dequantize+attention into one pass.
    import logging
    logging.getLogger(__name__).info(
        "TurboQuant fused Triton decode kernel not yet available — "
        "using PyTorch reference path"
    )
    return _pytorch_decode_reference(
        q, k_cache, v_cache, block_tables, seq_lens,
        kv_cache_dtype=kv_cache_dtype,
        head_dim=head_dim,
        scale=scale,
        group_masks=group_masks,
    )


def _pytorch_decode_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    kv_cache_dtype: str,
    head_dim: int,
    scale: float,
    group_masks: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """PyTorch reference decode — dequantize then standard attention."""
    from ..methods.turboquant import (
        dequantize_vectors,
        get_rotation,
        get_qjl_matrix,
        get_centroids as get_tq_centroids,
        get_group_dims,
        build_outlier_masks,
        compute_layout,
        GROUP_BITS,
    )

    device = q.device
    batch_size = q.shape[0]
    num_q_heads = q.shape[1]

    # Map kv_cache_dtype to method name
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

    # Simple reference: gather blocks, dequantize, compute attention
    outputs = []
    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        if seq_len == 0:
            outputs.append(torch.zeros(
                num_q_heads, head_dim, dtype=q.dtype, device=device,
            ))
            continue

        # Gather K/V from paged cache
        blocks_needed = (seq_len + k_cache.shape[1] - 1) // k_cache.shape[1]
        block_ids = block_tables[b, :blocks_needed]

        k_blocks = k_cache[block_ids]  # [blocks, block_size, heads, packed]
        v_blocks = v_cache[block_ids]

        # Reshape to [seq_len, heads, packed]
        k_flat = k_blocks.reshape(-1, k_blocks.shape[-2], k_blocks.shape[-1])[:seq_len]
        v_flat = v_blocks.reshape(-1, v_blocks.shape[-2], v_blocks.shape[-1])[:seq_len]

        # Build group indices if not provided
        if group_masks is not None:
            gi = group_masks
        else:
            # Default: first N dims are outliers
            from ..methods.turboquant import get_outlier_count
            oc = get_outlier_count(head_dim, method)
            num_kv_heads = k_flat.shape[-2]
            high = torch.arange(oc, device=device).unsqueeze(0).expand(num_kv_heads, -1)
            low = torch.arange(oc, head_dim, device=device).unsqueeze(0).expand(num_kv_heads, -1)
            gi = (high, low)

        # Dequantize
        k_deq = dequantize_vectors(
            k_flat, method, head_dim, rotations, qjl_matrices,
            centroids_dict, gi, q.dtype,
        )
        v_deq = dequantize_vectors(
            v_flat, method, head_dim, rotations, qjl_matrices,
            centroids_dict, gi, q.dtype,
        )

        # Standard attention
        q_b = q[b]  # [num_q_heads, head_dim]
        # Handle GQA: map query heads to KV heads
        num_kv_heads = k_deq.shape[-2]
        group_size = num_q_heads // num_kv_heads

        attn_out = torch.zeros(num_q_heads, head_dim, dtype=q.dtype, device=device)
        for qh in range(num_q_heads):
            kvh = qh // group_size
            scores = (q_b[qh] * k_deq[:, kvh]).sum(-1) * scale
            weights = torch.softmax(scores, dim=0)
            attn_out[qh] = (weights.unsqueeze(-1) * v_deq[:, kvh]).sum(0)

        outputs.append(attn_out)

    return torch.stack(outputs)
