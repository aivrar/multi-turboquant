# SPDX-License-Identifier: Apache-2.0 / MIT
# Derived from triton_attn_modified.py (vLLM project, Apache-2.0)
# Extended for Multi-TurboQuant unified method dispatch
"""Unified vLLM attention backend for all Multi-TurboQuant compression methods.

This module is the core integration point between Multi-TurboQuant and vLLM.
It extends the existing TurboQuant TritonAttentionImpl to dispatch across
all supported methods (TurboQuant, TCQ, IsoQuant, PlanarQuant) and optionally
apply TriAttention token eviction before the KV cache write.

Architecture:
    vLLM's forward() call
        -> forward() dispatcher
            -> _forward_turboquant()   if turbo*/turbo*_tcq
            -> _forward_isoquant()     if iso*
            -> _forward_planarquant()  if planar*
            -> unified_attention()     if baseline (f16/fp8)

    do_kv_cache_update() dispatcher
        -> [optional] TriAttention scoring & eviction mask
        -> turboquant_write_packed_kv()  if turbo*/turbo*_tcq
        -> iso_write_packed_kv()         if iso*
        -> planar_write_packed_kv()      if planar*
        -> triton_reshape_and_cache()    if baseline

The existing triton_attn_modified.py (1,529 lines) handles TurboQuant fully.
This module wraps it and adds dispatch for the other methods. When vLLM
updates its attention API, changes are isolated here.

NOTE: This module imports from vLLM and will only work inside a vLLM process.
It is NOT imported by the multi_turboquant package at top level.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Method family detection
_TURBOQUANT_DTYPES = {"turboquant25", "turboquant35"}
_ISOQUANT_DTYPES = {"isoquant3", "isoquant4"}
_PLANARQUANT_DTYPES = {"planarquant3", "planarquant4"}
_ALL_MULTI_TQ_DTYPES = _TURBOQUANT_DTYPES | _ISOQUANT_DTYPES | _PLANARQUANT_DTYPES


def is_multi_tq_kv_cache(kv_cache_dtype: str) -> bool:
    """Check if a kv_cache_dtype string is handled by Multi-TurboQuant."""
    return kv_cache_dtype in _ALL_MULTI_TQ_DTYPES


def get_method_family(kv_cache_dtype: str) -> str | None:
    """Map a kv_cache_dtype to its method family."""
    if kv_cache_dtype in _TURBOQUANT_DTYPES:
        return "turboquant"
    if kv_cache_dtype in _ISOQUANT_DTYPES:
        return "isoquant"
    if kv_cache_dtype in _PLANARQUANT_DTYPES:
        return "planarquant"
    return None


# ─── IsoQuant KV cache operations ──────────────────────────────────────────────

def iso_write_packed_kv(
    x, cache, slot_mapping, *, bits: int, head_dim: int,
):
    """Quantize and write IsoQuant-compressed vectors to paged KV cache.

    This is the IsoQuant equivalent of turboquant_write_packed_kv.
    Uses quaternion 4D rotation + uniform quantization.
    """
    from ...methods.isoquant import iso_encode

    # Encode all tokens
    packed = iso_encode(x.float(), bits)

    # Scatter to paged cache
    block_size = cache.shape[1]
    for i in range(slot_mapping.shape[0]):
        slot = slot_mapping[i].item()
        if slot < 0:
            continue
        block_idx = slot // block_size
        block_offset = slot % block_size
        cache[block_idx, block_offset] = packed[i]


def iso_decode_cached_kv(
    cache, block_table, seq_lens, *, bits: int, head_dim: int, dtype,
):
    """Read and decode IsoQuant vectors from paged KV cache."""
    import torch
    from ...methods.isoquant import iso_decode

    device = cache.device
    block_size = cache.shape[1]
    batch_size = seq_lens.shape[0]
    num_heads = cache.shape[2]
    results = []

    for b in range(batch_size):
        sl = seq_lens[b].item()
        if sl == 0:
            results.append(torch.zeros(0, num_heads, head_dim, dtype=dtype, device=device))
            continue
        blocks_needed = (sl + block_size - 1) // block_size
        block_ids = block_table[b, :blocks_needed]
        packed_blocks = cache[block_ids].reshape(-1, num_heads, cache.shape[-1])[:sl]
        decoded = iso_decode(packed_blocks, head_dim, bits, dtype)
        results.append(decoded)

    return results


# ─── PlanarQuant KV cache operations ───────────────────────────────────────────

def planar_write_packed_kv(
    x, cache, slot_mapping, *, bits: int, head_dim: int,
):
    """Quantize and write PlanarQuant-compressed vectors to paged KV cache."""
    from ...methods.planarquant import planar_encode

    packed = planar_encode(x.float(), bits)
    block_size = cache.shape[1]
    for i in range(slot_mapping.shape[0]):
        slot = slot_mapping[i].item()
        if slot < 0:
            continue
        block_idx = slot // block_size
        block_offset = slot % block_size
        cache[block_idx, block_offset] = packed[i]


def planar_decode_cached_kv(
    cache, block_table, seq_lens, *, bits: int, head_dim: int, dtype,
):
    """Read and decode PlanarQuant vectors from paged KV cache."""
    import torch
    from ...methods.planarquant import planar_decode

    device = cache.device
    block_size = cache.shape[1]
    batch_size = seq_lens.shape[0]
    num_heads = cache.shape[2]
    results = []

    for b in range(batch_size):
        sl = seq_lens[b].item()
        if sl == 0:
            results.append(torch.zeros(0, num_heads, head_dim, dtype=dtype, device=device))
            continue
        blocks_needed = (sl + block_size - 1) // block_size
        block_ids = block_table[b, :blocks_needed]
        packed_blocks = cache[block_ids].reshape(-1, num_heads, cache.shape[-1])[:sl]
        decoded = planar_decode(packed_blocks, head_dim, bits, dtype)
        results.append(decoded)

    return results


# ─── TriAttention eviction hook ─────────────────────────────────────────────────

def triattention_score_and_mask(
    key, *, budget: int, window: int, layer_idx: int = 0, stats=None,
):
    """Score pre-RoPE keys and return eviction mask.

    This is the hook point that feeds TriAttention real data.
    Called from do_kv_cache_update() BEFORE the KV write.

    Args:
        key: Pre-RoPE key tensor [num_tokens, num_kv_heads, head_dim].
        budget: Total token budget.
        window: Recent-window size (never evicted).
        layer_idx: Layer index for per-layer stats.
        stats: Pre-computed TriAttentionStats dict.

    Returns:
        Boolean mask [num_tokens] — True = keep, False = evict.
    """
    from ...methods.triattention import compute_frequency_scores, select_tokens

    layer_stats = None
    if stats is not None:
        layer_stats = stats.get(layer_idx)

    scores = compute_frequency_scores(key, layer_stats)
    mask = select_tokens(scores, budget=budget, window=window, seq_len=key.shape[0])
    return mask


# ─── Unified dispatch helpers ───────────────────────────────────────────────────

def get_packed_dim(kv_cache_dtype: str, head_dim: int) -> int:
    """Get the packed byte dimension for any Multi-TQ cache dtype."""
    family = get_method_family(kv_cache_dtype)
    if family == "turboquant":
        from ...methods.turboquant import compute_layout
        method_map = {"turboquant25": "turbo2", "turboquant35": "turbo3"}
        method = method_map[kv_cache_dtype]
        return compute_layout(method, head_dim).packed_dim
    elif family == "isoquant":
        from ...methods.isoquant import iso_packed_dim
        bits = 3 if kv_cache_dtype == "isoquant3" else 4
        return iso_packed_dim(head_dim, bits)
    elif family == "planarquant":
        from ...methods.planarquant import planar_packed_dim
        bits = 3 if kv_cache_dtype == "planarquant3" else 4
        return planar_packed_dim(head_dim, bits)
    else:
        raise ValueError(f"Unknown Multi-TQ dtype: {kv_cache_dtype}")


def dispatch_kv_cache_update(
    kv_cache_dtype: str,
    key, value, key_cache, value_cache, slot_mapping,
    head_dim: int,
    *,
    # TurboQuant-specific args (ignored for other methods)
    layout=None, key_masks=None, value_masks=None,
    mse_transform_matrices=None, qjl_transform_matrices=None,
    mse_to_qjl_matrices=None, centroids=None,
    # TriAttention args (optional, composable with any method)
    triattention_mask=None,
):
    """Dispatch KV cache update to the correct method's write kernel.

    This is the unified entry point called from do_kv_cache_update().
    """
    # Apply TriAttention eviction mask if present
    if triattention_mask is not None:
        key = key[triattention_mask]
        value = value[triattention_mask]
        slot_mapping = slot_mapping[triattention_mask]

    family = get_method_family(kv_cache_dtype)

    if family == "turboquant":
        # Use the existing fused Triton kernel
        try:
            from vllm.v1.attention.ops.triton_turboquant_kv_update import (
                turboquant_write_packed_kv,
            )
            turboquant_write_packed_kv(
                x=key, cache=key_cache, slot_mapping=slot_mapping,
                layout=layout, group_indices=key_masks,
                mse_transform_matrices=mse_transform_matrices,
                qjl_transform_matrices=qjl_transform_matrices,
                mse_to_qjl_matrices=mse_to_qjl_matrices,
                centroids=centroids,
            )
            turboquant_write_packed_kv(
                x=value, cache=value_cache, slot_mapping=slot_mapping,
                layout=layout, group_indices=value_masks,
                mse_transform_matrices=mse_transform_matrices,
                qjl_transform_matrices=qjl_transform_matrices,
                mse_to_qjl_matrices=mse_to_qjl_matrices,
                centroids=centroids,
            )
        except ImportError:
            logger.warning("vLLM TurboQuant kernel not available, using reference path")
            from .turboquant_update import _pytorch_update_reference
            _pytorch_update_reference(
                key, value, key_cache, value_cache, slot_mapping,
                kv_cache_dtype=kv_cache_dtype, head_dim=head_dim,
            )

    elif family == "isoquant":
        bits = 3 if kv_cache_dtype == "isoquant3" else 4
        iso_write_packed_kv(key, key_cache, slot_mapping, bits=bits, head_dim=head_dim)
        iso_write_packed_kv(value, value_cache, slot_mapping, bits=bits, head_dim=head_dim)

    elif family == "planarquant":
        bits = 3 if kv_cache_dtype == "planarquant3" else 4
        planar_write_packed_kv(key, key_cache, slot_mapping, bits=bits, head_dim=head_dim)
        planar_write_packed_kv(value, value_cache, slot_mapping, bits=bits, head_dim=head_dim)

    else:
        raise ValueError(f"Unsupported Multi-TQ dtype for KV update: {kv_cache_dtype}")


def dispatch_decode_attention(
    kv_cache_dtype: str,
    query, key_cache, value_cache, block_table, seq_lens,
    head_dim: int, scale: float, num_heads: int, num_kv_heads: int,
    dtype,
    *,
    # TurboQuant-specific args
    key_masks=None, value_masks=None,
    rotations=None, qjl_matrices=None, centroids=None,
    norm_lut=None, **tq_kwargs,
):
    """Dispatch decode attention to the correct method.

    For TurboQuant: uses the existing fused decode kernel.
    For IsoQuant/PlanarQuant: dequantizes then computes standard attention.
    """
    import torch

    family = get_method_family(kv_cache_dtype)

    if family == "turboquant":
        # Use existing fused TurboQuant decode kernel
        try:
            from vllm.v1.attention.ops.triton_turboquant_decode import (
                turboquant_decode_attention_fwd,
            )
            return turboquant_decode_attention_fwd(
                query=query,
                key_cache=key_cache, value_cache=value_cache,
                block_table=block_table, seq_lens=seq_lens,
                key_group_indices=key_masks, value_group_indices=value_masks,
                key_rotations=rotations, key_qjl_matrices=qjl_matrices,
                value_rotations=rotations, value_qjl_matrices=qjl_matrices,
                centroids=centroids, norm_lut=norm_lut,
                softmax_scale=scale, kv_cache_dtype=kv_cache_dtype,
                **tq_kwargs,
            )
        except ImportError:
            logger.warning("vLLM TurboQuant decode kernel not available, using reference")
            from .turboquant_decode import _pytorch_decode_reference
            return _pytorch_decode_reference(
                query, key_cache, value_cache, block_table, seq_lens,
                kv_cache_dtype=kv_cache_dtype, head_dim=head_dim,
                scale=scale, group_masks=key_masks,
            )

    elif family in ("isoquant", "planarquant"):
        # Dequantize from paged cache, then standard attention
        bits = 3 if kv_cache_dtype.endswith("3") else 4
        decode_fn = (
            iso_decode_cached_kv if family == "isoquant"
            else planar_decode_cached_kv
        )
        k_lists = decode_fn(
            key_cache, block_table, seq_lens,
            bits=bits, head_dim=head_dim, dtype=dtype,
        )
        v_lists = decode_fn(
            value_cache, block_table, seq_lens,
            bits=bits, head_dim=head_dim, dtype=dtype,
        )

        batch_size = query.shape[0]
        group_size = num_heads // num_kv_heads
        output = torch.zeros_like(query)

        for b in range(batch_size):
            if len(k_lists[b]) == 0:
                continue
            k_deq = k_lists[b]  # [seq_len, num_kv_heads, head_dim]
            v_deq = v_lists[b]

            q_b = query[b]  # [num_heads, head_dim]
            for qh in range(num_heads):
                kvh = qh // group_size
                scores = (q_b[qh] * k_deq[:, kvh]).sum(-1) * scale
                weights = torch.softmax(scores, dim=0)
                output[b, qh] = (weights.unsqueeze(-1) * v_deq[:, kvh]).sum(0)

        return output

    else:
        raise ValueError(f"Unsupported Multi-TQ dtype for decode: {kv_cache_dtype}")
