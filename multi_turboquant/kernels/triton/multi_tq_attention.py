# SPDX-License-Identifier: Apache-2.0 / MIT
"""MultiTQAttentionImpl — unified vLLM attention backend for all methods.

This is the single class that vLLM needs to know about. It handles:
  - TurboQuant/TCQ: delegates to the existing TritonAttentionImpl
  - IsoQuant: quaternion rotation encode/decode via vectorized kernels
  - PlanarQuant: Givens rotation encode/decode via vectorized kernels
  - TriAttention: pre-RoPE key scoring and eviction before KV write
  - Combined modes: TriAttention mask applied before quantized write

Install via:
    from multi_turboquant.integration import patch_vllm
    patch_vllm(config)

Architecture:
    MultiTQAttentionImpl wraps the dispatch logic. For TurboQuant dtypes
    it delegates to the original TritonAttentionImpl paths (fully working
    fused Triton kernels). For iso/planar it uses the vectorized encode/
    decode kernels. For TriAttention it hooks into do_rope_and_kv_cache_update
    to intercept pre-RoPE keys.

NOTE: This module is only importable inside a vLLM process. It is NOT
imported at package init time — the vllm_patch module loads it on demand.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from .attention_backend import (
    is_multi_tq_kv_cache,
    get_method_family,
    get_packed_dim,
    triattention_score_and_mask,
    iso_write_packed_kv,
    iso_decode_cached_kv,
    planar_write_packed_kv,
    planar_decode_cached_kv,
)

logger = logging.getLogger(__name__)


# ─── TriAttention configuration from environment ───────────────────────────────

_TRIATTENTION_ENABLED = os.environ.get("TRIATTENTION_ENABLED", "").lower() in ("1", "true", "yes")
_TRIATTENTION_BUDGET = int(os.environ.get("TRIATTENTION_BUDGET", "4096"))
_TRIATTENTION_WINDOW = int(os.environ.get("TRIATTENTION_WINDOW", "512"))
_TRIATTENTION_STATS_PATH = os.environ.get("TRIATTENTION_STATS_PATH", "")

# Load TriAttention stats once at module level
_triattention_stats = None
if _TRIATTENTION_ENABLED and _TRIATTENTION_STATS_PATH:
    try:
        from ...methods.triattention import load_stats
        _triattention_stats = load_stats(_TRIATTENTION_STATS_PATH)
        logger.info(
            "TriAttention stats loaded from %s (%d layers)",
            _TRIATTENTION_STATS_PATH, len(_triattention_stats),
        )
    except Exception as e:
        logger.warning("Failed to load TriAttention stats: %s", e)


def _get_iso_planar_bits(kv_cache_dtype: str) -> int:
    """Extract bit count from dtype string."""
    if kv_cache_dtype.endswith("3"):
        return 3
    if kv_cache_dtype.endswith("4"):
        return 4
    raise ValueError(f"Cannot extract bits from dtype: {kv_cache_dtype}")


class MultiTQKVCacheManager:
    """Manages KV cache operations for IsoQuant and PlanarQuant methods.

    TurboQuant is handled by the existing TritonAttentionImpl infrastructure.
    This class handles the new methods that don't have fused Triton kernels
    in the upstream vLLM code.
    """

    def __init__(
        self,
        kv_cache_dtype: str,
        head_dim: int,
        num_kv_heads: int,
        num_heads: int,
        scale: float,
        layer_idx: int = 0,
    ):
        self.kv_cache_dtype = kv_cache_dtype
        self.family = get_method_family(kv_cache_dtype)
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.scale = scale
        self.layer_idx = layer_idx
        self.bits = _get_iso_planar_bits(kv_cache_dtype)
        self.packed_dim = get_packed_dim(kv_cache_dtype, head_dim)
        self.group_size = num_heads // num_kv_heads

        # TriAttention state
        self.triattention_enabled = _TRIATTENTION_ENABLED
        self.triattention_budget = _TRIATTENTION_BUDGET
        self.triattention_window = _TRIATTENTION_WINDOW
        self._eviction_stats = _triattention_stats

        if self.triattention_enabled:
            logger.info(
                "TriAttention enabled: budget=%d window=%d layer=%d",
                self.triattention_budget, self.triattention_window, layer_idx,
            )

    def write_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        pre_rope_key: torch.Tensor | None = None,
    ) -> None:
        """Quantize and write K/V to paged cache.

        Args:
            key: Post-RoPE key tensor [num_tokens, num_kv_heads, head_dim].
            value: Value tensor [num_tokens, num_kv_heads, head_dim].
            key_cache: Paged K cache [num_blocks, block_size, heads, packed_dim].
            value_cache: Paged V cache.
            slot_mapping: Flat slot indices [num_tokens].
            pre_rope_key: Pre-RoPE key for TriAttention scoring (optional).
        """
        # TriAttention eviction — applied before quantized write
        if self.triattention_enabled and pre_rope_key is not None:
            mask = triattention_score_and_mask(
                pre_rope_key,
                budget=self.triattention_budget,
                window=self.triattention_window,
                layer_idx=self.layer_idx,
                stats=self._eviction_stats,
            )
            # Only write surviving tokens
            key = key[mask]
            value = value[mask]
            slot_mapping = slot_mapping[mask]

        if self.family == "isoquant":
            from .iso_encode_kernel import iso_encode_vectorized
            k_packed = iso_encode_vectorized(key, self.bits)
            v_packed = iso_encode_vectorized(value, self.bits)
        elif self.family == "planarquant":
            from .planar_encode_kernel import planar_encode_vectorized
            k_packed = planar_encode_vectorized(key, self.bits)
            v_packed = planar_encode_vectorized(value, self.bits)
        else:
            raise ValueError(f"Unsupported family for MultiTQKVCacheManager: {self.family}")

        # Scatter to paged cache
        block_size = key_cache.shape[1]
        for i in range(slot_mapping.shape[0]):
            slot = slot_mapping[i].item()
            if slot < 0:
                continue
            block_idx = slot // block_size
            block_offset = slot % block_size
            key_cache[block_idx, block_offset] = k_packed[i]
            value_cache[block_idx, block_offset] = v_packed[i]

    def decode_attention(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Decode: dequantize from cache and compute attention.

        Args:
            query: [num_tokens, num_heads, head_dim].
            key_cache: Paged K cache.
            value_cache: Paged V cache.
            block_table: [batch, max_blocks].
            seq_lens: [batch].
            output: Pre-allocated output tensor.

        Returns:
            output tensor filled with attention results.
        """
        if self.family == "isoquant":
            from .iso_encode_kernel import iso_decode_vectorized
            decode_fn = iso_decode_vectorized
        elif self.family == "planarquant":
            from .planar_encode_kernel import planar_decode_vectorized
            decode_fn = planar_decode_vectorized
        else:
            raise ValueError(f"Unsupported family: {self.family}")

        device = query.device
        block_size = key_cache.shape[1]
        dtype = query.dtype

        # Determine batch handling
        # For decode (1 token per sequence), query shape is [batch, num_heads, head_dim]
        batch_size = seq_lens.shape[0]

        for b in range(batch_size):
            sl = seq_lens[b].item()
            if sl == 0:
                continue

            blocks_needed = (sl + block_size - 1) // block_size
            block_ids = block_table[b, :blocks_needed]

            # Gather packed blocks
            k_blocks = key_cache[block_ids]  # [blocks, block_size, heads, packed]
            v_blocks = value_cache[block_ids]

            k_flat = k_blocks.reshape(-1, k_blocks.shape[-2], k_blocks.shape[-1])[:sl]
            v_flat = v_blocks.reshape(-1, v_blocks.shape[-2], v_blocks.shape[-1])[:sl]

            # Dequantize
            k_deq = decode_fn(k_flat, self.head_dim, self.bits, dtype)
            v_deq = decode_fn(v_flat, self.head_dim, self.bits, dtype)

            # Attention per query head
            q_b = query[b]  # [num_heads, head_dim]
            for qh in range(self.num_heads):
                kvh = qh // self.group_size
                scores = (q_b[qh] * k_deq[:, kvh]).sum(-1) * self.scale
                weights = torch.softmax(scores.float(), dim=0).to(dtype)
                output[b, qh] = (weights.unsqueeze(-1) * v_deq[:, kvh]).sum(0)

        return output

    def prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill: direct attention on raw K/V (no cache dequant needed).

        During prefill, K/V are in full precision. We compute attention
        directly, then the KV write path handles quantization to cache.
        """
        dtype = query.dtype
        kv_head_for_query_head = (
            torch.arange(self.num_heads, device=query.device, dtype=torch.int64)
            // self.group_size
        )
        num_seqs = seq_lens.shape[0]

        for seq_idx in range(num_seqs):
            q_start = int(query_start_loc[seq_idx].item())
            q_end = int(query_start_loc[seq_idx + 1].item())
            seq_len = int(seq_lens[seq_idx].item())

            q = query[q_start:q_end]      # [q_len, num_heads, head_dim]
            k = key[q_start:q_start + seq_len]
            v = value[q_start:q_start + seq_len]

            q_fp32 = q.permute(1, 0, 2).float()  # [heads, q_len, dim]
            k_fp32 = k.permute(1, 0, 2).index_select(0, kv_head_for_query_head).float()
            v_fp32 = v.permute(1, 0, 2).index_select(0, kv_head_for_query_head).float()

            logits = torch.einsum("hqd,hkd->hqk", q_fp32, k_fp32) * self.scale

            # Causal mask
            q_positions = torch.arange(seq_len - q.shape[0], seq_len, device=query.device)
            k_positions = torch.arange(seq_len, device=query.device)
            causal_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
            logits = logits.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

            attn = torch.softmax(logits, dim=-1)
            seq_output = torch.einsum("hqk,hkd->hqd", attn, v_fp32)
            output[q_start:q_end] = seq_output.permute(1, 0, 2).to(dtype)

        return output


def create_kv_cache_manager(
    kv_cache_dtype: str,
    head_dim: int,
    num_kv_heads: int,
    num_heads: int,
    scale: float,
    layer_idx: int = 0,
) -> MultiTQKVCacheManager | None:
    """Create a KV cache manager for iso/planar methods.

    Returns None for TurboQuant (handled by existing infrastructure)
    or non-Multi-TQ dtypes.
    """
    family = get_method_family(kv_cache_dtype)
    if family in ("isoquant", "planarquant"):
        return MultiTQKVCacheManager(
            kv_cache_dtype=kv_cache_dtype,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale=scale,
            layer_idx=layer_idx,
        )
    return None  # TurboQuant uses existing path, baselines use vLLM default
