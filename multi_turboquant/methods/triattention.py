# SPDX-License-Identifier: MIT
# Derived from WeianMao/triattention (Apache-2.0)
"""TriAttention — Trigonometric frequency-based KV cache token eviction.

Unlike quantization methods that compress each token's representation,
TriAttention reduces the NUMBER of tokens in the KV cache by scoring
token importance via frequency-domain analysis of pre-RoPE Q/K vectors.

This is orthogonal to all quantization methods and can be composed with
any of them for multiplicative memory savings:
  - TriAttention alone: 10-16x token reduction (full precision survivors)
  - TriAttention + TurboQuant: 50-80x total KV reduction
  - TriAttention + IsoQuant K-only: 16-20x with zero speed cost

Key properties:
  - Requires per-model calibration (.pt stat files)
  - Incompatible with prefix caching
  - Quality degrades at low budgets (2048 tokens: -26% on AIME24)
  - Best for long-context reasoning workloads with extended thinking chains
  - vLLM only (monkeypatch), no llama.cpp support

EXPERIMENTAL: Paper published April 2026, implementation may change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..config import CacheMethod, MethodFamily
from ..registry import register_method
from .base import CompressionMethod, CompressedKV, MethodInfo


# ─── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_BUDGET = 4096
DEFAULT_WINDOW = 512  # recent tokens never evicted
MIN_BUDGET = 256


# ─── Frequency scoring ─────────────────────────────────────────────────────────

@dataclass
class TriAttentionStats:
    """Per-layer trigonometric frequency statistics for token scoring.

    These are pre-computed from calibration data and stored as .pt files.
    Each tensor has shape [num_heads, head_dim].
    """
    freq_means: torch.Tensor    # mean frequency magnitudes per dimension
    freq_stds: torch.Tensor     # std of frequency magnitudes
    phase_means: torch.Tensor   # mean phase angles
    importance_weights: torch.Tensor  # learned dimension importance weights


def compute_frequency_scores(
    keys: torch.Tensor,
    stats: TriAttentionStats | None = None,
) -> torch.Tensor:
    """Score token importance using trigonometric frequency analysis.

    Applies DFT to pre-RoPE key vectors and scores based on frequency
    magnitude patterns relative to calibration statistics.

    Args:
        keys: Key vectors [..., seq_len, num_heads, head_dim].
        stats: Pre-computed frequency statistics (optional).

    Returns:
        Importance scores [..., seq_len, num_heads] in [0, 1].
    """
    keys_fp32 = keys.to(torch.float32)

    # Per-head frequency analysis via real DFT
    # [batch, seq, heads, dim] -> [batch, seq, heads, dim//2+1] complex
    freq = torch.fft.rfft(keys_fp32, dim=-1)
    magnitudes = freq.abs()  # [batch, seq, heads, freq_bins]

    if stats is not None:
        # Score relative to calibration distribution
        # Dimensions with unusual frequency patterns -> high importance
        z_scores = (magnitudes - stats.freq_means.unsqueeze(-3)) / (
            stats.freq_stds.unsqueeze(-3).clamp_min(1e-8)
        )
        # Weight by learned importance
        weighted = z_scores.abs() * stats.importance_weights.unsqueeze(-3)
        scores = weighted.mean(dim=-1)  # [..., seq, heads]
    else:
        # Fallback: score by total spectral energy (no calibration)
        scores = magnitudes.sum(dim=-1)  # [..., seq, heads]

    # Normalize to [0, 1]
    scores_min = scores.amin(dim=-2, keepdim=True)
    scores_max = scores.amax(dim=-2, keepdim=True)
    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)
    return scores


def select_tokens(
    scores: torch.Tensor,
    budget: int,
    window: int = DEFAULT_WINDOW,
    seq_len: int | None = None,
) -> torch.Tensor:
    """Select which tokens to retain in the KV cache.

    Always retains:
    1. The most recent `window` tokens (sliding window).
    2. The top `budget - window` tokens by importance score.

    Args:
        scores: Importance scores [..., seq_len, num_heads].
        budget: Total token budget.
        window: Recent-token window size (never evicted).
        seq_len: Actual sequence length (if scores are padded).

    Returns:
        Boolean mask [..., seq_len] of tokens to keep.
    """
    if seq_len is None:
        seq_len = scores.shape[-2]

    # Average across heads for selection
    avg_scores = scores.mean(dim=-1)  # [..., seq_len]

    if seq_len <= budget:
        # Under budget — keep everything
        return torch.ones(
            avg_scores.shape, dtype=torch.bool, device=scores.device,
        )

    # Window tokens are always kept
    keep_mask = torch.zeros(
        avg_scores.shape, dtype=torch.bool, device=scores.device,
    )
    keep_mask[..., -window:] = True

    # Select top-k from non-window tokens
    eviction_budget = budget - window
    non_window_scores = avg_scores.clone()
    non_window_scores[..., -window:] = float("-inf")  # exclude window from selection

    _, top_indices = non_window_scores.topk(
        min(eviction_budget, seq_len - window), dim=-1,
    )
    keep_mask.scatter_(-1, top_indices, True)

    return keep_mask


def apply_token_eviction(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    keep_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply token eviction mask to KV caches.

    Args:
        k_cache: Key cache [..., seq_len, num_heads, head_dim].
        v_cache: Value cache [..., seq_len, num_heads, head_dim].
        keep_mask: Boolean mask [..., seq_len].

    Returns:
        Tuple of (k_evicted, v_evicted) with reduced sequence dimension.
    """
    # Expand mask for broadcasting
    mask_expanded = keep_mask.unsqueeze(-1).unsqueeze(-1)
    mask_expanded = mask_expanded.expand_as(k_cache)

    # Gather kept tokens (flatten then reshape)
    batch_shape = k_cache.shape[:-3]
    seq_len = k_cache.shape[-3]
    num_heads = k_cache.shape[-2]
    head_dim = k_cache.shape[-1]

    num_kept = keep_mask.sum(dim=-1).max().item()

    # Use masked_select + reshape for variable-length output
    k_out = torch.zeros(
        *batch_shape, num_kept, num_heads, head_dim,
        dtype=k_cache.dtype, device=k_cache.device,
    )
    v_out = torch.zeros_like(k_out)

    # Per-batch gather (handles variable keep counts)
    flat_k = k_cache.reshape(-1, seq_len, num_heads, head_dim)
    flat_v = v_cache.reshape(-1, seq_len, num_heads, head_dim)
    flat_mask = keep_mask.reshape(-1, seq_len)

    for i in range(flat_k.shape[0]):
        kept = flat_mask[i].nonzero(as_tuple=True)[0][:num_kept]
        k_out.reshape(-1, num_kept, num_heads, head_dim)[i, :len(kept)] = flat_k[i, kept]
        v_out.reshape(-1, num_kept, num_heads, head_dim)[i, :len(kept)] = flat_v[i, kept]

    return k_out, v_out


# ─── Stats I/O ──────────────────────────────────────────────────────────────────

def load_stats(path: str | Path) -> dict[int, TriAttentionStats]:
    """Load per-layer TriAttention statistics from a .pt file.

    Returns:
        Dict mapping layer_idx -> TriAttentionStats.
    """
    path = Path(path)
    data = torch.load(path, map_location="cpu", weights_only=True)

    stats: dict[int, TriAttentionStats] = {}
    for layer_idx, layer_data in data.items():
        stats[int(layer_idx)] = TriAttentionStats(
            freq_means=layer_data["freq_means"],
            freq_stds=layer_data["freq_stds"],
            phase_means=layer_data["phase_means"],
            importance_weights=layer_data["importance_weights"],
        )
    return stats


def save_stats(stats: dict[int, TriAttentionStats], path: str | Path) -> None:
    """Save per-layer TriAttention statistics to a .pt file."""
    data = {}
    for layer_idx, s in stats.items():
        data[layer_idx] = {
            "freq_means": s.freq_means.cpu(),
            "freq_stds": s.freq_stds.cpu(),
            "phase_means": s.phase_means.cpu(),
            "importance_weights": s.importance_weights.cpu(),
        }
    torch.save(data, Path(path))


# ─── Method class ───────────────────────────────────────────────────────────────

@register_method(CacheMethod.TRIATTENTION)
class TriAttentionMethod(CompressionMethod):
    """TriAttention token eviction — not a compression method per se.

    This method doesn't compress individual tokens; it selects which
    tokens to keep. It's designed to compose with quantization methods
    via CacheConfig.triattention_enabled.

    The encode/decode interface here wraps the scoring and eviction
    logic for consistency with the method protocol.
    """

    def __init__(
        self,
        budget: int = DEFAULT_BUDGET,
        window: int = DEFAULT_WINDOW,
        stats_path: str | None = None,
    ):
        self._budget = budget
        self._window = window
        self._stats: dict[int, TriAttentionStats] | None = None
        if stats_path:
            self._stats = load_stats(stats_path)

    def info(self) -> MethodInfo:
        return MethodInfo(
            method=CacheMethod.TRIATTENTION,
            family=MethodFamily.TRIATTENTION,
            bits=16.0,  # full precision, token eviction only
            requires_calibration=True,
            supports_asymmetric=False,  # eviction applies to both K and V
            supported_head_dims=(64, 128, 192, 256),
            transform_name="Trigonometric frequency scoring (DFT)",
            description="Token eviction via frequency analysis — compose with quantization",
            fma_count=0,  # scoring cost, not per-element
            param_count=0,
        )

    def packed_dim(self, head_dim: int) -> int:
        # TriAttention doesn't change the per-token representation
        return head_dim * 2  # fp16 = 2 bytes per element

    def encode(
        self, x: torch.Tensor, *,
        head_indices: torch.Tensor | None = None, layer_idx: int = 0,
    ) -> CompressedKV:
        """Score tokens and return eviction metadata.

        Note: actual eviction happens at the cache management level,
        not here. This just computes the scores.
        """
        stats = self._stats.get(layer_idx) if self._stats else None
        if stats is not None:
            stats = TriAttentionStats(
                freq_means=stats.freq_means.to(x.device),
                freq_stds=stats.freq_stds.to(x.device),
                phase_means=stats.phase_means.to(x.device),
                importance_weights=stats.importance_weights.to(x.device),
            )
        scores = compute_frequency_scores(x, stats)
        keep_mask = select_tokens(scores, self._budget, self._window)

        return CompressedKV(
            data=x.to(torch.float16).contiguous().view(torch.uint8),
            head_dim=x.shape[-1],
            method=CacheMethod.TRIATTENTION,
            metadata={
                "scores": scores,
                "keep_mask": keep_mask,
                "budget": self._budget,
                "window": self._window,
            },
        )

    def decode(
        self, compressed: CompressedKV, *, dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Reconstruct from raw bytes (no compression, just reinterpret)."""
        raw = compressed.data.contiguous().view(torch.float16)
        target_shape = (*compressed.data.shape[:-1], compressed.head_dim)
        return raw.reshape(target_shape).to(dtype)
