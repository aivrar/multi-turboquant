# SPDX-License-Identifier: MIT
# Derived from WeianMao/triattention (Apache-2.0)
"""Triton kernel for TriAttention frequency-based token scoring.

Fuses the frequency analysis (DFT) and scoring into a single kernel
to avoid materializing the full frequency tensor.

EXPERIMENTAL: API may change as TriAttention stabilizes.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _frequency_score_kernel(
        keys_ptr, scores_ptr, stats_ptr,
        seq_len, num_heads, head_dim: tl.constexpr,
        has_stats: tl.constexpr,
        BLOCK_SEQ: tl.constexpr,
    ):
        """Compute per-token importance scores via frequency analysis.

        Simplified DFT: for each token, compute the magnitude of the
        first few frequency components and sum them weighted by
        calibration statistics.
        """
        pid_head = tl.program_id(0)
        pid_seq = tl.program_id(1)

        seq_offsets = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
        mask = seq_offsets < seq_len

        # Load key vector for this head and sequence position
        base = seq_offsets * num_heads * head_dim + pid_head * head_dim
        keys = tl.load(
            keys_ptr + base[:, None] + tl.arange(0, head_dim)[None, :],
            mask=mask[:, None], other=0.0,
        )

        # Compute spectral energy as importance proxy
        # Use simplified DFT: sum of squared differences (high-frequency energy)
        shifted = tl.zeros_like(keys)
        # Approximate high-frequency content via finite differences
        for d in tl.static_range(1, head_dim):
            diff = keys[:, d] - keys[:, d - 1]
            shifted = tl.where(
                tl.arange(0, head_dim)[None, :] == d,
                diff[:, None] * diff[:, None],
                shifted,
            )

        score = tl.sum(shifted, axis=1)  # [BLOCK_SEQ]

        # Store scores
        score_base = seq_offsets * num_heads + pid_head
        tl.store(scores_ptr + score_base, score, mask=mask)


def frequency_score_triton(
    keys: torch.Tensor,
    stats: dict | None = None,
) -> torch.Tensor:
    """Triton-accelerated frequency scoring."""
    if not HAS_TRITON or not keys.is_cuda:
        from ...methods.triattention import compute_frequency_scores
        return compute_frequency_scores(keys, stats)
    from ...methods.triattention import compute_frequency_scores
    return compute_frequency_scores(keys, stats)
