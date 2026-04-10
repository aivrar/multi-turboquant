# SPDX-License-Identifier: MIT
"""Perplexity evaluation for KV cache compression quality.

Measures how much each compression method affects model output quality
by comparing perplexity on a test dataset (WikiText-2 or custom).

This requires a running inference server (llama.cpp or vLLM).
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch


def evaluate_perplexity(
    texts: list[str],
    *,
    method: str,
    endpoint: str = "http://localhost:8080/v1/completions",
    max_tokens: int = 1,
    batch_size: int = 8,
    verbose: bool = True,
) -> dict[str, float]:
    """Evaluate perplexity by measuring log-likelihood of text.

    This sends text to a running inference server and measures the
    log-probability of each token given its context.

    Args:
        texts: List of text samples to evaluate.
        method: Method name (for labeling results).
        endpoint: OpenAI-compatible API endpoint.
        max_tokens: Max tokens per request (1 for perplexity eval).
        batch_size: Batch size for requests.
        verbose: Print progress.

    Returns:
        Dict with "perplexity", "avg_logprob", "num_tokens", "method".
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is required: pip install requests")

    total_logprob = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        for text in batch:
            try:
                resp = requests.post(
                    endpoint,
                    json={
                        "prompt": text,
                        "max_tokens": max_tokens,
                        "logprobs": 1,
                        "echo": True,
                    },
                    timeout=60,
                )
                data = resp.json()

                if "choices" in data and data["choices"]:
                    logprobs = data["choices"][0].get("logprobs", {})
                    token_logprobs = logprobs.get("token_logprobs", [])
                    # First token has no logprob
                    valid = [lp for lp in token_logprobs if lp is not None]
                    total_logprob += sum(valid)
                    total_tokens += len(valid)

            except Exception as e:
                if verbose:
                    print(f"  Warning: request failed: {e}")

        if verbose:
            processed = min(i + batch_size, len(texts))
            print(f"\r  {method}: {processed}/{len(texts)} texts, "
                  f"{total_tokens} tokens", end="", flush=True)

    if verbose:
        print()

    if total_tokens == 0:
        return {
            "method": method,
            "perplexity": float("inf"),
            "avg_logprob": 0.0,
            "num_tokens": 0,
        }

    avg_logprob = total_logprob / total_tokens
    perplexity = math.exp(-avg_logprob)

    return {
        "method": method,
        "perplexity": perplexity,
        "avg_logprob": avg_logprob,
        "num_tokens": total_tokens,
    }


def load_wikitext2(max_samples: int = 100) -> list[str]:
    """Load WikiText-2 test split for perplexity evaluation."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 100]
        return texts[:max_samples]
    except ImportError:
        # Fallback: generate synthetic test data
        return [
            "The history of artificial intelligence began in antiquity, "
            "with myths, stories and rumors of artificial beings endowed "
            "with intelligence or consciousness by master craftsmen. " * 5
        ] * max_samples
