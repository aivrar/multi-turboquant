# SPDX-License-Identifier: MIT
# Derived from WeianMao/triattention calibration code (Apache-2.0)
"""Generate TriAttention frequency statistics for token eviction.

Analyzes a model's attention patterns using calibration prompts to
compute per-layer frequency statistics that guide token importance scoring.

Usage as CLI:
    python -m multi_turboquant.calibration.generate_stats /path/to/model \
        --prompts calibration_prompts.jsonl --output stats.pt

Usage as library:
    from multi_turboquant.calibration import generate_triattention_stats
    stats = generate_triattention_stats(model, tokenizer, prompts)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


def generate_triattention_stats(
    model: Any = None,
    tokenizer: Any = None,
    prompts: list[str] | None = None,
    *,
    model_path: str | None = None,
    num_prompts: int = 32,
    max_seq_len: int = 2048,
    verbose: bool = True,
) -> dict[int, dict[str, torch.Tensor]]:
    """Generate per-layer frequency statistics for TriAttention.

    This function hooks into the model's attention layers during forward
    passes to collect frequency statistics from the key projections.

    Args:
        model: HuggingFace model (or None to load from model_path).
        tokenizer: HuggingFace tokenizer.
        prompts: Calibration prompts. If None, uses built-in defaults.
        model_path: Path to model directory (used if model is None).
        num_prompts: Number of prompts to use for calibration.
        max_seq_len: Maximum sequence length per prompt.
        verbose: Print progress.

    Returns:
        Dict mapping layer_idx -> {freq_means, freq_stds, phase_means,
        importance_weights} tensors.
    """
    if prompts is None:
        prompts = _default_calibration_prompts()

    prompts = prompts[:num_prompts]

    if model is None and model_path is not None:
        if verbose:
            print(f"Loading model from {model_path}...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except ImportError:
            raise ImportError(
                "transformers is required for TriAttention calibration. "
                "Install with: pip install transformers"
            )

    if model is None:
        raise ValueError("Either model or model_path must be provided")

    device = next(model.parameters()).device

    # Determine number of layers
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
    head_dim = getattr(
        model.config, "head_dim",
        model.config.hidden_size // num_heads,
    )
    freq_bins = head_dim // 2 + 1

    if verbose:
        print(f"Model: {num_layers} layers, {num_kv_heads} KV heads, head_dim={head_dim}")
        print(f"Collecting frequency stats from {len(prompts)} prompts...")

    # Accumulators for frequency statistics
    freq_sum = torch.zeros(num_layers, num_kv_heads, freq_bins, device=device)
    freq_sq_sum = torch.zeros_like(freq_sum)
    phase_sum = torch.zeros_like(freq_sum)
    count = torch.zeros(num_layers, 1, 1, device=device)

    # Hook into attention layers to capture key projections
    hooks = []
    captured_keys: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # Extract key projection from attention output
            # This is model-architecture specific
            if hasattr(output, 'past_key_value') and output.past_key_value is not None:
                k = output.past_key_value[0]  # [batch, heads, seq, dim]
                if k is not None:
                    captured_keys[layer_idx].append(k.detach())
        return hook_fn

    try:
        # Register hooks on attention layers
        for i, layer in enumerate(_get_attention_layers(model)):
            h = layer.register_forward_hook(make_hook(i))
            hooks.append(h)

        # Run calibration prompts
        model.eval()
        with torch.no_grad():
            for prompt_idx, prompt in enumerate(prompts):
                if tokenizer is None:
                    continue

                inputs = tokenizer(
                    prompt, return_tensors="pt",
                    max_length=max_seq_len, truncation=True,
                ).to(device)

                try:
                    model(**inputs, use_cache=True)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: prompt {prompt_idx} failed: {e}")
                    continue

                # Process captured keys
                for layer_idx, key_list in captured_keys.items():
                    if not key_list:
                        continue
                    for k_tensor in key_list:
                        # k_tensor: [batch, heads, seq, dim]
                        k_flat = k_tensor.reshape(-1, k_tensor.shape[-1]).float()
                        freq = torch.fft.rfft(k_flat, dim=-1)
                        mag = freq.abs()
                        phase = freq.angle()

                        # Reshape to [heads, freq_bins]
                        mag = mag.reshape(-1, num_kv_heads, freq_bins)
                        phase = phase.reshape(-1, num_kv_heads, freq_bins)

                        freq_sum[layer_idx] += mag.sum(0)
                        freq_sq_sum[layer_idx] += (mag ** 2).sum(0)
                        phase_sum[layer_idx] += phase.sum(0)
                        count[layer_idx] += mag.shape[0]

                    key_list.clear()

                if verbose:
                    sys.stdout.write(
                        f"\r  Processing prompt {prompt_idx + 1}/{len(prompts)}..."
                    )
                    sys.stdout.flush()

    finally:
        for h in hooks:
            h.remove()

    if verbose:
        print(" done")

    # Compute statistics
    stats: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx in range(num_layers):
        n = count[layer_idx].clamp_min(1)
        means = freq_sum[layer_idx] / n
        sq_means = freq_sq_sum[layer_idx] / n
        stds = (sq_means - means ** 2).clamp_min(0).sqrt()
        phase_means = phase_sum[layer_idx] / n

        # Importance weights: dimensions with high variance are more discriminative
        importance = stds / (stds.sum(dim=-1, keepdim=True).clamp_min(1e-8))

        stats[layer_idx] = {
            "freq_means": means.cpu(),
            "freq_stds": stds.cpu(),
            "phase_means": phase_means.cpu(),
            "importance_weights": importance.cpu(),
        }

    return stats


def save_stats(stats: dict[int, dict[str, torch.Tensor]], path: str | Path) -> str:
    """Save TriAttention statistics to a .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, path)
    return str(path)


def _get_attention_layers(model: Any) -> list:
    """Extract attention layer modules from various model architectures."""
    # Try common patterns
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [layer.self_attn for layer in model.model.layers]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [layer.attn for layer in model.transformer.h]
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return [layer.attention for layer in model.gpt_neox.layers]
    raise ValueError(
        "Could not detect attention layer structure. "
        "Supported: LlamaForCausalLM, GPT2, GPTNeoX"
    )


def _default_calibration_prompts() -> list[str]:
    """Return default calibration prompts for frequency analysis."""
    return [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "In mathematics, a group is a set equipped with an operation. " * 10,
        "Machine learning is a subset of artificial intelligence. " * 10,
        "The capital of France is Paris, known for the Eiffel Tower. " * 10,
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)\n" * 10,
        "SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.user_id\n" * 5,
        "HTTP/1.1 200 OK\nContent-Type: application/json\n{\"status\": \"ok\"}\n" * 10,
        "The mitochondria is the powerhouse of the cell. In biology, " * 10,
    ] * 4  # 32 prompts


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate TriAttention frequency statistics"
    )
    parser.add_argument("model_path", help="Path to HuggingFace model")
    parser.add_argument("--output", default="triattention_stats.pt")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--prompts", default=None, help="JSONL file with prompts")
    args = parser.parse_args()

    prompts = None
    if args.prompts:
        with open(args.prompts) as f:
            prompts = [json.loads(line)["text"] for line in f]

    stats = generate_triattention_stats(
        model_path=args.model_path,
        prompts=prompts,
        num_prompts=args.num_prompts,
        max_seq_len=args.max_seq_len,
    )
    out_path = save_stats(stats, args.output)
    print(f"\nSaved {out_path}")
    print(f"  {len(stats)} layers calibrated")


if __name__ == "__main__":
    main()
