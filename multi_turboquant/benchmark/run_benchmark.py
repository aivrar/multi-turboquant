# SPDX-License-Identifier: MIT
"""Unified benchmark runner — test all methods head-to-head.

Measures encode/decode speed, reconstruction quality (MSE, cosine sim),
and compression ratios for all registered methods.

Usage as CLI:
    python -m multi_turboquant.benchmark.run_benchmark --head-dim 128

Usage as library:
    from multi_turboquant.benchmark import run_benchmark
    results = run_benchmark(head_dim=128, seq_len=1024)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from ..config import CacheMethod, METHOD_BITS
from ..registry import registered_methods, get_method


@dataclass
class BenchmarkResult:
    """Results for a single method benchmark run."""
    method: str
    head_dim: int
    seq_len: int
    num_heads: int
    bits: float
    compression_ratio: float
    encode_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    mse: float              # mean squared error vs original
    cosine_sim: float       # mean cosine similarity vs original
    packed_bytes: int       # total packed cache size
    original_bytes: int     # original fp16 cache size
    encode_throughput_mbs: float  # MB/s encode
    decode_throughput_mbs: float  # MB/s decode

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "head_dim": self.head_dim,
            "seq_len": self.seq_len,
            "num_heads": self.num_heads,
            "bits": self.bits,
            "compression_ratio": self.compression_ratio,
            "encode_time_ms": round(self.encode_time_ms, 3),
            "decode_time_ms": round(self.decode_time_ms, 3),
            "total_time_ms": round(self.total_time_ms, 3),
            "mse": round(self.mse, 6),
            "cosine_sim": round(self.cosine_sim, 6),
            "packed_bytes": self.packed_bytes,
            "original_bytes": self.original_bytes,
            "encode_throughput_mbs": round(self.encode_throughput_mbs, 1),
            "decode_throughput_mbs": round(self.decode_throughput_mbs, 1),
        }


def run_benchmark(
    *,
    head_dim: int = 128,
    seq_len: int = 1024,
    num_heads: int = 8,
    num_runs: int = 3,
    warmup_runs: int = 1,
    device: str = "cpu",
    methods: list[str] | None = None,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """Run benchmarks on all (or specified) methods.

    Args:
        head_dim: Head dimension to test.
        seq_len: Sequence length (number of tokens).
        num_heads: Number of KV heads.
        num_runs: Number of timed runs (best is reported).
        warmup_runs: Warmup runs before timing.
        device: "cpu" or "cuda".
        methods: List of method names to test (None = all registered).
        verbose: Print progress and results.

    Returns:
        List of BenchmarkResult, sorted by method name.
    """
    if methods is None:
        methods_to_test = registered_methods()
    else:
        methods_to_test = [CacheMethod(m) for m in methods]

    # Filter to methods that support this head_dim
    valid_methods = []
    for m in methods_to_test:
        if m in (CacheMethod.FP16, CacheMethod.Q8_0, CacheMethod.TRIATTENTION):
            continue  # Skip baselines and TriAttention (not applicable)
        try:
            method_inst = get_method(m)
            if method_inst.validate_head_dim(head_dim):
                valid_methods.append(m)
            elif verbose:
                print(f"  Skipping {m.value}: doesn't support head_dim={head_dim}")
        except Exception as e:
            if verbose:
                print(f"  Skipping {m.value}: {e}")

    if verbose:
        print(f"Benchmarking {len(valid_methods)} methods")
        print(f"  head_dim={head_dim}  seq_len={seq_len}  "
              f"num_heads={num_heads}  device={device}")
        print(f"  {num_runs} runs + {warmup_runs} warmup")
        print()

    # Generate random test data
    torch_device = torch.device(device)
    x = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32, device=torch_device)
    original_bytes = seq_len * num_heads * head_dim * 2  # fp16

    results: list[BenchmarkResult] = []

    for method_enum in valid_methods:
        method = get_method(method_enum)
        info = method.info()

        if verbose:
            print(f"  {method_enum.value}...", end="", flush=True)

        # Warmup
        for _ in range(warmup_runs):
            compressed = method.encode(x)
            _ = method.decode(compressed)

        # Timed runs
        encode_times = []
        decode_times = []
        mses = []
        cosines = []

        for _ in range(num_runs):
            # Encode
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            compressed = method.encode(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            encode_times.append((t1 - t0) * 1000)

            # Decode
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            decoded = method.decode(compressed, dtype=torch.float32)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            decode_times.append((t1 - t0) * 1000)

            # Quality metrics
            mse = (x - decoded).square().mean().item()
            cos = torch.nn.functional.cosine_similarity(
                x.reshape(-1, head_dim), decoded.reshape(-1, head_dim), dim=-1
            ).mean().item()
            mses.append(mse)
            cosines.append(cos)

        # Best run
        best_idx = encode_times.index(min(encode_times))
        encode_ms = encode_times[best_idx]
        decode_ms = decode_times[best_idx]

        packed_bytes = compressed.data.numel() * compressed.data.element_size()
        data_mb = original_bytes / (1024 * 1024)

        result = BenchmarkResult(
            method=method_enum.value,
            head_dim=head_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            bits=info.bits,
            compression_ratio=16.0 / info.bits,
            encode_time_ms=encode_ms,
            decode_time_ms=decode_ms,
            total_time_ms=encode_ms + decode_ms,
            mse=sum(mses) / len(mses),
            cosine_sim=sum(cosines) / len(cosines),
            packed_bytes=packed_bytes,
            original_bytes=original_bytes,
            encode_throughput_mbs=data_mb / (encode_ms / 1000) if encode_ms > 0 else 0,
            decode_throughput_mbs=data_mb / (decode_ms / 1000) if decode_ms > 0 else 0,
        )
        results.append(result)

        if verbose:
            print(f" {encode_ms:.1f}ms enc / {decode_ms:.1f}ms dec | "
                  f"MSE={result.mse:.6f} cos={result.cosine_sim:.6f} | "
                  f"{result.compression_ratio:.1f}x")

    if verbose:
        print()
        _print_table(results)

    return results


def _print_table(results: list[BenchmarkResult]) -> None:
    """Print results as a formatted table."""
    print(f"{'Method':<15} {'Bits':>5} {'Comp':>5} {'Enc ms':>8} "
          f"{'Dec ms':>8} {'MSE':>10} {'Cosine':>8}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x.bits):
        print(f"{r.method:<15} {r.bits:>5.2f} {r.compression_ratio:>5.1f}x "
              f"{r.encode_time_ms:>8.1f} {r.decode_time_ms:>8.1f} "
              f"{r.mse:>10.6f} {r.cosine_sim:>8.6f}")


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark KV cache methods")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = run_benchmark(
        head_dim=args.head_dim,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        num_runs=args.num_runs,
        device=args.device,
        methods=args.methods,
        verbose=not args.json,
    )

    if args.json:
        import json
        print(json.dumps([r.to_dict() for r in results], indent=2))


if __name__ == "__main__":
    main()
