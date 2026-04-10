# SPDX-License-Identifier: MIT
"""VRAM profiling for KV cache compression methods.

Measures actual GPU memory usage at various context lengths to
validate theoretical compression ratios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..config import CacheConfig, CacheMethod, METHOD_BITS
from ..presets import PRESETS


@dataclass
class VRAMProfile:
    """VRAM usage measurement for a configuration."""
    config_name: str
    context_length: int
    k_method: str
    v_method: str
    estimated_kv_bytes: int
    measured_kv_bytes: int | None
    theoretical_compression: float
    actual_compression: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config_name,
            "context_length": self.context_length,
            "k_method": self.k_method,
            "v_method": self.v_method,
            "estimated_kv_mb": round(self.estimated_kv_bytes / (1024**2), 2),
            "measured_kv_mb": (
                round(self.measured_kv_bytes / (1024**2), 2)
                if self.measured_kv_bytes else None
            ),
            "theoretical_compression": round(self.theoretical_compression, 2),
            "actual_compression": (
                round(self.actual_compression, 2)
                if self.actual_compression else None
            ),
        }


def profile_vram(
    configs: dict[str, CacheConfig] | None = None,
    *,
    context_lengths: list[int] | None = None,
    head_dim: int = 128,
    num_kv_heads: int = 8,
    num_layers: int = 32,
    device: str = "cuda",
    verbose: bool = True,
) -> list[VRAMProfile]:
    """Profile VRAM usage across configs and context lengths.

    Args:
        configs: Named configs to profile (default: all presets).
        context_lengths: Context lengths to test.
        head_dim: Model head dimension.
        num_kv_heads: Number of KV attention heads.
        num_layers: Number of transformer layers.
        device: Device to profile on.
        verbose: Print progress.

    Returns:
        List of VRAMProfile measurements.
    """
    if configs is None:
        configs = {
            name: CacheConfig(
                k_method=p.k_method, v_method=p.v_method,
                triattention_enabled=p.triattention_enabled,
                triattention_budget=p.triattention_budget,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                num_layers=num_layers,
            )
            for name, p in PRESETS.items()
            if p.k_method != CacheMethod.Q8_0  # skip Q8 for profiling
        }

    if context_lengths is None:
        context_lengths = [1024, 4096, 8192, 16384, 32768]

    # Baseline FP16 config
    fp16_config = CacheConfig(
        k_method=CacheMethod.FP16, v_method=CacheMethod.FP16,
        head_dim=head_dim, num_kv_heads=num_kv_heads, num_layers=num_layers,
    )

    results: list[VRAMProfile] = []

    for ctx_len in context_lengths:
        fp16_bytes = fp16_config.estimate_kv_bytes(ctx_len)

        for name, config in configs.items():
            estimated = config.estimate_kv_bytes(ctx_len)
            theoretical = fp16_bytes / max(estimated, 1)

            # Try to measure actual GPU memory if CUDA is available
            measured = None
            actual_comp = None
            if device == "cuda" and torch.cuda.is_available():
                try:
                    measured = _measure_actual_vram(
                        config, ctx_len, head_dim, num_kv_heads, num_layers,
                    )
                    if measured > 0:
                        actual_comp = fp16_bytes / measured
                except Exception:
                    pass

            profile = VRAMProfile(
                config_name=name,
                context_length=ctx_len,
                k_method=config.k_method.value,
                v_method=config.v_method.value,
                estimated_kv_bytes=estimated,
                measured_kv_bytes=measured,
                theoretical_compression=theoretical,
                actual_compression=actual_comp,
            )
            results.append(profile)

    if verbose:
        _print_profiles(results, context_lengths)

    return results


def _measure_actual_vram(
    config: CacheConfig,
    ctx_len: int,
    head_dim: int,
    num_kv_heads: int,
    num_layers: int,
) -> int:
    """Measure actual VRAM usage by allocating compressed KV cache."""
    from ..registry import get_method

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(device)

    # Allocate compressed cache for one layer
    try:
        k_method = get_method(config.k_method)
        v_method = get_method(config.v_method)

        k_packed_dim = k_method.packed_dim(head_dim)
        v_packed_dim = v_method.packed_dim(head_dim)

        effective_len = ctx_len
        if config.triattention_enabled:
            effective_len = min(
                ctx_len,
                config.triattention_budget + config.triattention_window,
            )

        # Allocate for all layers
        caches = []
        for _ in range(num_layers):
            k_cache = torch.zeros(
                effective_len, num_kv_heads, k_packed_dim,
                dtype=torch.uint8, device=device,
            )
            v_cache = torch.zeros(
                effective_len, num_kv_heads, v_packed_dim,
                dtype=torch.uint8, device=device,
            )
            caches.append((k_cache, v_cache))

        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated(device)

        # Cleanup
        del caches
        torch.cuda.empty_cache()

        return mem_after - mem_before

    except Exception:
        return 0


def _print_profiles(
    results: list[VRAMProfile],
    context_lengths: list[int],
) -> None:
    """Print VRAM profiles as a formatted table."""
    configs = sorted(set(r.config_name for r in results))

    print(f"\n{'Config':<25}", end="")
    for ctx in context_lengths:
        print(f" {ctx:>8}", end="")
    print()
    print("-" * (25 + 9 * len(context_lengths)))

    for config_name in configs:
        config_results = [r for r in results if r.config_name == config_name]
        print(f"{config_name:<25}", end="")
        for ctx in context_lengths:
            r = next((r for r in config_results if r.context_length == ctx), None)
            if r:
                mb = r.estimated_kv_bytes / (1024**2)
                print(f" {mb:>7.1f}M", end="")
            else:
                print(f" {'N/A':>8}", end="")
        print()

    print(f"\n{'Config':<25}", end="")
    for ctx in context_lengths:
        print(f" {ctx:>8}", end="")
    print("  (compression ratio)")
    print("-" * (25 + 9 * len(context_lengths)))

    for config_name in configs:
        config_results = [r for r in results if r.config_name == config_name]
        print(f"{config_name:<25}", end="")
        for ctx in context_lengths:
            r = next((r for r in config_results if r.context_length == ctx), None)
            if r:
                print(f" {r.theoretical_compression:>7.1f}x", end="")
            else:
                print(f" {'N/A':>8}", end="")
        print()
