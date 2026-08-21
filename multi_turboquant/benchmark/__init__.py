# SPDX-License-Identifier: MIT
"""Unified benchmark suite for all KV cache compression methods."""

from .run_benchmark import run_benchmark, BenchmarkResult
from .perplexity import evaluate_perplexity
from .vram_profile import profile_vram
from .composition import (
    BenchmarkComparison,
    BenchmarkManifest,
    CompositionBenchmarkResult,
    ResultProvenance,
    compare_results,
    record_local_result,
    run_local_benchmark,
)
from .capacity import (
    CachePolicy,
    CapacityScenario,
    CapacitySimulationResult,
    KVModelShape,
    simulate_capacity,
)

__all__ = [
    "run_benchmark",
    "BenchmarkResult",
    "evaluate_perplexity",
    "profile_vram",
    "BenchmarkComparison",
    "BenchmarkManifest",
    "CompositionBenchmarkResult",
    "ResultProvenance",
    "compare_results",
    "record_local_result",
    "run_local_benchmark",
    "CachePolicy",
    "CapacityScenario",
    "CapacitySimulationResult",
    "KVModelShape",
    "simulate_capacity",
]
