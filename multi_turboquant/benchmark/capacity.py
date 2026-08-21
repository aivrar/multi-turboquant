# SPDX-License-Identifier: MIT
"""Analytical KV-cache memory and concurrency simulator.

The simulator performs byte accounting only. It does not predict latency,
throughput, output quality, or compounded optimization speedups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .composition import ResultProvenance


@dataclass(frozen=True)
class KVModelShape:
    """Transformer dimensions that determine KV-cache storage."""

    num_layers: int
    num_kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        for name in ("num_layers", "num_kv_heads", "head_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class CachePolicy:
    """Explicit storage assumptions for one candidate cache representation."""

    k_bits: float = 16.0
    v_bits: float = 16.0
    retained_fraction: float = 1.0
    allocator_efficiency: float = 1.0
    per_token_metadata_bytes: int = 0
    fixed_overhead_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("k_bits", "v_bits"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        for name in ("retained_fraction", "allocator_efficiency"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"{name} must be in the interval (0, 1]")
        for name in ("per_token_metadata_bytes", "fixed_overhead_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class CapacityScenario:
    """Memory budget and workload dimensions for a capacity calculation."""

    context_tokens: int
    available_memory_bytes: int
    model_weights_bytes: int = 0
    runtime_overhead_bytes: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.context_tokens, bool)
            or not isinstance(self.context_tokens, int)
            or self.context_tokens <= 0
        ):
            raise ValueError("context_tokens must be positive")
        for name in (
            "available_memory_bytes",
            "model_weights_bytes",
            "runtime_overhead_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        reserved = self.model_weights_bytes + self.runtime_overhead_bytes
        if reserved > self.available_memory_bytes:
            raise ValueError("Model weights and runtime overhead exceed available memory")


@dataclass(frozen=True)
class CapacitySimulationResult:
    """Byte-accounting output with an immutable analytical provenance label."""

    baseline_bytes_per_sequence: int
    candidate_bytes_per_sequence: int
    available_kv_bytes: int
    baseline_max_concurrency: int
    candidate_max_concurrency: int
    storage_ratio: float
    assumptions: tuple[str, ...]
    provenance: ResultProvenance = ResultProvenance.ANALYTICAL_SIMULATION

    def __post_init__(self) -> None:
        if self.provenance is not ResultProvenance.ANALYTICAL_SIMULATION:
            raise ValueError("Capacity results must use analytical-simulation provenance")
        if not self.assumptions or any(
            not isinstance(item, str) or not item.strip() for item in self.assumptions
        ):
            raise ValueError("Capacity results require explicit assumptions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.value,
            "baseline_bytes_per_sequence": self.baseline_bytes_per_sequence,
            "candidate_bytes_per_sequence": self.candidate_bytes_per_sequence,
            "available_kv_bytes": self.available_kv_bytes,
            "baseline_max_concurrency": self.baseline_max_concurrency,
            "candidate_max_concurrency": self.candidate_max_concurrency,
            "storage_ratio": self.storage_ratio,
            "assumptions": list(self.assumptions),
        }


def simulate_capacity(
    shape: KVModelShape,
    scenario: CapacityScenario,
    candidate: CachePolicy,
    *,
    baseline: CachePolicy | None = None,
) -> CapacitySimulationResult:
    """Calculate KV bytes and maximum whole-sequence concurrency.

    ``retained_fraction`` is applied before byte packing. Allocator efficiency
    is then applied conservatively by rounding the physical allocation upward.
    """

    baseline = baseline or CachePolicy()
    baseline_bytes = _sequence_bytes(shape, scenario.context_tokens, baseline)
    candidate_bytes = _sequence_bytes(shape, scenario.context_tokens, candidate)
    available = (
        scenario.available_memory_bytes
        - scenario.model_weights_bytes
        - scenario.runtime_overhead_bytes
    )
    assumptions = (
        "K and V storage is derived from layers, KV heads, head dimension, and bit width.",
        "Retained tokens are rounded up before storage is calculated.",
        "Allocator efficiency is applied to cache bytes after metadata and fixed overhead.",
        "The result excludes latency, throughput, transfer bandwidth, and output quality.",
    )
    return CapacitySimulationResult(
        baseline_bytes_per_sequence=baseline_bytes,
        candidate_bytes_per_sequence=candidate_bytes,
        available_kv_bytes=available,
        baseline_max_concurrency=available // baseline_bytes,
        candidate_max_concurrency=available // candidate_bytes,
        storage_ratio=candidate_bytes / baseline_bytes,
        assumptions=assumptions,
    )


def _sequence_bytes(shape: KVModelShape, context_tokens: int, policy: CachePolicy) -> int:
    retained = Fraction(str(policy.retained_fraction))
    retained_tokens = _ceil_fraction(context_tokens * retained)
    element_count = shape.num_layers * shape.num_kv_heads * shape.head_dim * retained_tokens
    total_bits = Fraction(str(policy.k_bits)) + Fraction(str(policy.v_bits))
    packed_bytes = _ceil_fraction(element_count * total_bits / 8)
    logical_bytes = (
        packed_bytes
        + retained_tokens * policy.per_token_metadata_bytes
        + policy.fixed_overhead_bytes
    )
    efficiency = Fraction(str(policy.allocator_efficiency))
    return _ceil_fraction(Fraction(logical_bytes, 1) / efficiency)


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)
