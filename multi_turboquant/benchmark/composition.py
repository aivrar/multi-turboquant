# SPDX-License-Identifier: MIT
"""Provenance-safe records for composition benchmarks.

This module records results; it deliberately does not launch third-party engines.
The strict contracts prevent upstream claims and analytical projections from being
presented as measurements made on the local machine.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from statistics import median
from typing import Any, Callable, Mapping
from urllib.parse import urlparse


class ResultProvenance(str, Enum):
    """The only supported origins for composition benchmark data."""

    MEASURED_LOCAL = "measured-local"
    REPORTED_UPSTREAM = "reported-upstream"
    ANALYTICAL_SIMULATION = "analytical-simulation"


@dataclass(frozen=True)
class BenchmarkManifest:
    """Reproducibility information shared by all benchmark result types."""

    run_id: str
    profile_id: str
    engine: str
    model: str
    model_revision: str
    workload: str
    prompt_tokens: int
    output_tokens: int
    concurrency: int = 1
    warmup_runs: int = 0
    measured_runs: int = 0
    software_revisions: Mapping[str, str] = field(default_factory=dict)
    hardware: Mapping[str, str] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("run_id", "profile_id", "engine", "model", "model_revision", "workload"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.schema_version != 1:
            raise ValueError("Unsupported benchmark manifest schema version")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.prompt_tokens, self.output_tokens)
        ):
            raise ValueError("Token counts must be non-negative")
        if isinstance(self.concurrency, bool) or not isinstance(self.concurrency, int) or self.concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.warmup_runs, self.measured_runs)
        ):
            raise ValueError("Run counts must be non-negative")
        for label, values in (
            ("software_revisions", self.software_revisions),
            ("hardware", self.hardware),
        ):
            if any(
                not isinstance(key, str)
                or not key.strip()
                or not isinstance(value, str)
                or not value.strip()
                for key, value in values.items()
            ):
                raise ValueError(f"{label} must contain non-empty string keys and values")
        object.__setattr__(self, "software_revisions", dict(self.software_revisions))
        object.__setattr__(self, "hardware", dict(self.hardware))

    def comparison_key(self) -> tuple[Any, ...]:
        """Return workload dimensions that must match for a comparison."""

        return (
            self.engine,
            self.model,
            self.model_revision,
            self.workload,
            self.prompt_tokens,
            self.output_tokens,
            self.concurrency,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "profile_id": self.profile_id,
            "engine": self.engine,
            "model": self.model,
            "model_revision": self.model_revision,
            "workload": self.workload,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "concurrency": self.concurrency,
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "software_revisions": dict(self.software_revisions),
            "hardware": dict(self.hardware),
        }


@dataclass(frozen=True)
class CompositionBenchmarkResult:
    """A validated result whose provenance cannot be omitted or relabelled."""

    manifest: BenchmarkManifest
    provenance: ResultProvenance
    metrics: Mapping[str, float]
    raw_samples: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    source_url: str | None = None
    source_reference: str | None = None
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, ResultProvenance):
            raise TypeError("provenance must be a ResultProvenance value")
        metrics = dict(self.metrics)
        if not metrics:
            raise ValueError("At least one metric is required")
        if any(not isinstance(name, str) or not name.strip() for name in metrics):
            raise ValueError("Metric names must not be empty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in metrics.values()
        ):
            raise ValueError("Metric values must be finite numbers")
        samples = {name: tuple(values) for name, values in self.raw_samples.items()}
        for values in samples.values():
            if not values or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values
            ):
                raise ValueError("Raw sample sets must contain finite numbers")

        if self.provenance is ResultProvenance.MEASURED_LOCAL:
            if self.manifest.measured_runs <= 0 or not samples:
                raise ValueError("Local measurements require measured_runs and raw samples")
            if not self.manifest.hardware or not self.manifest.software_revisions:
                raise ValueError("Local measurements require hardware and software revisions")
            if self.source_url or self.source_reference or self.assumptions:
                raise ValueError("Local measurements cannot carry upstream or simulation metadata")
            if any(len(values) != self.manifest.measured_runs for values in samples.values()):
                raise ValueError("Every local sample set must match measured_runs")
            if metrics.keys() != samples.keys():
                raise ValueError("Every local metric must have a matching raw sample set")
            if any(metrics[name] != float(median(samples[name])) for name in metrics):
                raise ValueError("Local metric summaries must equal their sample medians")
        elif self.provenance is ResultProvenance.REPORTED_UPSTREAM:
            if self.manifest.measured_runs != 0:
                raise ValueError("Upstream reports must set measured_runs to zero")
            if samples or self.assumptions:
                raise ValueError("Upstream reports cannot contain local samples or assumptions")
            _require_https_url(self.source_url)
            if not isinstance(self.source_reference, str) or not self.source_reference.strip():
                raise ValueError("Upstream reports require a source reference")
        else:
            if self.manifest.measured_runs != 0:
                raise ValueError("Simulations must set measured_runs to zero")
            if samples or self.source_url or self.source_reference:
                raise ValueError("Simulations cannot contain measured or upstream evidence")
            if not self.assumptions or any(
                not isinstance(item, str) or not item.strip() for item in self.assumptions
            ):
                raise ValueError("Simulations require explicit assumptions")
            if any(not name.startswith(("estimated_", "simulated_")) for name in metrics):
                raise ValueError("Simulation metric names must start with estimated_ or simulated_")

        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "raw_samples", samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "provenance": self.provenance.value,
            "metrics": dict(self.metrics),
            "raw_samples": {name: list(values) for name, values in self.raw_samples.items()},
            "source_url": self.source_url,
            "source_reference": self.source_reference,
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class BenchmarkComparison:
    """A direct comparison, never a product of unrelated claimed speedups."""

    provenance: ResultProvenance
    baseline_run_id: str
    candidate_run_id: str
    ratios: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.value,
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "ratios": dict(self.ratios),
        }


def record_local_result(
    manifest: BenchmarkManifest,
    samples: Mapping[str, tuple[float, ...] | list[float]],
) -> CompositionBenchmarkResult:
    """Record medians from locally collected samples."""

    normalized = {name: tuple(values) for name, values in samples.items()}
    if any(
        not values
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        )
        for values in normalized.values()
    ):
        raise ValueError("Raw sample sets must contain finite numbers")
    metrics = {name: float(median(values)) for name, values in normalized.items() if values}
    return CompositionBenchmarkResult(
        manifest=manifest,
        provenance=ResultProvenance.MEASURED_LOCAL,
        metrics=metrics,
        raw_samples=normalized,
    )


def run_local_benchmark(
    manifest: BenchmarkManifest,
    operation: Callable[[], Mapping[str, float] | None],
    *,
    synchronize: Callable[[], None] | None = None,
) -> CompositionBenchmarkResult:
    """Warm up and repeatedly measure one local operation.

    The operation may return additional per-run metrics. Wall-clock latency is
    always collected by this function, and metric names must remain stable.
    """

    if manifest.measured_runs <= 0:
        raise ValueError("A local benchmark requires measured_runs greater than zero")
    synchronize = synchronize or (lambda: None)
    for _ in range(manifest.warmup_runs):
        synchronize()
        operation()
        synchronize()

    samples: dict[str, list[float]] = {"latency_ms": []}
    expected_metrics: set[str] | None = None
    for _ in range(manifest.measured_runs):
        synchronize()
        started = time.perf_counter()
        returned = dict(operation() or {})
        synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if "latency_ms" in returned:
            raise ValueError("operation metrics cannot override latency_ms")
        metric_names = set(returned)
        if expected_metrics is None:
            expected_metrics = metric_names
            samples.update({name: [] for name in returned})
        elif metric_names != expected_metrics:
            raise ValueError("operation metric names must be stable across measured runs")
        samples["latency_ms"].append(elapsed_ms)
        for name, value in returned.items():
            samples[name].append(value)
    return record_local_result(manifest, samples)


def compare_results(
    baseline: CompositionBenchmarkResult,
    candidate: CompositionBenchmarkResult,
) -> BenchmarkComparison:
    """Compare like-for-like results without mixing evidence classes."""

    if baseline.provenance is not candidate.provenance:
        raise ValueError("Cannot compare or combine results with different provenance")
    if baseline.manifest.comparison_key() != candidate.manifest.comparison_key():
        raise ValueError("Benchmark workload, model, and engine dimensions must match")
    if baseline.provenance is ResultProvenance.MEASURED_LOCAL:
        if baseline.manifest.hardware != candidate.manifest.hardware:
            raise ValueError("Local benchmark hardware must match")
        if baseline.manifest.software_revisions != candidate.manifest.software_revisions:
            raise ValueError("Local benchmark software revisions must match")
    elif baseline.provenance is ResultProvenance.REPORTED_UPSTREAM:
        if (baseline.source_url, baseline.source_reference) != (
            candidate.source_url,
            candidate.source_reference,
        ):
            raise ValueError("Upstream results must come from the same cited experiment")
    else:
        if baseline.assumptions != candidate.assumptions:
            raise ValueError("Simulation assumptions must match")

    common = baseline.metrics.keys() & candidate.metrics.keys()
    if not common:
        raise ValueError("Results do not share a metric")
    ratios: dict[str, float] = {}
    for name in sorted(common):
        base_value = baseline.metrics[name]
        if base_value == 0:
            raise ValueError(f"Cannot ratio zero baseline metric: {name}")
        ratios[name] = candidate.metrics[name] / base_value
    return BenchmarkComparison(
        provenance=baseline.provenance,
        baseline_run_id=baseline.manifest.run_id,
        candidate_run_id=candidate.manifest.run_id,
        ratios=ratios,
    )


def _require_https_url(value: str | None) -> None:
    parsed = urlparse(value or "")
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Upstream reports require an HTTPS source URL")
