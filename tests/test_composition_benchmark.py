from __future__ import annotations

from dataclasses import replace

import pytest

from multi_turboquant.benchmark.composition import (
    BenchmarkManifest,
    CompositionBenchmarkResult,
    ResultProvenance,
    compare_results,
    record_local_result,
    run_local_benchmark,
)


def manifest(**changes):
    base = BenchmarkManifest(
        run_id="local-baseline",
        profile_id="baseline",
        engine="vllm",
        model="example/model",
        model_revision="abc123",
        workload="chat",
        prompt_tokens=1024,
        output_tokens=128,
        warmup_runs=2,
        measured_runs=3,
        software_revisions={"vllm": "1.2.3"},
        hardware={"gpu": "Example GPU"},
    )
    return replace(base, **changes)


def test_local_result_uses_median_and_preserves_samples():
    result = record_local_result(manifest(), {"latency_ms": [12.0, 10.0, 11.0]})
    assert result.provenance is ResultProvenance.MEASURED_LOCAL
    assert result.metrics == {"latency_ms": 11.0}
    assert result.raw_samples["latency_ms"] == (12.0, 10.0, 11.0)


def test_local_runner_executes_warmups_and_measured_runs():
    calls = 0
    syncs = 0

    def operation():
        nonlocal calls
        calls += 1
        return {"tokens_processed": 128.0}

    def synchronize():
        nonlocal syncs
        syncs += 1

    result = run_local_benchmark(manifest(warmup_runs=2), operation, synchronize=synchronize)
    assert calls == 5
    assert syncs == 10
    assert len(result.raw_samples["latency_ms"]) == 3
    assert result.metrics["tokens_processed"] == 128.0


def test_local_runner_rejects_unstable_or_reserved_metrics():
    calls = 0

    def unstable():
        nonlocal calls
        calls += 1
        return {"first" if calls == 1 else "second": 1.0}

    with pytest.raises(ValueError, match="stable"):
        run_local_benchmark(manifest(warmup_runs=0), unstable)
    with pytest.raises(ValueError, match="cannot override"):
        run_local_benchmark(
            manifest(warmup_runs=0),
            lambda: {"latency_ms": 1.0},
        )


def test_local_result_requires_reproducibility_details_and_exact_sample_count():
    with pytest.raises(ValueError, match="hardware and software"):
        record_local_result(manifest(hardware={}), {"latency_ms": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="match measured_runs"):
        record_local_result(manifest(), {"latency_ms": [1.0]})
    with pytest.raises(ValueError, match="sample medians"):
        CompositionBenchmarkResult(
            manifest=manifest(),
            provenance=ResultProvenance.MEASURED_LOCAL,
            metrics={"latency_ms": 99.0},
            raw_samples={"latency_ms": (1.0, 2.0, 3.0)},
        )


def test_upstream_result_requires_https_citation_and_cannot_claim_samples():
    kwargs = dict(
        manifest=manifest(measured_runs=0),
        provenance=ResultProvenance.REPORTED_UPSTREAM,
        metrics={"tokens_per_second": 42.0},
        source_reference="Table 2, commit abc123",
    )
    with pytest.raises(ValueError, match="HTTPS"):
        CompositionBenchmarkResult(source_url="http://example.com", **kwargs)
    result = CompositionBenchmarkResult(source_url="https://example.com/paper", **kwargs)
    assert result.provenance.value == "reported-upstream"
    with pytest.raises(ValueError, match="cannot contain local"):
        CompositionBenchmarkResult(
            source_url="https://example.com/paper",
            raw_samples={"tokens_per_second": (42.0,)},
            **kwargs,
        )


def test_simulation_requires_explicit_assumptions_and_estimated_metric_names():
    kwargs = dict(
        manifest=manifest(measured_runs=0),
        provenance=ResultProvenance.ANALYTICAL_SIMULATION,
        assumptions=("No allocator fragmentation beyond stated efficiency",),
    )
    with pytest.raises(ValueError, match="start with"):
        CompositionBenchmarkResult(metrics={"latency_ms": 2.0}, **kwargs)
    result = CompositionBenchmarkResult(metrics={"estimated_kv_bytes": 2048.0}, **kwargs)
    assert result.to_dict()["provenance"] == "analytical-simulation"
    with pytest.raises(ValueError, match="measured_runs to zero"):
        CompositionBenchmarkResult(
            manifest=manifest(),
            metrics={"estimated_kv_bytes": 2048.0},
            provenance=ResultProvenance.ANALYTICAL_SIMULATION,
            assumptions=("An assumption",),
        )


def test_comparison_rejects_mixed_provenance_and_mismatched_environments():
    baseline = record_local_result(manifest(), {"latency_ms": [10.0, 10.0, 10.0]})
    candidate = record_local_result(
        manifest(run_id="candidate", profile_id="candidate"),
        {"latency_ms": [5.0, 5.0, 5.0]},
    )
    comparison = compare_results(baseline, candidate)
    assert comparison.ratios == {"latency_ms": 0.5}

    upstream = CompositionBenchmarkResult(
        manifest=manifest(run_id="upstream", measured_runs=0),
        provenance=ResultProvenance.REPORTED_UPSTREAM,
        metrics={"latency_ms": 3.0},
        source_url="https://example.com/results",
        source_reference="commit abc",
    )
    with pytest.raises(ValueError, match="different provenance"):
        compare_results(baseline, upstream)
    changed_hardware = record_local_result(
        manifest(run_id="other", hardware={"gpu": "Other GPU"}),
        {"latency_ms": [5.0, 5.0, 5.0]},
    )
    with pytest.raises(ValueError, match="hardware"):
        compare_results(baseline, changed_hardware)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "fast"])
def test_non_finite_metrics_fail_closed(value):
    with pytest.raises(ValueError, match="finite"):
        record_local_result(manifest(), {"latency_ms": [value, value, value]})


def test_manifest_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="non-negative"):
        manifest(prompt_tokens=-1)
    with pytest.raises(ValueError, match="positive"):
        manifest(concurrency=0)
    with pytest.raises(ValueError, match="must not be empty"):
        manifest(run_id=1)
    with pytest.raises(ValueError, match="non-negative"):
        manifest(prompt_tokens=True)
    with pytest.raises(ValueError, match="non-empty string"):
        manifest(hardware={"gpu": ""})
