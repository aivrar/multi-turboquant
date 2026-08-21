from __future__ import annotations

import json

import pytest

from multi_turboquant.benchmark.capacity import (
    CachePolicy,
    CapacityScenario,
    CapacitySimulationResult,
    KVModelShape,
    simulate_capacity,
)
from multi_turboquant.benchmark.composition import ResultProvenance
from multi_turboquant.optimizations.composition_cli import main


SHAPE = KVModelShape(num_layers=2, num_kv_heads=4, head_dim=8)


def test_fp16_formula_and_capacity_budget_are_exact():
    scenario = CapacityScenario(
        context_tokens=100,
        available_memory_bytes=100_000,
        model_weights_bytes=10_000,
        runtime_overhead_bytes=10_000,
    )
    result = simulate_capacity(SHAPE, scenario, CachePolicy())
    expected = 2 * 4 * 8 * 100 * (16 + 16) // 8
    assert result.baseline_bytes_per_sequence == expected
    assert result.candidate_bytes_per_sequence == expected
    assert result.available_kv_bytes == 80_000
    assert result.baseline_max_concurrency == 80_000 // expected


def test_quantization_and_token_retention_increase_estimated_capacity():
    scenario = CapacityScenario(context_tokens=100, available_memory_bytes=1_000_000)
    result = simulate_capacity(
        SHAPE,
        scenario,
        CachePolicy(k_bits=4, v_bits=4, retained_fraction=0.5),
    )
    assert result.candidate_bytes_per_sequence == result.baseline_bytes_per_sequence // 8
    assert result.candidate_max_concurrency >= result.baseline_max_concurrency * 8
    assert result.provenance is ResultProvenance.ANALYTICAL_SIMULATION


def test_metadata_fixed_overhead_and_allocator_efficiency_are_accounted_for():
    result = simulate_capacity(
        KVModelShape(1, 1, 1),
        CapacityScenario(context_tokens=3, available_memory_bytes=100),
        CachePolicy(
            k_bits=4,
            v_bits=4,
            retained_fraction=0.5,
            per_token_metadata_bytes=2,
            fixed_overhead_bytes=2,
            allocator_efficiency=0.5,
        ),
    )
    # ceil(3 * .5) = 2 tokens; 2 packed + 4 metadata + 2 fixed, / .5.
    assert result.candidate_bytes_per_sequence == 16


@pytest.mark.parametrize(
    "factory",
    (
        lambda: KVModelShape(0, 1, 1),
        lambda: KVModelShape(1.5, 1, 1),
        lambda: CachePolicy(k_bits=0),
        lambda: CachePolicy(k_bits="4"),
        lambda: CachePolicy(retained_fraction=0),
        lambda: CachePolicy(retained_fraction="0.5"),
        lambda: CachePolicy(allocator_efficiency=1.1),
        lambda: CapacityScenario(0, 1),
        lambda: CapacityScenario(1, 10, model_weights_bytes=11),
        lambda: CapacityScenario(1, 10.5),
    ),
)
def test_invalid_inputs_fail_closed(factory):
    with pytest.raises(ValueError):
        factory()


def test_result_rejects_a_measurement_label():
    with pytest.raises(ValueError, match="analytical-simulation"):
        CapacitySimulationResult(
            baseline_bytes_per_sequence=1,
            candidate_bytes_per_sequence=1,
            available_kv_bytes=1,
            baseline_max_concurrency=1,
            candidate_max_concurrency=1,
            storage_ratio=1,
            assumptions=("explicit",),
            provenance=ResultProvenance.MEASURED_LOCAL,
        )


def test_cli_simulation_is_labelled_and_contains_no_performance_claim(capsys):
    assert main([
        "simulate-capacity",
        "--layers", "2",
        "--kv-heads", "4",
        "--head-dim", "8",
        "--context-tokens", "100",
        "--available-memory-gib", "1",
        "--k-bits", "4",
        "--v-bits", "4",
        "--retained-fraction", "0.5",
        "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provenance"] == "analytical-simulation"
    assert not ({"latency", "throughput", "speedup"} & payload.keys())


def test_cli_rejects_impossible_memory_budget(capsys):
    assert main([
        "simulate-capacity",
        "--layers", "2",
        "--kv-heads", "4",
        "--head-dim", "8",
        "--context-tokens", "100",
        "--available-memory-gib", "1",
        "--model-weights-gib", "2",
    ]) == 2
    assert "exceed available memory" in capsys.readouterr().err
