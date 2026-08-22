from __future__ import annotations

from itertools import combinations

import pytest

from multi_turboquant.optimizations import (
    BUILTIN_DESCRIPTORS,
    CompositionDisposition,
    build_composition_matrix,
    create_builtin_registry,
)


def test_builtin_matrix_covers_every_unordered_pair_once():
    matrix = build_composition_matrix(BUILTIN_DESCRIPTORS)
    ids = tuple(sorted(descriptor.id for descriptor in BUILTIN_DESCRIPTORS))

    assert matrix.optimization_ids == ids
    assert len(matrix.rules) == len(ids) * (len(ids) - 1) // 2
    assert {rule.pair for rule in matrix.rules} == set(combinations(ids, 2))
    assert matrix.to_dict()["pair_count"] == len(set(combinations(ids, 2)))


def test_lookup_is_symmetric_and_serializable():
    matrix = build_composition_matrix(BUILTIN_DESCRIPTORS)
    forward = matrix.rule("fastdms", "flashattention")
    reverse = matrix.rule("flashattention", "fastdms")

    assert forward == reverse
    assert forward.to_dict()["disposition"] == "direct"
    assert forward.requirements
    assert forward.validation_gates


def test_reviewed_direct_compositions_are_explicit():
    matrix = build_composition_matrix(BUILTIN_DESCRIPTORS)
    direct_pairs = {
        ("fastdms", "flashattention"),
        ("flashattention", "jetspec"),
        ("flashattention", "jetlong"),
        ("flashattention", "triattention"),
        ("gigatoken", "triattention"),
        ("jetlong", "resonance_jetlong"),
        ("lmcache", "maru"),
    }

    for pair in direct_pairs:
        assert matrix.rule(*pair).disposition == CompositionDisposition.DIRECT


def test_known_runtime_and_cache_conflicts_fail_closed():
    matrix = build_composition_matrix(BUILTIN_DESCRIPTORS)
    prohibited_pairs = {
        ("flashattention", "sageattention"),
        ("fastdms", "triattention"),
        ("lmcache", "triattention"),
        ("maru", "triattention"),
        ("proxima", "triattention"),
    }

    for pair in prohibited_pairs:
        assert matrix.rule(*pair).disposition == CompositionDisposition.PROHIBITED


def test_duplicate_descriptor_ids_are_rejected_instead_of_silently_overwritten():
    descriptor = BUILTIN_DESCRIPTORS[0]
    with pytest.raises(ValueError, match="unique IDs"):
        build_composition_matrix((descriptor, descriptor))


def test_adapter_work_and_separate_runtime_paths_are_distinguished():
    matrix = build_composition_matrix(BUILTIN_DESCRIPTORS)

    assert matrix.rule("lmcache", "proxima").disposition == CompositionDisposition.CONDITIONAL
    assert matrix.rule("minference", "triattention").disposition == CompositionDisposition.CONDITIONAL
    assert matrix.rule("jetspec", "lmcache").disposition == CompositionDisposition.ROUTED
    assert matrix.rule("jetlong", "triattention").disposition == CompositionDisposition.ROUTED


def test_recent_research_compositions_fail_closed_by_domain_and_maturity():
    matrix = build_composition_matrix(BUILTIN_DESCRIPTORS)

    assert matrix.rule("restorekv", "triattention").disposition == (
        CompositionDisposition.PROHIBITED
    )
    assert matrix.rule("novakv", "proxima").disposition == CompositionDisposition.PROHIBITED
    assert matrix.rule("dspark", "jetspec").disposition == CompositionDisposition.PROHIBITED
    assert matrix.rule("archead", "restorekv").disposition == (
        CompositionDisposition.CONDITIONAL
    )


def test_triattention_catalog_tracks_current_runtime_boundaries():
    descriptor = create_builtin_registry().get("triattention").descriptor

    assert {"godzilla", "vllm", "sglang"} <= set(descriptor.supported_engines)
    assert descriptor.reviewed_source_commit == "a4bc3c8f709db60f016ef42c3feb290fd0c00c1b"
    assert any("prefix/radix caching" in item for item in descriptor.limitations)
