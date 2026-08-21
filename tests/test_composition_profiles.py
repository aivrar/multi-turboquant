from __future__ import annotations

import pytest

from multi_turboquant.optimizations import QualityRisk
from multi_turboquant.optimizations.profiles import (
    BUILTIN_EXECUTABLE_PROFILES,
    ProfileHost,
    get_executable_profile,
    plan_execution_profile,
)


LINUX_CUDA = ProfileHost("linux", "cuda", gpu_memory_gb=24)


def test_profile_ids_and_optimization_sets_are_unique():
    ids = [profile.id for profile in BUILTIN_EXECUTABLE_PROFILES]
    assert len(ids) == len(set(ids))
    assert all(len(profile.optimizations) == len(set(profile.optimizations)) for profile in BUILTIN_EXECUTABLE_PROFILES)


def test_every_builtin_profile_has_a_valid_direct_contract():
    for profile in BUILTIN_EXECUTABLE_PROFILES:
        plan = plan_execution_profile(
            profile.id,
            LINUX_CUDA,
            available_artifacts=frozenset(profile.artifact_keys),
        )
        assert not any(issue.code in {"unknown_optimization", "profile_not_direct"} for issue in plan.issues)


def test_profile_plan_fails_closed_for_artifacts_host_and_features():
    missing = plan_execution_profile("vllm_triattention", LINUX_CUDA)
    assert any(issue.code == "missing_artifact" for issue in missing.issues)

    windows = plan_execution_profile(
        "vllm_triattention",
        ProfileHost("windows", "cuda"),
        available_artifacts=frozenset({"triattention_stats"}),
    )
    assert any(issue.code == "unsupported_os" for issue in windows.issues)

    prefix = plan_execution_profile(
        "vllm_triattention",
        LINUX_CUDA,
        available_artifacts=frozenset({"triattention_stats"}),
        active_features=frozenset({"prefix_cache"}),
    )
    assert any(issue.code == "forbidden_feature" for issue in prefix.issues)

    upstream_spelling = plan_execution_profile(
        "vllm_triattention",
        LINUX_CUDA,
        available_artifacts=frozenset({"triattention_stats"}),
        active_features=frozenset({"enable_prefix_caching"}),
    )
    assert any(issue.code == "forbidden_feature" for issue in upstream_spelling.issues)


def test_exact_output_gate_allows_only_exact_profiles():
    lmcache = plan_execution_profile(
        "vllm_lmcache", LINUX_CUDA, exact_output_required=True
    )
    assert lmcache.ready
    assert lmcache.profile.quality_risk == QualityRisk.EXACT

    proxima = plan_execution_profile(
        "vllm_proxima",
        LINUX_CUDA,
        available_artifacts=frozenset({"star_kv_checkpoint"}),
        exact_output_required=True,
    )
    assert not proxima.ready
    assert any(issue.code == "exact_output_required" for issue in proxima.issues)


def test_plan_serialization_contains_constraints_and_no_actions():
    plan = plan_execution_profile(
        "jetspec_flash",
        LINUX_CUDA,
        available_artifacts=frozenset({"jetspec_draft_head"}),
    )
    data = plan.to_dict()
    assert data["ready"]
    assert data["profile"]["version_constraints"]
    assert data["profile"]["artifact_keys"] == ["jetspec_draft_head"]
    assert "commands" not in data


def test_unknown_profile_is_rejected():
    with pytest.raises(KeyError, match="Unknown execution profile"):
        get_executable_profile("unknown")


def test_profile_plan_rejects_non_boolean_and_empty_inputs():
    with pytest.raises(ValueError, match="boolean"):
        plan_execution_profile(
            "python_flashattention", LINUX_CUDA, exact_output_required=1
        )
    with pytest.raises(ValueError, match="available_artifacts"):
        plan_execution_profile(
            "python_flashattention",
            LINUX_CUDA,
            available_artifacts=frozenset({""}),
        )


def test_profile_host_rejects_empty_or_non_finite_dimensions():
    with pytest.raises(ValueError, match="non-empty"):
        ProfileHost("", "cuda")
    with pytest.raises(ValueError, match="finite"):
        ProfileHost("linux", "cuda", gpu_memory_gb=float("nan"))
