from __future__ import annotations

import json

import pytest

from multi_turboquant.optimizations.composition_cli import main
from multi_turboquant.optimizations.profiles import ProfileHost
from multi_turboquant.optimizations.routing import WorkloadRequest, route_workload


LINUX_CUDA = ProfileHost("linux", "cuda", gpu_memory_gb=24)


def test_repeated_prefix_routes_to_lmcache():
    decision = route_workload(
        WorkloadRequest("rag", 12000, repeated_prefix=True),
        LINUX_CUDA,
    )
    assert decision.selected_profile == "vllm_lmcache"
    assert decision.routed


def test_long_reasoning_routes_to_triattention_only_with_stats():
    missing = route_workload(
        WorkloadRequest("reasoning", 8192, expected_output_tokens=4096),
        LINUX_CUDA,
        candidate_profile_ids=("vllm_triattention",),
    )
    assert not missing.routed
    assert any(issue.code == "missing_artifact" for issue in missing.candidates[0].issues)

    ready = route_workload(
        WorkloadRequest(
            "reasoning",
            8192,
            expected_output_tokens=4096,
            artifacts=frozenset({"triattention_stats"}),
        ),
        LINUX_CUDA,
        candidate_profile_ids=("vllm_triattention",),
    )
    assert ready.selected_profile == "vllm_triattention"


def test_exact_output_rejects_lossy_or_conditional_profile():
    decision = route_workload(
        WorkloadRequest(
            "completion",
            4096,
            exact_output_required=True,
            artifacts=frozenset({"star_kv_checkpoint"}),
            model_traits=frozenset({"star_kv_calibrated", "memory_bound"}),
        ),
        LINUX_CUDA,
        candidate_profile_ids=("vllm_proxima",),
    )
    assert not decision.routed
    assert any(issue.code == "exact_output_required" for issue in decision.candidates[0].issues)


def test_router_requires_positive_activation_signal_and_falls_back():
    decision = route_workload(WorkloadRequest("chat", 512), LINUX_CUDA)
    assert decision.selected_profile is None
    assert "baseline" in decision.reason
    assert all(not candidate.eligible for candidate in decision.candidates)


def test_engine_preference_is_a_hard_guard():
    decision = route_workload(
        WorkloadRequest(
            "rag", 4096, repeated_prefix=True, preferred_engine="godzilla"
        ),
        LINUX_CUDA,
        candidate_profile_ids=("vllm_lmcache",),
    )
    assert not decision.routed
    assert any(issue.code == "engine_not_preferred" for issue in decision.candidates[0].issues)


def test_routing_is_deterministic_when_multiple_profiles_pass():
    request = WorkloadRequest(
        "reasoning",
        8192,
        expected_output_tokens=4096,
        artifacts=frozenset({"triattention_stats", "jetspec_draft_head"}),
        model_traits=frozenset({"jetspec_target", "speculative_target"}),
    )
    first = route_workload(
        request,
        LINUX_CUDA,
        candidate_profile_ids=("vllm_triattention", "jetspec_flash"),
    )
    second = route_workload(
        request,
        LINUX_CUDA,
        candidate_profile_ids=("jetspec_flash", "vllm_triattention"),
    )
    assert first.selected_profile == second.selected_profile == "jetspec_flash"


@pytest.mark.parametrize(
    ("profile_id", "workload"),
    (
        ("fastdms_flash", WorkloadRequest(
            "completion", 1024,
            artifacts=frozenset({"dms_checkpoint"}),
            model_traits=frozenset({"dms_trained"}),
        )),
        ("vllm_lmcache", WorkloadRequest("rag", 4096, repeated_prefix=True)),
        ("vllm_minference", WorkloadRequest(
            "long_context", 32768,
            artifacts=frozenset({"minference_pattern"}),
        )),
        ("vllm_triattention", WorkloadRequest(
            "reasoning", 8192, expected_output_tokens=1024,
            artifacts=frozenset({"triattention_stats"}),
        )),
        ("vllm_proxima", WorkloadRequest(
            "chat", 2048,
            artifacts=frozenset({"star_kv_checkpoint"}),
            model_traits=frozenset({"star_kv_calibrated", "memory_bound"}),
        )),
        ("jetspec_flash", WorkloadRequest(
            "code", 2048,
            artifacts=frozenset({"jetspec_draft_head"}),
            model_traits=frozenset({"jetspec_target", "speculative_target"}),
        )),
        ("jetlong_flash", WorkloadRequest(
            "long_context", 16384,
            artifacts=frozenset({"qwen3_checkpoint"}),
            model_traits=frozenset({"qwen3", "extended_context"}),
        )),
        ("lucebox_guarded", WorkloadRequest(
            "completion", 1024,
            artifacts=frozenset({"lucebox_source", "gguf_model"}),
            model_traits=frozenset({"lucebox_supported"}),
        )),
        ("lucebox_qwen36_composed", WorkloadRequest(
            "long_context", 32768,
            artifacts=frozenset({
                "lucebox_source", "gguf_model", "lucebox_drafter",
                "lucebox_prefill_drafter",
            }),
            model_traits=frozenset({"qwen36_27b", "lucebox_supported_gpu"}),
        )),
        ("python_flashattention", WorkloadRequest(
            "completion", 1024,
            model_traits=frozenset({"prefer_flashattention"}),
        )),
        ("python_sageattention", WorkloadRequest(
            "completion", 1024,
            model_traits=frozenset({"prefer_sageattention"}),
        )),
        ("godzilla_guarded", WorkloadRequest(
            "completion", 2048,
            artifacts=frozenset({"godzilla_09214b160_source", "gguf_model"}),
            model_traits=frozenset({"gguf"}),
        )),
        ("godzilla_triattention", WorkloadRequest(
            "reasoning", 8192, expected_output_tokens=1024,
            artifacts=frozenset({"triattention_stats", "gguf_model"}),
            model_traits=frozenset({"gguf"}),
        )),
    ),
)
def test_every_builtin_profile_is_reachable_through_explicit_traits(profile_id, workload):
    decision = route_workload(
        workload,
        LINUX_CUDA,
        candidate_profile_ids=(profile_id,),
    )
    assert decision.selected_profile == profile_id


def test_request_validation_and_duplicate_candidates_fail_closed():
    with pytest.raises(ValueError, match="Unknown task"):
        WorkloadRequest("unknown", 1)
    with pytest.raises(ValueError, match="non-negative"):
        WorkloadRequest("chat", -1)
    with pytest.raises(ValueError, match="non-negative"):
        WorkloadRequest("chat", True)
    with pytest.raises(ValueError, match="non-empty"):
        WorkloadRequest("chat", 1, preferred_engine="")
    with pytest.raises(ValueError, match="non-empty strings"):
        WorkloadRequest("chat", 1, artifacts=frozenset({""}))
    with pytest.raises(ValueError, match="repeated_prefix"):
        WorkloadRequest("chat", 1, repeated_prefix=1)
    with pytest.raises(ValueError, match="exact_output_required"):
        WorkloadRequest("chat", 1, exact_output_required="yes")
    with pytest.raises(ValueError, match="unique"):
        route_workload(
            WorkloadRequest("chat", 1),
            LINUX_CUDA,
            candidate_profile_ids=("vllm_lmcache", "vllm_lmcache"),
        )


def test_cli_plan_and_route_are_read_only(capsys):
    assert main([
        "plan", "vllm_lmcache", "--os", "linux", "--compute", "cuda", "--json"
    ]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["ready"]
    assert "commands" not in plan

    assert main([
        "route", "--task", "rag", "--prompt-tokens", "4096",
        "--repeated-prefix", "--os", "linux", "--compute", "cuda", "--json",
    ]) == 0
    routed = json.loads(capsys.readouterr().out)
    assert routed["selected_profile"] == "vllm_lmcache"

    assert main([
        "route", "--task", "chat", "--prompt-tokens", "-1",
        "--os", "linux", "--compute", "cuda",
    ]) == 2
    assert "non-negative" in capsys.readouterr().err
