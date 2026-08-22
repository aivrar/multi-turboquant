from __future__ import annotations

import pytest

from multi_turboquant.optimizations import (
    ManifestPlugin,
    OptimizationContext,
    OptimizationDescriptor,
    OptimizationKind,
    OptimizationMaturity,
    OptimizationRegistry,
    QualityRisk,
    IntegrationMode,
    create_builtin_registry,
    plan_optimizations,
)


def context(**overrides) -> OptimizationContext:
    values = {
        "engine": "vllm",
        "os": "linux",
        "compute": "cuda",
        "kv_format": "fp16",
        "python_version": (3, 12),
        "capabilities": frozenset(),
        "active_features": frozenset(),
        "installed_modules": frozenset({"lmcache", "minference", "speculative_prefill"}),
        "installed_executables": frozenset({"lmcache"}),
    }
    values.update(overrides)
    return OptimizationContext(**values)


def test_builtin_catalog_is_explicit_and_disabled_by_default():
    plugins = create_builtin_registry().list()
    ids = [plugin.descriptor.id for plugin in plugins]
    assert len(ids) == len(set(ids))
    assert "lmcache" in ids
    assert "resonance_yarn" in ids
    assert {"triattention", "gigatoken", "cuda_weight_share"} <= set(ids)
    assert {
        "jetspec",
        "lucebox",
        "proxima",
        "jetlong",
        "chunkllama",
        "rabitqcache",
        "scope_pe",
        "duoattention",
        "icecache",
        "pflash_llamacpp",
        "godzilla_composition",
        "resonance_jetlong",
    } <= set(ids)
    assert all(not plugin.descriptor.default_enabled for plugin in plugins)


def test_issue_43_catalog_records_quality_and_validation_boundaries():
    registry = create_builtin_registry()

    jetspec = registry.get("jetspec").descriptor
    assert jetspec.quality_risk == QualityRisk.EXACT
    assert jetspec.required_artifacts
    assert jetspec.validation_gates
    assert jetspec.reviewed_source_commit

    rabitq = registry.get("rabitqcache").descriptor
    assert rabitq.maturity == OptimizationMaturity.BLOCKED
    assert "No repository software license" == rabitq.license

    icecache = registry.get("icecache").descriptor
    assert {"openblas", "high_cpu_parallelism"} <= set(icecache.required_capabilities)


def test_unreviewed_pipeline_composition_fails_closed():
    plan = plan_optimizations(
        ["proxima", "sageattention"],
        context(
            installed_modules=frozenset({"proxima_vllm", "sageattention"}),
        ),
    )
    assert not plan.ready
    issue = next(item for item in plan.issues if item.code == "unvalidated_composition")
    assert "attention_kernel" in issue.message


def test_conditional_and_routed_compositions_fail_closed_with_actionable_rules():
    conditional = plan_optimizations(
        ["lmcache", "proxima"],
        context(installed_modules=frozenset({"lmcache", "proxima_vllm"})),
    )
    issue = next(item for item in conditional.issues if item.code == "conditional_composition")
    assert "serializer" in issue.message
    assert conditional.to_dict()["composition_rules"][0]["disposition"] == "conditional"

    routed = plan_optimizations(
        ["lmcache", "jetspec"],
        context(
            engine="jetspec",
            installed_modules=frozenset({"lmcache", "jetspec"}),
        ),
    )
    assert any(item.code == "separate_runtime_required" for item in routed.issues)


def test_mutually_allowlisted_resonance_jetlong_composition_is_not_rejected_as_overlap():
    plan = plan_optimizations(
        ["jetlong", "resonance_jetlong"],
        context(
            engine="transformers",
            installed_modules=frozenset({"jetlm"}),
        ),
    )
    assert not any(item.code == "unvalidated_composition" for item in plan.issues)
    assert any(item.code == "native_backend_required" for item in plan.issues)
    data = plan.to_dict()
    assert data["quality_risks"] == {
        "jetlong": "conditional",
        "resonance_jetlong": "research",
    }
    assert any("PG-19" in gate for gate in data["validation_gates"])
    assert any("Qwen3" in artifact for artifact in data["required_artifacts"])


def test_lmcache_is_ready_only_for_validated_vllm_formats():
    plan = plan_optimizations(["lmcache"], context())
    assert plan.ready

    custom = plan_optimizations(["lmcache"], context(kv_format="turbo3"))
    assert not custom.ready
    assert any(issue.code == "unsupported_kv_format" for issue in custom.issues)


def test_lmcache_does_not_claim_llamacpp_support():
    plan = plan_optimizations(["lmcache"], context(engine="llamacpp"))
    assert not plan.ready
    assert any(issue.code == "unsupported_engine" for issue in plan.issues)


def test_flashattention_is_optional_linux_cuda_backend():
    ready = plan_optimizations(
        ["flashattention"],
        context(
            engine="python",
            installed_modules=frozenset({"flash_attn"}),
        ),
    )
    assert ready.ready

    unsupported = plan_optimizations(
        ["flashattention"],
        context(
            engine="python",
            os="windows",
            installed_modules=frozenset({"flash_attn"}),
        ),
    )
    assert not unsupported.ready
    assert any(issue.code == "unsupported_os" for issue in unsupported.issues)


def test_fastdms_declares_flashattention_dependency():
    missing = plan_optimizations(
        ["fastdms"],
        context(
            engine="fastdms",
            installed_modules=frozenset({"fastdms", "flash_attn"}),
            installed_executables=frozenset(),
        ),
    )
    assert any(issue.code == "missing_optimization_dependency" for issue in missing.issues)

    ready = plan_optimizations(
        ["flashattention", "fastdms"],
        context(
            engine="fastdms",
            installed_modules=frozenset({"fastdms", "flash_attn"}),
            installed_executables=frozenset(),
        ),
    )
    assert ready.ready


def test_maru_requires_lmcache_cxl_and_python_312():
    base = context(
        installed_modules=frozenset({"lmcache", "maru_lmcache"}),
        installed_executables=frozenset({"lmcache", "maru-server"}),
        python_version=(3, 11),
    )
    plan = plan_optimizations(["maru"], base)
    codes = {issue.code for issue in plan.issues}
    assert {"missing_capability", "unsupported_python", "missing_optimization_dependency"} <= codes

    ready = plan_optimizations(
        ["lmcache", "maru"],
        context(
            capabilities=frozenset({"cxl_devdax"}),
            installed_modules=frozenset({"lmcache", "maru_lmcache"}),
            installed_executables=frozenset({"lmcache", "maru-server"}),
        ),
    )
    assert ready.ready


def test_known_prefill_conflict_is_reported_once():
    plan = plan_optimizations(["minference", "speculative_prefill"], context())
    conflicts = [issue for issue in plan.issues if issue.code == "optimization_conflict"]
    assert len(conflicts) == 1
    assert not plan.ready


def test_active_existing_feature_can_conflict_with_plugin():
    plan = plan_optimizations(
        ["fastdms"],
        context(
            engine="fastdms",
            active_features=frozenset({"triattention"}),
            installed_modules=frozenset({"fastdms"}),
            installed_executables=frozenset(),
        ),
    )
    assert any(issue.code == "optimization_conflict" for issue in plan.issues)

    reverse_sort = plan_optimizations(
        ["rocketkv"],
        context(engine="gpt-fast", active_features=frozenset({"triattention"})),
    )
    assert any(issue.code == "optimization_conflict" for issue in reverse_sort.issues)


def test_godzilla_gigatoken_requires_triattention_and_exact_linux_cuda_stack():
    missing = plan_optimizations(
        ["gigatoken"],
        context(
            engine="godzilla",
            installed_modules=frozenset({"gigatoken"}),
            installed_executables=frozenset(),
        ),
    )
    assert any(issue.code == "missing_optimization_dependency" for issue in missing.issues)

    ready = plan_optimizations(
        ["triattention", "gigatoken"],
        context(
            engine="godzilla",
            installed_modules=frozenset({"triattention", "gigatoken"}),
            installed_executables=frozenset(),
        ),
    )
    assert ready.ready


def test_godzilla_triattention_rejects_active_kvarn():
    plan = plan_optimizations(
        ["triattention"],
        context(
            engine="godzilla",
            active_features=frozenset({"kvarn"}),
            installed_modules=frozenset({"triattention"}),
            installed_executables=frozenset(),
        ),
    )
    assert any(issue.code == "optimization_conflict" for issue in plan.issues)


def test_cuda_weight_share_is_limited_to_linux_cuda_x86_64():
    ready = plan_optimizations(
        ["cuda_weight_share"],
        context(
            engine="godzilla",
            architecture="x86_64",
            installed_modules=frozenset(),
            installed_executables=frozenset({"gcc"}),
        ),
    )
    assert ready.ready

    arm = plan_optimizations(
        ["cuda_weight_share"],
        context(
            engine="godzilla",
            architecture="aarch64",
            installed_modules=frozenset(),
            installed_executables=frozenset({"gcc"}),
        ),
    )
    assert any(issue.code == "unsupported_architecture" for issue in arm.issues)


def test_context_values_are_normalized():
    normalized = context(engine=" VLLM ", compute=" CUDA ", kv_format=" FP16 ")
    assert normalized.engine == "vllm"
    assert normalized.compute == "cuda"
    assert normalized.kv_format == "fp16"


def test_research_and_unlicensed_projects_are_not_eligible():
    rocket = plan_optimizations(["rocketkv"], context(engine="gpt-fast"))
    adadecode = plan_optimizations(["adadecode"], context(engine="transformers"))
    assert not rocket.ready
    assert any(issue.code == "research_only" for issue in rocket.issues)
    assert not adadecode.ready
    assert any(issue.code == "blocked" for issue in adadecode.issues)


def test_unknown_and_duplicate_registration_are_rejected():
    unknown = plan_optimizations(["does_not_exist"], context())
    assert any(issue.code == "unknown_optimization" for issue in unknown.issues)

    descriptor = OptimizationDescriptor(
        id="example",
        name="Example",
        source_url="https://example.com",
        kind=OptimizationKind.PREFILL,
        maturity=OptimizationMaturity.EXPERIMENTAL,
        integration_mode=IntegrationMode.OPTIONAL_PYTHON,
        license="MIT",
        summary="test",
        supported_engines=("vllm",),
    )
    registry = OptimizationRegistry()
    registry.register(ManifestPlugin(descriptor))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ManifestPlugin(descriptor))
