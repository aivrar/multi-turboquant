# SPDX-License-Identifier: MIT
"""Guarded executable profiles built from the compatibility matrix."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

from .catalog import BUILTIN_DESCRIPTORS
from .composition import CompositionDisposition, build_composition_matrix
from .core import QualityRisk


@dataclass(frozen=True)
class ProfileHost:
    os: str
    compute: str
    architecture: str = "x86_64"
    gpu_memory_gb: float = 0.0

    def __post_init__(self) -> None:
        for name in ("os", "compute", "architecture"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip().lower())
        if (
            isinstance(self.gpu_memory_gb, bool)
            or not isinstance(self.gpu_memory_gb, (int, float))
            or not math.isfinite(self.gpu_memory_gb)
            or self.gpu_memory_gb < 0
        ):
            raise ValueError("gpu_memory_gb must be finite and non-negative")


@dataclass(frozen=True)
class ExecutableProfile:
    id: str
    name: str
    engine: str
    optimizations: tuple[str, ...]
    summary: str
    quality_risk: QualityRisk
    supported_os: tuple[str, ...] = ("linux",)
    supported_compute: tuple[str, ...] = ("cuda",)
    artifact_keys: tuple[str, ...] = ()
    version_constraints: tuple[str, ...] = ()
    forbidden_features: tuple[str, ...] = ()
    supported_tasks: tuple[str, ...] = (
        "chat", "completion", "code", "reasoning", "rag", "long_context"
    )
    required_traits: tuple[str, ...] = ()
    activation_traits: tuple[str, ...] = ()
    minimum_prompt_tokens: int = 0
    priority: int = 0
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or self.id.lower() != self.id:
            raise ValueError("Profile IDs must be non-empty lowercase strings")
        if not self.optimizations or len(set(self.optimizations)) != len(self.optimizations):
            raise ValueError("Profiles require unique optimization IDs")
        if self.minimum_prompt_tokens < 0:
            raise ValueError("minimum_prompt_tokens must be non-negative")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "engine": self.engine,
            "optimizations": list(self.optimizations),
            "summary": self.summary,
            "quality_risk": self.quality_risk.value,
            "supported_os": list(self.supported_os),
            "supported_compute": list(self.supported_compute),
            "artifact_keys": list(self.artifact_keys),
            "version_constraints": list(self.version_constraints),
            "forbidden_features": list(self.forbidden_features),
            "supported_tasks": list(self.supported_tasks),
            "required_traits": list(self.required_traits),
            "activation_traits": list(self.activation_traits),
            "minimum_prompt_tokens": self.minimum_prompt_tokens,
            "priority": self.priority,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class ProfileIssue:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class ExecutionProfilePlan:
    profile: ExecutableProfile
    host: ProfileHost
    available_artifacts: frozenset[str] = field(default_factory=frozenset)
    active_features: frozenset[str] = field(default_factory=frozenset)
    issues: tuple[ProfileIssue, ...] = ()

    @property
    def ready(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict:
        return {
            "profile": self.profile.to_dict(),
            "host": {
                "os": self.host.os,
                "compute": self.host.compute,
                "architecture": self.host.architecture,
                "gpu_memory_gb": self.host.gpu_memory_gb,
            },
            "available_artifacts": sorted(self.available_artifacts),
            "active_features": sorted(self.active_features),
            "ready": self.ready,
            "issues": [issue.to_dict() for issue in self.issues],
        }


BUILTIN_EXECUTABLE_PROFILES = (
    ExecutableProfile(
        id="fastdms_flash",
        name="FastDMS with FlashAttention",
        engine="fastdms",
        optimizations=("fastdms", "flashattention"),
        summary="Compact DMS serving for an explicitly supported DMS-trained checkpoint.",
        quality_risk=QualityRisk.CONDITIONAL,
        artifact_keys=("dms_checkpoint",),
        version_constraints=("One isolated FastDMS/Torch/Triton/FlashAttention environment",),
        required_traits=("dms_trained",),
        activation_traits=("dms_trained",),
        priority=90,
    ),
    ExecutableProfile(
        id="vllm_lmcache",
        name="vLLM with LMCache",
        engine="vllm",
        optimizations=("lmcache",),
        summary="Standard-layout reusable KV storage for repeated-prefix and RAG workloads.",
        quality_risk=QualityRisk.EXACT,
        version_constraints=("A connector version matching the exact vLLM release",),
        forbidden_features=("custom_kv_layout",),
        activation_traits=("repeated_prefix",),
        priority=80,
    ),
    ExecutableProfile(
        id="vllm_minference",
        name="vLLM with MInference",
        engine="vllm",
        optimizations=("minference",),
        summary="Sparse prefill for a model with a reviewed attention-pattern configuration.",
        quality_risk=QualityRisk.CONDITIONAL,
        artifact_keys=("minference_pattern",),
        version_constraints=("One exact supported vLLM/Torch/Triton stack",),
        forbidden_features=("speculative_prefill",),
        supported_tasks=("completion", "reasoning", "rag", "long_context"),
        activation_traits=("long_prefill",),
        minimum_prompt_tokens=32768,
        priority=70,
    ),
    ExecutableProfile(
        id="vllm_triattention",
        name="vLLM with TriAttention",
        engine="vllm",
        optimizations=("triattention", "flashattention"),
        summary="Calibrated KV selection for long reasoning under an explicit cache budget.",
        quality_risk=QualityRisk.CONDITIONAL,
        artifact_keys=("triattention_stats",),
        version_constraints=("An upstream-supported vLLM/TriAttention/FlashAttention stack",),
        forbidden_features=(
            "enable_prefix_caching",
            "prefix_cache",
            "prefix_caching",
            "radix_cache",
            "speculative_decode",
            "speculative_decoding",
        ),
        supported_tasks=("completion", "code", "reasoning", "long_context"),
        activation_traits=("long_reasoning",),
        priority=85,
    ),
    ExecutableProfile(
        id="vllm_proxima",
        name="vLLM with Proxima STAR-KV",
        engine="vllm",
        optimizations=("proxima",),
        summary="Calibrated low-rank paged KV serving for memory-bound concurrency.",
        quality_risk=QualityRisk.LOSSY,
        artifact_keys=("star_kv_checkpoint",),
        version_constraints=("vLLM==0.10.1.1", "transformers>=4.55,<5"),
        forbidden_features=("external_kv_serializer", "alternate_attention_kernel"),
        required_traits=("star_kv_calibrated",),
        activation_traits=("memory_bound", "high_concurrency"),
        priority=75,
    ),
    ExecutableProfile(
        id="jetspec_flash",
        name="JetSpec tree decoding",
        engine="jetspec",
        optimizations=("jetspec", "flashattention"),
        summary="Lossless tree-speculative decoding with a model-matched draft head.",
        quality_risk=QualityRisk.EXACT,
        artifact_keys=("jetspec_draft_head",),
        version_constraints=("A JetSpec revision matching the selected draft head",),
        supported_tasks=("chat", "completion", "code", "reasoning"),
        required_traits=("jetspec_target",),
        activation_traits=("speculative_target",),
        priority=95,
        limitations=("Published headline throughput uses B200 GPUs.",),
    ),
    ExecutableProfile(
        id="jetlong_flash",
        name="Jet-Long with FlashAttention",
        engine="transformers",
        optimizations=("jetlong", "flashattention"),
        summary="Training-free Qwen3 context extension using the reviewed non-fused path.",
        quality_risk=QualityRisk.CONDITIONAL,
        artifact_keys=("qwen3_checkpoint",),
        version_constraints=("Torch 2.9.1", "Transformers 5.3.0"),
        supported_tasks=("completion", "reasoning", "rag", "long_context"),
        required_traits=("qwen3",),
        activation_traits=("extended_context",),
        priority=88,
        limitations=("The fused FlashAttention-4 path requires SM90/H100 and CUDA 13.",),
    ),
    ExecutableProfile(
        id="lucebox_guarded",
        name="Guarded LuceBox runtime",
        engine="lucebox",
        optimizations=("lucebox",),
        summary="Separate LuceBox service with only source-reviewed, model-supported paths enabled.",
        quality_risk=QualityRisk.CONDITIONAL,
        supported_compute=("cuda", "rocm"),
        artifact_keys=("lucebox_source", "gguf_model"),
        version_constraints=("LuceBox commit ac22a3ed2b555e7211d730e7372807ae07e6df3b",),
        forbidden_features=("unvalidated_lucebox_combination",),
        required_traits=("lucebox_supported",),
        activation_traits=("lucebox_supported",),
        priority=65,
        limitations=(
            "LuceBox remains a separate runtime and is not overlaid onto Godzilla.",
            "Model-specific kernels and headline results do not transfer to unsupported models or GPUs.",
        ),
    ),
    ExecutableProfile(
        id="lucebox_qwen36_composed",
        name="LuceBox Qwen 3.6 composed path",
        engine="lucebox",
        optimizations=("lucebox",),
        summary="Guarded DFlash/DDTree, PFlash, and KVFlash route for the documented Qwen 3.6 27B path.",
        quality_risk=QualityRisk.LOSSY,
        supported_compute=("cuda",),
        artifact_keys=(
            "lucebox_source",
            "gguf_model",
            "lucebox_drafter",
            "lucebox_prefill_drafter",
        ),
        version_constraints=("LuceBox commit ac22a3ed2b555e7211d730e7372807ae07e6df3b",),
        forbidden_features=("exact_extraction", "unvalidated_lucebox_combination"),
        supported_tasks=("chat", "completion", "reasoning", "rag", "long_context"),
        required_traits=("qwen36_27b", "lucebox_supported_gpu"),
        activation_traits=("long_prefill", "high_concurrency"),
        minimum_prompt_tokens=32768,
        priority=92,
        limitations=(
            "PFlash may omit prompt facts; compare output quality against the direct path.",
            "Use LuceBox's own model cards and benchmark harness for the selected GPU.",
        ),
    ),
    ExecutableProfile(
        id="python_flashattention",
        name="Python FlashAttention",
        engine="python",
        optimizations=("flashattention",),
        summary="Exact fused attention for an explicitly compatible PyTorch model.",
        quality_risk=QualityRisk.EXACT,
        activation_traits=("prefer_flashattention",),
        priority=40,
    ),
    ExecutableProfile(
        id="python_sageattention",
        name="Python SageAttention",
        engine="python",
        optimizations=("sageattention",),
        summary="Quantized attention kernel selected instead of FlashAttention.",
        quality_risk=QualityRisk.CONDITIONAL,
        forbidden_features=("flashattention",),
        activation_traits=("prefer_sageattention",),
        priority=40,
    ),
    ExecutableProfile(
        id="godzilla_guarded",
        name="Guarded Godzilla PFlash/KVFlash",
        engine="godzilla",
        optimizations=("godzilla_composition",),
        summary="Exact-source Godzilla server with opt-in PFlash and bounded whole-slot residency.",
        quality_risk=QualityRisk.CONDITIONAL,
        artifact_keys=("godzilla_09214b160_source", "gguf_model"),
        version_constraints=("Godzilla commit 09214b160b402011359f0ef9d5fa8f8be1112e85",),
        required_traits=("gguf",),
        activation_traits=("gguf",),
        priority=60,
    ),
    ExecutableProfile(
        id="godzilla_triattention",
        name="Godzilla TriAttention",
        engine="godzilla",
        optimizations=("triattention",),
        summary="Patched Godzilla KV selection with a model-matched binary statistics artifact.",
        quality_risk=QualityRisk.CONDITIONAL,
        artifact_keys=("triattention_stats", "gguf_model"),
        forbidden_features=("kvarn",),
        supported_tasks=("completion", "code", "reasoning", "long_context"),
        required_traits=("gguf", "long_reasoning"),
        activation_traits=("long_reasoning",),
        priority=70,
    ),
)


def get_executable_profile(profile_id: str) -> ExecutableProfile:
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise KeyError("Execution profile ID must be a non-empty string")
    normalized = profile_id.strip().lower()
    for profile in BUILTIN_EXECUTABLE_PROFILES:
        if profile.id == normalized:
            return profile
    available = ", ".join(profile.id for profile in BUILTIN_EXECUTABLE_PROFILES)
    raise KeyError(f"Unknown execution profile {profile_id!r}. Available: {available}")


def _contract_issues(profile: ExecutableProfile) -> tuple[ProfileIssue, ...]:
    descriptor_ids = {descriptor.id for descriptor in BUILTIN_DESCRIPTORS}
    unknown = sorted(set(profile.optimizations) - descriptor_ids)
    issues = [
        ProfileIssue("error", "unknown_optimization", f"Unknown optimization {item!r}.")
        for item in unknown
    ]
    if unknown or len(profile.optimizations) < 2:
        return tuple(issues)

    matrix = build_composition_matrix(BUILTIN_DESCRIPTORS)
    for left, right in combinations(profile.optimizations, 2):
        rule = matrix.rule(left, right)
        if rule.disposition != CompositionDisposition.DIRECT:
            issues.append(ProfileIssue(
                "error",
                "profile_not_direct",
                f"Profile {profile.id!r} cannot activate {left!r} with {right!r}: {rule.reason}",
            ))
    return tuple(issues)


def plan_execution_profile(
    profile_id: str,
    host: ProfileHost,
    *,
    available_artifacts: frozenset[str] = frozenset(),
    active_features: frozenset[str] = frozenset(),
    exact_output_required: bool = False,
) -> ExecutionProfilePlan:
    """Validate one profile without installing or launching anything."""
    if not isinstance(exact_output_required, bool):
        raise ValueError("exact_output_required must be a boolean")
    for label, values in (
        ("available_artifacts", available_artifacts),
        ("active_features", active_features),
    ):
        if any(not isinstance(item, str) or not item.strip() for item in values):
            raise ValueError(f"{label} must contain non-empty strings")
    profile = get_executable_profile(profile_id)
    artifacts = frozenset(item.strip().lower() for item in available_artifacts)
    features = frozenset(item.strip().lower() for item in active_features)
    issues = list(_contract_issues(profile))

    if profile.supported_os and host.os not in profile.supported_os:
        issues.append(ProfileIssue(
            "error", "unsupported_os",
            f"Profile {profile.id!r} supports {profile.supported_os}, not {host.os!r}.",
        ))
    if profile.supported_compute and host.compute not in profile.supported_compute:
        issues.append(ProfileIssue(
            "error", "unsupported_compute",
            f"Profile {profile.id!r} supports {profile.supported_compute}, not {host.compute!r}.",
        ))

    for artifact in profile.artifact_keys:
        if artifact not in artifacts:
            issues.append(ProfileIssue(
                "error", "missing_artifact", f"Required artifact {artifact!r} was not provided."
            ))
    for feature in sorted(set(profile.forbidden_features) & set(features)):
        issues.append(ProfileIssue(
            "error", "forbidden_feature",
            f"Active feature {feature!r} is incompatible with profile {profile.id!r}.",
        ))
    if exact_output_required and profile.quality_risk != QualityRisk.EXACT:
        issues.append(ProfileIssue(
            "error", "exact_output_required",
            f"Profile {profile.id!r} has {profile.quality_risk.value} output risk.",
        ))
    elif profile.quality_risk != QualityRisk.EXACT:
        issues.append(ProfileIssue(
            "warning", "quality_validation_required",
            "This profile requires workload-specific output-quality validation.",
        ))

    return ExecutionProfilePlan(profile, host, artifacts, features, tuple(issues))
