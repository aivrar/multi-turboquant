# SPDX-License-Identifier: MIT
"""Deterministic, side-effect-free workload routing for composition profiles."""

from __future__ import annotations

from dataclasses import dataclass, field

from .profiles import (
    BUILTIN_EXECUTABLE_PROFILES,
    ExecutionProfilePlan,
    ProfileHost,
    ProfileIssue,
    get_executable_profile,
    plan_execution_profile,
)


SUPPORTED_TASKS = ("chat", "completion", "code", "long_context", "rag", "reasoning")
_TASKS = frozenset(SUPPORTED_TASKS)


@dataclass(frozen=True)
class WorkloadRequest:
    task: str
    prompt_tokens: int
    expected_output_tokens: int = 0
    repeated_prefix: bool = False
    exact_output_required: bool = False
    preferred_engine: str | None = None
    artifacts: frozenset[str] = field(default_factory=frozenset)
    model_traits: frozenset[str] = field(default_factory=frozenset)
    active_features: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.task, str):
            raise ValueError("task must be a string")
        task = self.task.strip().lower()
        if task not in _TASKS:
            raise ValueError(f"Unknown task {self.task!r}; choose one of {tuple(sorted(_TASKS))}")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.prompt_tokens, self.expected_output_tokens)
        ):
            raise ValueError("Token counts must be non-negative")
        for name in ("repeated_prefix", "exact_output_required"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        object.__setattr__(self, "task", task)
        if self.preferred_engine is not None:
            if not isinstance(self.preferred_engine, str) or not self.preferred_engine.strip():
                raise ValueError("preferred_engine must be a non-empty string")
            object.__setattr__(self, "preferred_engine", self.preferred_engine.strip().lower())
        for name in ("artifacts", "model_traits", "active_features"):
            values = getattr(self, name)
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise ValueError(f"{name} must contain non-empty strings")
            object.__setattr__(
                self,
                name,
                frozenset(item.strip().lower() for item in values),
            )

    @property
    def effective_traits(self) -> frozenset[str]:
        traits = set(self.model_traits)
        if self.repeated_prefix:
            traits.add("repeated_prefix")
        if self.prompt_tokens >= 32768:
            traits.add("long_prefill")
        if (
            self.task == "reasoning"
            and self.prompt_tokens + self.expected_output_tokens >= 8192
        ):
            traits.add("long_reasoning")
        return frozenset(traits)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "prompt_tokens": self.prompt_tokens,
            "expected_output_tokens": self.expected_output_tokens,
            "repeated_prefix": self.repeated_prefix,
            "exact_output_required": self.exact_output_required,
            "preferred_engine": self.preferred_engine,
            "artifacts": sorted(self.artifacts),
            "model_traits": sorted(self.model_traits),
            "active_features": sorted(self.active_features),
            "effective_traits": sorted(self.effective_traits),
        }


@dataclass(frozen=True)
class RouteCandidate:
    profile_id: str
    eligible: bool
    score: int
    matched_traits: tuple[str, ...]
    issues: tuple[ProfileIssue, ...]

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "eligible": self.eligible,
            "score": self.score,
            "matched_traits": list(self.matched_traits),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class RouteDecision:
    request: WorkloadRequest
    host: ProfileHost
    selected_profile: str | None
    selected_plan: ExecutionProfilePlan | None
    candidates: tuple[RouteCandidate, ...]
    reason: str

    @property
    def routed(self) -> bool:
        return self.selected_profile is not None

    def to_dict(self) -> dict:
        return {
            "routed": self.routed,
            "selected_profile": self.selected_profile,
            "reason": self.reason,
            "request": self.request.to_dict(),
            "host": {
                "os": self.host.os,
                "compute": self.host.compute,
                "architecture": self.host.architecture,
                "gpu_memory_gb": self.host.gpu_memory_gb,
            },
            "selected_plan": self.selected_plan.to_dict() if self.selected_plan else None,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def _candidate(
    profile_id: str,
    request: WorkloadRequest,
    host: ProfileHost,
) -> tuple[RouteCandidate, ExecutionProfilePlan]:
    profile = get_executable_profile(profile_id)
    plan = plan_execution_profile(
        profile_id,
        host,
        available_artifacts=request.artifacts,
        active_features=request.active_features,
        exact_output_required=request.exact_output_required,
    )
    issues = list(plan.issues)
    traits = request.effective_traits
    matched = tuple(sorted(set(profile.activation_traits) & set(traits)))

    if request.task not in profile.supported_tasks:
        issues.append(ProfileIssue(
            "error", "unsupported_task",
            f"Task {request.task!r} is unsupported by profile {profile.id!r}.",
        ))
    missing_traits = sorted(set(profile.required_traits) - set(traits))
    for trait in missing_traits:
        issues.append(ProfileIssue(
            "error", "missing_model_trait", f"Required model trait {trait!r} is absent."
        ))
    if request.prompt_tokens < profile.minimum_prompt_tokens:
        issues.append(ProfileIssue(
            "error", "prompt_too_short",
            f"Profile {profile.id!r} requires at least {profile.minimum_prompt_tokens} prompt tokens.",
        ))
    if profile.activation_traits and not matched:
        issues.append(ProfileIssue(
            "error", "no_activation_signal",
            f"No activation trait matched profile {profile.id!r}.",
        ))
    if request.preferred_engine and profile.engine != request.preferred_engine:
        issues.append(ProfileIssue(
            "error", "engine_not_preferred",
            f"Profile engine {profile.engine!r} does not match {request.preferred_engine!r}.",
        ))

    eligible = not any(issue.severity == "error" for issue in issues)
    score = profile.priority + 100 * len(matched)
    if request.preferred_engine == profile.engine:
        score += 1000
    return RouteCandidate(profile.id, eligible, score, matched, tuple(issues)), plan


def route_workload(
    request: WorkloadRequest,
    host: ProfileHost,
    *,
    candidate_profile_ids: tuple[str, ...] | None = None,
) -> RouteDecision:
    """Choose one eligible profile or fail closed to the unmodified baseline."""
    profile_ids = candidate_profile_ids or tuple(
        profile.id for profile in BUILTIN_EXECUTABLE_PROFILES
    )
    if not profile_ids:
        raise ValueError("At least one candidate profile is required")
    if len(set(profile_ids)) != len(profile_ids):
        raise ValueError("Candidate profile IDs must be unique")

    candidates: list[RouteCandidate] = []
    plans: dict[str, ExecutionProfilePlan] = {}
    for profile_id in profile_ids:
        candidate, plan = _candidate(profile_id, request, host)
        candidates.append(candidate)
        plans[profile_id] = plan

    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        return RouteDecision(
            request,
            host,
            None,
            None,
            tuple(candidates),
            "No guarded profile passed every host, artifact, model, task, and quality gate; use the unmodified baseline.",
        )

    selected = sorted(eligible, key=lambda item: (-item.score, item.profile_id))[0]
    return RouteDecision(
        request,
        host,
        selected.profile_id,
        plans[selected.profile_id],
        tuple(candidates),
        f"Selected {selected.profile_id!r} from explicit request traits and passed guardrails.",
    )
