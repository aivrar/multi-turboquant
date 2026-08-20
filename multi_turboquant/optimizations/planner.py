# SPDX-License-Identifier: MIT
"""Compatibility planning for explicitly selected optional optimizations."""

from __future__ import annotations

import platform as platform_module
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from .catalog import create_builtin_registry
from .core import (
    OptimizationContext,
    OptimizationIssue,
    OptimizationProbe,
    OptimizationRegistry,
)


@dataclass(frozen=True)
class OptimizationPlan:
    selected: tuple[str, ...]
    probes: tuple[OptimizationProbe, ...]
    issues: tuple[OptimizationIssue, ...]

    @property
    def ready(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def required_artifacts(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            artifact
            for probe in self.probes
            for artifact in probe.descriptor.required_artifacts
        ))

    @property
    def validation_gates(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            gate
            for probe in self.probes
            for gate in probe.descriptor.validation_gates
        ))

    def to_dict(self) -> dict:
        return {
            "selected": list(self.selected),
            "ready": self.ready,
            "probes": [probe.to_dict() for probe in self.probes],
            "required_artifacts": list(self.required_artifacts),
            "validation_gates": list(self.validation_gates),
            "quality_risks": {
                probe.descriptor.id: probe.descriptor.quality_risk.value
                for probe in self.probes
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }


def detect_optimization_context(
    *,
    engine: str,
    kv_format: str = "fp16",
    active_features: tuple[str, ...] = (),
) -> OptimizationContext:
    """Build a planner context from Multi-TurboQuant's hardware detector."""
    from ..hardware import detect_platform

    detected = detect_platform()
    capabilities: set[str] = set()
    if detected.os == "linux":
        try:
            if any(Path("/dev").glob("dax*")):
                capabilities.add("cxl_devdax")
        except OSError:
            pass
    return OptimizationContext(
        engine=engine.strip().lower(),
        os=detected.os,
        compute=detected.primary_compute,
        architecture=detected.arch or platform_module.machine().lower(),
        kv_format=kv_format.strip().lower(),
        python_version=(sys.version_info.major, sys.version_info.minor),
        capabilities=frozenset(capabilities),
        active_features=frozenset(feature.strip().lower() for feature in active_features),
    )


def plan_optimizations(
    selected: list[str] | tuple[str, ...],
    context: OptimizationContext,
    *,
    registry: OptimizationRegistry | None = None,
) -> OptimizationPlan:
    """Probe a requested set and report every dependency or conflict."""
    registry = registry or create_builtin_registry()
    normalized = tuple(dict.fromkeys(item.strip().lower() for item in selected if item.strip()))
    selected_set = set(normalized)
    probes: list[OptimizationProbe] = []
    issues: list[OptimizationIssue] = []
    reported_conflicts: set[tuple[str, str]] = set()
    descriptors = {}

    for optimization_id in normalized:
        try:
            plugin = registry.get(optimization_id)
        except KeyError:
            issues.append(OptimizationIssue(
                "error",
                "unknown_optimization",
                optimization_id,
                f"Optimization {optimization_id!r} is not registered.",
            ))
            continue
        probe = plugin.probe(context)
        probes.append(probe)
        issues.extend(probe.issues)

        descriptor = plugin.descriptor
        descriptors[optimization_id] = descriptor
        for requirement in descriptor.requires:
            if requirement not in selected_set:
                issues.append(OptimizationIssue(
                    "error",
                    "missing_optimization_dependency",
                    optimization_id,
                    f"{optimization_id!r} requires {requirement!r} to be selected.",
                ))

        active = selected_set | set(context.active_features)
        for conflict in descriptor.conflicts:
            if conflict in active:
                pair = tuple(sorted((optimization_id, conflict)))
                if pair in reported_conflicts:
                    continue
                reported_conflicts.add(pair)
                issues.append(OptimizationIssue(
                    "error",
                    "optimization_conflict",
                    optimization_id,
                    f"{optimization_id!r} conflicts with {conflict!r}; choose one.",
                ))

    for left_id, right_id in combinations(descriptors, 2):
        left = descriptors[left_id]
        right = descriptors[right_id]
        shared_domains = sorted(set(left.composition_domains) & set(right.composition_domains))
        if not shared_domains:
            continue
        explicitly_allowed = (
            right_id in left.allows_composition_with
            and left_id in right.allows_composition_with
        )
        pair = tuple(sorted((left_id, right_id)))
        if explicitly_allowed or pair in reported_conflicts:
            continue
        reported_conflicts.add(pair)
        issues.append(OptimizationIssue(
            "error",
            "unvalidated_composition",
            left_id,
            f"{left_id!r} and {right_id!r} both modify {', '.join(shared_domains)}; "
            "no reviewed composition profile allows them together.",
        ))

    return OptimizationPlan(normalized, tuple(probes), tuple(issues))
