# SPDX-License-Identifier: MIT
"""Compatibility planning for explicitly selected optional optimizations."""

from __future__ import annotations

import platform as platform_module
import sys
from dataclasses import dataclass
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

    def to_dict(self) -> dict:
        return {
            "selected": list(self.selected),
            "ready": self.ready,
            "probes": [probe.to_dict() for probe in self.probes],
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

    return OptimizationPlan(normalized, tuple(probes), tuple(issues))
