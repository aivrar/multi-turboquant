# SPDX-License-Identifier: MIT
"""Optional optimization catalog and compatibility planner."""

from .catalog import BUILTIN_DESCRIPTORS, create_builtin_registry
from .core import (
    IntegrationMode,
    ManifestPlugin,
    OptimizationContext,
    OptimizationDescriptor,
    OptimizationIssue,
    OptimizationKind,
    OptimizationMaturity,
    OptimizationPlugin,
    OptimizationProbe,
    OptimizationRegistry,
)
from .planner import OptimizationPlan, detect_optimization_context, plan_optimizations

__all__ = [
    "BUILTIN_DESCRIPTORS",
    "IntegrationMode",
    "ManifestPlugin",
    "OptimizationContext",
    "OptimizationDescriptor",
    "OptimizationIssue",
    "OptimizationKind",
    "OptimizationMaturity",
    "OptimizationPlan",
    "OptimizationPlugin",
    "OptimizationProbe",
    "OptimizationRegistry",
    "create_builtin_registry",
    "detect_optimization_context",
    "plan_optimizations",
]
