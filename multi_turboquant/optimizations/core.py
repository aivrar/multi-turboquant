# SPDX-License-Identifier: MIT
"""Side-effect-free metadata and registry types for optional optimizations."""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class OptimizationKind(str, Enum):
    CACHE_STORAGE = "cache_storage"
    CACHE_REPRESENTATION = "cache_representation"
    TOKEN_POLICY = "token_policy"
    PREFILL = "prefill"
    ATTENTION_BACKEND = "attention_backend"
    DECODE = "decode"
    POSITION_ENCODING = "position_encoding"
    TOKENIZER = "tokenizer"
    RESOURCE_SHARING = "resource_sharing"


class OptimizationMaturity(str, Enum):
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    RESEARCH = "research"
    BLOCKED = "blocked"


class QualityRisk(str, Enum):
    """Expected effect on model outputs when an optimization is enabled."""

    EXACT = "exact"
    CONDITIONAL = "conditional"
    LOSSY = "lossy"
    RESEARCH = "research"


class IntegrationMode(str, Enum):
    EXTERNAL_SERVICE = "external_service"
    OPTIONAL_PYTHON = "optional_python"
    NATIVE_BACKEND_REQUIRED = "native_backend_required"
    RESEARCH_ONLY = "research_only"
    PRELOAD_LIBRARY = "preload_library"


@dataclass(frozen=True)
class OptimizationDescriptor:
    """Static, reviewable contract for one optional optimization."""

    id: str
    name: str
    source_url: str
    kind: OptimizationKind
    maturity: OptimizationMaturity
    integration_mode: IntegrationMode
    license: str
    summary: str
    supported_engines: tuple[str, ...]
    supported_compute: tuple[str, ...] = ()
    supported_os: tuple[str, ...] = ()
    supported_architectures: tuple[str, ...] = ()
    supported_kv_formats: tuple[str, ...] = ()
    required_modules: tuple[str, ...] = ()
    required_executables: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    composition_domains: tuple[str, ...] = ()
    allows_composition_with: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    validation_gates: tuple[str, ...] = ()
    quality_risk: QualityRisk = QualityRisk.CONDITIONAL
    reviewed_source_commit: str | None = None
    min_python: tuple[int, int] | None = None
    default_enabled: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "source_url": self.source_url,
            "kind": self.kind.value,
            "maturity": self.maturity.value,
            "integration_mode": self.integration_mode.value,
            "license": self.license,
            "summary": self.summary,
            "supported_engines": list(self.supported_engines),
            "supported_compute": list(self.supported_compute),
            "supported_os": list(self.supported_os),
            "supported_architectures": list(self.supported_architectures),
            "supported_kv_formats": list(self.supported_kv_formats),
            "required_modules": list(self.required_modules),
            "required_executables": list(self.required_executables),
            "required_capabilities": list(self.required_capabilities),
            "requires": list(self.requires),
            "conflicts": list(self.conflicts),
            "limitations": list(self.limitations),
            "composition_domains": list(self.composition_domains),
            "allows_composition_with": list(self.allows_composition_with),
            "required_artifacts": list(self.required_artifacts),
            "validation_gates": list(self.validation_gates),
            "quality_risk": self.quality_risk.value,
            "reviewed_source_commit": self.reviewed_source_commit,
            "min_python": list(self.min_python) if self.min_python else None,
            "default_enabled": self.default_enabled,
        }


@dataclass(frozen=True)
class OptimizationContext:
    """Runtime facts used to evaluate optimization eligibility.

    Installed module/executable sets are injectable for deterministic tests. A
    value of ``None`` asks the probe to inspect the current process host.
    """

    engine: str
    os: str
    compute: str
    architecture: str = "x86_64"
    kv_format: str = "fp16"
    python_version: tuple[int, int] = (3, 10)
    capabilities: frozenset[str] = field(default_factory=frozenset)
    active_features: frozenset[str] = field(default_factory=frozenset)
    installed_modules: frozenset[str] | None = None
    installed_executables: frozenset[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "engine", self.engine.strip().lower())
        object.__setattr__(self, "os", self.os.strip().lower())
        object.__setattr__(self, "compute", self.compute.strip().lower())
        object.__setattr__(self, "architecture", self.architecture.strip().lower())
        object.__setattr__(self, "kv_format", self.kv_format.strip().lower())
        object.__setattr__(
            self,
            "capabilities",
            frozenset(item.strip().lower() for item in self.capabilities),
        )
        object.__setattr__(
            self,
            "active_features",
            frozenset(item.strip().lower() for item in self.active_features),
        )


@dataclass(frozen=True)
class OptimizationIssue:
    severity: str
    code: str
    optimization: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "optimization": self.optimization,
            "message": self.message,
        }


@dataclass(frozen=True)
class OptimizationProbe:
    descriptor: OptimizationDescriptor
    installed: bool
    eligible: bool
    issues: tuple[OptimizationIssue, ...] = ()

    def to_dict(self) -> dict:
        return {
            "descriptor": self.descriptor.to_dict(),
            "installed": self.installed,
            "eligible": self.eligible,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class OptimizationPlugin(Protocol):
    """Common interface for explicitly registered optimization plugins."""

    descriptor: OptimizationDescriptor

    def probe(self, context: OptimizationContext) -> OptimizationProbe:
        ...


def _module_available(module: str, context: OptimizationContext) -> bool:
    if context.installed_modules is not None:
        return module in context.installed_modules
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _executable_available(executable: str, context: OptimizationContext) -> bool:
    if context.installed_executables is not None:
        return executable in context.installed_executables
    return shutil.which(executable) is not None


class ManifestPlugin:
    """Default plugin implementation backed only by a static descriptor."""

    def __init__(self, descriptor: OptimizationDescriptor):
        self.descriptor = descriptor

    def probe(self, context: OptimizationContext) -> OptimizationProbe:
        descriptor = self.descriptor
        issues: list[OptimizationIssue] = []

        def add(severity: str, code: str, message: str) -> None:
            issues.append(OptimizationIssue(severity, code, descriptor.id, message))

        if descriptor.integration_mode == IntegrationMode.RESEARCH_ONLY:
            add("error", "research_only", "No maintained runtime integration is available.")
        elif descriptor.integration_mode == IntegrationMode.NATIVE_BACKEND_REQUIRED:
            add(
                "error",
                "native_backend_required",
                "This optimization requires native inference-backend changes; a wrapper is insufficient.",
            )
        if descriptor.maturity == OptimizationMaturity.BLOCKED:
            add("error", "blocked", "This integration is blocked and must not be enabled.")

        if descriptor.supported_engines and context.engine not in descriptor.supported_engines:
            add(
                "error",
                "unsupported_engine",
                f"Engine {context.engine!r} is unsupported; use one of {descriptor.supported_engines}.",
            )
        if descriptor.supported_os and context.os not in descriptor.supported_os:
            add(
                "error",
                "unsupported_os",
                f"Operating system {context.os!r} is unsupported; use one of {descriptor.supported_os}.",
            )
        if descriptor.supported_compute and context.compute not in descriptor.supported_compute:
            add(
                "error",
                "unsupported_compute",
                f"Compute backend {context.compute!r} is unsupported; use one of {descriptor.supported_compute}.",
            )
        if (
            descriptor.supported_architectures
            and context.architecture not in descriptor.supported_architectures
        ):
            add(
                "error",
                "unsupported_architecture",
                f"Architecture {context.architecture!r} is unsupported; use one of "
                f"{descriptor.supported_architectures}.",
            )
        if (
            descriptor.supported_kv_formats
            and context.kv_format not in descriptor.supported_kv_formats
        ):
            add(
                "error",
                "unsupported_kv_format",
                f"KV format {context.kv_format!r} has not been validated; supported formats are "
                f"{descriptor.supported_kv_formats}.",
            )
        if descriptor.min_python and context.python_version < descriptor.min_python:
            add(
                "error",
                "unsupported_python",
                f"Python {descriptor.min_python[0]}.{descriptor.min_python[1]} or newer is required.",
            )

        for capability in descriptor.required_capabilities:
            if capability not in context.capabilities:
                add(
                    "error",
                    "missing_capability",
                    f"Required host capability {capability!r} was not detected.",
                )

        missing_modules = [
            module for module in descriptor.required_modules
            if not _module_available(module, context)
        ]
        missing_executables = [
            executable for executable in descriptor.required_executables
            if not _executable_available(executable, context)
        ]
        for module in missing_modules:
            add("error", "missing_module", f"Python module {module!r} is not installed.")
        for executable in missing_executables:
            add("error", "missing_executable", f"Executable {executable!r} was not found.")

        installed = not missing_modules and not missing_executables
        eligible = not any(issue.severity == "error" for issue in issues)
        return OptimizationProbe(descriptor, installed, eligible, tuple(issues))


class OptimizationRegistry:
    """Explicit registry; importing the package never discovers third-party code."""

    def __init__(self):
        self._plugins: dict[str, OptimizationPlugin] = {}

    def register(self, plugin: OptimizationPlugin) -> None:
        optimization_id = plugin.descriptor.id
        if not optimization_id or optimization_id.lower() != optimization_id:
            raise ValueError("Optimization IDs must be non-empty lowercase strings")
        if optimization_id in self._plugins:
            raise ValueError(f"Optimization {optimization_id!r} is already registered")
        self._plugins[optimization_id] = plugin

    def get(self, optimization_id: str) -> OptimizationPlugin:
        try:
            return self._plugins[optimization_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._plugins))
            raise KeyError(
                f"Unknown optimization {optimization_id!r}. Available: {available}"
            ) from exc

    def list(self) -> tuple[OptimizationPlugin, ...]:
        return tuple(self._plugins[key] for key in sorted(self._plugins))
