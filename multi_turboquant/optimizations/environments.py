# SPDX-License-Identifier: MIT
"""Isolated, opt-in environments for third-party optimization runtimes.

The compatibility planner in :mod:`multi_turboquant.optimizations.planner`
answers whether an already-installed optimization can run.  This module is a
separate layer: it describes reproducible uv projects and can materialize one
only after an explicit request.  Importing it never creates files, installs
packages, or launches a third-party process.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence


PROFILE_SCHEMA_VERSION = 1
DEFAULT_ENVIRONMENT_ROOT = Path(".mtq") / "environments"
_PROFILE_ID = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True)
class DependencyProfile:
    """A reviewable dependency contract for one isolated runtime."""

    id: str
    name: str
    optimization_id: str
    source_url: str
    python_spec: str
    default_python: str
    packages: tuple[str, ...]
    supported_os: tuple[str, ...]
    supported_compute: tuple[str, ...]
    required_executables: tuple[str, ...] = ()
    no_build_isolation_packages: tuple[str, ...] = ()
    build_environment: tuple[tuple[str, str], ...] = ()
    package_sources: tuple[tuple[str, str], ...] = ()
    package_indexes: tuple[tuple[str, str], ...] = ()
    dependency_metadata: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    cuda_toolkit_major: int | None = None
    torch_cuda_major: int | None = None
    validation_modules: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    build_may_compile: bool = False
    installable: bool = True
    blocked_reason: str | None = None
    source_build_packages: tuple[str, ...] = ()
    source_build_environment: tuple[tuple[str, str], ...] = ()
    local_source_package: str | None = None
    local_source_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _PROFILE_ID.fullmatch(self.id):
            raise ValueError("Profile IDs must be lowercase filesystem-safe names")
        if self.installable and not self.packages:
            raise ValueError("Dependency profiles must declare at least one package")
        if self.installable and not self.validation_modules:
            raise ValueError("Dependency profiles must declare validation modules")
        if self.installable and self.blocked_reason is not None:
            raise ValueError("Installable dependency profiles cannot have a blocked reason")
        if not self.installable and not self.blocked_reason:
            raise ValueError("Blocked dependency profiles must explain why they are blocked")
        if self.source_build_environment and not self.source_build_packages:
            raise ValueError("Source-build environment variables require source-build packages")
        if bool(self.local_source_package) != bool(self.local_source_markers):
            raise ValueError("Local source packages must declare reviewed checkout markers")
        if self.local_source_package and not self.installable:
            raise ValueError("Blocked profiles cannot accept a local source checkout")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "optimization_id": self.optimization_id,
            "source_url": self.source_url,
            "python_spec": self.python_spec,
            "default_python": self.default_python,
            "packages": list(self.packages),
            "supported_os": list(self.supported_os),
            "supported_compute": list(self.supported_compute),
            "required_executables": list(self.required_executables),
            "no_build_isolation_packages": list(self.no_build_isolation_packages),
            "build_environment": dict(self.build_environment),
            "source_build_packages": list(self.source_build_packages),
            "source_build_environment": dict(self.source_build_environment),
            "local_source_package": self.local_source_package,
            "local_source_markers": list(self.local_source_markers),
            "package_sources": dict(self.package_sources),
            "package_indexes": dict(self.package_indexes),
            "dependency_metadata": [
                {
                    "name": name,
                    "version": version,
                    "requires_dist": list(requires_dist),
                }
                for name, version, requires_dist in self.dependency_metadata
            ],
            "cuda_toolkit_major": self.cuda_toolkit_major,
            "torch_cuda_major": self.torch_cuda_major,
            "validation_modules": list(self.validation_modules),
            "notes": list(self.notes),
            "build_may_compile": self.build_may_compile,
            "status": "installable" if self.installable else "blocked",
            "blocked_reason": self.blocked_reason,
        }


BUILTIN_ENVIRONMENT_PROFILES = (
    DependencyProfile(
        id="flashattention",
        name="FlashAttention 2",
        optimization_id="flashattention",
        source_url="https://github.com/Dao-AILab/flash-attention",
        python_spec=">=3.10,<3.14",
        default_python="3.11",
        packages=(
            "torch==2.7.1",
            "packaging",
            "psutil",
            "ninja",
            "setuptools",
            "wheel",
            "flash-attn>=2.7,<3",
        ),
        supported_os=("linux",),
        supported_compute=("cuda",),
        required_executables=("nvcc",),
        no_build_isolation_packages=("flash-attn",),
        build_environment=(("MAX_JOBS", "4"),),
        source_build_packages=("flash-attn",),
        source_build_environment=(("FLASH_ATTENTION_FORCE_BUILD", "TRUE"),),
        local_source_package="flash-attn",
        local_source_markers=("setup.py", "flash_attn", "csrc"),
        package_sources=(("torch", "pytorch-cu126"),),
        package_indexes=(("pytorch-cu126", "https://download.pytorch.org/whl/cu126"),),
        cuda_toolkit_major=12,
        torch_cuda_major=12,
        validation_modules=("torch", "flash_attn"),
        notes=(
            "Uses a separate environment and never changes Multi-TurboQuant's environment.",
            "Requires a CUDA toolkit compatible with the selected PyTorch build.",
            "Windows remains unsupported by this conservative profile.",
        ),
        build_may_compile=True,
    ),
    DependencyProfile(
        id="fastdms",
        name="FastDMS",
        optimization_id="fastdms",
        source_url="https://github.com/shisa-ai/FastDMS",
        python_spec=">=3.10,<3.14",
        default_python="3.11",
        packages=(
            "torch==2.7.1",
            "packaging",
            "psutil",
            "ninja",
            "setuptools>=77",
            "wheel",
            "fastdms>=0.2,<0.3",
        ),
        supported_os=("linux",),
        supported_compute=("cuda",),
        required_executables=("nvcc",),
        no_build_isolation_packages=("flash-attn",),
        build_environment=(("MAX_JOBS", "4"),),
        source_build_packages=("flash-attn",),
        source_build_environment=(("FLASH_ATTENTION_FORCE_BUILD", "TRUE"),),
        local_source_package="fastdms",
        local_source_markers=("pyproject.toml", "fastdms"),
        package_sources=(("torch", "pytorch-cu126"),),
        package_indexes=(("pytorch-cu126", "https://download.pytorch.org/whl/cu126"),),
        cuda_toolkit_major=12,
        torch_cuda_major=12,
        validation_modules=("torch", "triton", "flash_attn", "fastdms"),
        notes=(
            "FastDMS is a standalone engine rather than a vLLM plugin.",
            "Only DMS-trained checkpoints are supported upstream.",
            "FastDMS depends on FlashAttention and may compile it during installation.",
        ),
        build_may_compile=True,
    ),
    DependencyProfile(
        id="lmcache",
        name="LMCache",
        optimization_id="lmcache",
        source_url="https://github.com/LMCache/LMCache",
        python_spec=">=3.10,<3.14",
        default_python="3.12",
        packages=(
            "torch==2.11.0",
            "lmcache==0.5.2",
            "openai==2.46.0",
        ),
        supported_os=("linux",),
        supported_compute=("cuda",),
        package_sources=(("torch", "pytorch-cu130"),),
        package_indexes=(("pytorch-cu130", "https://download.pytorch.org/whl/cu130"),),
        torch_cuda_major=13,
        validation_modules=("torch", "lmcache", "lmcache.c_ops", "openai"),
        local_source_package="lmcache",
        local_source_markers=("pyproject.toml", "setup.py", "lmcache"),
        notes=(
            "Uses the upstream CUDA 13 stable wheel and a matching official PyTorch wheel.",
            "Pins the OpenAI client imported by LMCache's CLI but omitted from its metadata.",
            "This profile installs the standalone LMCache service, not vLLM.",
            "Keep the serving engine isolated and qualify its connector version separately.",
        ),
    ),
    DependencyProfile(
        id="minference",
        name="MInference",
        optimization_id="minference",
        source_url="https://github.com/microsoft/MInference",
        python_spec=">=3.10,<3.13",
        default_python="3.11",
        packages=(
            "torch==2.7.1",
            "packaging",
            "psutil",
            "ninja",
            "setuptools",
            "wheel",
            "transformers>=4.37,<5",
            "minference @ git+https://github.com/microsoft/MInference.git@d76b76e89cb59817c89e1777c4c51b1c7f233335",
        ),
        supported_os=("linux",),
        supported_compute=("cuda",),
        required_executables=("git", "nvcc"),
        no_build_isolation_packages=("minference",),
        build_environment=(("MAX_JOBS", "1"), ("MINFERENCE_FORCE_BUILD", "TRUE")),
        package_sources=(("torch", "pytorch-cu126"),),
        package_indexes=(("pytorch-cu126", "https://download.pytorch.org/whl/cu126"),),
        dependency_metadata=(
            (
                "minference",
                "0.1.6.0",
                ("transformers>=4.37.0", "torch", "triton", "einops"),
            ),
        ),
        cuda_toolkit_major=12,
        torch_cuda_major=12,
        validation_modules=("torch", "triton", "minference"),
        notes=(
            "The official v0.1.6 source commit fixes the PyPI release's kivi_gemv import.",
            "Transformers stays below 5 because its private package probe changed return type.",
            "Local compilation is forced instead of relying on upstream's guessed wheel URL.",
            "The CUDA extension compiles locally when no matching wheel exists.",
            "Model-specific sparse-head configuration is still required for real workloads.",
            "vLLM or SGLang is deliberately not installed into this kernel profile.",
        ),
        build_may_compile=True,
        local_source_package="minference",
        local_source_markers=("setup.py", "minference", "csrc"),
    ),
    DependencyProfile(
        id="sageattention",
        name="SageAttention 2",
        optimization_id="sageattention",
        source_url="https://github.com/thu-ml/SageAttention",
        python_spec=">=3.10,<3.13",
        default_python="3.11",
        packages=(
            "torch==2.7.1",
            "numpy==2.2.6",
            "packaging",
            "ninja",
            "setuptools",
            "wheel",
            "sageattention @ git+https://github.com/thu-ml/SageAttention.git@d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5",
        ),
        supported_os=("linux",),
        supported_compute=("cuda",),
        required_executables=("git", "nvcc"),
        no_build_isolation_packages=("sageattention",),
        build_environment=(
            ("EXT_PARALLEL", "2"),
            ("MAX_JOBS", "4"),
        ),
        package_sources=(("torch", "pytorch-cu126"),),
        package_indexes=(("pytorch-cu126", "https://download.pytorch.org/whl/cu126"),),
        dependency_metadata=(("sageattention", "2.2.0", ()),),
        cuda_toolkit_major=12,
        torch_cuda_major=12,
        validation_modules=("torch", "triton", "numpy", "sageattention"),
        notes=(
            "Pinned to the audited upstream commit because the documented 2.2.0 package "
            "is not published on PyPI.",
            "Pins NumPy because the exercised Torch/SageAttention path imports it at runtime.",
            "Upstream supports Ampere and newer GPUs; model integration remains explicit.",
        ),
        build_may_compile=True,
        local_source_package="sageattention",
        local_source_markers=("setup.py", "sageattention", "csrc"),
    ),
    DependencyProfile(
        id="maru",
        name="Maru",
        optimization_id="maru",
        source_url="https://github.com/xcena-dev/maru",
        python_spec=">=3.12,<3.14",
        default_python="3.12",
        packages=(),
        supported_os=("linux",),
        supported_compute=("cuda",),
        blocked_reason=(
            "Upstream installation builds and installs a C++ resource manager and requires "
            "CXL /dev/dax hardware; mtq-env does not perform privileged host installation."
        ),
        notes=("Use the upstream installation guide on a dedicated CXL host.",),
        installable=False,
    ),
    DependencyProfile(
        id="speculative_prefill",
        name="Speculative Prefill",
        optimization_id="speculative_prefill",
        source_url="https://github.com/Jingyu6/speculative_prefill",
        python_spec=">=3.10,<3.11",
        default_python="3.10",
        packages=(),
        supported_os=("linux",),
        supported_compute=("cuda",),
        blocked_reason=(
            "Upstream is an unpackaged source monkeypatch pinned to Torch 2.4.0 and "
            "vLLM 0.6.3.post1, so it is not safe to expose as a maintained runtime profile."
        ),
        installable=False,
    ),
    DependencyProfile(
        id="rocketkv",
        name="RocketKV",
        optimization_id="rocketkv",
        source_url="https://github.com/NVlabs/RocketKV",
        python_spec=">=3.10,<3.11",
        default_python="3.10",
        packages=(),
        supported_os=("linux",),
        supported_compute=("cuda",),
        blocked_reason=(
            "Upstream is an unpackaged research snapshot under a non-commercial research "
            "license and is not a supported serving add-on."
        ),
        installable=False,
    ),
    DependencyProfile(
        id="lexico",
        name="Lexico",
        optimization_id="lexico",
        source_url="https://github.com/krafton-ai/lexico",
        python_spec=">=3.10,<3.13",
        default_python="3.11",
        packages=(),
        supported_os=("linux",),
        supported_compute=("cuda",),
        blocked_reason=(
            "Upstream is an unpackaged WIP source tree and requires a trained dictionary "
            "for each model/configuration."
        ),
        installable=False,
    ),
    DependencyProfile(
        id="adadecode",
        name="AdaDecode",
        optimization_id="adadecode",
        source_url="https://github.com/weizhepei/AdaDecode",
        python_spec=">=3.10,<3.11",
        default_python="3.10",
        packages=(),
        supported_os=("linux",),
        supported_compute=("cuda",),
        blocked_reason=(
            "The source repository has no software license and runtime use requires "
            "task-specific trained prediction heads."
        ),
        installable=False,
    ),
    DependencyProfile(
        id="resonance_yarn",
        name="Resonance YaRN",
        optimization_id="resonance_yarn",
        source_url="https://github.com/sheryc/resonance_rope",
        python_spec=">=3.10,<3.11",
        default_python="3.10",
        packages=(),
        supported_os=("linux",),
        supported_compute=("cuda",),
        blocked_reason=(
            "Upstream provides an old training environment and a Hugging Face LLaMA fork, "
            "not an installable llama.cpp or serving-runtime plugin."
        ),
        installable=False,
    ),
)


def get_environment_profile(profile_id: str) -> DependencyProfile:
    normalized = profile_id.strip().lower()
    for profile in BUILTIN_ENVIRONMENT_PROFILES:
        if profile.id == normalized:
            return profile
    available = ", ".join(profile.id for profile in BUILTIN_ENVIRONMENT_PROFILES)
    raise KeyError(f"Unknown environment profile {profile_id!r}. Available: {available}")


def inspect_profile_source(profile_id: str, path: str | Path) -> dict[str, object]:
    """Validate a local checkout against one reviewed dependency profile."""
    profile = get_environment_profile(profile_id)
    raw_path = str(path).strip()
    supported = profile.installable and profile.local_source_package is not None
    if not raw_path:
        return {
            "profile": profile.id,
            "package": profile.local_source_package,
            "path": raw_path,
            "supported": supported,
            "valid": False,
            "markers": {},
            "issues": ["Local source checkout is not configured."],
        }

    resolved = Path(raw_path).expanduser().resolve()
    markers = {
        marker: (resolved / marker).exists()
        for marker in profile.local_source_markers
    }
    issues: list[str] = []
    if not supported:
        issues.append(f"{profile.name} does not support installation from a local checkout.")
    if not resolved.is_dir():
        issues.append(f"Local source checkout is not a directory: {resolved}")
    else:
        issues.extend(f"Missing reviewed marker: {marker}" for marker, found in markers.items() if not found)
    return {
        "profile": profile.id,
        "package": profile.local_source_package,
        "path": str(resolved),
        "supported": supported,
        "valid": supported and not issues,
        "markers": markers,
        "issues": issues,
    }


@dataclass(frozen=True)
class EnvironmentContext:
    """Host facts used to decide if an environment can be created safely."""

    os: str
    compute: str
    available_executables: frozenset[str] = field(default_factory=frozenset)
    cuda_toolkit_version: tuple[int, int] | None = None
    cuda_toolkit_root: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "os", self.os.strip().lower())
        object.__setattr__(self, "compute", self.compute.strip().lower())
        object.__setattr__(
            self,
            "available_executables",
            frozenset(item.strip().lower() for item in self.available_executables),
        )
        if self.cuda_toolkit_root is not None:
            object.__setattr__(
                self,
                "cuda_toolkit_root",
                str(Path(self.cuda_toolkit_root).expanduser().resolve()),
            )


def _cuda_toolkit_paths(cuda_toolkit: str | Path | None) -> tuple[Path | None, Path | None]:
    executable_names = ("nvcc.exe", "nvcc") if os.name == "nt" else ("nvcc", "nvcc.exe")
    if cuda_toolkit is None:
        discovered = shutil.which("nvcc")
        if discovered is None:
            return None, None
        nvcc = Path(discovered).resolve()
        return nvcc.parent.parent, nvcc

    selected = Path(cuda_toolkit).expanduser().resolve()
    if selected.name.lower() in executable_names:
        return selected.parent.parent, selected
    if selected.name.lower() == "bin":
        root = selected.parent
        candidates = [selected / name for name in executable_names]
    else:
        root = selected
        candidates = [root / "bin" / name for name in executable_names]
    nvcc = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    return root, nvcc


def detect_environment_context(
    *, cuda_toolkit: str | Path | None = None
) -> EnvironmentContext:
    """Detect only the host facts needed by the environment planner."""
    from ..hardware import detect_platform

    platform_info = detect_platform()
    known_executables = ("uv", "pyenv", "git", "nvidia-smi", "rocminfo")
    available = frozenset(name for name in known_executables if shutil.which(name))
    toolkit_root, nvcc = _cuda_toolkit_paths(cuda_toolkit)
    cuda_toolkit_version = None
    if nvcc is not None and nvcc.is_file():
        available = available | {"nvcc"}
        try:
            result = subprocess.run(
                [str(nvcc), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            match = re.search(r"release\s+(\d+)\.(\d+)", result.stdout or result.stderr)
            if match:
                cuda_toolkit_version = (int(match.group(1)), int(match.group(2)))
        except (OSError, subprocess.SubprocessError):
            pass
    return EnvironmentContext(
        os=platform_info.os,
        compute=platform_info.primary_compute,
        available_executables=available,
        cuda_toolkit_version=cuda_toolkit_version,
        cuda_toolkit_root=str(toolkit_root) if toolkit_root is not None else None,
    )


@dataclass(frozen=True)
class EnvironmentIssue:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class EnvironmentCommand:
    description: str
    argv: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"description": self.description, "argv": list(self.argv)}


@dataclass(frozen=True)
class EnvironmentPlan:
    profile: DependencyProfile
    target: Path
    python_request: str
    project_toml: str
    commands: tuple[EnvironmentCommand, ...]
    issues: tuple[EnvironmentIssue, ...]
    cuda_toolkit_root: Path | None = None
    cuda_toolkit_version: tuple[int, int] | None = None
    local_source: Path | None = None
    build_from_source: bool = False

    @property
    def ready(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict:
        return {
            "profile": self.profile.to_dict(),
            "target": str(self.target),
            "python_request": self.python_request,
            "cuda_toolkit_root": (
                str(self.cuda_toolkit_root) if self.cuda_toolkit_root is not None else None
            ),
            "cuda_toolkit_version": (
                list(self.cuda_toolkit_version)
                if self.cuda_toolkit_version is not None
                else None
            ),
            "local_source": str(self.local_source) if self.local_source is not None else None,
            "local_source_package": (
                self.profile.local_source_package if self.local_source is not None else None
            ),
            "build_from_source": self.build_from_source,
            "source_build_packages": (
                list(self.profile.source_build_packages) if self.build_from_source else []
            ),
            "ready": self.ready,
            "commands": [command.to_dict() for command in self.commands],
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _toml_string(value: str) -> str:
    # JSON strings are valid TOML basic strings and give us reliable escaping.
    return json.dumps(value, ensure_ascii=False)


def _normalized_package_name(value: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", value)
    return re.sub(r"[-_.]+", "-", match.group(1)).lower() if match else ""


def render_profile_project(
    profile: DependencyProfile,
    *,
    build_from_source: bool = False,
    local_source: str | Path | None = None,
) -> str:
    """Render the independent uv project used to lock one profile."""
    if not profile.installable:
        raise ValueError(f"Dependency profile {profile.id!r} is blocked")
    selected_source = Path(local_source).expanduser().resolve() if local_source is not None else None
    if selected_source is not None and profile.local_source_package is None:
        raise ValueError(f"Dependency profile {profile.id!r} has no reviewed local source")
    local_package = _normalized_package_name(profile.local_source_package or "")
    dependencies = [
        profile.local_source_package
        if selected_source is not None and _normalized_package_name(item) == local_package
        else item
        for item in profile.packages
    ]
    dependency_lines = "\n".join(f"    {_toml_string(item)}," for item in dependencies)
    lines = [
        "# Generated by Multi-TurboQuant. Edit the dependency profile, not this file.",
        "[project]",
        f"name = {_toml_string(f'multi-turboquant-env-{profile.id}')}",
        'version = "0.0.0"',
        f"requires-python = {_toml_string(profile.python_spec)}",
        "dependencies = [",
        dependency_lines,
        "]",
        "",
        "[tool.uv]",
        "package = false",
    ]
    if profile.no_build_isolation_packages:
        rendered = ", ".join(_toml_string(item) for item in profile.no_build_isolation_packages)
        lines.append(f"no-build-isolation-package = [{rendered}]")
    if build_from_source and profile.source_build_packages:
        rendered = ", ".join(_toml_string(item) for item in profile.source_build_packages)
        lines.append(f"no-binary-package = [{rendered}]")
    if profile.package_sources or selected_source is not None:
        lines.extend(["", "[tool.uv.sources]"])
        for package, index in profile.package_sources:
            lines.append(f"{package} = {{ index = {_toml_string(index)} }}")
        if selected_source is not None and profile.local_source_package is not None:
            lines.append(
                f"{profile.local_source_package} = "
                f"{{ path = {_toml_string(str(selected_source))} }}"
            )
    for name, url in profile.package_indexes:
        lines.extend(
            [
                "",
                "[[tool.uv.index]]",
                f"name = {_toml_string(name)}",
                f"url = {_toml_string(url)}",
                "explicit = true",
            ]
        )
    for name, version, requires_dist in profile.dependency_metadata:
        rendered_requires = ", ".join(_toml_string(item) for item in requires_dist)
        lines.extend(
            [
                "",
                "[[tool.uv.dependency-metadata]]",
                f"name = {_toml_string(name)}",
                f"version = {_toml_string(version)}",
                f"requires-dist = [{rendered_requires}]",
            ]
        )
    lines.extend(
        [
            "",
            "[tool.multi-turboquant]",
            f"profile = {_toml_string(profile.id)}",
            f"schema = {PROFILE_SCHEMA_VERSION}",
        ]
    )
    if selected_source is not None:
        lines.append(f"local-source = {_toml_string(str(selected_source))}")
    lines.append("")
    return "\n".join(lines)


def plan_environment(
    profile_id: str,
    *,
    root: str | Path = DEFAULT_ENVIRONMENT_ROOT,
    python: str | None = None,
    cuda_toolkit: str | Path | None = None,
    local_source: str | Path | None = None,
    build_from_source: bool = False,
    context: EnvironmentContext | None = None,
) -> EnvironmentPlan:
    """Plan environment creation without writing files or running commands."""
    profile = get_environment_profile(profile_id)
    if context is not None and cuda_toolkit is not None:
        raise ValueError("Pass either context or cuda_toolkit, not both")
    context = context or detect_environment_context(cuda_toolkit=cuda_toolkit)
    toolkit_root = (
        Path(context.cuda_toolkit_root) if context.cuda_toolkit_root is not None else None
    )
    python_request = (python or profile.default_python).strip()
    if not python_request:
        raise ValueError("Python request must not be empty")
    target = (Path(root).expanduser() / profile.id).resolve()
    issues: list[EnvironmentIssue] = []
    local_source_path: Path | None = None
    if local_source is not None:
        inspection = inspect_profile_source(profile.id, local_source)
        inspected_path = str(inspection["path"])
        if inspected_path:
            local_source_path = Path(inspected_path)
        if not inspection["valid"]:
            issues.append(
                EnvironmentIssue(
                    "error",
                    "invalid_local_source",
                    "; ".join(str(item) for item in inspection["issues"]),
                )
            )
        else:
            issues.append(
                EnvironmentIssue(
                    "warning",
                    "local_source_selected",
                    f"The isolated environment will build {profile.local_source_package} from "
                    f"the reviewed local checkout at {local_source_path}.",
                )
            )

    if not profile.installable:
        issues.append(
            EnvironmentIssue(
                "error",
                "profile_blocked",
                profile.blocked_reason or "This dependency profile is blocked.",
            )
        )
        return EnvironmentPlan(
            profile=profile,
            target=target,
            python_request=python_request,
            build_from_source=build_from_source,
            project_toml="",
            commands=(),
            issues=tuple(issues),
            cuda_toolkit_root=toolkit_root,
            cuda_toolkit_version=context.cuda_toolkit_version,
            local_source=local_source_path,
        )

    if context.os not in profile.supported_os:
        issues.append(
            EnvironmentIssue(
                "error",
                "unsupported_os",
                f"{profile.name} supports {profile.supported_os}, not {context.os!r}.",
            )
        )
    if context.compute not in profile.supported_compute:
        issues.append(
            EnvironmentIssue(
                "error",
                "unsupported_compute",
                f"{profile.name} requires {profile.supported_compute}, not {context.compute!r}.",
            )
        )
    if "uv" not in context.available_executables:
        issues.append(
            EnvironmentIssue(
                "error",
                "missing_uv",
                "uv was not found; install uv or put its executable on PATH.",
            )
        )
    for executable in profile.required_executables:
        if executable.lower() not in context.available_executables:
            location = (
                "the selected CUDA toolkit"
                if executable.lower() == "nvcc" and context.cuda_toolkit_root
                else "PATH"
            )
            issues.append(
                EnvironmentIssue(
                    "error",
                    "missing_build_tool",
                    f"Required build tool {executable!r} was not found in {location}.",
                )
            )
    if profile.cuda_toolkit_major is not None and "cuda" in profile.supported_compute:
        detected_cuda = context.cuda_toolkit_version
        if detected_cuda is None:
            issues.append(
                EnvironmentIssue(
                    "error",
                    "unknown_cuda_toolkit",
                    "The CUDA toolkit version could not be read from nvcc. Select a toolkit "
                    "root with --cuda-toolkit or in Setup & Add-ons.",
                )
            )
        elif detected_cuda[0] != profile.cuda_toolkit_major:
            issues.append(
                EnvironmentIssue(
                    "error",
                    "unsupported_cuda_toolkit",
                    f"This profile requires CUDA {profile.cuda_toolkit_major}.x, but nvcc reports "
                    f"{detected_cuda[0]}.{detected_cuda[1]}. Native extensions must use the same "
                    "CUDA major as the profile's PyTorch build. Select a matching side-by-side "
                    "toolkit; the newer NVIDIA driver can remain installed.",
                )
            )
        elif toolkit_root is not None:
            issues.append(
                EnvironmentIssue(
                    "info",
                    "cuda_toolkit_selected",
                    f"Using CUDA toolkit {detected_cuda[0]}.{detected_cuda[1]} at {toolkit_root}.",
                )
            )
    if profile.build_may_compile:
        issues.append(
            EnvironmentIssue(
                "warning",
                "native_build_possible",
                "Installation may compile CUDA/C++ extensions and can take several minutes.",
            )
        )
    if build_from_source:
        if not profile.source_build_packages:
            issues.append(
                EnvironmentIssue(
                    "error",
                    "source_build_unavailable",
                    f"{profile.name} does not declare a reviewed source-build path.",
                )
            )
        else:
            packages = ", ".join(profile.source_build_packages)
            issues.append(
                EnvironmentIssue(
                    "warning",
                    "source_build_forced",
                    f"Source compilation is forced for: {packages}.",
                )
            )
    if "pyenv" in context.available_executables:
        issues.append(
            EnvironmentIssue(
                "info",
                "pyenv_available",
                "A pyenv interpreter can be selected by passing its path with --python.",
            )
        )

    command_argv = ["uv", "sync", "--project", str(target), "--python", python_request]
    reinstall_packages: list[str] = []
    if build_from_source:
        reinstall_packages.extend(profile.source_build_packages)
    if local_source_path is not None and profile.local_source_package is not None:
        reinstall_packages.append(profile.local_source_package)
    if reinstall_packages:
        command_argv.append("--no-cache")
        for package in dict.fromkeys(reinstall_packages):
            command_argv.extend(("--reinstall-package", package))
    command = EnvironmentCommand(
        "Resolve, lock, and synchronize the isolated environment",
        tuple(command_argv),
    )
    return EnvironmentPlan(
        profile=profile,
        target=target,
        python_request=python_request,
        build_from_source=build_from_source,
        project_toml=render_profile_project(
            profile,
            build_from_source=build_from_source,
            local_source=local_source_path,
        ),
        commands=(command,),
        issues=tuple(issues),
        cuda_toolkit_root=toolkit_root,
        cuda_toolkit_version=context.cuda_toolkit_version,
        local_source=local_source_path,
    )


def _read_owned_profile(project_file: Path) -> str | None:
    """Return the generated profile marker without requiring a TOML dependency."""
    if not project_file.is_file():
        return None
    in_marker = False
    for raw_line in project_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("["):
            in_marker = line == "[tool.multi-turboquant]"
            continue
        if in_marker and line.startswith("profile") and "=" in line:
            value = line.split("=", 1)[1].strip()
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, str) else None
    return None


def materialize_environment_project(plan: EnvironmentPlan) -> Path:
    """Write only the generated project metadata, refusing foreign targets."""
    project_file = plan.target / "pyproject.toml"
    if plan.target.exists() and not plan.target.is_dir():
        raise RuntimeError(f"Environment target is not a directory: {plan.target}")
    if project_file.exists():
        owner = _read_owned_profile(project_file)
        if owner != plan.profile.id:
            raise RuntimeError(
                f"Refusing to overwrite an unmanaged environment project: {project_file}"
            )
    plan.target.mkdir(parents=True, exist_ok=True)
    project_file.write_text(plan.project_toml, encoding="utf-8", newline="\n")
    return project_file


RunCommand = Callable[..., subprocess.CompletedProcess]


def _plan_child_environment(plan: EnvironmentPlan) -> dict[str, str]:
    child_environment = dict(os.environ)
    if plan.cuda_toolkit_root is not None:
        toolkit_root = str(plan.cuda_toolkit_root)
        child_environment["CUDA_HOME"] = toolkit_root
        child_environment["CUDA_PATH"] = toolkit_root
        child_environment["PATH"] = (
            str(plan.cuda_toolkit_root / "bin")
            + os.pathsep
            + child_environment.get("PATH", "")
        )
    return child_environment


def synchronize_environment(
    plan: EnvironmentPlan,
    *,
    upgrade: bool = False,
    runner: RunCommand = subprocess.run,
) -> None:
    """Materialize and sync an explicitly requested, compatible plan."""
    if not plan.ready:
        errors = "; ".join(issue.message for issue in plan.issues if issue.severity == "error")
        raise RuntimeError(f"Environment plan is not ready: {errors}")
    materialize_environment_project(plan)
    argv = list(plan.commands[0].argv)
    if upgrade:
        argv.append("--upgrade")
    child_environment = _plan_child_environment(plan)
    child_environment.update(dict(plan.profile.build_environment))
    if plan.build_from_source:
        child_environment.update(dict(plan.profile.source_build_environment))
    result = runner(argv, cwd=plan.target, env=child_environment, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"uv sync failed with exit code {result.returncode}")


def environment_python(target: Path, *, os_name: str | None = None) -> Path:
    os_name = (os_name or ("windows" if os.name == "nt" else "linux")).lower()
    relative = (
        Path(".venv/Scripts/python.exe") if os_name == "windows" else Path(".venv/bin/python")
    )
    return target / relative


def validation_script(profile: DependencyProfile) -> str:
    modules = json.dumps(list(profile.validation_modules))
    return (
        "import importlib, importlib.metadata, json\n"
        f"modules = {modules}\n"
        "versions = {}\n"
        "for name in modules:\n"
        "    module = importlib.import_module(name)\n"
        "    version = getattr(module, '__version__', None)\n"
        "    if version is None:\n"
        "        try:\n"
        "            version = importlib.metadata.version(name.replace('_', '-'))\n"
        "        except importlib.metadata.PackageNotFoundError:\n"
        "            version = 'unknown'\n"
        "    versions[name] = version\n"
        "try:\n"
        "    import torch\n"
        "    versions['torch_cuda'] = torch.version.cuda\n"
        "    versions['cuda_available'] = torch.cuda.is_available()\n"
        "except ImportError:\n"
        "    pass\n"
        "print(json.dumps(versions, sort_keys=True))\n"
    )


def check_environment(
    plan: EnvironmentPlan,
    *,
    runner: RunCommand = subprocess.run,
) -> Mapping[str, object]:
    """Import profile modules with the isolated interpreter and return its report."""
    interpreter = environment_python(plan.target)
    if not interpreter.is_file():
        raise RuntimeError(f"Environment has not been created: {interpreter}")
    result = runner(
        [str(interpreter), "-c", validation_script(plan.profile)],
        cwd=plan.target,
        env=_plan_child_environment(plan),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Environment validation failed: {stderr}")
    output_lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    try:
        parsed = json.loads(output_lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("Environment validation returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Environment validation returned an unexpected result")
    if "cuda" in plan.profile.supported_compute:
        if not parsed.get("torch_cuda"):
            raise RuntimeError("Environment validation found a CPU-only PyTorch build")
        if parsed.get("cuda_available") is not True:
            raise RuntimeError("PyTorch cannot access CUDA from the isolated environment")
        expected_major = plan.profile.torch_cuda_major
        torch_cuda = str(parsed["torch_cuda"])
        if expected_major is not None and not torch_cuda.startswith(f"{expected_major}."):
            raise RuntimeError(
                f"PyTorch uses CUDA {torch_cuda}, but the profile requires CUDA {expected_major}.x"
            )
    return parsed


def run_in_environment(
    plan: EnvironmentPlan,
    command: Sequence[str],
    *,
    runner: RunCommand = subprocess.run,
) -> int:
    """Run an explicit command with the isolated environment activated."""
    if not command:
        raise ValueError("A command is required")
    interpreter = environment_python(plan.target)
    if not interpreter.is_file():
        raise RuntimeError(f"Environment has not been created: {interpreter}")
    binary_dir = interpreter.parent
    child_environment = _plan_child_environment(plan)
    child_environment["VIRTUAL_ENV"] = str(interpreter.parent.parent)
    child_environment["PATH"] = str(binary_dir) + os.pathsep + child_environment.get("PATH", "")
    result = runner(list(command), cwd=Path.cwd(), env=child_environment, check=False)
    return int(result.returncode)
