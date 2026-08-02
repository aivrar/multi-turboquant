# SPDX-License-Identifier: MIT
"""Launch helpers for CUDA LLM weight sharing.

This module integrates with the reviewed pontostroy/cuda-llm-weight-share
source and prepares its LD_PRELOAD launch environment. Source inspection and
build planning are read-only; compilation requires explicit confirmation.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


CUDA_WEIGHT_SHARE_URL = "https://github.com/pontostroy/cuda-llm-weight-share"
CUDA_WEIGHT_SHARE_COMMIT = "15bcecaebdbcec479f13df1c4396d5318b5bb85d"
CUDA_WEIGHT_SHARE_SOURCE = "cuda-llm-weight-share.c"
CUDA_WEIGHT_SHARE_LIBRARY = "cuda-llm-weight-share.so"


@dataclass(frozen=True)
class CudaWeightShareConfig:
    """Configuration for wrapping a llama.cpp launch with CUDA weight sharing."""

    enabled: bool = False
    library_path: str = "./cuda-llm-weight-share.so"
    model_size_bytes: int | None = None
    model_size_tolerance: int = 0
    ipc_name: str = "/cuda_vram_ipc_auto"
    shm_wait_sec: int | None = None
    suppress_master_free: bool = False
    trace_callers: bool = False
    trace_depth: int | None = None
    trace_normal_allocs: bool = False

    def validate(self) -> list[str]:
        """Return configuration warnings."""
        warnings: list[str] = []
        if self.enabled and not self.library_path:
            warnings.append("CUDA weight sharing requires a preload library path")
        if self.model_size_bytes is not None and self.model_size_bytes < 0:
            warnings.append("MODEL_SIZE must be >= 0")
        if self.model_size_tolerance < 0:
            warnings.append("MODEL_SIZE_TOLERANCE must be >= 0")
        if not re.fullmatch(r"/[A-Za-z0-9_.-]+", self.ipc_name):
            warnings.append(
                "CUDA_VRAM_IPC_NAME must start with / and contain only letters, digits, '.', '_', or '-'"
            )
        if self.model_size_bytes and self.ipc_name == "/cuda_vram_ipc_auto":
            warnings.append(
                "Production weight sharing requires a unique CUDA_VRAM_IPC_NAME for this "
                "model, GPU, allocation size, and runtime build"
            )
        if self.shm_wait_sec is not None and self.shm_wait_sec < 0:
            warnings.append("CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC must be >= 0")
        if self.trace_depth is not None and self.trace_depth <= 0:
            warnings.append("CUDA_VRAM_IPC_TRACE_DEPTH must be > 0")
        if self.suppress_master_free and not self.model_size_bytes:
            warnings.append("Do not suppress the master free during MODEL_SIZE=0 reconnaissance")
        return warnings


def get_cuda_weight_share_env(config: CudaWeightShareConfig) -> dict[str, str]:
    """Return environment variables for cuda-llm-weight-share."""
    if not config.enabled:
        return {}

    warnings = config.validate()
    if warnings:
        raise ValueError("; ".join(warnings))

    env = {
        "LD_PRELOAD": config.library_path,
        "MODEL_SIZE": str(config.model_size_bytes or 0),
        "MODEL_SIZE_TOLERANCE": str(config.model_size_tolerance),
        "CUDA_VRAM_IPC_NAME": config.ipc_name,
    }
    if config.shm_wait_sec is not None:
        env["CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC"] = str(config.shm_wait_sec)
    if config.suppress_master_free:
        env["CUDA_VRAM_IPC_SUPPRESS_MASTER_FREE"] = "1"
    if config.trace_callers:
        env["CUDA_VRAM_IPC_TRACE_CALLERS"] = "1"
    if config.trace_depth is not None:
        env["CUDA_VRAM_IPC_TRACE_DEPTH"] = str(config.trace_depth)
    if config.trace_normal_allocs:
        env["CUDA_VRAM_IPC_TRACE_NORMAL_ALLOCS"] = "1"
    return env


def wrap_cuda_weight_share_command(
    command: list[str],
    config: CudaWeightShareConfig,
) -> list[str]:
    """Prefix a command with env assignments for CUDA weight sharing."""
    env = get_cuda_weight_share_env(config)
    if not env:
        return command
    return ["env", *(f"{key}={value}" for key, value in env.items()), *command]


def _git_value(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _expected_weight_share_remote(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower().removesuffix(".git").replace("\\", "/")
    normalized = normalized.replace("github.com:", "github.com/")
    return normalized.endswith("github.com/pontostroy/cuda-llm-weight-share")


def inspect_cuda_weight_share_source(path: str | Path) -> dict[str, object]:
    """Validate markers and exact provenance without compiling checkout code."""
    root = Path(path).expanduser().resolve()
    markers = {
        "README.md": (root / "README.md").is_file(),
        "LICENSE": (root / "LICENSE").is_file(),
        CUDA_WEIGHT_SHARE_SOURCE: (root / CUDA_WEIGHT_SHARE_SOURCE).is_file(),
    }
    issues: list[str] = []
    if not root.is_dir():
        issues.append(f"CUDA weight-share source is not a directory: {root}")
    issues.extend(f"Missing reviewed source marker: {name}" for name, found in markers.items() if not found)
    remote = _git_value(root, "remote", "get-url", "origin") if root.is_dir() else None
    revision = _git_value(root, "rev-parse", "HEAD") if root.is_dir() else None
    if root.is_dir() and not _expected_weight_share_remote(remote):
        issues.append(f"Source origin is not {CUDA_WEIGHT_SHARE_URL}: {remote or 'unavailable'}")
    if root.is_dir() and revision != CUDA_WEIGHT_SHARE_COMMIT:
        issues.append(
            f"Source revision must be {CUDA_WEIGHT_SHARE_COMMIT}; found {revision or 'unavailable'}."
        )
    return {
        "profile": "cuda_weight_share",
        "name": "CUDA LLM Weight Share",
        "path": str(root),
        "valid": not issues,
        "status": "source_build_available" if not issues else "invalid_source",
        "source_url": CUDA_WEIGHT_SHARE_URL,
        "expected_commit": CUDA_WEIGHT_SHARE_COMMIT,
        "git_remote": remote,
        "git_revision": revision,
        "markers": markers,
        "issues": issues,
        "setup": {
            "mode": "reviewed_source_build",
            "automatic": True,
            "requirements": ["Linux x86_64", "CUDA toolkit headers", "GCC", "libdl"],
            "next_steps": [
                "Build and validate the shared object with mtq-weight-share.",
                "Run reconnaissance with MODEL_SIZE=0 before production sharing.",
                "Use a unique IPC name for each model, GPU, allocation, and runtime build.",
            ],
        },
    }


@dataclass(frozen=True)
class WeightShareBuildIssue:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class CudaWeightShareBuildPlan:
    source: Path
    output: Path
    cuda_root: Path | None
    compiler: str | None
    command: tuple[str, ...]
    issues: tuple[WeightShareBuildIssue, ...]

    @property
    def ready(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "output": str(self.output),
            "cuda_root": str(self.cuda_root) if self.cuda_root is not None else None,
            "compiler": self.compiler,
            "command": list(self.command),
            "ready": self.ready,
            "issues": [issue.to_dict() for issue in self.issues],
            "pin": CUDA_WEIGHT_SHARE_COMMIT,
        }


def _cuda_toolkit_root(value: str | Path | None) -> Path | None:
    candidates: list[Path] = []
    if value is not None:
        selected = Path(value).expanduser().resolve()
        if selected.name.lower() in {"nvcc", "nvcc.exe"}:
            selected = selected.parent.parent
        elif selected.name.lower() == "bin":
            selected = selected.parent
        candidates.append(selected)
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(variable):
            candidates.append(Path(os.environ[variable]).expanduser().resolve())
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(Path(nvcc).resolve().parent.parent)
    candidates.append(Path("/usr/local/cuda"))
    return next(
        (root for root in candidates if (root / "include" / "cuda_runtime_api.h").is_file()),
        None,
    )


def plan_cuda_weight_share_build(
    source: str | Path,
    *,
    output: str | Path | None = None,
    cuda_toolkit: str | Path | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> CudaWeightShareBuildPlan:
    """Create a read-only, fail-closed build plan for the pinned source."""
    root = Path(source).expanduser().resolve()
    output_path = (
        Path(output).expanduser().resolve() if output is not None else root / CUDA_WEIGHT_SHARE_LIBRARY
    )
    issues: list[WeightShareBuildIssue] = []
    inspection = inspect_cuda_weight_share_source(root)
    issues.extend(
        WeightShareBuildIssue("error", "invalid_source", str(message))
        for message in inspection["issues"]
    )
    normalized_system = (system or platform.system()).strip().lower()
    normalized_machine = (machine or platform.machine()).strip().lower()
    if normalized_system != "linux" or normalized_machine not in {"x86_64", "amd64"}:
        issues.append(
            WeightShareBuildIssue(
                "error",
                "unsupported_platform",
                "CUDA weight-share builds are supported only on Linux x86_64.",
            )
        )
    cuda_root = _cuda_toolkit_root(cuda_toolkit)
    if cuda_root is None:
        issues.append(
            WeightShareBuildIssue(
                "error",
                "missing_cuda_headers",
                "CUDA runtime headers were not found; select a CUDA toolkit root or nvcc.",
            )
        )
    compiler = shutil.which("gcc")
    if compiler is None:
        issues.append(WeightShareBuildIssue("error", "missing_gcc", "Required compiler not found: gcc"))
    for tool in ("file", "nm", "ldd"):
        if shutil.which(tool) is None:
            issues.append(
                WeightShareBuildIssue("error", f"missing_{tool}", f"Required validator not found: {tool}")
            )
    if output_path.exists():
        issues.append(
            WeightShareBuildIssue(
                "error",
                "output_exists",
                f"Build output already exists and will not be overwritten: {output_path}",
            )
        )
    command: tuple[str, ...] = ()
    if compiler is not None and cuda_root is not None:
        command = (
            compiler,
            "-shared",
            "-fPIC",
            "-O2",
            "-g",
            "-Wall",
            "-Wextra",
            f"-I{cuda_root / 'include'}",
            str(root / CUDA_WEIGHT_SHARE_SOURCE),
            "-o",
            str(output_path),
            "-ldl",
        )
    issues.append(
        WeightShareBuildIssue(
            "warning",
            "reconnaissance_required",
            "A successful build is not a production MODEL_SIZE. Run one trusted llama.cpp "
            "process with MODEL_SIZE=0 and record the exact model-weight allocation first.",
        )
    )
    return CudaWeightShareBuildPlan(
        source=root,
        output=output_path,
        cuda_root=cuda_root,
        compiler=compiler,
        command=command,
        issues=tuple(issues),
    )


RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def _run_build_command(
    argv: Sequence[str],
    *,
    runner: RunCommand,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    result = runner(
        list(argv),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(argv)}\n{detail}".rstrip())
    return result


def validate_cuda_weight_share_library(
    library: str | Path,
    *,
    runner: RunCommand = subprocess.run,
) -> dict[str, object]:
    """Validate the ELF shared object, exported hooks, and dynamic dependencies."""
    path = Path(library).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"CUDA weight-share library not found: {path}")
    file_report = _run_build_command(("file", str(path)), runner=runner, cwd=path.parent)
    if "ELF" not in file_report.stdout or "shared object" not in file_report.stdout:
        raise RuntimeError(f"Build output is not an ELF shared object: {file_report.stdout.strip()}")
    symbols = _run_build_command(("nm", "-D", str(path)), runner=runner, cwd=path.parent)
    missing = [
        name
        for name in ("cudaMalloc", "cudaFree")
        if re.search(rf"\bT\s+{name}$", symbols.stdout, flags=re.MULTILINE) is None
    ]
    if missing:
        raise RuntimeError("CUDA weight-share hooks are not exported: " + ", ".join(missing))
    dependencies = _run_build_command(("ldd", str(path)), runner=runner, cwd=path.parent)
    combined_dependencies = (dependencies.stdout + "\n" + dependencies.stderr).strip()
    if "libcudart" in combined_dependencies.lower():
        raise RuntimeError("CUDA weight-share library has an unexpected hard libcudart dependency")
    return {
        "valid": True,
        "library": str(path),
        "file": file_report.stdout.strip(),
        "exported_hooks": ["cudaMalloc", "cudaFree"],
        "dependencies": combined_dependencies,
        "hard_libcudart_dependency": False,
        "reconnaissance_required": True,
    }


def build_cuda_weight_share(
    plan: CudaWeightShareBuildPlan,
    *,
    confirmed: bool = False,
    runner: RunCommand = subprocess.run,
) -> dict[str, object]:
    """Compile and validate only after an explicit confirmation."""
    if not confirmed:
        raise RuntimeError("CUDA weight-share build was not confirmed")
    if not plan.ready:
        errors = "; ".join(issue.message for issue in plan.issues if issue.severity == "error")
        raise RuntimeError(f"CUDA weight-share build plan is not ready: {errors}")
    if plan.output.exists():
        raise RuntimeError(f"Build output appeared after planning and will not be overwritten: {plan.output}")
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    _run_build_command(plan.command, runner=runner, cwd=plan.source)
    return validate_cuda_weight_share_library(plan.output, runner=runner)
