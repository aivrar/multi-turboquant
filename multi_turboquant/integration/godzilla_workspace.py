# SPDX-License-Identifier: MIT
"""Read-only Godzilla checkout inspection and explicit TriAttention preparation."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


def _file_contains(path: Path, token: str, *, max_bytes: int = 4_000_000) -> bool:
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return False
        return token.lower() in path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False


def _candidate_binaries(root: Path) -> list[Path]:
    names = ("llama-server.exe", "llama-server")
    directories = (
        root / "bin",
        root / "build" / "bin",
        root / "build" / "bin" / "Release",
        root / "build-cuda" / "bin",
        root / "build-cuda" / "bin" / "Release",
        root / "build-king" / "bin",
        root / "build-king" / "bin" / "Release",
    )
    return [directory / name for directory in directories for name in names if (directory / name).is_file()]


def inspect_godzilla_checkout(path: str | Path) -> dict[str, object]:
    """Inspect a configured Godzilla checkout without executing repository code."""
    raw_path = str(path).strip()
    if not raw_path:
        return {
            "path": raw_path,
            "valid": False,
            "kind": "godzilla",
            "issues": ["Godzilla source checkout is not configured."],
        }

    root = Path(raw_path).expanduser().resolve()
    markers = {
        "CMakeLists.txt": (root / "CMakeLists.txt").is_file(),
        "ggml": (root / "ggml").is_dir(),
        "common/arg.cpp": (root / "common" / "arg.cpp").is_file(),
        "godzilla_identity": (
            (root / "GODZILLA_KING.md").is_file()
            or (root / "scripts" / "godzilla-paths.ps1").is_file()
        ),
    }
    issues: list[str] = []
    if not root.is_dir():
        issues.append(f"Godzilla source checkout is not a directory: {root}")
    else:
        issues.extend(f"Missing Godzilla marker: {name}" for name, found in markers.items() if not found)

    arg_source = root / "common" / "arg.cpp"
    triattention_source = root / "src" / "llama-triattention.cpp"
    ensure_script = root / "scripts" / "ensure-triattention.ps1"
    resolver_script = root / "scripts" / "resolve-triattention-hf.py"
    bundled_calibrator = root / "scripts" / "calibrate-triattention.py"
    binaries = _candidate_binaries(root) if root.is_dir() else []
    return {
        "path": str(root),
        "valid": not issues,
        "kind": "godzilla",
        "markers": markers,
        "git_remote": None,
        "features": {
            "kvarn": _file_contains(arg_source, "kvarn"),
            "triattention": (
                triattention_source.is_file()
                and _file_contains(arg_source, "--triattention-stats")
            ),
            "triattention_prepare": ensure_script.is_file(),
            "triattention_auto_resolver": resolver_script.is_file(),
            "bundled_calibrator": bundled_calibrator.is_file(),
        },
        "paths": {
            "ensure_triattention": str(ensure_script) if ensure_script.is_file() else None,
            "resolve_triattention_hf": str(resolver_script) if resolver_script.is_file() else None,
            "bundled_calibrator": str(bundled_calibrator) if bundled_calibrator.is_file() else None,
            "windows_build_script": (
                str(root / "scripts" / "build_cuda.ps1")
                if (root / "scripts" / "build_cuda.ps1").is_file()
                else None
            ),
        },
        "binaries": [str(item.resolve()) for item in binaries],
        "preferred_binary": str(binaries[0].resolve()) if binaries else None,
        "issues": issues,
        "notes": [
            "KVarN is a runtime cache-type selection and does not use a calibration file.",
            "TriAttention calibration is model-specific and may download the matching source model.",
            "Godzilla currently expects TRIATTENTION_PYTHON and TRIATTENTION_CALIBRATE_PY.",
        ],
    }


@dataclass(frozen=True)
class GodzillaIssue:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class GodzillaCalibrationPlan:
    checkout: Path
    gguf: Path
    output: Path
    python: Path | None
    calibrator: Path | None
    hf_model: str | None
    n_tokens: int
    device: str
    command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    issues: tuple[GodzillaIssue, ...]

    @property
    def ready(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "checkout": str(self.checkout),
            "gguf": str(self.gguf),
            "output": str(self.output),
            "python": str(self.python) if self.python is not None else None,
            "calibrator": str(self.calibrator) if self.calibrator is not None else None,
            "hf_model": self.hf_model,
            "n_tokens": self.n_tokens,
            "device": self.device,
            "command": list(self.command),
            "environment": dict(self.environment),
            "issues": [issue.to_dict() for issue in self.issues],
            "ready": self.ready,
            "kvarn_calibration_required": False,
        }


def plan_godzilla_triattention(
    checkout: str | Path,
    gguf: str | Path,
    *,
    output: str | Path | None = None,
    python: str | Path | None = None,
    calibrator: str | Path | None = None,
    hf_model: str | None = None,
    n_tokens: int = 2048,
    device: str = "cuda",
    shell_executable: str | None = None,
) -> GodzillaCalibrationPlan:
    """Plan Godzilla's own ensure-triattention workflow without running it."""
    checkout_path = Path(checkout).expanduser().resolve()
    gguf_path = Path(gguf).expanduser().resolve()
    output_path = (
        Path(output).expanduser().resolve()
        if output is not None and str(output).strip()
        else checkout_path / "calibrations" / f"{gguf_path.stem}.triattention"
    )
    python_value = python or os.environ.get("TRIATTENTION_PYTHON")
    calibrator_value = calibrator or os.environ.get("TRIATTENTION_CALIBRATE_PY")
    python_path = Path(python_value).expanduser().resolve() if python_value else None
    calibrator_path = Path(calibrator_value).expanduser().resolve() if calibrator_value else None
    normalized_hf = hf_model.strip() if hf_model and hf_model.strip() else None
    normalized_device = device.strip().lower()
    inspection = inspect_godzilla_checkout(checkout_path)
    issues: list[GodzillaIssue] = []

    if not inspection["valid"]:
        issues.append(
            GodzillaIssue(
                "error",
                "invalid_godzilla_checkout",
                "; ".join(str(item) for item in inspection.get("issues", [])),
            )
        )
    ensure_script = checkout_path / "scripts" / "ensure-triattention.ps1"
    resolver_script = checkout_path / "scripts" / "resolve-triattention-hf.py"
    if not ensure_script.is_file():
        issues.append(
            GodzillaIssue(
                "error",
                "missing_prepare_script",
                f"Godzilla TriAttention preparation script was not found: {ensure_script}",
            )
        )
    if not gguf_path.is_file() or gguf_path.suffix.lower() != ".gguf":
        issues.append(GodzillaIssue("error", "invalid_gguf", f"GGUF model not found: {gguf_path}"))
    if output_path.suffix.lower() != ".triattention":
        issues.append(
            GodzillaIssue(
                "error",
                "invalid_output",
                "Godzilla TriAttention output must end in .triattention.",
            )
        )
    if python_path is None or not python_path.is_file():
        issues.append(
            GodzillaIssue(
                "error",
                "missing_calibration_python",
                "Select TRIATTENTION_PYTHON: a Python executable with Torch and calibrator dependencies.",
            )
        )
    if calibrator_path is None or not calibrator_path.is_file():
        issues.append(
            GodzillaIssue(
                "error",
                "missing_calibrator",
                "Select a Godzilla-compatible calibrate-triattention.py script.",
            )
        )
    if not 128 <= n_tokens <= 32_768:
        issues.append(
            GodzillaIssue("error", "invalid_token_count", "Calibration tokens must be 128 to 32768.")
        )
    if normalized_device not in {"cuda", "cpu"}:
        issues.append(GodzillaIssue("error", "invalid_device", "Device must be 'cuda' or 'cpu'."))
    if normalized_hf is None:
        if resolver_script.is_file():
            issues.append(
                GodzillaIssue(
                    "warning",
                    "automatic_model_resolution",
                    "Godzilla will try to resolve the matching Hugging Face model from GGUF metadata; "
                    "provide it explicitly if the mapping is unknown or ambiguous.",
                )
            )
        else:
            issues.append(
                GodzillaIssue(
                    "error",
                    "missing_hf_model",
                    "Provide the matching Hugging Face model because this checkout has no resolver.",
                )
            )
    issues.append(
        GodzillaIssue(
            "info",
            "kvarn_no_calibration",
            "KVarN does not require calibration; select its target K/V cache types at launch.",
        )
    )
    if output_path.is_file():
        issues.append(
            GodzillaIssue(
                "info",
                "calibration_present",
                f"Existing calibration will be reused: {output_path}",
            )
        )
    else:
        issues.append(
            GodzillaIssue(
                "warning",
                "model_download_possible",
                "Calibration can download and load the matching Hugging Face checkpoint.",
            )
        )

    shell = shell_executable or shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        issues.append(
            GodzillaIssue(
                "error",
                "missing_powershell",
                "PowerShell was not found; Godzilla's ensure-triattention.ps1 cannot run.",
            )
        )
    command: list[str] = []
    if shell is not None:
        command = [
            shell,
            "-NoProfile",
            "-File",
            str(ensure_script),
            "-Gguf",
            str(gguf_path),
            "-Output",
            str(output_path),
            "-NTokens",
            str(n_tokens),
            "-Device",
            normalized_device,
        ]
        if normalized_hf is not None:
            command.extend(("-HfModel", normalized_hf))
    environment: list[tuple[str, str]] = [("GODZILLA_ROOT", str(checkout_path))]
    if python_path is not None:
        environment.append(("TRIATTENTION_PYTHON", str(python_path)))
    if calibrator_path is not None:
        environment.append(("TRIATTENTION_CALIBRATE_PY", str(calibrator_path)))
    return GodzillaCalibrationPlan(
        checkout=checkout_path,
        gguf=gguf_path,
        output=output_path,
        python=python_path,
        calibrator=calibrator_path,
        hf_model=normalized_hf,
        n_tokens=n_tokens,
        device=normalized_device,
        command=tuple(command),
        environment=tuple(environment),
        issues=tuple(issues),
    )


RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def run_godzilla_triattention(
    plan: GodzillaCalibrationPlan,
    *,
    runner: RunCommand = subprocess.run,
) -> dict[str, object]:
    """Run one explicitly confirmed plan and verify its output artifact."""
    if not plan.ready:
        errors = "; ".join(issue.message for issue in plan.issues if issue.severity == "error")
        raise RuntimeError(f"Godzilla calibration plan is not ready: {errors}")
    existed = plan.output.is_file()
    environment: Mapping[str, str] = {**os.environ, **dict(plan.environment)}
    result = runner(
        list(plan.command),
        cwd=plan.checkout,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Godzilla TriAttention preparation failed with exit code {result.returncode}"
            + (f": {details}" if details else "")
        )
    if not plan.output.is_file():
        raise RuntimeError(f"Godzilla calibration command did not create {plan.output}")
    return {
        "output": str(plan.output),
        "reused": existed,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }
