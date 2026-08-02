# SPDX-License-Identifier: MIT
"""Read-only Godzilla inspection and explicit TriAttention preparation."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .._paths import lexical_absolute_path

from ..calibration.godzilla_triattention import (
    LONG_CALIBRATION_THRESHOLD,
    MAX_CALIBRATION_TOKENS,
    convert_domvox_triattention_stats,
    inspect_calibration_python,
    inspect_domvox_triattention_calibrator,
    inspect_godzilla_triattention_file,
    load_huggingface_model_metadata,
    inspect_official_triattention_calibrator,
)


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
        root / "build-gigatoken-cpu" / "bin",
        root / "build-gigatoken-cpu" / "bin" / "Release",
        root / "build-gigatoken-cuda" / "bin",
        root / "build-gigatoken-cuda" / "bin" / "Release",
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
            "gigatoken": (
                (root / "src" / "llama-gigatoken.cpp").is_file()
                and (root / "cmake" / "gigatoken.cmake").is_file()
            ),
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
            "TriAttention is experimental, opt-in, and not recommended by the current Godzilla lab policy.",
            "TriAttention calibration is a separate manual, model-specific operation that may download the matching source model.",
            (
                "This checkout contains a Gigatoken C++ integration; qualify token-ID parity for the selected model."
                if (root / "src" / "llama-gigatoken.cpp").is_file()
                else "The Python Gigatoken calibration option does not add Gigatoken to this Godzilla runtime checkout."
            ),
            (
                "This checkout bundles a calibration script."
                if bundled_calibrator.is_file()
                else "This checkout does not bundle calibrate-triattention.py; use the official "
                "WeianMao/triattention calibrator plus conversion, or a validated checkout-owned script."
            ),
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
    mode: str
    calibration_input: Path | None
    official_stats: Path | None
    official_stats_input: Path | None
    attention_implementation: str
    dependency_validation: Mapping[str, object] | None
    dependency_override: bool
    domvox_calibrator: Path | None
    domvox_accept_lossy: bool
    allow_long_calibration: bool
    command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    issues: tuple[GodzillaIssue, ...]
    tokenizer_backend: str = "transformers"

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
            "mode": self.mode,
            "calibration_input": (
                str(self.calibration_input) if self.calibration_input is not None else None
            ),
            "official_stats": str(self.official_stats) if self.official_stats is not None else None,
            "official_stats_input": (
                str(self.official_stats_input) if self.official_stats_input is not None else None
            ),
            "domvox_calibrator": (
                str(self.domvox_calibrator) if self.domvox_calibrator is not None else None
            ),
            "domvox_accept_lossy": self.domvox_accept_lossy,
            "allow_long_calibration": self.allow_long_calibration,
            "attention_implementation": self.attention_implementation,
            "tokenizer_backend": self.tokenizer_backend,
            "dependency_validation": self.dependency_validation,
            "dependency_override": self.dependency_override,
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
    calibration_input: str | Path | None = None,
    official_stats_input: str | Path | None = None,
    domvox_calibrator: str | Path | None = None,
    domvox_accept_lossy: bool = False,
    allow_long_calibration: bool = False,
    hf_model: str | None = None,
    n_tokens: int = 2048,
    device: str = "cuda",
    mode: str = "official_python",
    attention_implementation: str = "sdpa",
    tokenizer_backend: str = "transformers",
    verify_dependencies: bool = False,
    dependency_override: bool = False,
    dependency_runner=subprocess.run,
    shell_executable: str | None = None,
) -> GodzillaCalibrationPlan:
    """Plan a validated official-Python or checkout-owned calibration workflow."""
    checkout_path = Path(checkout).expanduser().resolve()
    gguf_path = Path(gguf).expanduser().resolve()
    output_path = (
        Path(output).expanduser().resolve()
        if output is not None and str(output).strip()
        else checkout_path / "calibrations" / f"{gguf_path.stem}.triattention"
    )
    normalized_hf = hf_model.strip() if hf_model and hf_model.strip() else None
    normalized_device = device.strip().lower()
    normalized_mode = mode.strip().lower()
    normalized_attention = attention_implementation.strip().lower()
    normalized_tokenizer = tokenizer_backend.strip().lower()
    inspection = inspect_godzilla_checkout(checkout_path)
    python_value = python or os.environ.get("TRIATTENTION_PYTHON")
    inspection_paths = inspection.get("paths")
    bundled_calibrator = (
        inspection_paths.get("bundled_calibrator")
        if isinstance(inspection_paths, dict)
        else None
    )
    calibrator_value = (
        calibrator
        or os.environ.get("TRIATTENTION_CALIBRATE_PY")
        or (bundled_calibrator if normalized_mode == "godzilla_script" else None)
    )
    domvox_calibrator_value = (
        domvox_calibrator
        or os.environ.get("TRIATTENTION_DOMVOX_CALIBRATE_PY")
        or (calibrator if normalized_mode == "domvox" else None)
    )
    calibration_input_value = calibration_input or os.environ.get("TRIATTENTION_CALIBRATION_TEXT")
    official_stats_input_value = official_stats_input or os.environ.get(
        "TRIATTENTION_OFFICIAL_STATS"
    )
    python_path = lexical_absolute_path(python_value) if python_value else None
    calibrator_path = Path(calibrator_value).expanduser().resolve() if calibrator_value else None
    domvox_calibrator_path = (
        Path(domvox_calibrator_value).expanduser().resolve()
        if domvox_calibrator_value
        else None
    )
    calibration_input_path = (
        Path(calibration_input_value).expanduser().resolve() if calibration_input_value else None
    )
    official_stats_input_path = (
        Path(official_stats_input_value).expanduser().resolve()
        if official_stats_input_value
        else None
    )
    official_stats = (
        official_stats_input_path
        if normalized_mode == "official_convert"
        else output_path.with_suffix(".domvox.bin")
        if normalized_mode == "domvox"
        else output_path.with_suffix(".official.pt")
    )
    output_exists = output_path.is_file()
    issues: list[GodzillaIssue] = []
    dependency_validation: Mapping[str, object] | None = None

    if not inspection["valid"]:
        issues.append(
            GodzillaIssue(
                "error",
                "invalid_godzilla_checkout",
                "; ".join(str(item) for item in inspection.get("issues", [])),
            )
        )
    if normalized_mode not in {"official_python", "official_convert", "godzilla_script", "domvox"}:
        issues.append(
            GodzillaIssue(
                "error",
                "invalid_calibration_mode",
                "Calibration mode must be official_python, official_convert, godzilla_script, or domvox.",
            )
        )
    if normalized_tokenizer not in {"transformers", "gigatoken"}:
        issues.append(
            GodzillaIssue(
                "error",
                "invalid_tokenizer_backend",
                "Tokenizer backend must be transformers or gigatoken.",
            )
        )
    elif normalized_tokenizer == "gigatoken" and normalized_mode not in {
        "official_python",
        "domvox",
    }:
        issues.append(
            GodzillaIssue(
                "error",
                "gigatoken_mode_unsupported",
                "Gigatoken is supported only by the reviewed official and domvox Python calibrators; existing-stat conversion does not tokenize, and the checkout-owned calibrator is not API-qualified.",
            )
        )
    ensure_script = checkout_path / "scripts" / "ensure-triattention.ps1"
    resolver_script = checkout_path / "scripts" / "resolve-triattention-hf.py"
    if normalized_mode == "godzilla_script" and not ensure_script.is_file():
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
    if not output_exists and (python_path is None or not python_path.is_file()):
        issues.append(
            GodzillaIssue(
                "error",
                "missing_calibration_python",
                "Select TRIATTENTION_PYTHON: a Python executable with Torch and calibrator dependencies.",
            )
        )
    if not output_exists and normalized_mode in {"official_python", "godzilla_script"}:
        if calibrator_path is None or not calibrator_path.is_file():
            message = (
                "Select the official WeianMao/triattention scripts/calibrate.py."
                if normalized_mode == "official_python"
                else "This Godzilla checkout does not provide calibrate-triattention.py. Select a "
                "separately validated script compatible with this checkout's binary format."
            )
            issues.append(GodzillaIssue("error", "missing_calibrator", message))
        elif normalized_mode == "official_python":
            calibrator_inspection = inspect_official_triattention_calibrator(calibrator_path)
            if not calibrator_inspection["valid"]:
                domvox_inspection = inspect_domvox_triattention_calibrator(calibrator_path)
                message = (
                    "The selected script matches domvox, not the official WeianMao calibrator. "
                    "Choose domvox mode and acknowledge its lossy Godzilla v1 conversion, or "
                    "select WeianMao/triattention/scripts/calibrate.py."
                    if domvox_inspection["valid"]
                    else "; ".join(str(item) for item in calibrator_inspection["issues"])
                )
                issues.append(
                    GodzillaIssue(
                        "error",
                        "invalid_official_calibrator",
                        message,
                    )
                )
    if not output_exists and normalized_mode == "domvox":
        if domvox_calibrator_path is None or not domvox_calibrator_path.is_file():
            issues.append(
                GodzillaIssue(
                    "error",
                    "missing_domvox_calibrator",
                    "Select domvox/triattention-ggml/triattention_calibrate.py.",
                )
            )
        else:
            domvox_inspection = inspect_domvox_triattention_calibrator(domvox_calibrator_path)
            if not domvox_inspection["valid"]:
                official_inspection = inspect_official_triattention_calibrator(
                    domvox_calibrator_path
                )
                message = (
                    "The selected script matches the official WeianMao calibrator, not domvox. "
                    "Choose the recommended Generate stats + convert mode."
                    if official_inspection["valid"]
                    else "; ".join(str(item) for item in domvox_inspection["issues"])
                )
                issues.append(
                    GodzillaIssue(
                        "error",
                        "invalid_domvox_calibrator",
                        message,
                    )
                )
        if not domvox_accept_lossy:
            issues.append(
                GodzillaIssue(
                    "error",
                    "domvox_lossy_confirmation",
                    "domvox TRIA v2 to Godzilla v1 conversion drops layer-budget and attention-scale fields; explicitly acknowledge this experimental conversion.",
                )
            )
    if not output_exists and normalized_mode in {"official_python", "domvox"} and (
        calibration_input_path is None
        or not calibration_input_path.is_file()
        or calibration_input_path.stat().st_size == 0
    ):
        issues.append(
            GodzillaIssue(
                "error",
                "missing_calibration_input",
                "Select a non-empty plain-text calibration input file.",
            )
        )
    if not output_exists and normalized_mode == "official_convert" and (
        official_stats_input_path is None
        or not official_stats_input_path.is_file()
        or official_stats_input_path.suffix.lower() not in {".pt", ".pth"}
    ):
        issues.append(
            GodzillaIssue(
                "error",
                "missing_official_stats",
                "Select an existing official TriAttention .pt statistics file to convert.",
            )
        )
    if not 128 <= n_tokens <= MAX_CALIBRATION_TOKENS:
        issues.append(
            GodzillaIssue(
                "error",
                "invalid_token_count",
                f"Calibration tokens must be 128 to {MAX_CALIBRATION_TOKENS}.",
            )
        )
    elif n_tokens > LONG_CALIBRATION_THRESHOLD:
        if not allow_long_calibration:
            issues.append(
                GodzillaIssue(
                    "error",
                    "long_calibration_confirmation",
                    f"Calibration above {LONG_CALIBRATION_THRESHOLD} tokens is one-shot and may exhaust memory; explicitly enable long calibration to continue.",
                )
            )
        else:
            issues.append(
                GodzillaIssue(
                    "warning",
                    "long_calibration_one_shot",
                    "The upstream calibrator processes one long sequence; no chunked aggregation is being assumed. Expect substantially higher memory and runtime.",
                )
            )
    if normalized_device not in {"cuda", "cpu"}:
        issues.append(GodzillaIssue("error", "invalid_device", "Device must be 'cuda' or 'cpu'."))
    if normalized_mode == "official_python" and normalized_attention not in {
        "eager",
        "sdpa",
        "flash_attention_2",
    }:
        issues.append(
            GodzillaIssue(
                "error",
                "invalid_attention_implementation",
                "Attention implementation must be eager, sdpa, or flash_attention_2.",
            )
        )
    if (
        not output_exists
        and normalized_hf is None
        and normalized_mode in {"official_python", "official_convert", "domvox"}
    ):
        issues.append(
            GodzillaIssue(
                "error",
                "missing_hf_model",
                "Provide the matching Hugging Face model ID or local Transformers directory "
                "for model-shape metadata.",
            )
        )
    elif not output_exists and normalized_hf is None and normalized_mode == "godzilla_script":
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
    issues.append(
        GodzillaIssue(
            "warning",
            "triattention_experimental",
            "Current Godzilla documentation treats TriAttention as experimental, opt-in, and "
            "manually calibrated; validate retrieval quality before relying on it.",
        )
    )
    if output_exists:
        try:
            artifact = inspect_godzilla_triattention_file(output_path)
        except (OSError, ValueError) as exc:
            issues.append(
                GodzillaIssue(
                    "error",
                    "invalid_existing_calibration",
                    f"Existing calibration is not a valid Godzilla v1 artifact: {exc}",
                )
            )
        else:
            issues.append(
                GodzillaIssue(
                    "info",
                    "calibration_present",
                    f"Validated existing calibration will be reused: {output_path} "
                    f"({artifact['sampled_heads']} sampled heads).",
                )
            )
    elif normalized_mode == "official_python":
        issues.append(
            GodzillaIssue(
                "warning",
                "model_download_possible",
                "Calibration can download and load the matching Hugging Face checkpoint.",
            )
        )
    elif normalized_mode == "official_convert":
        issues.append(
            GodzillaIssue(
                "info",
                "model_config_download_possible",
                "Conversion loads only matching Hugging Face configuration metadata, not model weights.",
            )
        )
    elif normalized_mode == "domvox":
        issues.append(
            GodzillaIssue(
                "warning",
                "domvox_experimental_adapter",
                "domvox TRIA v2 is adapted to Godzilla v1; validate model metadata and retrieval quality before serving.",
            )
        )
    if (
        not output_exists
        and normalized_tokenizer == "gigatoken"
        and normalized_mode in {"official_python", "domvox"}
    ):
        issues.append(
            GodzillaIssue(
                "info",
                "gigatoken_parity_required",
                "Gigatoken will be enabled only after exact token-ID parity passes for the selected calibration text; any mismatch stops calibration.",
            )
        )

    if (
        not output_exists
        and verify_dependencies
        and normalized_mode in {"official_python", "official_convert", "domvox"}
        and python_path is not None
        and python_path.is_file()
    ):
        python_check = inspect_calibration_python(
            python_path,
            device=normalized_device,
            tokenizer_backend=normalized_tokenizer,
            runner=dependency_runner,
        )
        report = python_check.get("report")
        dependency_validation = report if isinstance(report, Mapping) else None
        if python_check["valid"]:
            ready_dependencies = "torch, transformers, and accelerate"
            if normalized_tokenizer == "gigatoken":
                ready_dependencies += ", plus the reviewed Gigatoken backend"
            issues.append(
                GodzillaIssue(
                    "info",
                    "calibration_dependencies_ready",
                    f"Calibration Python successfully imported {ready_dependencies}.",
                )
            )
            if dependency_validation and dependency_validation.get("cuda_total_memory_bytes"):
                gib = 1024**3
                total_gib = float(dependency_validation["cuda_total_memory_bytes"]) / gib
                free_gib = float(dependency_validation.get("cuda_free_memory_bytes", 0)) / gib
                device_name = dependency_validation.get("cuda_device", "selected CUDA device")
                issues.append(
                    GodzillaIssue(
                        "info",
                        "calibration_device_memory",
                        f"{device_name}: {free_gib:.1f} GiB free of {total_gib:.1f} GiB VRAM "
                        "at preflight time. This is capacity information, not a 200k-token "
                        "memory guarantee.",
                    )
                )
            elif dependency_validation and dependency_validation.get("cuda_memory_error"):
                issues.append(
                    GodzillaIssue(
                        "warning",
                        "calibration_device_memory_unavailable",
                        "Calibration dependencies are valid, but CUDA memory capacity could not "
                        f"be read: {dependency_validation['cuda_memory_error']}",
                    )
                )
        else:
            message = "; ".join(str(item) for item in python_check["issues"])
            if dependency_override:
                issues.append(
                    GodzillaIssue(
                        "warning",
                        "calibration_dependency_override",
                        "Automatic dependency validation failed, but the manual override is active: "
                        f"{message}. Calibration may still fail.",
                    )
                )
            else:
                issues.append(
                    GodzillaIssue(
                        "error",
                        "calibration_dependencies_missing",
                        f"Calibration dependency check failed: {message}",
                    )
                )

    if normalized_mode in {"official_python", "official_convert", "domvox"}:
        issues.append(
            GodzillaIssue(
                "info",
                "no_llama_cli_required",
                (
                    "This mode runs the official Python/Hugging Face calibrator and a validated "
                    "Godzilla format conversion; it does not use llama-cli."
                    if normalized_mode == "official_python"
                    else (
                        "This mode converts existing official .pt statistics to the validated "
                        "Godzilla format; it does not rerun calibration or use llama-cli."
                        if normalized_mode == "official_convert"
                        else "This mode runs the domvox Python calibrator, then adapts its TRIA v2 output to Godzilla v1; it does not use llama-cli."
                    )
                ),
            )
        )
        if not output_exists:
            issues.append(
                GodzillaIssue(
                    "warning",
                    "official_remote_code",
                    "Hugging Face metadata/model loading uses trust_remote_code=True; review and "
                    "trust the selected model source before continuing.",
                )
            )

    shell = shell_executable or shutil.which("pwsh") or shutil.which("powershell")
    if not output_exists and normalized_mode == "godzilla_script" and shell is None:
        issues.append(
            GodzillaIssue(
                "error",
                "missing_powershell",
                "PowerShell was not found; Godzilla's ensure-triattention.ps1 cannot run.",
            )
        )
    command: list[str] = []
    if (
        not output_exists
        and normalized_mode == "official_python"
        and python_path is not None
        and calibrator_path is not None
        and calibration_input_path is not None
        and normalized_hf is not None
    ):
        converter = Path(__file__).resolve().parents[1] / "calibration" / "godzilla_triattention.py"
        command = [
            str(python_path),
            str(converter),
            "calibrate",
            "--calibrator",
            str(calibrator_path) if calibrator_path is not None else "",
            "--model",
            normalized_hf or "",
            "--input",
            str(calibration_input_path) if calibration_input_path is not None else "",
            "--output",
            str(output_path),
            "--stats-output",
            str(official_stats),
            "--max-length",
            str(n_tokens),
            "--device",
            normalized_device,
            "--attn-implementation",
            normalized_attention,
            "--tokenizer-backend",
            normalized_tokenizer,
        ]
    elif (
        not output_exists
        and normalized_mode == "domvox"
        and python_path is not None
        and domvox_calibrator_path is not None
        and calibration_input_path is not None
        and normalized_hf is not None
    ):
        executable = [str(python_path), str(domvox_calibrator_path)]
        if normalized_tokenizer == "gigatoken":
            wrapper = Path(__file__).resolve().parents[1] / "calibration" / "gigatoken_runner.py"
            executable = [
                str(python_path),
                str(wrapper),
                "--kind",
                "domvox",
                "--calibrator",
                str(domvox_calibrator_path),
            ]
        command = [
            *executable,
            "--model",
            normalized_hf,
            "--input",
            str(calibration_input_path),
            "--output",
            str(official_stats),
            "--max-length",
            str(n_tokens),
            "--device",
            normalized_device,
        ]
    elif (
        not output_exists
        and normalized_mode == "official_convert"
        and python_path is not None
        and official_stats_input_path is not None
        and normalized_hf is not None
    ):
        converter = Path(__file__).resolve().parents[1] / "calibration" / "godzilla_triattention.py"
        command = [
            str(python_path),
            str(converter),
            "convert",
            str(official_stats_input_path),
            str(output_path),
            "--model",
            normalized_hf,
        ]
    elif not output_exists and normalized_mode == "godzilla_script" and shell is not None:
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
    if domvox_calibrator_path is not None:
        environment.append(("TRIATTENTION_DOMVOX_CALIBRATE_PY", str(domvox_calibrator_path)))
    return GodzillaCalibrationPlan(
        checkout=checkout_path,
        gguf=gguf_path,
        output=output_path,
        python=python_path,
        calibrator=calibrator_path,
        hf_model=normalized_hf,
        n_tokens=n_tokens,
        device=normalized_device,
        mode=normalized_mode,
        calibration_input=calibration_input_path,
        official_stats=(
            official_stats
            if normalized_mode in {"official_python", "official_convert", "domvox"}
            else None
        ),
        official_stats_input=official_stats_input_path,
        attention_implementation=normalized_attention,
        dependency_validation=dependency_validation,
        dependency_override=dependency_override,
        domvox_calibrator=domvox_calibrator_path,
        domvox_accept_lossy=domvox_accept_lossy,
        allow_long_calibration=allow_long_calibration,
        command=tuple(command),
        environment=tuple(environment),
        issues=tuple(issues),
        tokenizer_backend=normalized_tokenizer,
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
    if plan.output.is_file():
        artifact = inspect_godzilla_triattention_file(plan.output)
        return {
            "output": str(plan.output),
            "reused": True,
            "artifact": artifact,
            "stdout": "",
            "stderr": "",
        }
    existed = plan.output.is_file()
    environment: Mapping[str, str] = {**os.environ, **dict(plan.environment)}
    run_cwd = (
        plan.domvox_calibrator.parent
        if plan.mode == "domvox" and plan.domvox_calibrator is not None
        else plan.checkout
    )
    result = runner(
        list(plan.command),
        cwd=run_cwd,
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
        if plan.mode == "domvox":
            if plan.official_stats is None or not plan.official_stats.is_file():
                raise RuntimeError(
                    f"domvox calibration command did not create {plan.official_stats}"
                )
            if not plan.hf_model:
                raise RuntimeError("domvox conversion requires a matching Hugging Face model")
            model_metadata = load_huggingface_model_metadata(plan.hf_model)
            display_name = (
                plan.hf_model.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]
            )
            conversion = convert_domvox_triattention_stats(
                plan.official_stats,
                plan.output,
                model_name=display_name,
                num_layers=int(model_metadata["num_layers"]),
                num_attention_heads=int(model_metadata["num_attention_heads"]),
                num_key_value_heads=int(model_metadata["num_key_value_heads"]),
                rope_theta=float(model_metadata["rope_theta"]),
                expected_head_dim=int(model_metadata["head_dim"]),
                accept_lossy=plan.domvox_accept_lossy,
            )
            artifact = inspect_godzilla_triattention_file(plan.output)
            return {
                "output": str(plan.output),
                "reused": existed,
                "artifact": artifact,
                "domvox_stats": str(plan.official_stats),
                "conversion": conversion,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
            }
        raise RuntimeError(f"Godzilla calibration command did not create {plan.output}")
    try:
        artifact = inspect_godzilla_triattention_file(plan.output)
    except ValueError as exc:
        raise RuntimeError(f"Godzilla calibration command created an invalid artifact: {exc}") from exc
    return {
        "output": str(plan.output),
        "reused": existed,
        "artifact": artifact,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }
