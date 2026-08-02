# SPDX-License-Identifier: MIT
"""Official TriAttention calibration and Godzilla v1 format conversion.

The official WeianMao/triattention calibrator writes a PyTorch ``.pt`` payload.
Godzilla's patched llama.cpp loader expects a different ``TRIA`` binary layout.
This module validates and converts between those formats without invoking
``llama-cli``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .._paths import lexical_absolute_path
from ..tokenizer_backends import gigatoken_version_is_reviewed


GODZILLA_TRIATTENTION_MAGIC = 0x54524941
GODZILLA_TRIATTENTION_VERSION = 1
_HEADER = struct.Struct("<6Id3I")
_U32 = struct.Struct("<I")
_HEAD_INDEX = struct.Struct("<II")
_DOMVOX_HEADER = struct.Struct("<7I2f")
_DOMVOX_HEADER_SIZE = 64
_DOMVOX_VERSION = 2
_DOMVOX_MAX_BYTES = 512 * 1024 * 1024
MAX_CALIBRATION_TOKENS = 200_000
LONG_CALIBRATION_THRESHOLD = 32_768
_STAT_KEY = re.compile(r"layer(\d+)_head(\d+)")
_OFFICIAL_CALIBRATOR_MARKERS = (
    "AutoModelForCausalLM",
    "AutoTokenizer",
    "--max-length",
    "--attn-implementation",
    "q_mean_real",
    "q_mean_imag",
    "q_abs_mean",
)
_DOMVOX_CALIBRATOR_MARKERS = (
    "--model",
    "--input",
    "--output",
    "--max-length",
    "--device",
    "TRIA",
)


def inspect_calibration_python(
    python: str | Path,
    *,
    device: str = "cuda",
    tokenizer_backend: str = "transformers",
    runner=subprocess.run,
) -> dict[str, object]:
    """Verify the interpreter used for official calibration before model loading."""
    interpreter = lexical_absolute_path(python)
    normalized_device = device.strip().lower()
    normalized_tokenizer = tokenizer_backend.strip().lower()
    issues: list[str] = []
    if normalized_tokenizer not in {"transformers", "gigatoken"}:
        issues.append("Tokenizer backend must be 'transformers' or 'gigatoken'")
        return {"python": str(interpreter), "valid": False, "report": None, "issues": issues}
    if not interpreter.is_file():
        issues.append(f"Calibration Python was not found: {interpreter}")
        return {"python": str(interpreter), "valid": False, "report": None, "issues": issues}
    tokenizer_import = (
        "import gigatoken, importlib.metadata\n"
        "gigatoken_version = importlib.metadata.version('gigatoken')\n"
        if normalized_tokenizer == "gigatoken"
        else "gigatoken_version = None\n"
    )
    script = (
        "import json\n"
        "import accelerate, torch, transformers\n"
        f"{tokenizer_import}"
        "report = {"
        "'torch': torch.__version__, "
        "'torch_cuda': torch.version.cuda, "
        "'cuda_available': torch.cuda.is_available(), "
        "'transformers': transformers.__version__, "
        "'accelerate': accelerate.__version__, "
        "'gigatoken': gigatoken_version"
        "}\n"
        "if report['cuda_available']:\n"
        "    try:\n"
        "        device = torch.cuda.current_device()\n"
        "        free_bytes, total_bytes = torch.cuda.mem_get_info(device)\n"
        "        report.update({'cuda_device': torch.cuda.get_device_name(device), "
        "'cuda_device_index': device, 'cuda_free_memory_bytes': free_bytes, "
        "'cuda_total_memory_bytes': total_bytes})\n"
        "    except Exception as exc:\n"
        "        report['cuda_memory_error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(report, sort_keys=True))\n"
    )
    try:
        result = runner(
            [str(interpreter), "-I", "-c", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        issues.append(f"Calibration dependency check could not run: {exc}")
        return {"python": str(interpreter), "valid": False, "report": None, "issues": issues}
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        dependencies = "torch, transformers, and accelerate"
        if normalized_tokenizer == "gigatoken":
            dependencies += ", plus the reviewed Gigatoken backend"
        issues.append(
            f"Calibration Python is missing or cannot import {dependencies}"
            + (f": {details}" if details else "")
        )
        return {"python": str(interpreter), "valid": False, "report": None, "issues": issues}
    output_lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    try:
        report = json.loads(output_lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        issues.append(f"Calibration dependency check returned invalid output: {exc}")
        return {"python": str(interpreter), "valid": False, "report": None, "issues": issues}
    if not isinstance(report, dict):
        issues.append("Calibration dependency check returned an unexpected result")
    elif normalized_tokenizer == "gigatoken" and not gigatoken_version_is_reviewed(
        report.get("gigatoken")
    ):
        issues.append(
            "Gigatoken must use the reviewed 0.10.x compatibility API; found "
            f"{report.get('gigatoken') or 'no importable version'}"
        )
    elif normalized_device == "cuda":
        if not report.get("torch_cuda"):
            issues.append("Calibration Python has a CPU-only PyTorch build")
        elif report.get("cuda_available") is not True:
            issues.append("Calibration Python cannot access CUDA")
    return {
        "python": str(interpreter),
        "valid": not issues,
        "report": report if isinstance(report, dict) else None,
        "issues": issues,
    }


def inspect_official_triattention_checkout(path: str | Path) -> dict[str, object]:
    """Inspect an official-style TriAttention checkout without importing it."""
    root = Path(path).expanduser().resolve()
    calibrator = root / "scripts" / "calibrate.py"
    markers = {
        "scripts/calibrate.py": calibrator.is_file(),
        "triattention": (root / "triattention").is_dir(),
        "calibration_docs": (root / "docs" / "calibration.md").is_file(),
    }
    issues = [f"Missing TriAttention marker: {name}" for name, found in markers.items() if not found]
    if calibrator.is_file():
        script_inspection = inspect_official_triattention_calibrator(calibrator)
        issues.extend(str(item) for item in script_inspection["issues"])
    return {
        "path": str(root),
        "valid": root.is_dir() and not issues,
        "kind": "triattention",
        "markers": markers,
        "calibrator": str(calibrator) if calibrator.is_file() else None,
        "issues": issues,
    }


def inspect_official_triattention_calibrator(path: str | Path) -> dict[str, object]:
    """Check that a script exposes the official calibration payload and CLI."""
    calibrator = Path(path).expanduser().resolve()
    issues: list[str] = []
    if not calibrator.is_file():
        issues.append(f"Official TriAttention calibrator not found: {calibrator}")
        text = ""
    else:
        try:
            if calibrator.stat().st_size > 2_000_000:
                issues.append(f"Calibration script is unexpectedly large: {calibrator}")
                text = ""
            else:
                text = calibrator.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            issues.append(f"Could not inspect calibration script: {exc}")
            text = ""
    missing = [marker for marker in _OFFICIAL_CALIBRATOR_MARKERS if marker not in text]
    if missing:
        issues.append(
            "Script does not match the official TriAttention calibration interface; missing "
            + ", ".join(missing)
        )
    return {
        "path": str(calibrator),
        "valid": not issues,
        "issues": issues,
    }


def inspect_domvox_triattention_calibrator(path: str | Path) -> dict[str, object]:
    """Check a domvox calibrator without importing or executing it."""
    calibrator = Path(path).expanduser().resolve()
    issues: list[str] = []
    if not calibrator.is_file():
        issues.append(f"domvox TriAttention calibrator not found: {calibrator}")
        text = ""
    else:
        try:
            if calibrator.stat().st_size > 2_000_000:
                issues.append(f"domvox calibration script is unexpectedly large: {calibrator}")
                text = ""
            else:
                text = calibrator.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            issues.append(f"Could not inspect domvox calibration script: {exc}")
            text = ""
    missing = [marker for marker in _DOMVOX_CALIBRATOR_MARKERS if marker not in text]
    if missing:
        issues.append(
            "Script does not match the reviewed domvox calibration interface; missing "
            + ", ".join(missing)
        )
    return {
        "path": str(calibrator),
        "valid": not issues,
        "issues": issues,
    }


def inspect_domvox_triattention_checkout(path: str | Path) -> dict[str, object]:
    """Inspect a domvox/triattention-ggml checkout without executing it."""
    root = Path(path).expanduser().resolve()
    calibrator = root / "triattention_calibrate.py"
    markers = {
        "triattention_calibrate.py": calibrator.is_file(),
        "triattention_common.py": (root / "triattention_common.py").is_file(),
        "TRIA_FORMAT.md": (root / "TRIA_FORMAT.md").is_file(),
    }
    issues = [f"Missing domvox marker: {name}" for name, present in markers.items() if not present]
    if calibrator.is_file():
        script_inspection = inspect_domvox_triattention_calibrator(calibrator)
        issues.extend(str(item) for item in script_inspection["issues"])
    return {
        "path": str(root),
        "valid": root.is_dir() and not issues,
        "kind": "domvox_triattention",
        "markers": markers,
        "calibrator": str(calibrator) if calibrator.is_file() else None,
        "issues": issues,
        "status": "experimental" if not issues else "invalid_source",
        "notes": [
            "domvox TRIA v2 statistics require an explicit experimental conversion to Godzilla v1.",
            "Layer budget scales, attention scale, and domvox RoPE assumptions are not preserved by the Godzilla v1 format.",
        ],
    }


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _validate_calibration_length(value: Any, *, allow_long: bool) -> int:
    length = _require_positive_int(value, "max_length")
    if not 128 <= length <= MAX_CALIBRATION_TOKENS:
        raise ValueError(f"max_length must be 128 to {MAX_CALIBRATION_TOKENS}")
    if length > LONG_CALIBRATION_THRESHOLD and not allow_long:
        raise ValueError(
            f"max_length above {LONG_CALIBRATION_THRESHOLD} requires explicit long-calibration acknowledgement"
        )
    return length


def _load_official_payload(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - package dependency
        raise RuntimeError("PyTorch is required to read official TriAttention statistics") from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - protects unsupported old torch
        raise RuntimeError("PyTorch with weights_only loading support is required") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Official TriAttention stats must contain a mapping payload")
    return payload


def _float_vector(value: Any, *, key: str, field: str, expected: int):
    import torch

    try:
        tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu().flatten().contiguous()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{key}.{field} is not a numeric vector") from exc
    if tensor.numel() != expected:
        raise ValueError(
            f"{key}.{field} has {tensor.numel()} values; expected {expected}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{key}.{field} contains NaN or infinite values")
    return tensor


def _parse_stats(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[tuple[int, int]]]:
    metadata = payload.get("metadata")
    stats = payload.get("stats")
    if not isinstance(metadata, Mapping) or not isinstance(stats, Mapping) or not stats:
        raise ValueError("Expected official payload with non-empty metadata and stats mappings")

    parsed_keys: dict[tuple[int, int], str] = {}
    for raw_key in stats:
        if not isinstance(raw_key, str):
            raise ValueError("TriAttention stat keys must be strings")
        match = _STAT_KEY.fullmatch(raw_key)
        if match is None:
            raise ValueError(f"Unexpected TriAttention stat key: {raw_key}")
        pair = (int(match.group(1)), int(match.group(2)))
        if pair in parsed_keys:
            raise ValueError(f"Duplicate TriAttention layer/head entry: {pair}")
        parsed_keys[pair] = raw_key

    raw_sampled = metadata.get("sampled_heads")
    if raw_sampled is None:
        sampled = sorted(parsed_keys)
    elif not isinstance(raw_sampled, Sequence) or isinstance(raw_sampled, (str, bytes)):
        raise ValueError("metadata.sampled_heads must be a sequence of [layer, head] pairs")
    else:
        sampled = []
        for item in raw_sampled:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                raise ValueError("metadata.sampled_heads contains an invalid pair")
            layer, head = int(item[0]), int(item[1])
            if layer < 0 or head < 0:
                raise ValueError("metadata.sampled_heads indices must be non-negative")
            sampled.append((layer, head))
    if not sampled or len(sampled) != len(set(sampled)):
        raise ValueError("metadata.sampled_heads must contain unique entries")
    if set(sampled) != set(parsed_keys):
        raise ValueError("metadata.sampled_heads does not match the stats entries")
    return stats, sampled


def _stat_key(stats: Mapping[str, Any], layer: int, head: int) -> str:
    expected_pair = (layer, head)
    for raw_key in stats:
        match = _STAT_KEY.fullmatch(str(raw_key))
        if match and (int(match.group(1)), int(match.group(2))) == expected_pair:
            return str(raw_key)
    raise ValueError(f"Missing stats for layer {layer}, head {head}")


def _rope_style_value(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "half":
            return 0
        if normalized == "interleaved":
            return 1
    elif value in {0, 1}:
        return int(value)
    raise ValueError("RoPE style must be 'half' or 'interleaved'")


def convert_official_triattention_stats(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str,
    num_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    rope_theta: float,
    rope_style: str | int | None = None,
    expected_head_dim: int | None = None,
) -> dict[str, object]:
    """Convert official ``.pt`` stats to Godzilla's v1 ``.triattention`` format."""
    import torch

    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Official TriAttention stats not found: {source}")
    if output.suffix.lower() != ".triattention":
        raise ValueError("Godzilla output must end in .triattention")

    payload = _load_official_payload(source)
    metadata = payload.get("metadata")
    assert isinstance(metadata, Mapping)
    stats, sampled = _parse_stats(payload)
    head_dim = _require_positive_int(metadata.get("head_dim"), "metadata.head_dim")
    if head_dim % 2:
        raise ValueError("metadata.head_dim must be even")
    if expected_head_dim is not None and head_dim != _require_positive_int(
        expected_head_dim, "expected_head_dim"
    ):
        raise ValueError("Official stats head_dim does not match the model configuration")
    freq_count = head_dim // 2
    layers = _require_positive_int(num_layers, "num_layers")
    attention_heads = _require_positive_int(num_attention_heads, "num_attention_heads")
    kv_heads = _require_positive_int(num_key_value_heads, "num_key_value_heads")
    if attention_heads % kv_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if any(layer >= layers or head >= attention_heads for layer, head in sampled):
        raise ValueError("Official stats contain a layer/head outside the model configuration")
    try:
        theta = float(rope_theta)
    except (TypeError, ValueError) as exc:
        raise ValueError("rope_theta must be a positive finite number") from exc
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("rope_theta must be a positive finite number")
    style = _rope_style_value(metadata.get("rope_style") if rope_style is None else rope_style)

    encoded_name = model_name.strip().encode("utf-8") + b"\0"
    if len(encoded_name) <= 1 or len(encoded_name) > 255:
        raise ValueError("model_name must encode to 1-254 UTF-8 bytes")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(
                _HEADER.pack(
                    GODZILLA_TRIATTENTION_MAGIC,
                    GODZILLA_TRIATTENTION_VERSION,
                    head_dim,
                    layers,
                    attention_heads,
                    kv_heads,
                    theta,
                    style,
                    len(sampled),
                    freq_count,
                )
            )
            handle.write(_U32.pack(len(encoded_name)))
            handle.write(encoded_name)
            vector_format = struct.Struct(f"<{freq_count}f")
            for layer, head in sampled:
                key = _stat_key(stats, layer, head)
                entry = stats[key]
                if not isinstance(entry, Mapping):
                    raise ValueError(f"{key} must contain a stats mapping")
                q_mean_real = _float_vector(
                    entry.get("q_mean_real"), key=key, field="q_mean_real", expected=freq_count
                )
                q_mean_imag = _float_vector(
                    entry.get("q_mean_imag"), key=key, field="q_mean_imag", expected=freq_count
                )
                q_abs_mean = _float_vector(
                    entry.get("q_abs_mean"), key=key, field="q_abs_mean", expected=freq_count
                )
                if bool((q_abs_mean < 0).any()):
                    raise ValueError(f"{key}.q_abs_mean contains negative values")
                mean_magnitude = torch.hypot(q_mean_real, q_mean_imag)
                ratio = torch.where(q_abs_mean > 0, mean_magnitude / q_abs_mean, 0.0)
                ratio = ratio.clamp(0.0, 1.0)

                handle.write(_HEAD_INDEX.pack(layer, head))
                for vector in (q_mean_real, q_mean_imag, q_abs_mean, ratio):
                    handle.write(vector_format.pack(*vector.tolist()))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass

    report = inspect_godzilla_triattention_file(output)
    return {"input": str(source), "output": str(output), **report}


def _read_exact(handle, count: int, label: str) -> bytes:
    value = handle.read(count)
    if len(value) != count:
        raise ValueError(f"Truncated Godzilla TriAttention file while reading {label}")
    return value


def _domvox_vector(handle, vector: struct.Struct, *, label: str) -> tuple[float, ...]:
    values = vector.unpack(_read_exact(handle, vector.size, label))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite values in domvox {label}")
    return values


def inspect_domvox_triattention_file(path: str | Path) -> dict[str, object]:
    """Strictly inspect a domvox TRIA v2 calibration file."""
    calibration = Path(path).expanduser().resolve()
    if not calibration.is_file():
        raise ValueError(f"domvox TriAttention calibration not found: {calibration}")
    file_size = calibration.stat().st_size
    if file_size > _DOMVOX_MAX_BYTES:
        raise ValueError(
            f"domvox TriAttention calibration is larger than the safe limit of {_DOMVOX_MAX_BYTES} bytes"
        )
    with calibration.open("rb") as handle:
        header = _read_exact(handle, _DOMVOX_HEADER_SIZE, "domvox header")
        (
            magic,
            version,
            num_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            freq_count,
            rope_theta,
            attention_scale,
        ) = _DOMVOX_HEADER.unpack(header[: _DOMVOX_HEADER.size])
        if magic != GODZILLA_TRIATTENTION_MAGIC or version != _DOMVOX_VERSION:
            raise ValueError("File is not a supported domvox TRIA v2 calibration")
        if not 0 < num_layers <= 10_000 or not 0 < num_attention_heads <= 10_000:
            raise ValueError("Invalid domvox layer or attention-head count")
        if not 0 < num_key_value_heads <= num_attention_heads:
            raise ValueError("Invalid domvox KV-head count")
        if num_attention_heads % num_key_value_heads:
            raise ValueError("domvox attention-head count is not divisible by KV-head count")
        if not 0 < head_dim <= 16_384 or head_dim % 2 or freq_count != head_dim // 2:
            raise ValueError("Invalid domvox head dimension or frequency count")
        if not math.isfinite(rope_theta) or rope_theta <= 0:
            raise ValueError("Invalid domvox RoPE theta")
        if not math.isfinite(attention_scale) or attention_scale <= 0:
            raise ValueError("Invalid domvox attention scale")
        expected_size = (
            _DOMVOX_HEADER_SIZE
            + num_layers * 4
            + num_layers * num_attention_heads * (4 * freq_count * 4)
        )
        if expected_size != file_size:
            raise ValueError(
                f"domvox TRIA v2 size mismatch: expected {expected_size} bytes, found {file_size}"
            )
        scales = struct.unpack(
            f"<{num_layers}f", _read_exact(handle, num_layers * 4, "layer budget scales")
        )
        if not all(math.isfinite(value) and value >= 0 for value in scales):
            raise ValueError("domvox layer budget scales contain invalid values")
        vector = struct.Struct(f"<{freq_count}f")
        for layer in range(num_layers):
            for head in range(num_attention_heads):
                _domvox_vector(handle, vector, label=f"q_mean_real layer {layer} head {head}")
                _domvox_vector(handle, vector, label=f"q_mean_imag layer {layer} head {head}")
                q_abs_mean = _domvox_vector(
                    handle, vector, label=f"q_abs_mean layer {layer} head {head}"
                )
                mrl = _domvox_vector(handle, vector, label=f"mrl layer {layer} head {head}")
                if any(value < 0 for value in q_abs_mean):
                    raise ValueError(f"Negative domvox q_abs_mean at layer {layer} head {head}")
                if any(value < 0 or value > 1 for value in mrl):
                    raise ValueError(f"Out-of-range domvox mrl at layer {layer} head {head}")
    return {
        "format": "domvox-tria-v2",
        "version": version,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "rope_theta": rope_theta,
        "attention_scale": attention_scale,
        "freq_count": freq_count,
        "layer_budget_scale_min": min(scales),
        "layer_budget_scale_max": max(scales),
        "size_bytes": file_size,
    }


def convert_domvox_triattention_stats(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str,
    num_layers: int | None = None,
    num_attention_heads: int | None = None,
    num_key_value_heads: int | None = None,
    rope_theta: float | None = None,
    rope_style: str | int = "half",
    expected_head_dim: int | None = None,
    accept_lossy: bool = False,
) -> dict[str, object]:
    """Convert domvox TRIA v2 to Godzilla v1 with explicit loss acknowledgement.

    Godzilla v1 has no fields for domvox layer budget scales or attention scale,
    so this adapter is deliberately opt-in and reports the dropped fields.
    """
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not accept_lossy:
        raise ValueError(
            "domvox-to-Godzilla conversion is lossy; set accept_lossy=True only after reviewing compatibility"
        )
    inspection = inspect_domvox_triattention_file(source)
    actual_layers = int(inspection["num_layers"])
    actual_heads = int(inspection["num_attention_heads"])
    actual_kv_heads = int(inspection["num_key_value_heads"])
    actual_head_dim = int(inspection["head_dim"])
    actual_theta = float(inspection["rope_theta"])
    for label, expected, actual in (
        ("num_layers", num_layers, actual_layers),
        ("num_attention_heads", num_attention_heads, actual_heads),
        ("num_key_value_heads", num_key_value_heads, actual_kv_heads),
        ("head_dim", expected_head_dim, actual_head_dim),
    ):
        if expected is not None and int(expected) != actual:
            raise ValueError(f"domvox {label} does not match the model configuration")
    selected_theta = actual_theta if rope_theta is None else float(rope_theta)
    if not math.isfinite(selected_theta) or selected_theta <= 0:
        raise ValueError("rope_theta must be a positive finite number")
    if not math.isclose(selected_theta, actual_theta, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("domvox RoPE theta does not match the model configuration")
    style = _rope_style_value(rope_style)
    encoded_name = model_name.strip().encode("utf-8") + b"\0"
    if len(encoded_name) <= 1 or len(encoded_name) > 255:
        raise ValueError("model_name must encode to 1-254 UTF-8 bytes")
    if output.suffix.lower() != ".triattention":
        raise ValueError("Godzilla output must end in .triattention")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with source.open("rb") as source_handle, tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(
                _HEADER.pack(
                    GODZILLA_TRIATTENTION_MAGIC,
                    GODZILLA_TRIATTENTION_VERSION,
                    actual_head_dim,
                    actual_layers,
                    actual_heads,
                    actual_kv_heads,
                    selected_theta,
                    style,
                    actual_layers * actual_heads,
                    actual_head_dim // 2,
                )
            )
            handle.write(_U32.pack(len(encoded_name)))
            handle.write(encoded_name)
            source_handle.seek(_DOMVOX_HEADER_SIZE + actual_layers * 4)
            vector = struct.Struct(f"<{actual_head_dim // 2}f")
            for layer in range(actual_layers):
                for head in range(actual_heads):
                    q_mean_real = _domvox_vector(
                        source_handle, vector, label=f"q_mean_real layer {layer} head {head}"
                    )
                    q_mean_imag = _domvox_vector(
                        source_handle, vector, label=f"q_mean_imag layer {layer} head {head}"
                    )
                    q_abs_mean = _domvox_vector(
                        source_handle, vector, label=f"q_abs_mean layer {layer} head {head}"
                    )
                    mrl = _domvox_vector(
                        source_handle, vector, label=f"mrl layer {layer} head {head}"
                    )
                    if any(value < 0 for value in q_abs_mean):
                        raise ValueError(f"Negative domvox q_abs_mean at layer {layer} head {head}")
                    if any(value < 0 or value > 1 for value in mrl):
                        raise ValueError(f"Out-of-range domvox mrl at layer {layer} head {head}")
                    handle.write(_HEAD_INDEX.pack(layer, head))
                    for values in (q_mean_real, q_mean_imag, q_abs_mean, mrl):
                        handle.write(vector.pack(*values))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    report = inspect_godzilla_triattention_file(output)
    return {
        "input": str(source),
        "output": str(output),
        "source_format": "domvox-tria-v2",
        "lossy": True,
        "dropped_fields": ["layer_budget_scales", "attention_scale"],
        **report,
    }


def inspect_godzilla_triattention_file(path: str | Path) -> dict[str, object]:
    """Read and strictly validate a Godzilla v1 calibration artifact."""
    calibration = Path(path).expanduser().resolve()
    if not calibration.is_file():
        raise ValueError(f"Godzilla TriAttention calibration not found: {calibration}")
    file_size = calibration.stat().st_size
    with calibration.open("rb") as handle:
        values = _HEADER.unpack(_read_exact(handle, _HEADER.size, "header"))
        (
            magic,
            version,
            head_dim,
            num_layers,
            num_attention_heads,
            num_key_value_heads,
            rope_theta,
            rope_style,
            sampled_count,
            freq_count,
        ) = values
        if magic != GODZILLA_TRIATTENTION_MAGIC or version != GODZILLA_TRIATTENTION_VERSION:
            raise ValueError("File is not a supported Godzilla v1 TriAttention calibration")
        if not 0 < head_dim <= 16_384 or head_dim % 2 or freq_count != head_dim // 2:
            raise ValueError("Invalid head dimension or frequency count")
        if not 0 < num_layers <= 10_000 or not 0 < num_attention_heads <= 10_000:
            raise ValueError("Invalid layer or attention-head count")
        if not 0 < num_key_value_heads <= num_attention_heads:
            raise ValueError("Invalid KV-head count")
        if num_attention_heads % num_key_value_heads:
            raise ValueError("Attention-head count is not divisible by KV-head count")
        if not math.isfinite(rope_theta) or rope_theta <= 0 or rope_style not in {0, 1}:
            raise ValueError("Invalid RoPE metadata")
        if not 0 < sampled_count <= num_layers * num_attention_heads:
            raise ValueError("Invalid sampled-head count")

        name_length = _U32.unpack(_read_exact(handle, _U32.size, "model name length"))[0]
        if not 0 < name_length <= 255:
            raise ValueError("Invalid model name length")
        encoded_name = _read_exact(handle, name_length, "model name")
        if encoded_name[-1:] != b"\0":
            raise ValueError("Godzilla model name is not null terminated")
        try:
            model_name = encoded_name[:-1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Godzilla model name is not valid UTF-8") from exc
        if not model_name:
            raise ValueError("Godzilla model name is empty")

        vector = struct.Struct(f"<{freq_count}f")
        expected_size = (
            _HEADER.size
            + _U32.size
            + name_length
            + sampled_count * (_HEAD_INDEX.size + 4 * vector.size)
        )
        if file_size < expected_size:
            raise ValueError("Truncated Godzilla TriAttention file")
        if file_size > expected_size:
            raise ValueError("Godzilla TriAttention file has unexpected trailing data")
        seen: set[tuple[int, int]] = set()
        for index in range(sampled_count):
            layer, head = _HEAD_INDEX.unpack(
                _read_exact(handle, _HEAD_INDEX.size, f"head index {index}")
            )
            if layer >= num_layers or head >= num_attention_heads or (layer, head) in seen:
                raise ValueError(f"Invalid or duplicate sampled head at entry {index}")
            seen.add((layer, head))
            for field in ("q_mean_real", "q_mean_imag", "q_abs_mean", "r_f"):
                values = vector.unpack(_read_exact(handle, vector.size, f"{field} entry {index}"))
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Non-finite values in {field} entry {index}")
                if field == "q_abs_mean" and any(value < 0 for value in values):
                    raise ValueError(f"Negative values in {field} entry {index}")
                if field == "r_f" and any(value < 0 or value > 1 for value in values):
                    raise ValueError(f"Out-of-range values in {field} entry {index}")

    return {
        "format": "godzilla-triattention-v1",
        "version": version,
        "model_name": model_name,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "rope_theta": rope_theta,
        "rope_style": "interleaved" if rope_style else "half",
        "sampled_heads": sampled_count,
        "freq_count": freq_count,
        "size_bytes": file_size,
    }


def load_huggingface_model_metadata(model: str) -> dict[str, object]:
    """Load only the matching Hugging Face config needed by the converter."""
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("transformers is required to load model calibration metadata") from exc
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    num_layers = _require_positive_int(
        getattr(text_config, "num_hidden_layers", None), "config.num_hidden_layers"
    )
    num_attention_heads = _require_positive_int(
        getattr(text_config, "num_attention_heads", None), "config.num_attention_heads"
    )
    num_key_value_heads = _require_positive_int(
        getattr(text_config, "num_key_value_heads", num_attention_heads),
        "config.num_key_value_heads",
    )
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is None:
        hidden_size = _require_positive_int(
            getattr(text_config, "hidden_size", None), "config.hidden_size"
        )
        if hidden_size % num_attention_heads:
            raise ValueError("config.hidden_size is not divisible by num_attention_heads")
        head_dim = hidden_size // num_attention_heads
    rope_theta = getattr(text_config, "rope_theta", None)
    if rope_theta is None:
        raise ValueError("The Hugging Face config does not declare rope_theta")
    return {
        "head_dim": _require_positive_int(head_dim, "config.head_dim"),
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "rope_theta": float(rope_theta),
    }


def calibrate_official_triattention_for_godzilla(
    *,
    calibrator: str | Path,
    model: str,
    input_path: str | Path,
    output_path: str | Path,
    stats_output_path: str | Path | None = None,
    max_length: int = 2048,
    allow_long_calibration: bool = False,
    device: str = "cuda",
    attention_implementation: str = "sdpa",
    tokenizer_backend: str = "transformers",
    runner=subprocess.run,
) -> dict[str, object]:
    """Run the official calibrator, then convert and verify its output."""
    max_length = _validate_calibration_length(max_length, allow_long=allow_long_calibration)
    script = Path(calibrator).expanduser().resolve()
    script_report = inspect_official_triattention_calibrator(script)
    if not script_report["valid"]:
        raise ValueError("; ".join(str(item) for item in script_report["issues"]))
    calibration_input = Path(input_path).expanduser().resolve()
    if not calibration_input.is_file() or calibration_input.stat().st_size == 0:
        raise ValueError(f"Calibration input must be a non-empty plain-text file: {calibration_input}")
    output = Path(output_path).expanduser().resolve()
    stats_output = (
        Path(stats_output_path).expanduser().resolve()
        if stats_output_path is not None
        else output.with_suffix(".official.pt")
    )
    if stats_output == output:
        raise ValueError("Intermediate .pt stats and final .triattention output must differ")
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    normalized_tokenizer = tokenizer_backend.strip().lower()
    if normalized_tokenizer not in {"transformers", "gigatoken"}:
        raise ValueError("Tokenizer backend must be 'transformers' or 'gigatoken'")
    executable = [sys.executable, str(script)]
    if normalized_tokenizer == "gigatoken":
        wrapper = Path(__file__).with_name("gigatoken_runner.py")
        executable = [sys.executable, str(wrapper), "--calibrator", str(script)]
    command = [
        *executable,
        "--model",
        model,
        "--input",
        str(calibration_input),
        "--output",
        str(stats_output),
        "--max-length",
        str(max_length),
        "--device",
        device,
        "--attn-implementation",
        attention_implementation,
    ]
    result = runner(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Official TriAttention calibration failed with exit code {result.returncode}")
    if not stats_output.is_file():
        raise RuntimeError(f"Official calibrator did not create {stats_output}")
    model_metadata = load_huggingface_model_metadata(model)
    payload = _load_official_payload(stats_output)
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Official calibrator output is missing metadata")
    if _require_positive_int(metadata.get("head_dim"), "metadata.head_dim") != model_metadata["head_dim"]:
        raise ValueError("Official stats head_dim does not match the Hugging Face model config")
    display_name = model.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]
    report = convert_official_triattention_stats(
        stats_output,
        output,
        model_name=display_name,
        num_layers=int(model_metadata["num_layers"]),
        num_attention_heads=int(model_metadata["num_attention_heads"]),
        num_key_value_heads=int(model_metadata["num_key_value_heads"]),
        rope_theta=float(model_metadata["rope_theta"]),
        expected_head_dim=int(model_metadata["head_dim"]),
    )
    return {
        "command": command,
        "official_stats": str(stats_output),
        "tokenizer_backend": normalized_tokenizer,
        **report,
    }


def calibrate_domvox_triattention_for_godzilla(
    *,
    calibrator: str | Path,
    python: str | Path,
    model: str,
    input_path: str | Path,
    output_path: str | Path,
    stats_output_path: str | Path | None = None,
    max_length: int = 2048,
    allow_long_calibration: bool = False,
    device: str = "cuda",
    rope_style: str | int = "half",
    accept_lossy: bool = False,
    runner=subprocess.run,
) -> dict[str, object]:
    """Run domvox calibration and explicitly adapt TRIA v2 to Godzilla v1."""
    max_length = _validate_calibration_length(max_length, allow_long=allow_long_calibration)
    script = Path(calibrator).expanduser().resolve()
    script_report = inspect_domvox_triattention_calibrator(script)
    if not script_report["valid"]:
        raise ValueError("; ".join(str(item) for item in script_report["issues"]))
    python_path = lexical_absolute_path(python)
    if not python_path.is_file():
        raise ValueError(f"domvox calibration Python was not found: {python_path}")
    calibration_input = Path(input_path).expanduser().resolve()
    if not calibration_input.is_file() or calibration_input.stat().st_size == 0:
        raise ValueError(f"Calibration input must be a non-empty plain-text file: {calibration_input}")
    output = Path(output_path).expanduser().resolve()
    stats_output = (
        Path(stats_output_path).expanduser().resolve()
        if stats_output_path is not None
        else output.with_suffix(".domvox.bin")
    )
    if stats_output == output:
        raise ValueError("Intermediate domvox stats and final .triattention output must differ")
    command = [
        str(python_path),
        str(script),
        "--model",
        model,
        "--input",
        str(calibration_input),
        "--output",
        str(stats_output),
        "--max-length",
        str(max_length),
        "--device",
        device,
    ]
    result = runner(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"domvox TriAttention calibration failed with exit code {result.returncode}")
    if not stats_output.is_file():
        raise RuntimeError(f"domvox calibrator did not create {stats_output}")
    model_metadata = load_huggingface_model_metadata(model)
    display_name = model.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]
    report = convert_domvox_triattention_stats(
        stats_output,
        output,
        model_name=display_name,
        num_layers=int(model_metadata["num_layers"]),
        num_attention_heads=int(model_metadata["num_attention_heads"]),
        num_key_value_heads=int(model_metadata["num_key_value_heads"]),
        rope_theta=float(model_metadata["rope_theta"]),
        rope_style=rope_style,
        expected_head_dim=int(model_metadata["head_dim"]),
        accept_lossy=accept_lossy,
    )
    return {"command": command, "domvox_stats": str(stats_output), **report}


def _add_model_shape_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help="Matching Hugging Face model ID or local directory")
    parser.add_argument("--model-name", help="Display name stored in the Godzilla artifact")
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-attention-heads", type=int)
    parser.add_argument("--num-kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--rope-theta", type=float)
    parser.add_argument("--rope-style", choices=("half", "interleaved"))


def _conversion_metadata(args: argparse.Namespace) -> dict[str, object]:
    loaded = load_huggingface_model_metadata(args.model) if args.model else {}
    fields = {
        "num_layers": args.num_layers or loaded.get("num_layers"),
        "num_attention_heads": args.num_attention_heads or loaded.get("num_attention_heads"),
        "num_key_value_heads": args.num_kv_heads or loaded.get("num_key_value_heads"),
        "rope_theta": args.rope_theta or loaded.get("rope_theta"),
        "expected_head_dim": args.head_dim or loaded.get("head_dim"),
    }
    missing = [key for key, value in fields.items() if value is None]
    if missing:
        raise ValueError("Provide --model or explicit " + ", ".join(missing))
    model_name = args.model_name
    if not model_name and args.model:
        model_name = args.model.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]
    if not model_name:
        raise ValueError("Provide --model-name when --model is omitted")
    return {"model_name": model_name, "rope_style": args.rope_style, **fields}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate or convert official TriAttention stats for Godzilla llama.cpp"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser(
        "calibrate", help="Run official scripts/calibrate.py and convert its .pt output"
    )
    calibrate.add_argument("--calibrator", required=True)
    calibrate.add_argument("--model", required=True)
    calibrate.add_argument("--input", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--stats-output")
    calibrate.add_argument("--max-length", type=int, default=2048)
    calibrate.add_argument(
        "--allow-long-calibration",
        action="store_true",
        help=f"Allow one-shot calibration above {LONG_CALIBRATION_THRESHOLD} tokens (maximum {MAX_CALIBRATION_TOKENS})",
    )
    calibrate.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    calibrate.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    calibrate.add_argument(
        "--tokenizer-backend",
        choices=("transformers", "gigatoken"),
        default="transformers",
        help="Tokenizer used by official calibration; Gigatoken requires exact ID parity",
    )

    domvox = subparsers.add_parser(
        "domvox", help="Run domvox TRIA v2 calibration and adapt it to Godzilla v1"
    )
    domvox.add_argument("--calibrator", required=True)
    domvox.add_argument("--python", required=True)
    domvox.add_argument("--model", required=True)
    domvox.add_argument("--input", required=True)
    domvox.add_argument("--output", required=True)
    domvox.add_argument("--stats-output")
    domvox.add_argument("--max-length", type=int, default=2048)
    domvox.add_argument("--allow-long-calibration", action="store_true")
    domvox.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    domvox.add_argument("--accept-lossy", action="store_true")

    convert = subparsers.add_parser("convert", help="Convert an existing official .pt payload")
    convert.add_argument("input")
    convert.add_argument("output")
    _add_model_shape_arguments(convert)

    inspect = subparsers.add_parser("inspect", help="Validate a Godzilla .triattention file")
    inspect.add_argument("path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "calibrate":
        _validate_calibration_length(args.max_length, allow_long=args.allow_long_calibration)
        report = calibrate_official_triattention_for_godzilla(
            calibrator=args.calibrator,
            model=args.model,
            input_path=args.input,
            output_path=args.output,
            stats_output_path=args.stats_output,
            max_length=args.max_length,
            allow_long_calibration=args.allow_long_calibration,
            device=args.device,
            attention_implementation=args.attn_implementation,
            tokenizer_backend=args.tokenizer_backend,
        )
    elif args.command == "domvox":
        _validate_calibration_length(args.max_length, allow_long=args.allow_long_calibration)
        report = calibrate_domvox_triattention_for_godzilla(
            calibrator=args.calibrator,
            python=args.python,
            model=args.model,
            input_path=args.input,
            output_path=args.output,
            stats_output_path=args.stats_output,
            max_length=args.max_length,
            allow_long_calibration=args.allow_long_calibration,
            device=args.device,
            accept_lossy=args.accept_lossy,
        )
    elif args.command == "convert":
        report = convert_official_triattention_stats(
            args.input, args.output, **_conversion_metadata(args)
        )
    else:
        report = inspect_godzilla_triattention_file(args.path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
