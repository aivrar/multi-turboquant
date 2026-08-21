from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from multi_turboquant.calibration import godzilla_triattention as calibration


def _official_payload(*, sampled_heads: list[list[int]] | None = None) -> dict[str, object]:
    sampled = sampled_heads or [[0, 0], [0, 1], [1, 0], [1, 1]]
    stats = {}
    for layer, head in sampled:
        key = f"layer{layer:02d}_head{head:02d}"
        stats[key] = {
            "q_mean_real": torch.tensor([0.1, 0.2]),
            "q_mean_imag": torch.tensor([0.2, 0.1]),
            "q_abs_mean": torch.tensor([0.5, 0.5]),
        }
    return {
        "metadata": {
            "head_dim": 4,
            "rope_style": "half",
            "sampled_heads": sampled,
        },
        "stats": stats,
    }


def _write_official_script(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(calibration._OFFICIAL_CALIBRATOR_MARKERS),
        encoding="utf-8",
    )


def _write_domvox_stats(path: Path, *, layers: int = 1, heads: int = 2) -> None:
    head_dim = 4
    freq_count = head_dim // 2
    header = struct.pack(
        "<7I2f",
        0x54524941,
        2,
        layers,
        heads,
        1,
        head_dim,
        freq_count,
        10_000.0,
        1.0,
    )
    path.write_bytes(
        header
        + (b"\0" * (64 - len(header)))
        + struct.pack(f"<{layers}f", *([1.0] * layers))
        + b"".join(
            struct.pack(f"<{freq_count}f", *values)
            for _layer in range(layers)
            for _head in range(heads)
            for values in (
                (0.1, 0.2),
                (0.2, 0.1),
                (0.5, 0.5),
                (0.4, 0.5),
            )
        )
    )


def _python_probe_report(
    *,
    torch_cuda: str | None = "12.6",
    cuda_available: bool = True,
    gigatoken: str | None = None,
    module_errors: dict[str, str] | None = None,
    cuda_device_index: int = 0,
) -> str:
    versions = {
        "torch": "2.7.1",
        "transformers": "4.57.6",
        "accelerate": "1.14.0",
        "numpy": "2.2.6",
        "safetensors": "0.6.2",
        "huggingface_hub": "0.35.0",
        "tokenizers": "0.22.0",
        "sentencepiece": "0.2.2",
    }
    if gigatoken is not None:
        versions["gigatoken"] = gigatoken
    errors = module_errors or {}
    modules = {
        name: (
            {"status": "error", "error": errors[name], "traceback": errors[name]}
            if name in errors
            else {"status": "ok", "version": version}
        )
        for name, version in versions.items()
    }
    for name, error in errors.items():
        modules.setdefault(name, {"status": "error", "error": error, "traceback": error})
    return json.dumps(
        {
            **versions,
            "runtime_executable": "/env/bin/python",
            "prefix": "/env",
            "base_prefix": "/base",
            "python_version": "3.11.9",
            "platform": "Linux",
            "modules": modules,
            "torch_cuda": torch_cuda,
            "cuda_available": cuda_available,
            "cuda_device": "Test GPU",
            "cuda_device_index": cuda_device_index,
            "cuda_free_memory_bytes": 24 * 1024**3,
            "cuda_total_memory_bytes": 24 * 1024**3,
        }
    )


def test_convert_official_stats_writes_strict_godzilla_v1(tmp_path: Path):
    source = tmp_path / "official.pt"
    output = tmp_path / "model.triattention"
    torch.save(_official_payload(), source)

    report = calibration.convert_official_triattention_stats(
        source,
        output,
        model_name="example-model",
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_theta=10_000.0,
    )

    assert output.read_bytes()[:4] == b"AIRT"
    assert report == {
        "input": str(source.resolve()),
        "output": str(output.resolve()),
        "format": "godzilla-triattention-v1",
        "version": 1,
        "model_name": "example-model",
        "head_dim": 4,
        "num_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rope_theta": 10_000.0,
        "rope_style": "half",
        "sampled_heads": 4,
        "freq_count": 2,
        "size_bytes": output.stat().st_size,
    }


def test_domvox_inspection_and_lossy_conversion_are_strict(tmp_path: Path):
    source = tmp_path / "domvox.bin"
    output = tmp_path / "model.triattention"
    _write_domvox_stats(source, layers=1, heads=2)

    inspection = calibration.inspect_domvox_triattention_file(source)
    assert inspection["format"] == "domvox-tria-v2"
    assert inspection["num_attention_heads"] == 2

    with pytest.raises(ValueError, match="lossy"):
        calibration.convert_domvox_triattention_stats(source, output, model_name="model")

    report = calibration.convert_domvox_triattention_stats(
        source,
        output,
        model_name="model",
        num_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_theta=10_000.0,
        expected_head_dim=4,
        accept_lossy=True,
    )
    assert report["source_format"] == "domvox-tria-v2"
    assert report["lossy"] is True
    assert report["sampled_heads"] == 2


def test_domvox_conversion_rejects_model_shape_mismatch(tmp_path: Path):
    source = tmp_path / "domvox.bin"
    _write_domvox_stats(source)

    with pytest.raises(ValueError, match="num_layers"):
        calibration.convert_domvox_triattention_stats(
            source,
            tmp_path / "model.triattention",
            model_name="model",
            num_layers=2,
            accept_lossy=True,
        )


def test_domvox_calibration_runs_then_converts(tmp_path: Path, monkeypatch):
    script = tmp_path / "triattention_calibrate.py"
    script.write_text(
        " ".join(calibration._DOMVOX_CALIBRATOR_MARKERS),
        encoding="utf-8",
    )
    (script.parent / "triattention_common.py").write_text("# helper\n", encoding="utf-8")
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")
    calibration_input = tmp_path / "calibration.txt"
    calibration_input.write_text("text", encoding="utf-8")
    stats = tmp_path / "domvox.bin"
    output = tmp_path / "model.triattention"
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        _write_domvox_stats(stats)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        calibration,
        "load_huggingface_model_metadata",
        lambda model, **_kwargs: {
            "head_dim": 4,
            "num_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rope_theta": 10_000.0,
        },
    )
    report = calibration.calibrate_domvox_triattention_for_godzilla(
        calibrator=script,
        python=python,
        model="org/model",
        input_path=calibration_input,
        output_path=output,
        stats_output_path=stats,
        max_length=512,
        device="cpu",
        accept_lossy=True,
        runner=runner,
    )

    assert calls[0][0] == str(python.resolve())
    assert "--max-length" in calls[0]
    assert report["source_format"] == "domvox-tria-v2"


def test_domvox_calibration_wraps_gigatoken_with_exact_parity_guard(tmp_path: Path, monkeypatch):
    script = tmp_path / "triattention_calibrate.py"
    script.write_text(
        " ".join(calibration._DOMVOX_CALIBRATOR_MARKERS),
        encoding="utf-8",
    )
    (script.parent / "triattention_common.py").write_text("# helper\n", encoding="utf-8")

    python = tmp_path / "python"
    python.write_bytes(b"python")
    calibration_input = tmp_path / "calibration.txt"
    calibration_input.write_text("text", encoding="utf-8")
    stats = tmp_path / "domvox.bin"
    output = tmp_path / "model.triattention"
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        _write_domvox_stats(stats)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        calibration,
        "load_huggingface_model_metadata",
        lambda model, **_kwargs: {
            "head_dim": 4,
            "num_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rope_theta": 10_000.0,
        },
    )
    report = calibration.calibrate_domvox_triattention_for_godzilla(
        calibrator=script,
        python=python,
        model="org/model",
        input_path=calibration_input,
        output_path=output,
        stats_output_path=stats,
        max_length=512,
        device="cpu",
        accept_lossy=True,
        tokenizer_backend="gigatoken",
        runner=runner,
    )

    assert Path(calls[0][1]).name == "triattention_runner.py"
    assert calls[0][2:8] == [
        "--kind", "domvox", "--tokenizer-backend", "gigatoken", "--calibrator", str(script.resolve())
    ]
    assert "--attn-implementation" not in calls[0]
    assert report["tokenizer_backend"] == "gigatoken"


def test_convert_rejects_metadata_stats_mismatch(tmp_path: Path):
    source = tmp_path / "official.pt"
    payload = _official_payload()
    payload["metadata"]["sampled_heads"] = [[0, 0]]
    torch.save(payload, source)

    with pytest.raises(ValueError, match="does not match"):
        calibration.convert_official_triattention_stats(
            source,
            tmp_path / "model.triattention",
            model_name="model",
            num_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            rope_theta=10_000.0,
        )


def test_convert_rejects_model_shape_mismatch(tmp_path: Path):
    source = tmp_path / "official.pt"
    torch.save(_official_payload(), source)

    with pytest.raises(ValueError, match="outside the model configuration"):
        calibration.convert_official_triattention_stats(
            source,
            tmp_path / "model.triattention",
            model_name="model",
            num_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            rope_theta=10_000.0,
        )

    with pytest.raises(ValueError, match="head_dim does not match"):
        calibration.convert_official_triattention_stats(
            source,
            tmp_path / "head-mismatch.triattention",
            model_name="model",
            num_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            rope_theta=10_000.0,
            expected_head_dim=8,
        )


def test_inspector_rejects_truncated_or_trailing_data(tmp_path: Path):
    invalid = tmp_path / "invalid.triattention"
    invalid.write_bytes(b"TRIA")
    with pytest.raises(ValueError, match="Truncated"):
        calibration.inspect_godzilla_triattention_file(invalid)

    source = tmp_path / "official.pt"
    valid = tmp_path / "valid.triattention"
    torch.save(_official_payload(sampled_heads=[[0, 0]]), source)
    calibration.convert_official_triattention_stats(
        source,
        valid,
        model_name="model",
        num_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        rope_theta=10_000.0,
    )
    valid.write_bytes(valid.read_bytes() + b"extra")
    with pytest.raises(ValueError, match="trailing data"):
        calibration.inspect_godzilla_triattention_file(valid)


def test_official_calibration_runs_script_then_converts(tmp_path: Path, monkeypatch):
    script = tmp_path / "triattention" / "scripts" / "calibrate.py"
    calibration_input = tmp_path / "calibration.txt"
    output = tmp_path / "model.triattention"
    _write_official_script(script)
    calibration_input.write_text("coherent calibration text", encoding="utf-8")
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        stats_output = Path(command[command.index("--output") + 1])
        torch.save(_official_payload(), stats_output)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        calibration,
        "load_huggingface_model_metadata",
        lambda model, **_kwargs: {
            "head_dim": 4,
            "num_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rope_theta": 10_000.0,
        },
    )

    report = calibration.calibrate_official_triattention_for_godzilla(
        calibrator=script,
        model="org/model",
        input_path=calibration_input,
        output_path=output,
        max_length=512,
        device="cpu",
        attention_implementation="eager",
        runner=runner,
    )

    assert calls[0][0] == calibration.sys.executable
    assert "--max-length" in calls[0]
    assert "--attn-implementation" in calls[0]
    assert report["format"] == "godzilla-triattention-v1"
    assert Path(report["official_stats"]).name == "model.official.pt"
    assert calibration.inspect_godzilla_triattention_file(output)["model_name"] == "model"


def test_official_checkout_inspection_requires_expected_markers(tmp_path: Path):
    checkout = tmp_path / "triattention"
    (checkout / "triattention").mkdir(parents=True)
    (checkout / "docs").mkdir()
    (checkout / "docs" / "calibration.md").write_text("docs", encoding="utf-8")
    _write_official_script(checkout / "scripts" / "calibrate.py")

    report = calibration.inspect_official_triattention_checkout(checkout)

    assert report["valid"] is True
    assert report["calibrator"] == str((checkout / "scripts" / "calibrate.py").resolve())


def test_domvox_calibrator_inspection_requires_sibling_helper(tmp_path: Path):
    calibrator = tmp_path / "triattention_calibrate.py"
    calibrator.write_text(
        " ".join(calibration._DOMVOX_CALIBRATOR_MARKERS),
        encoding="utf-8",
    )

    report = calibration.inspect_domvox_triattention_calibrator(calibrator)

    assert report["valid"] is False
    assert any("triattention_common.py" in issue for issue in report["issues"])


def test_calibration_python_preflight_checks_dependencies_and_cuda(tmp_path: Path):
    python = tmp_path / "python"
    python.write_bytes(b"python")
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_python_probe_report() + "\n",
            stderr="",
        )

    report = calibration.inspect_calibration_python(python, runner=runner)

    assert report["valid"] is True
    assert report["report"]["torch_cuda"] == "12.6"
    assert "torch.cuda.mem_get_info" in commands[0][-1]


def test_calibration_python_preflight_selects_explicit_cuda_device(tmp_path: Path):
    python = tmp_path / "python"
    python.write_bytes(b"python")
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_python_probe_report(cuda_device_index=1) + "\n",
            stderr="",
        )

    report = calibration.inspect_calibration_python(
        python, device="cuda:1", runner=runner
    )

    assert report["valid"] is True
    assert report["report"]["cuda_device_index"] == 1
    assert "requested_index = 1" in commands[0][-1]


def test_model_metadata_prefers_nested_rope_and_exposes_context(monkeypatch):
    import transformers

    config = SimpleNamespace(
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=128,
        hidden_size=2048,
        intermediate_size=11008,
        vocab_size=151936,
        tie_word_embeddings=True,
        max_position_embeddings=131072,
        rope_theta=10_000.0,
        rope_parameters={"rope_theta": 1_000_000.0, "rope_type": "default"},
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: config),
    )

    metadata = calibration.load_huggingface_model_metadata("org/model")

    assert metadata["rope_theta"] == 1_000_000.0
    assert metadata["legacy_rope_theta"] == 10_000.0
    assert metadata["rope_theta_conflict"] is True
    assert metadata["rope_theta_source"] == "rope_parameters"
    assert metadata["max_position_embeddings"] == 131_072


def test_mythos_shape_rejects_200k_and_reports_memory_floor():
    metadata = {
        "head_dim": 128,
        "num_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "hidden_size": 2048,
        "estimated_bf16_weight_bytes": 6_171_394_048,
        "max_position_embeddings": 131_072,
    }

    with pytest.raises(ValueError, match="131072"):
        calibration.validate_model_calibration_length(200_000, metadata)

    estimate = calibration.estimate_official_calibration_bytes(metadata, 200_000)
    assert estimate["captured_q_bytes"] == 29_491_200_000
    assert estimate["estimated_floor_bytes"] == 38_939_394_048


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"device": "metal"}, "device"),
        ({"tokenizer_backend": "unknown"}, "Tokenizer backend"),
        ({"attention_implementation": "unknown"}, "Attention implementation"),
        (
            {"device": "cpu", "attention_implementation": "flash_attention_2"},
            "requires the CUDA device",
        ),
    ],
)
def test_calibration_python_preflight_rejects_invalid_options(
    tmp_path: Path,
    arguments: dict[str, str],
    message: str,
):
    python = tmp_path / "python"
    python.write_bytes(b"python")

    report = calibration.inspect_calibration_python(python, **arguments)

    assert report["valid"] is False
    assert report["compatible"] is False
    assert report["report"] is None
    assert message in report["issues"][0]


def test_calibration_python_preflight_preserves_venv_symlink(tmp_path: Path):
    base = tmp_path / "base-python"
    interpreter = tmp_path / ".venv" / "bin" / "python"
    base.write_bytes(b"python")
    interpreter.parent.mkdir(parents=True)
    try:
        interpreter.symlink_to(base)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_python_probe_report() + "\n",
            stderr="",
        )

    report = calibration.inspect_calibration_python(interpreter, runner=runner)

    assert report["valid"] is True
    assert Path(commands[0][0]) == interpreter.absolute()
    assert Path(report["python"]) == interpreter.absolute()
    assert Path(commands[0][0]) != base.resolve()


def test_calibration_python_preflight_rejects_missing_cuda(tmp_path: Path):
    python = tmp_path / "python"
    python.write_bytes(b"python")

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_python_probe_report(torch_cuda=None, cuda_available=False) + "\n",
            stderr="",
        )

    report = calibration.inspect_calibration_python(python, runner=runner)

    assert report["valid"] is False
    assert "CPU-only" in report["issues"][0]


def test_calibration_python_preflight_requires_reviewed_gigatoken(tmp_path: Path):
    python = tmp_path / "python"
    python.write_bytes(b"python")

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                _python_probe_report(
                    torch_cuda=None,
                    cuda_available=False,
                    gigatoken="0.11.0",
                )
                + "\n"
            ),
            stderr="",
        )

    report = calibration.inspect_calibration_python(
        python,
        device="cpu",
        tokenizer_backend="gigatoken",
        runner=runner,
    )

    assert report["valid"] is False
    assert "reviewed 0.10.x" in report["issues"][0]


def test_calibration_python_preflight_reports_every_missing_module(tmp_path: Path):
    python = tmp_path / "python"
    python.write_bytes(b"python")

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                _python_probe_report(
                    torch_cuda=None,
                    cuda_available=False,
                    module_errors={
                        "torch": "ModuleNotFoundError: No module named 'torch'",
                        "accelerate": "ModuleNotFoundError: No module named 'accelerate'",
                    },
                )
                + "\n"
            ),
            stderr="",
        )

    report = calibration.inspect_calibration_python(python, runner=runner)

    assert report["valid"] is False
    assert any(issue.startswith("torch import failed:") for issue in report["issues"])
    assert any(issue.startswith("accelerate import failed:") for issue in report["issues"])
    assert report["report"]["runtime_executable"] == "/env/bin/python"
    assert report["report"]["modules"]["transformers"]["status"] == "ok"


def test_calibration_python_preflight_requires_flash_attention_only_when_selected(tmp_path: Path):
    python = tmp_path / "python"
    python.write_bytes(b"python")

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                _python_probe_report(
                    module_errors={
                        "flash_attn": "ModuleNotFoundError: No module named 'flash_attn'"
                    }
                )
                + "\n"
            ),
            stderr="",
        )

    sdpa = calibration.inspect_calibration_python(python, runner=runner)
    flash = calibration.inspect_calibration_python(
        python,
        attention_implementation="flash_attention_2",
        runner=runner,
    )

    assert sdpa["valid"] is True
    assert flash["valid"] is False
    assert any(issue.startswith("flash_attn import failed:") for issue in flash["issues"])


def test_calibration_python_selection_checks_bounded_candidates_in_priority_order(
    tmp_path: Path, monkeypatch
):
    managed = tmp_path / "envs" / "triattention" / ".venv" / "bin" / "python"
    current = tmp_path / "current" / "python"
    managed.parent.mkdir(parents=True)
    current.parent.mkdir()
    managed.write_bytes(b"python")
    current.write_bytes(b"python")
    candidates = [
        {"python": str(current), "sources": ["current"]},
        {"python": str(managed), "sources": ["managed"]},
    ]
    monkeypatch.setattr(calibration, "discover_python_interpreters", lambda **kwargs: candidates)
    calls: list[str] = []

    def runner(command, **kwargs):
        calls.append(command[0])
        if Path(command[0]) == managed:
            output = _python_probe_report(
                torch_cuda=None,
                cuda_available=False,
                module_errors={"torch": "ModuleNotFoundError: No module named 'torch'"},
            )
        else:
            output = _python_probe_report()
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    result = calibration.select_compatible_calibration_python(
        environment_root=tmp_path / "envs",
        runner=runner,
    )

    assert calls == [str(managed.resolve()), str(current.resolve())]
    assert result["selected"] == str(current.resolve())
    assert result["checked_count"] == 2
    assert result["attempts"][0]["valid"] is False
    assert result["attempts"][1]["valid"] is True


def test_official_calibration_uses_fail_closed_gigatoken_wrapper(tmp_path: Path, monkeypatch):
    script = tmp_path / "triattention" / "scripts" / "calibrate.py"
    calibration_input = tmp_path / "calibration.txt"
    output = tmp_path / "model.triattention"
    _write_official_script(script)
    calibration_input.write_text("coherent calibration text", encoding="utf-8")
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        stats_output = Path(command[command.index("--output") + 1])
        torch.save(_official_payload(), stats_output)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        calibration,
        "load_huggingface_model_metadata",
        lambda model, **_kwargs: {
            "head_dim": 4,
            "num_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rope_theta": 10_000.0,
        },
    )

    report = calibration.calibrate_official_triattention_for_godzilla(
        calibrator=script,
        model="org/model",
        input_path=calibration_input,
        output_path=output,
        max_length=512,
        device="cpu",
        tokenizer_backend="gigatoken",
        runner=runner,
    )

    assert Path(calls[0][1]).name == "triattention_runner.py"
    assert calls[0][2:8] == [
        "--kind",
        "official",
        "--tokenizer-backend",
        "gigatoken",
        "--calibrator",
        str(script.resolve()),
    ]
    assert report["tokenizer_backend"] == "gigatoken"
