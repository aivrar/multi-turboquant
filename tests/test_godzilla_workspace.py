from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import torch
import pytest

from multi_turboquant.calibration.godzilla_triattention import (
    convert_official_triattention_stats,
)
from multi_turboquant.integration.godzilla_workspace import (
    inspect_godzilla_checkout,
    plan_godzilla_triattention,
    run_godzilla_triattention,
)
from multi_turboquant.ui.discovery import scan_addon_roots


def _godzilla_checkout(root: Path) -> Path:
    checkout = root / "godzilla-llama.cpp"
    (checkout / "ggml").mkdir(parents=True)
    (checkout / "common").mkdir()
    (checkout / "src").mkdir()
    (checkout / "scripts").mkdir()
    (checkout / "build" / "bin").mkdir(parents=True)
    (checkout / "CMakeLists.txt").write_text("project(godzilla)\n", encoding="utf-8")
    (checkout / "GODZILLA_KING.md").write_text("Godzilla\n", encoding="utf-8")
    (checkout / "common" / "arg.cpp").write_text(
        "kvarn4 --triattention-stats\n", encoding="utf-8"
    )
    (checkout / "src" / "llama-triattention.cpp").write_text("// tri\n", encoding="utf-8")
    (checkout / "scripts" / "godzilla-paths.ps1").write_text("", encoding="utf-8")
    (checkout / "scripts" / "ensure-triattention.ps1").write_text("", encoding="utf-8")
    (checkout / "scripts" / "resolve-triattention-hf.py").write_text("", encoding="utf-8")
    (checkout / "build" / "bin" / "llama-server.exe").write_bytes(b"exe")
    return checkout


def _write_domvox_stats(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack("<7I2f", 0x54524941, 2, 1, 2, 1, 4, 2, 10_000.0, 1.0)
    vectors = b"".join(
        struct.pack("<2f", *values)
        for values in ((0.1, 0.2), (0.2, 0.1), (0.5, 0.5), (0.4, 0.5)) * 2
    )
    path.write_bytes(header + b"\0" * (64 - len(header)) + struct.pack("<f", 1.0) + vectors)


def _write_valid_calibration(output: Path) -> None:
    source = output.with_suffix(".pt")
    torch.save(
        {
            "metadata": {
                "head_dim": 4,
                "rope_style": "half",
                "sampled_heads": [[0, 0]],
            },
            "stats": {
                "layer00_head00": {
                    "q_mean_real": torch.tensor([0.1, 0.2]),
                    "q_mean_imag": torch.tensor([0.2, 0.1]),
                    "q_abs_mean": torch.tensor([0.5, 0.5]),
                }
            },
        },
        source,
    )
    convert_official_triattention_stats(
        source,
        output,
        model_name="model",
        num_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        rope_theta=10_000.0,
    )


def test_godzilla_checkout_inspection_reports_features_and_binary(tmp_path: Path):
    checkout = _godzilla_checkout(tmp_path)

    inspection = inspect_godzilla_checkout(checkout)

    assert inspection["valid"] is True
    assert inspection["features"] == {
        "kvarn": True,
        "triattention": True,
        "triattention_prepare": True,
        "triattention_auto_resolver": True,
        "bundled_calibrator": False,
        "gigatoken": False,
    }
    assert inspection["preferred_binary"].endswith("llama-server.exe")
    assert any("KVarN" in note for note in inspection["notes"])


def test_godzilla_checkout_is_recognized_as_an_addon(tmp_path: Path):
    checkout = _godzilla_checkout(tmp_path)

    result = scan_addon_roots([tmp_path])

    addon = next(item for item in result["addons"] if item["path"] == str(checkout.resolve()))
    assert addon["kind"] == "godzilla"
    assert addon["source"]["valid"] is True
    assert addon["source"]["features"]["kvarn"] is True


def test_godzilla_triattention_plan_uses_explicit_prerequisites(tmp_path: Path):
    checkout = _godzilla_checkout(tmp_path)
    model = tmp_path / "model.gguf"
    python = tmp_path / "python.exe"
    calibrator = tmp_path / "calibrate-triattention.py"
    model.write_bytes(b"gguf")
    python.write_bytes(b"python")
    calibrator.write_text("", encoding="utf-8")

    plan = plan_godzilla_triattention(
        checkout,
        model,
        python=python,
        calibrator=calibrator,
        hf_model="org/source-model",
        n_tokens=4096,
        device="CPU",
        mode="godzilla_script",
        shell_executable="pwsh-test",
    )

    assert plan.ready is True
    assert plan.device == "cpu"
    assert plan.command[0] == "pwsh-test"
    assert plan.command[-2:] == ("-HfModel", "org/source-model")
    assert dict(plan.environment) == {
        "GODZILLA_ROOT": str(checkout.resolve()),
        "TRIATTENTION_PYTHON": str(python.resolve()),
        "TRIATTENTION_CALIBRATE_PY": str(calibrator.resolve()),
    }
    assert plan.to_dict()["kvarn_calibration_required"] is False


def test_godzilla_triattention_plan_rejects_missing_prerequisites(tmp_path: Path):
    checkout = _godzilla_checkout(tmp_path)

    plan = plan_godzilla_triattention(
        checkout,
        tmp_path / "missing.gguf",
        n_tokens=64,
        device="metal",
        shell_executable="pwsh-test",
    )

    assert plan.ready is False
    codes = {issue.code for issue in plan.issues if issue.severity == "error"}
    assert {
        "invalid_gguf",
        "missing_calibration_python",
        "missing_calibrator",
        "invalid_token_count",
        "invalid_device",
    } <= codes


def test_godzilla_plan_uses_bundled_calibrator_when_present(tmp_path: Path):
    checkout = _godzilla_checkout(tmp_path)
    model = tmp_path / "model.gguf"
    python = tmp_path / "python"
    calibrator = checkout / "scripts" / "calibrate-triattention.py"
    model.write_bytes(b"gguf")
    python.write_bytes(b"python")
    calibrator.write_text("", encoding="utf-8")

    plan = plan_godzilla_triattention(
        checkout,
        model,
        python=python,
        hf_model="org/source-model",
        mode="godzilla_script",
        shell_executable="pwsh-test",
    )

    assert plan.ready is True
    assert plan.calibrator == calibrator.resolve()
    assert dict(plan.environment)["TRIATTENTION_CALIBRATE_PY"] == str(calibrator.resolve())


def test_godzilla_plan_reuses_existing_calibration_without_toolchain(tmp_path: Path):
    checkout = _godzilla_checkout(tmp_path)
    model = tmp_path / "model.gguf"
    output = tmp_path / "model.triattention"
    model.write_bytes(b"gguf")
    _write_valid_calibration(output)

    plan = plan_godzilla_triattention(checkout, model, output=output)
    runner_called = False

    def runner(*args, **kwargs):
        nonlocal runner_called
        runner_called = True
        raise AssertionError("existing calibration must not start a subprocess")

    report = run_godzilla_triattention(plan, runner=runner)

    assert plan.ready is True
    assert plan.command == ()
    assert report["reused"] is True
    assert runner_called is False


def test_godzilla_triattention_run_verifies_output(tmp_path: Path):
    checkout = _godzilla_checkout(tmp_path)
    model = tmp_path / "model.gguf"
    python = tmp_path / "python.exe"
    calibrator = tmp_path / "calibrate-triattention.py"
    model.write_bytes(b"gguf")
    python.write_bytes(b"python")
    calibrator.write_text("", encoding="utf-8")
    plan = plan_godzilla_triattention(
        checkout,
        model,
        python=python,
        calibrator=calibrator,
        hf_model="org/source-model",
        mode="godzilla_script",
        shell_executable="pwsh-test",
    )
    observed: dict[str, object] = {}

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["argv"] = argv
        observed.update(kwargs)
        plan.output.parent.mkdir(parents=True)
        _write_valid_calibration(plan.output)
        return subprocess.CompletedProcess(argv, 0, stdout="prepared\n", stderr="")

    report = run_godzilla_triattention(plan, runner=runner)

    assert report["output"] == str(plan.output)
    assert report["reused"] is False
    assert observed["argv"] == list(plan.command)
    assert observed["cwd"] == checkout.resolve()
    assert observed["check"] is False


def test_godzilla_official_python_plan_does_not_require_powershell(tmp_path: Path):
    checkout = _godzilla_checkout(tmp_path)
    model = tmp_path / "model.gguf"
    python = tmp_path / "python.exe"
    calibrator = tmp_path / "triattention" / "scripts" / "calibrate.py"
    calibration_input = tmp_path / "calibration.txt"
    model.write_bytes(b"gguf")
    python.write_bytes(b"python")
    calibrator.parent.mkdir(parents=True)
    calibrator.write_text(
        "AutoModelForCausalLM AutoTokenizer --max-length --attn-implementation "
        "q_mean_real q_mean_imag q_abs_mean",
        encoding="utf-8",
    )
    calibration_input.write_text("coherent calibration text", encoding="utf-8")

    plan = plan_godzilla_triattention(
        checkout,
        model,
        python=python,
        calibrator=calibrator,
        calibration_input=calibration_input,
        hf_model="org/model",
        n_tokens=1024,
        device="cuda",
        mode="official_python",
        attention_implementation="sdpa",
        shell_executable=None,
    )

    assert plan.ready is True
    assert plan.mode == "official_python"
    assert plan.command[0] == str(python.resolve())
    assert "calibrate" in plan.command
    assert "--stats-output" in plan.command
    assert not any("PowerShell" in issue.message for issue in plan.issues)


def test_godzilla_plan_preserves_linux_venv_python_symlink(tmp_path: Path):
    checkout = _godzilla_checkout(tmp_path)
    model = tmp_path / "model.gguf"
    base = tmp_path / "uv" / "python3.11"
    python = tmp_path / "triattention" / ".venv" / "bin" / "python"
    calibrator = tmp_path / "triattention" / "scripts" / "calibrate.py"
    calibration_input = tmp_path / "calibration.txt"
    model.write_bytes(b"gguf")
    base.parent.mkdir()
    base.write_bytes(b"python")
    python.parent.mkdir(parents=True)
    try:
        python.symlink_to(base)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    calibrator.parent.mkdir(parents=True, exist_ok=True)
    calibrator.write_text(
        "AutoModelForCausalLM AutoTokenizer --max-length --attn-implementation "
        "q_mean_real q_mean_imag q_abs_mean",
        encoding="utf-8",
    )
    calibration_input.write_text("coherent calibration text", encoding="utf-8")

    plan = plan_godzilla_triattention(
        checkout,
        model,
        python=python,
        calibrator=calibrator,
        calibration_input=calibration_input,
        hf_model="org/model",
        mode="official_python",
    )

    assert plan.ready
    assert plan.python == python.absolute()
    assert Path(plan.command[0]) == python.absolute()
    assert Path(plan.command[0]) != base.resolve()


def test_godzilla_official_convert_plan_skips_model_forward_pass(tmp_path: Path):
    checkout = _godzilla_checkout(tmp_path)
    model = tmp_path / "model.gguf"
    python = tmp_path / "python"
    official_stats = tmp_path / "official.pt"
    model.write_bytes(b"gguf")
    python.write_bytes(b"python")
    torch.save(
        {
            "metadata": {"head_dim": 4, "rope_style": "half", "sampled_heads": [[0, 0]]},
            "stats": {
                "layer00_head00": {
                    "q_mean_real": torch.tensor([0.1, 0.2]),
                    "q_mean_imag": torch.tensor([0.2, 0.1]),
                    "q_abs_mean": torch.tensor([0.5, 0.5]),
                }
            },
        },
        official_stats,
    )

    plan = plan_godzilla_triattention(
        checkout,
        model,
        python=python,
        official_stats_input=official_stats,
        hf_model="org/model",
        mode="official_convert",
    )

    assert plan.ready
    assert plan.official_stats_input == official_stats.resolve()
    assert "convert" in plan.command
    assert "calibrate" not in plan.command
    assert "--input" not in plan.command


def test_domvox_plan_requires_explicit_lossy_acknowledgement(tmp_path: Path):
    checkout = _godzilla_checkout(tmp_path)
    model = tmp_path / "model.gguf"
    python = tmp_path / "python.exe"
    calibrator = tmp_path / "triattention_calibrate.py"
    calibration_input = tmp_path / "calibration.txt"
    model.write_bytes(b"gguf")
    python.write_bytes(b"python")
    calibrator.write_text(
        "--model --input --output --max-length --device TRIA", encoding="utf-8"
    )
    calibration_input.write_text("text", encoding="utf-8")

    blocked = plan_godzilla_triattention(
        checkout,
        model,
        python=python,
        domvox_calibrator=calibrator,
        calibration_input=calibration_input,
        hf_model="org/model",
        mode="domvox",
    )
    assert blocked.ready is False
    assert any(issue.code == "domvox_lossy_confirmation" for issue in blocked.issues)

    plan = plan_godzilla_triattention(
        checkout,
        model,
        python=python,
        domvox_calibrator=calibrator,
        calibration_input=calibration_input,
        hf_model="org/model",
        mode="domvox",
        domvox_accept_lossy=True,
    )
    assert plan.ready is True
    assert plan.official_stats == (checkout / "calibrations" / "model.domvox.bin").resolve()
    assert plan.command[0] == str(python.resolve())
    assert "triattention_calibrate.py" in plan.command[1]


def test_domvox_run_converts_and_validates_output(tmp_path: Path, monkeypatch):
    checkout = _godzilla_checkout(tmp_path)
    model = tmp_path / "model.gguf"
    python = tmp_path / "python.exe"
    calibrator = tmp_path / "triattention_calibrate.py"
    calibration_input = tmp_path / "calibration.txt"
    model.write_bytes(b"gguf")
    python.write_bytes(b"python")
    calibrator.write_text(
        "--model --input --output --max-length --device TRIA", encoding="utf-8"
    )
    calibration_input.write_text("text", encoding="utf-8")
    plan = plan_godzilla_triattention(
        checkout,
        model,
        python=python,
        domvox_calibrator=calibrator,
        calibration_input=calibration_input,
        hf_model="org/model",
        mode="domvox",
        domvox_accept_lossy=True,
    )

    monkeypatch.setattr(
        "multi_turboquant.integration.godzilla_workspace.load_huggingface_model_metadata",
        lambda model: {
            "head_dim": 4,
            "num_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rope_theta": 10_000.0,
        },
    )

    def runner(argv, **kwargs):
        _write_domvox_stats(plan.official_stats)
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    report = run_godzilla_triattention(plan, runner=runner)

    assert report["reused"] is False
    assert report["conversion"]["source_format"] == "domvox-tria-v2"
    assert plan.output.is_file()


def test_long_calibration_requires_acknowledgement_and_has_hard_cap(tmp_path: Path):
    checkout = _godzilla_checkout(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")

    blocked = plan_godzilla_triattention(
        checkout,
        model,
        n_tokens=200_000,
    )
    assert any(issue.code == "long_calibration_confirmation" for issue in blocked.issues)

    capped = plan_godzilla_triattention(
        checkout,
        model,
        n_tokens=200_001,
        allow_long_calibration=True,
    )
    assert any(issue.code == "invalid_token_count" for issue in capped.issues)


def test_gigatoken_plan_is_official_only_and_forwards_backend(tmp_path: Path):
    checkout = _godzilla_checkout(tmp_path)
    model = tmp_path / "model.gguf"
    python = tmp_path / "python"
    calibrator = tmp_path / "triattention" / "scripts" / "calibrate.py"
    calibration_input = tmp_path / "calibration.txt"
    model.write_bytes(b"gguf")
    python.write_bytes(b"python")
    calibrator.parent.mkdir(parents=True)
    calibrator.write_text(
        "AutoModelForCausalLM AutoTokenizer --max-length --attn-implementation "
        "q_mean_real q_mean_imag q_abs_mean",
        encoding="utf-8",
    )
    calibration_input.write_text("text", encoding="utf-8")

    plan = plan_godzilla_triattention(
        checkout,
        model,
        python=python,
        calibrator=calibrator,
        calibration_input=calibration_input,
        hf_model="org/model",
        mode="official_python",
        tokenizer_backend="gigatoken",
    )

    assert plan.ready
    assert plan.tokenizer_backend == "gigatoken"
    assert plan.command[-2:] == ("--tokenizer-backend", "gigatoken")

    blocked = plan_godzilla_triattention(
        checkout,
        model,
        python=python,
        mode="official_convert",
        tokenizer_backend="gigatoken",
    )
    assert any(issue.code == "gigatoken_mode_unsupported" for issue in blocked.issues)


def test_planner_identifies_calibrator_selected_in_the_wrong_mode(tmp_path: Path):
    checkout = _godzilla_checkout(tmp_path)
    model = tmp_path / "model.gguf"
    python = tmp_path / "python"
    calibration_input = tmp_path / "calibration.txt"
    domvox = tmp_path / "triattention_calibrate.py"
    official = tmp_path / "scripts" / "calibrate.py"
    model.write_bytes(b"gguf")
    python.write_bytes(b"python")
    calibration_input.write_text("text", encoding="utf-8")
    domvox.write_text("--model --input --output --max-length --device TRIA", encoding="utf-8")
    official.parent.mkdir()
    official.write_text(
        "AutoModelForCausalLM AutoTokenizer --max-length --attn-implementation "
        "q_mean_real q_mean_imag q_abs_mean",
        encoding="utf-8",
    )

    official_plan = plan_godzilla_triattention(
        checkout,
        model,
        python=python,
        calibrator=domvox,
        calibration_input=calibration_input,
        hf_model="org/model",
        mode="official_python",
    )
    domvox_plan = plan_godzilla_triattention(
        checkout,
        model,
        python=python,
        domvox_calibrator=official,
        calibration_input=calibration_input,
        hf_model="org/model",
        mode="domvox",
        domvox_accept_lossy=True,
    )

    official_error = next(
        issue for issue in official_plan.issues if issue.code == "invalid_official_calibrator"
    )
    domvox_error = next(
        issue for issue in domvox_plan.issues if issue.code == "invalid_domvox_calibrator"
    )
    assert "matches domvox" in official_error.message
    assert "recommended Generate stats + convert" in domvox_error.message
