from __future__ import annotations

import subprocess
from pathlib import Path

import torch

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
        "AutoModelForCausalLM --max-length --attn-implementation "
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
