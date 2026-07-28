from __future__ import annotations

import subprocess
from pathlib import Path

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
        shell_executable="pwsh-test",
    )
    observed: dict[str, object] = {}

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["argv"] = argv
        observed.update(kwargs)
        plan.output.parent.mkdir(parents=True)
        plan.output.write_bytes(b"stats")
        return subprocess.CompletedProcess(argv, 0, stdout="prepared\n", stderr="")

    report = run_godzilla_triattention(plan, runner=runner)

    assert report["output"] == str(plan.output)
    assert report["reused"] is False
    assert observed["argv"] == list(plan.command)
    assert observed["cwd"] == checkout.resolve()
    assert observed["check"] is False
