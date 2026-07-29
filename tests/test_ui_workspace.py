from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_turboquant.optimizations import EnvironmentContext
from multi_turboquant.optimizations.environments import environment_python
from multi_turboquant.ui.discovery import (
    inspect_flashattention_source,
    scan_addon_roots,
    scan_environment_profiles,
    scan_models,
)
from multi_turboquant.ui.runtime import (
    EnvironmentJobManager,
    GodzillaCalibrationJobManager,
    ManagedProcess,
    _split_env_wrapper,
)
from multi_turboquant.ui.settings import DEFAULT_UI_SETTINGS, UISettingsStore, validate_ui_settings


def test_settings_store_defaults_and_persists_atomically(tmp_path: Path):
    settings_file = tmp_path / "settings" / "ui.json"
    store = UISettingsStore(settings_file)

    assert store.load() == DEFAULT_UI_SETTINGS
    assert not settings_file.exists()

    saved = store.save(
        {
            **DEFAULT_UI_SETTINGS,
            "model_root": str(tmp_path / "models"),
            "addon_roots": [str(tmp_path / "addons")],
            "form_values": {"cmd-context": "8192", "cmd-tri": True},
        }
    )

    assert store.load() == saved
    assert json.loads(settings_file.read_text(encoding="utf-8"))["schema"] == 1
    assert list(settings_file.parent.glob("*.tmp")) == []


def test_settings_validation_rejects_complex_form_values():
    with pytest.raises(ValueError, match="JSON scalar"):
        validate_ui_settings(
            {**DEFAULT_UI_SETTINGS, "form_values": {"unsafe": {"nested": True}}}
        )


def test_model_scan_is_bounded_to_the_configured_root(tmp_path: Path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    (model_root / "main.gguf").write_bytes(b"gguf")
    transformers = model_root / "transformers-model"
    transformers.mkdir()
    (transformers / "config.json").write_text("{}", encoding="utf-8")
    (transformers / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "outside.gguf").write_bytes(b"outside")

    result = scan_models(model_root)

    paths = {item["path"] for item in result["models"]}
    assert str((model_root / "main.gguf").resolve()) in paths
    assert str(transformers.resolve()) in paths
    assert str((tmp_path / "outside.gguf").resolve()) not in paths
    gguf = next(item for item in result["models"] if item["name"] == "main.gguf")
    assert gguf["launchable"] is True


def test_flashattention_source_inspection_requires_reviewed_markers(tmp_path: Path):
    source = tmp_path / "flash-attention"
    source.mkdir()
    (source / "setup.py").write_text("", encoding="utf-8")
    (source / "flash_attn").mkdir()

    incomplete = inspect_flashattention_source(source)
    assert incomplete["valid"] is False
    assert "Missing csrc" in incomplete["issues"]

    (source / "csrc").mkdir()
    (source / "version.txt").write_text("2.8.3\n", encoding="utf-8")
    complete = inspect_flashattention_source(source)
    assert complete["valid"] is True
    assert complete["version"] == "2.8.3"


def test_addon_scan_finds_flashattention_under_explicit_root(tmp_path: Path):
    addon_root = tmp_path / "addons"
    source = addon_root / "flash-attention"
    (source / "flash_attn").mkdir(parents=True)
    (source / "csrc").mkdir()
    (source / "setup.py").write_text("", encoding="utf-8")

    result = scan_addon_roots([addon_root])

    assert result["errors"] == []
    assert result["count"] == 1
    assert result["addons"][0]["kind"] == "flashattention"
    assert result["addons"][0]["source"]["valid"] is True
    assert result["addons"][0]["environment_profile"] == "flashattention"
    assert result["addons"][0]["local_source"]["valid"] is True
    assert result["scanned_directories"] >= 2
    assert result["max_depth"] == 3


def test_addon_scan_recognizes_renamed_godzilla_checkout(tmp_path: Path):
    checkout = tmp_path / "custom-llama.cpp"
    (checkout / "ggml").mkdir(parents=True)
    (checkout / "common").mkdir()
    (checkout / "scripts").mkdir()
    (checkout / "src").mkdir()
    (checkout / "CMakeLists.txt").write_text("project(godzilla)\n", encoding="utf-8")
    (checkout / "common" / "arg.cpp").write_text(
        "kvarn --triattention-stats\n", encoding="utf-8"
    )
    (checkout / "scripts" / "godzilla-paths.ps1").write_text("", encoding="utf-8")
    (checkout / "src" / "llama-triattention.cpp").write_text("", encoding="utf-8")

    result = scan_addon_roots([tmp_path])

    addon = next(item for item in result["addons"] if item["path"] == str(checkout.resolve()))
    assert addon["kind"] == "godzilla"
    assert addon["source"]["valid"] is True


def test_addon_scan_recognizes_official_triattention_checkout(tmp_path: Path):
    checkout = tmp_path / "renamed-triattention"
    (checkout / "triattention").mkdir(parents=True)
    (checkout / "docs").mkdir()
    (checkout / "scripts").mkdir()
    (checkout / "setup.py").write_text("", encoding="utf-8")
    (checkout / "docs" / "calibration.md").write_text("calibration", encoding="utf-8")
    (checkout / "scripts" / "calibrate.py").write_text(
        "AutoModelForCausalLM --max-length --attn-implementation "
        "q_mean_real q_mean_imag q_abs_mean",
        encoding="utf-8",
    )

    result = scan_addon_roots([tmp_path])

    addon = next(item for item in result["addons"] if item["path"] == str(checkout.resolve()))
    assert addon["kind"] == "triattention"
    assert addon["source"]["valid"] is True
    assert Path(addon["source"]["calibrator"]).name == "calibrate.py"
    assert addon["environment_profile"] == "triattention"
    assert addon["local_source"]["valid"] is True


def test_environment_scan_validates_before_suggesting_rebuild(tmp_path: Path, monkeypatch):
    from multi_turboquant.optimizations import environments

    context = EnvironmentContext(
        os="linux",
        compute="cuda",
        available_executables=frozenset({"uv", "git", "nvcc"}),
        cuda_toolkit_version=(12, 6),
    )
    monkeypatch.setattr(environments, "detect_environment_context", lambda **kwargs: context)
    monkeypatch.setattr(
        environments,
        "check_environment",
        lambda plan: {"torch": "2.7.1", plan.profile.validation_modules[-1]: "test"},
    )
    interpreter = environment_python((tmp_path / "triattention").resolve(), os_name="linux")
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("python", encoding="utf-8")

    result = scan_environment_profiles(tmp_path)
    profile = next(item for item in result["profiles"] if item["id"] == "triattention")

    assert profile["status"] == "installed"
    assert profile["validation"]["torch"] == "2.7.1"


def test_environment_scan_manual_override_warns_without_suggesting_rebuild(
    tmp_path: Path, monkeypatch
):
    from multi_turboquant.optimizations import environments

    context = EnvironmentContext(
        os="linux",
        compute="cuda",
        available_executables=frozenset({"uv", "git", "nvcc"}),
        cuda_toolkit_version=(12, 6),
    )
    monkeypatch.setattr(environments, "detect_environment_context", lambda **kwargs: context)
    monkeypatch.setattr(
        environments,
        "check_environment",
        lambda plan: (_ for _ in ()).throw(RuntimeError("extension import failed")),
    )
    interpreter = environment_python((tmp_path / "triattention").resolve(), os_name="linux")
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("python", encoding="utf-8")

    result = scan_environment_profiles(tmp_path, manual_dependency_override=True)
    profile = next(item for item in result["profiles"] if item["id"] == "triattention")

    assert profile["status"] == "manual"
    assert any(issue["code"] == "manual_dependency_override" for issue in profile["issues"])


def test_addon_scan_explains_empty_configuration():
    result = scan_addon_roots([])

    assert result["addons"] == []
    assert result["scanned_directories"] == 0
    assert "No add-on roots" in result["warnings"][0]


def test_addon_scan_does_not_classify_the_parent_as_a_python_addon(tmp_path: Path):
    addon_root = tmp_path / "addons"
    source = addon_root / "FastDMS"
    (source / "fastdms").mkdir(parents=True)
    (source / "pyproject.toml").write_text("[project]\nname='fastdms'\n", encoding="utf-8")

    result = scan_addon_roots([addon_root])

    paths = [item["path"] for item in result["addons"]]
    assert paths == [str(source.resolve())]
    assert result["addons"][0]["local_source"]["valid"] is True


def test_env_wrapper_is_split_without_a_shell():
    argv, environment = _split_env_wrapper(
        ["env", "LD_PRELOAD=/tmp/share.so", "MODEL_SIZE=10", "llama-server", "--port", "8080"]
    )

    assert argv == ["llama-server", "--port", "8080"]
    assert environment == {"LD_PRELOAD": "/tmp/share.so", "MODEL_SIZE": "10"}


def test_managed_process_starts_collects_output_and_stops(tmp_path: Path):
    manager = ManagedProcess()
    manager.start(
        [
            sys.executable,
            "-u",
            "-c",
            "import time; print('ready', flush=True); time.sleep(30)",
        ],
        cwd=tmp_path,
    )

    deadline = time.time() + 5
    while "ready" not in manager.status()["log"] and time.time() < deadline:
        time.sleep(0.02)

    assert manager.status()["running"] is True
    assert "ready" in manager.status()["log"]
    assert manager.stop(timeout=2)["running"] is False


def test_environment_job_reports_completion(tmp_path: Path, monkeypatch):
    from multi_turboquant.optimizations import environments

    plan = SimpleNamespace(
        ready=True,
        target=tmp_path / "fastdms",
        issues=[],
        cuda_toolkit_root=None,
        local_source=None,
    )
    monkeypatch.setattr(environments, "plan_environment", lambda *args, **kwargs: plan)
    monkeypatch.setattr(environments, "synchronize_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        environments,
        "check_environment",
        lambda *args, **kwargs: {"fastdms": "0.2.0"},
    )
    manager = EnvironmentJobManager()

    job = manager.start_create("fastdms", root=tmp_path)
    deadline = time.time() + 5
    while manager.get(job["id"])["status"] not in {"completed", "failed"}:
        assert time.time() < deadline
        time.sleep(0.02)

    completed = manager.get(job["id"])
    assert completed["status"] == "completed"
    assert completed["report"] == {"fastdms": "0.2.0"}


def test_godzilla_job_reports_completion(tmp_path: Path, monkeypatch):
    from multi_turboquant.integration import godzilla_workspace

    plan = SimpleNamespace(
        ready=True,
        checkout=tmp_path / "godzilla",
        gguf=tmp_path / "model.gguf",
        output=tmp_path / "model.triattention",
        mode="official_python",
        issues=[],
    )
    monkeypatch.setattr(
        godzilla_workspace,
        "plan_godzilla_triattention",
        lambda *args, **kwargs: plan,
    )
    monkeypatch.setattr(
        godzilla_workspace,
        "run_godzilla_triattention",
        lambda *args, **kwargs: {"output": str(plan.output), "reused": False},
    )
    manager = GodzillaCalibrationJobManager()

    job = manager.start(plan.checkout, plan.gguf)
    deadline = time.time() + 5
    while manager.get(job["id"])["status"] not in {"completed", "failed"}:
        assert time.time() < deadline
        time.sleep(0.02)

    completed = manager.get(job["id"])
    assert completed["status"] == "completed"
    assert completed["report"] == {"output": str(plan.output), "reused": False}
