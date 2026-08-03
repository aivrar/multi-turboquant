from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_turboquant.optimizations import EnvironmentContext
from multi_turboquant.optimizations.environments import environment_python
from multi_turboquant.ui.discovery import (
    inspect_addon_source,
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
        validate_ui_settings({**DEFAULT_UI_SETTINGS, "form_values": {"unsafe": {"nested": True}}})


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


def test_informational_addon_source_inspection_is_read_only(tmp_path: Path):
    source = tmp_path / "RocketKV"
    (source / "rocketkv").mkdir(parents=True)
    (source / "README.md").write_text("RocketKV", encoding="utf-8")
    (source / "requirements.txt").write_text("torch", encoding="utf-8")

    result = inspect_addon_source("rocketkv", source)

    assert result["valid"] is True
    assert result["status"] == "informational_only"
    assert result["setup"]["automatic"] is False
    assert result["setup"]["requirements"]
    assert result["setup"]["next_steps"]
    assert result["path"] == str(source.resolve())
    assert all(result["marker_groups"].values())
    assert not (source / ".mtq").exists()


def test_maru_source_reports_current_host_setup_requirements(tmp_path: Path):
    source = tmp_path / "maru"
    (source / "maru_resource_manager").mkdir(parents=True)
    (source / "README.md").write_text("Maru", encoding="utf-8")
    (source / "pyproject.toml").write_text("[project]", encoding="utf-8")

    result = inspect_addon_source("maru", source)

    assert result["valid"] is True
    assert result["setup"]["mode"] == "guided_host_setup"
    assert any("/dev/dax" in item for item in result["setup"]["requirements"])
    assert any("install.sh" in item for item in result["setup"]["next_steps"])


def test_informational_addon_source_inspection_reports_missing_markers(tmp_path: Path):
    source = tmp_path / "Lexico"
    source.mkdir()

    result = inspect_addon_source("lexico", source)

    assert result["valid"] is False
    assert result["status"] == "invalid_source"
    assert len(result["issues"]) == 3


@pytest.mark.parametrize(
    ("profile", "markers"),
    [
        ("maru", ("README.md", "pyproject.toml", "maru_resource_manager")),
        ("speculative_prefill", ("README.md", "requirements.txt", "speculative_prefill")),
        ("rocketkv", ("README.md", "requirements.txt", "rocketkv")),
        ("lexico", ("README.md", "setup.py", "lexico")),
        ("adadecode", ("README.md", "requirements.txt", "adadecode")),
        ("resonance_yarn", ("README.md", "requirements.txt", "src")),
    ],
)
def test_each_blocked_addon_profile_has_a_read_only_source_contract(
    tmp_path: Path, profile: str, markers: tuple[str, ...]
):
    source = tmp_path / profile
    source.mkdir()
    for marker in markers:
        target = source / marker
        if "." in marker:
            target.write_text("marker", encoding="utf-8")
        else:
            target.mkdir()

    result = inspect_addon_source(profile, source)

    assert result["valid"] is True
    assert result["status"] == "informational_only"
    assert result["setup"]["automatic"] is False


def test_addon_scan_recognizes_blocked_source_as_informational(tmp_path: Path):
    source = tmp_path / "rocketkv"
    (source / "rocketkv").mkdir(parents=True)
    (source / "README.md").write_text("RocketKV", encoding="utf-8")
    (source / "requirements.txt").write_text("torch", encoding="utf-8")

    result = scan_addon_roots([tmp_path])

    addon = next(item for item in result["addons"] if item["path"] == str(source.resolve()))
    assert addon["kind"] == "rocketkv"
    assert addon["source_profile"] == "rocketkv"
    assert addon["source"]["status"] == "informational_only"
    assert "environment_profile" not in addon


def test_addon_scan_does_not_classify_an_empty_blocked_named_folder(tmp_path: Path):
    (tmp_path / "rocketkv").mkdir()

    result = scan_addon_roots([tmp_path])

    assert result["addons"] == []


def test_addon_scan_recognizes_domvox_triattention_checkout(tmp_path: Path):
    source = tmp_path / "triattention-ggml"
    source.mkdir()
    for marker in ("triattention_calibrate.py", "triattention_common.py", "TRIA_FORMAT.md"):
        (source / marker).write_text(
            "--model --input --output --max-length --device TRIA triattention_common",
            encoding="utf-8",
        )

    result = scan_addon_roots([tmp_path])

    addon = next(item for item in result["addons"] if item["path"] == str(source.resolve()))
    assert addon["kind"] == "domvox_triattention"
    assert addon["source"]["valid"] is True
    assert addon["source_profile"] == "domvox_triattention"


def test_addon_scan_recognizes_renamed_godzilla_checkout(tmp_path: Path):
    checkout = tmp_path / "custom-llama.cpp"
    (checkout / "ggml").mkdir(parents=True)
    (checkout / "common").mkdir()
    (checkout / "scripts").mkdir()
    (checkout / "src").mkdir()
    (checkout / "CMakeLists.txt").write_text("project(godzilla)\n", encoding="utf-8")
    (checkout / "common" / "arg.cpp").write_text("kvarn --triattention-stats\n", encoding="utf-8")
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
        "AutoModelForCausalLM AutoTokenizer --max-length --attn-implementation "
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


def test_addon_scan_recognizes_gigatoken_llamacpp_as_separate_runtime(tmp_path: Path):
    source = tmp_path / "gigatoken-llama.cpp"
    (source / "docs").mkdir(parents=True)
    (source / "cmake").mkdir()
    (source / "src").mkdir()
    (source / "patches").mkdir()
    (source / "ggml").mkdir()
    (source / "CMakeLists.txt").write_text("project(llama)", encoding="utf-8")
    (source / "docs" / "gigatoken.md").write_text("docs", encoding="utf-8")
    (source / "cmake" / "gigatoken.cmake").write_text("cmake", encoding="utf-8")
    (source / "src" / "llama-gigatoken.cpp").write_text("cpp", encoding="utf-8")
    (source / "patches" / "gigatoken-llama-cpp.patch").write_text("patch", encoding="utf-8")

    result = scan_addon_roots([tmp_path])

    addon = next(item for item in result["addons"] if item["path"] == str(source.resolve()))
    assert addon["kind"] == "gigatoken_llamacpp"
    assert addon["source"]["valid"] is True
    assert addon["source"]["setup"]["automatic"] is False
    assert "Godzilla" in addon["source"]["summary"]


def test_addon_scan_keeps_combined_godzilla_gigatoken_tree_as_godzilla(tmp_path: Path):
    source = tmp_path / "godzilla-gigatoken"
    (source / "ggml").mkdir(parents=True)
    (source / "common").mkdir()
    (source / "src").mkdir()
    (source / "scripts").mkdir()
    (source / "cmake").mkdir()
    (source / "patches").mkdir()
    (source / "CMakeLists.txt").write_text("project(godzilla)", encoding="utf-8")
    (source / "GODZILLA_KING.md").write_text("Godzilla", encoding="utf-8")
    (source / "common" / "arg.cpp").write_text("kvarn", encoding="utf-8")
    (source / "src" / "llama-gigatoken.cpp").write_text("cpp", encoding="utf-8")
    (source / "cmake" / "gigatoken.cmake").write_text("cmake", encoding="utf-8")
    (source / "patches" / "gigatoken-llama-cpp.patch").write_text("patch", encoding="utf-8")

    result = scan_addon_roots([tmp_path])

    addon = next(item for item in result["addons"] if item["path"] == str(source.resolve()))
    assert addon["kind"] == "godzilla"
    assert addon["source"]["features"]["gigatoken"] is True


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
    assert manager._collector is None
    assert manager._process is not None
    assert manager._process.stdout is not None
    assert manager._process.stdout.closed


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

    python = tmp_path / "env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    plan = SimpleNamespace(
        ready=True,
        checkout=tmp_path / "godzilla",
        gguf=tmp_path / "model.gguf",
        output=tmp_path / "model.triattention",
        python=python,
        command=(str(python), "calibrate.py"),
        dependency_validation={"torch": "2.7.1"},
        device="cuda",
        tokenizer_backend="transformers",
        mode="official_python",
        issues=[],
    )
    monkeypatch.setattr(
        godzilla_workspace,
        "plan_godzilla_triattention",
        lambda *args, **kwargs: plan,
    )

    def run_plan(*args, **kwargs):
        kwargs["runner"]([sys.executable, "-c", "print('HF_TOKEN=hf_abcdefghijklmnop')"])
        return {"output": str(plan.output), "reused": False}

    monkeypatch.setattr(godzilla_workspace, "run_godzilla_triattention", run_plan)
    monkeypatch.setattr(
        "multi_turboquant.calibration.godzilla_triattention.inspect_calibration_python",
        lambda *args, **kwargs: {"valid": True, "issues": [], "report": {}},
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
    assert "hf_abcdefghijklmnop" not in "\n".join(completed["log"])
    assert "<redacted>" in "\n".join(completed["log"])


def test_godzilla_jobs_limit_calibration_to_one_process(tmp_path: Path, monkeypatch):
    from multi_turboquant.integration import godzilla_workspace

    started = threading.Event()
    release = threading.Event()
    python = tmp_path / "env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")

    def make_plan(checkout, gguf, **kwargs):
        return SimpleNamespace(
            ready=True,
            checkout=Path(checkout),
            gguf=Path(gguf),
            output=Path(kwargs["output"]),
            python=python,
            command=(str(python), "calibrate.py"),
            dependency_validation={"torch": "2.7.1"},
            device="cuda",
            tokenizer_backend="transformers",
            mode="official_python",
            issues=[],
        )

    def run_plan(plan, **kwargs):
        started.set()
        assert release.wait(5)
        return {"output": str(plan.output), "reused": False}

    monkeypatch.setattr(godzilla_workspace, "plan_godzilla_triattention", make_plan)
    monkeypatch.setattr(godzilla_workspace, "run_godzilla_triattention", run_plan)
    monkeypatch.setattr(
        "multi_turboquant.calibration.godzilla_triattention.inspect_calibration_python",
        lambda *args, **kwargs: {"valid": True, "issues": [], "report": {}},
    )
    manager = GodzillaCalibrationJobManager()
    manager.start(
        tmp_path / "godzilla",
        tmp_path / "model.gguf",
        output=tmp_path / "first.triattention",
    )
    assert started.wait(5)

    with pytest.raises(RuntimeError, match="one process at a time"):
        manager.start(
            tmp_path / "godzilla",
            tmp_path / "model.gguf",
            output=tmp_path / "second.triattention",
        )

    release.set()


def test_godzilla_job_rechecks_dependencies_and_writes_failure_diagnostics(
    tmp_path: Path, monkeypatch
):
    from multi_turboquant.integration import godzilla_workspace

    python = tmp_path / "env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    plan = SimpleNamespace(
        ready=True,
        checkout=tmp_path / "godzilla",
        gguf=tmp_path / "model.gguf",
        output=tmp_path / "calibrations" / "model.triattention",
        python=python,
        command=(str(python), "triattention_calibrate.py"),
        dependency_validation={"torch": "2.7.1"},
        device="cuda",
        tokenizer_backend="transformers",
        mode="domvox",
        issues=[],
    )
    monkeypatch.setattr(
        godzilla_workspace,
        "plan_godzilla_triattention",
        lambda *args, **kwargs: plan,
    )
    run_calls = []
    monkeypatch.setattr(
        godzilla_workspace,
        "run_godzilla_triattention",
        lambda *args, **kwargs: run_calls.append(args) or {},
    )
    monkeypatch.setattr(
        "multi_turboquant.calibration.godzilla_triattention.inspect_calibration_python",
        lambda *args, **kwargs: {
            "valid": False,
            "issues": ["torch import failed: ModuleNotFoundError"],
            "report": {"modules": {"torch": {"status": "error"}}},
        },
    )
    monkeypatch.setattr(
        godzilla_workspace,
        "collect_godzilla_calibration_diagnostics",
        lambda *args, **kwargs: {
            "schema": 1,
            "failure": {"message": str(kwargs["failure"])},
        },
    )
    manager = GodzillaCalibrationJobManager()

    job = manager.start(plan.checkout, plan.gguf)
    deadline = time.time() + 5
    while manager.get(job["id"])["status"] not in {"completed", "failed"}:
        assert time.time() < deadline
        time.sleep(0.02)

    failed = manager.get(job["id"])
    diagnostics_path = Path(failed["diagnostics_path"])
    assert failed["status"] == "failed"
    assert "final dependency preflight" in failed["error"]
    assert failed["runtime_preflight"]["valid"] is False
    assert run_calls == []
    assert diagnostics_path.is_file()
    assert json.loads(diagnostics_path.read_text(encoding="utf-8"))["schema"] == 1
