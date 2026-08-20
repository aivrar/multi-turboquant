# SPDX-License-Identifier: MIT
"""Tests for the lightweight web UI command API."""

import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_turboquant import CacheMethod
from multi_turboquant.hardware import GPU, PlatformInfo
from multi_turboquant.ui import DEFAULT_UI_SETTINGS, UISettingsStore
import run_ui
from run_ui import (
    _command_config,
    _command_context_extension_config,
    api_generate_command,
    api_methods,
    api_presets,
    api_scan_llamacpp,
)


def _valid_calibration_python(path) -> dict[str, object]:
    return {
        "python": str(Path(path).absolute()),
        "python_resolved": str(Path(path).resolve()),
        "valid": True,
        "compatible": True,
        "required_modules": [
            "torch",
            "transformers",
            "accelerate",
            "numpy",
            "safetensors",
            "huggingface_hub",
            "tokenizers",
            "sentencepiece",
        ],
        "report": {
            "runtime_executable": str(Path(path).absolute()),
            "prefix": str(Path(path).parent),
            "base_prefix": "/base",
            "torch": "2.7.1",
            "transformers": "4.57.6",
            "accelerate": "1.14.0",
            "torch_cuda": "12.6",
            "cuda_available": True,
            "modules": {
                name: {"status": "ok", "version": version}
                for name, version in {
                    "torch": "2.7.1",
                    "transformers": "4.57.6",
                    "accelerate": "1.14.0",
                    "numpy": "2.2.6",
                    "safetensors": "0.6.2",
                    "huggingface_hub": "0.35.0",
                    "tokenizers": "0.22.0",
                    "sentencepiece": "0.2.2",
                }.items()
            },
        },
        "issues": [],
    }


def test_command_config_translates_triattention_k_method_to_flag():
    config = _command_config({"k_method": "triattention", "v_method": "f16"})

    assert config.k_method == CacheMethod.FP16
    assert config.v_method == CacheMethod.FP16
    assert config.triattention_enabled is True


def test_api_generate_command_accepts_legacy_triattention_selection():
    with pytest.warns(RuntimeWarning, match="TriAttention"):
        result = api_generate_command(
            {
                "k_method": "triattention",
                "v_method": "f16",
                "model_path": "/opt/models/model.gguf",
                "port": 8080,
                "context": 4096,
                "parallel": 1,
            }
        )

    assert "--cache-type-k f16" in result["command"]
    assert "--cache-type-v f16" in result["command"]
    assert any(issue["method"] == "triattention" for issue in result["issues"])


def test_api_generate_command_supports_patched_triattention_flags():
    result = api_generate_command(
        {
            "k_method": "turbo3",
            "v_method": "turbo3",
            "triattention": True,
            "use_custom_triattention_llamacpp": True,
            "triattention_stats_path": "model.triattention",
            "triattention_budget": 2048,
            "triattention_window": 256,
            "triattention_log": True,
            "model_path": "/opt/models/model.gguf",
            "port": 8080,
            "context": 4096,
            "parallel": 1,
        }
    )

    assert "--cache-type-k turbo3" in result["command"]
    assert "--cache-type-v turbo3" in result["command"]
    assert "--triattention-stats model.triattention" in result["command"]
    assert "--triattention-budget 2048" in result["command"]
    assert "--triattention-window 256" in result["command"]
    assert "--triattention-log" in result["command"]
    assert not any(issue["method"] == "command" for issue in result["issues"])


def test_api_generate_command_returns_error_for_missing_triattention_stats():
    result = api_generate_command(
        {
            "k_method": "turbo3",
            "v_method": "turbo3",
            "use_custom_triattention_llamacpp": True,
            "model_path": "/opt/models/model.gguf",
            "port": 8080,
            "context": 4096,
            "parallel": 1,
        }
    )

    assert result["command"] == ""
    triattention_issues = [issue for issue in result["issues"] if issue["method"] == "triattention"]
    assert len(triattention_issues) == 1
    assert "Stats Path" in triattention_issues[0]["message"]
    assert "matching Hugging Face checkpoint" in triattention_issues[0]["suggestion"]
    assert "GGUF alone is not sufficient" in triattention_issues[0]["suggestion"]
    assert not any(issue["method"] == "command" for issue in result["issues"])


def test_api_generate_command_supports_cuda_weight_share_wrapper():
    result = api_generate_command(
        {
            "k_method": "f16",
            "v_method": "f16",
            "model_path": "/opt/models/model.gguf",
            "port": 8080,
            "context": 4096,
            "parallel": 1,
            "cuda_weight_share": True,
            "cuda_weight_share_library": "/opt/cuda-llm-weight-share.so",
            "cuda_weight_share_model_size": 123456,
            "cuda_weight_share_tolerance": 1024,
            "cuda_weight_share_ipc_name": "/cuda_vram_ipc_test",
            "cuda_weight_share_shm_wait_sec": 15,
            "cuda_weight_share_suppress_master_free": True,
            "cuda_weight_share_trace": True,
            "cuda_weight_share_trace_depth": 8,
            "cuda_weight_share_trace_normal_allocs": True,
        }
    )

    assert result["command"].startswith("env LD_PRELOAD=/opt/cuda-llm-weight-share.so ")
    assert "MODEL_SIZE=123456" in result["command"]
    assert "MODEL_SIZE_TOLERANCE=1024" in result["command"]
    assert "CUDA_VRAM_IPC_NAME=/cuda_vram_ipc_test" in result["command"]
    assert "CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC=15" in result["command"]
    assert "CUDA_VRAM_IPC_SUPPRESS_MASTER_FREE=1" in result["command"]
    assert "CUDA_VRAM_IPC_TRACE_CALLERS=1" in result["command"]
    assert "CUDA_VRAM_IPC_TRACE_DEPTH=8" in result["command"]
    assert "CUDA_VRAM_IPC_TRACE_NORMAL_ALLOCS=1" in result["command"]
    assert "llama-server" in result["command"]


def test_command_context_extension_config_parses_ui_values():
    assert _command_context_extension_config({"rope_scaling": "off"}) is None

    config = _command_context_extension_config(
        {
            "rope_scale": "8",
            "yarn_orig_ctx": "4096",
            "yarn_ext_factor": "0",
            "yarn_attn_factor": "1.1",
        }
    )

    assert config.rope_scaling is None
    assert config.rope_scale == 8.0
    assert config.yarn_orig_ctx == 4096
    assert config.yarn_ext_factor == 0.0
    assert config.yarn_attn_factor == 1.1


def test_api_generate_command_supports_context_extension_flags():
    result = api_generate_command(
        {
            "k_method": "f16",
            "v_method": "f16",
            "model_path": "/opt/models/model.gguf",
            "port": 8080,
            "context": 32768,
            "parallel": 1,
            "rope_scale": "8",
            "yarn_orig_ctx": "4096",
        }
    )

    assert "-c 32768" in result["command"]
    assert "--rope-scaling yarn" in result["command"]
    assert "--rope-scale 8.0" in result["command"]
    assert "--yarn-orig-ctx 4096" in result["command"]
    assert not any(issue["method"] == "command" for issue in result["issues"])


def test_api_generate_command_reports_invalid_context_extension():
    result = api_generate_command(
        {
            "k_method": "f16",
            "v_method": "f16",
            "model_path": "/opt/models/model.gguf",
            "port": 8080,
            "context": 32768,
            "parallel": 1,
            "rope_scaling": "linear",
            "yarn_orig_ctx": "4096",
        }
    )

    assert result["command"] == ""
    assert any(
        issue["method"] == "command" and "YaRN options require" in issue["message"]
        for issue in result["issues"]
    )


def test_api_scan_llamacpp_reports_missing_binary():
    result = api_scan_llamacpp(
        {
            "binary": "__definitely_missing_llama_server__",
            "timeout_seconds": "0.1",
        }
    )

    assert result["binary"] == "__definitely_missing_llama_server__"
    assert result["scanned"] is False
    assert result["error"]


def test_api_methods_include_backend_only_kvarn():
    methods = api_methods()
    kvarn = [m for m in methods if m["value"] == "kvarn4"]
    assert len(kvarn) == 1
    assert kvarn[0]["family"] == "kvarn"
    assert kvarn[0]["backend_only"] is True


def test_api_presets_include_godzilla_kvarn():
    presets = api_presets()
    preset = [p for p in presets if p["name"] == "godzilla_kvarn4"]
    assert len(preset) == 1
    assert preset[0]["k_method"] == "kvarn4"
    assert preset[0]["v_method"] == "kvarn4"


def test_api_generate_command_supports_godzilla_kvarn_dflash():
    result = api_generate_command(
        {
            "fork_profile": "godzilla",
            "k_method": "kvarn4",
            "v_method": "kvarn4",
            "model_path": "/opt/models/model.gguf",
            "port": 8080,
            "context": 8192,
            "parallel": 1,
            "spec_dflash": True,
            "spec_draft_model": "/opt/models/draft.gguf",
            "spec_draft_n_max": 16,
            "spec_branch_budget": 0,
            "spec_dflash_cross_ctx": 512,
            "spec_draft_gpu_layers": "all",
        }
    )

    assert "--cache-type-k kvarn4" in result["command"]
    assert "--cache-type-v kvarn4" in result["command"]
    assert "--spec-type dflash" in result["command"]
    assert "--spec-draft-model /opt/models/draft.gguf" in result["command"]
    assert "--spec-draft-n-max 16" in result["command"]
    assert not any(issue["method"] == "command" for issue in result["issues"])


def test_api_generate_command_rejects_kvarn_without_godzilla_profile():
    result = api_generate_command(
        {
            "k_method": "kvarn4",
            "v_method": "kvarn4",
            "model_path": "/opt/models/model.gguf",
            "port": 8080,
            "context": 8192,
            "parallel": 1,
        }
    )

    assert result["command"] == ""
    assert any(issue["method"] == "command" for issue in result["issues"])


def test_api_settings_persist_workspace_and_form_values(tmp_path, monkeypatch):
    store = UISettingsStore(tmp_path / "ui.json")
    monkeypatch.setattr(run_ui, "SETTINGS_STORE", store)
    monkeypatch.setattr(run_ui, "UI_MUTATIONS_ENABLED", True)

    state = run_ui.api_save_settings(
        {
            "schema": 1,
            "model_root": str(tmp_path / "models"),
            "environment_root": str(tmp_path / "envs"),
            "flashattention_source": "",
            "addon_roots": [str(tmp_path / "addons")],
            "form_values": {"cmd-k": "f16", "cmd-context": "8192"},
        }
    )

    assert state["settings"]["form_values"]["cmd-k"] == "f16"
    assert run_ui.api_settings()["settings"] == state["settings"]


def test_api_settings_returns_defaults_when_file_is_absent(tmp_path, monkeypatch):
    store = UISettingsStore(tmp_path / "missing" / "ui.json")
    monkeypatch.setattr(run_ui, "SETTINGS_STORE", store)

    state = run_ui.api_settings()

    assert state["settings"] == DEFAULT_UI_SETTINGS
    assert state["path"] == str(store.path.resolve())
    assert not store.path.exists()


def test_api_addon_scan_respects_explicit_empty_roots(tmp_path, monkeypatch):
    saved_root = tmp_path / "addons"
    saved_root.mkdir()
    store = UISettingsStore(tmp_path / "ui.json")
    store.save(
        {
            **DEFAULT_UI_SETTINGS,
            "addon_roots": [str(saved_root)],
        }
    )
    monkeypatch.setattr(run_ui, "SETTINGS_STORE", store)

    result = run_ui.api_scan_addons({"roots": []})

    assert result["roots"] == []
    assert "No add-on roots" in result["warnings"][0]


def test_api_status_reports_ram_and_vram_separately(monkeypatch):
    monkeypatch.setattr(
        run_ui,
        "detect_platform",
        lambda: PlatformInfo(
            os="linux",
            arch="x86_64",
            gpus=[GPU(0, "Test GPU", 24 * 1024, vram_used_mb=8 * 1024)],
            system_memory_total_mb=64 * 1024,
            system_memory_available_mb=40 * 1024,
        ),
    )

    result = run_ui.api_status()

    assert result["system_ram_gb"] == 64
    assert result["available_system_ram_gb"] == 40
    assert result["total_vram_gb"] == 24
    assert result["available_vram_gb"] == 16
    assert result["combined_memory_gb"] == 88


def test_api_inspects_informational_addon_source(tmp_path):
    source = tmp_path / "maru"
    (source / "maru_resource_manager").mkdir(parents=True)
    (source / "README.md").write_text("Maru", encoding="utf-8")
    (source / "pyproject.toml").write_text("[project]", encoding="utf-8")

    result = run_ui.api_inspect_addon_source({"profile": "maru", "path": str(source)})

    assert result["valid"] is True
    assert result["status"] == "informational_only"
    assert result["name"] == "Maru"
    assert "CMakeLists.txt" not in result["marker_groups"]


def test_evaluated_ui_javascript_has_valid_syntax():
    match = re.search(r"(?s)<script>(.*?)</script>", run_ui.UI_HTML)
    assert match is not None
    script = match.group(1)
    assert r".split(/\r?\n/)" in script
    assert r".join('\n')" in script
    assert "\r" not in script

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available for embedded JavaScript syntax validation")
    result = subprocess.run(
        [node, "--check", "-"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_issue_43_profiles_and_capabilities_are_exposed_in_ui():
    for profile in ("jetspec", "proxima", "jetlong"):
        assert f'<option value="{profile}">' in run_ui.UI_HTML

    for profile in (
        "lucebox",
        "chunkllama",
        "rabitqcache",
        "scope_pe",
        "duoattention",
        "icecache",
        "pflash_llamacpp",
    ):
        assert f'<option value="{profile}">' in run_ui.UI_HTML

    for capability in ("PFlash", "KVFlash", "DDTree", "SpecLA"):
        assert f"capabilityTag('{capability}'" in run_ui.UI_HTML


def test_tokenizer_source_actions_keep_supported_calibration_modes():
    assert (
        "function useGigatokenPython(encodedPath) {\n"
        "  document.getElementById('godzilla-mode').value = 'official_python';" in run_ui.UI_HTML
    )
    assert (
        "function useDomvoxSource(encodedCalibrator, encodedPath) {\n"
        "  document.getElementById('godzilla-mode').value = 'domvox';\n"
        "  document.getElementById('godzilla-tokenizer').value = 'transformers';" in run_ui.UI_HTML
    )


def test_json_response_treats_client_disconnect_as_cancelled_request():
    class DisconnectedWriter:
        def write(self, payload):
            raise BrokenPipeError("client closed the socket")

    handler = object.__new__(run_ui.UIHandler)
    handler.send_response = lambda status: None
    handler.send_header = lambda name, value: None
    handler.end_headers = lambda: None
    handler.wfile = DisconnectedWriter()
    handler.close_connection = False

    assert handler._json({"ok": True}) is False
    assert handler.close_connection is True


def test_post_disconnect_does_not_attempt_a_second_response(monkeypatch):
    handler = object.__new__(run_ui.UIHandler)
    handler.path = "/api/settings"
    handler._read_json = lambda: {}
    calls = []

    def disconnected_response(*args, **kwargs):
        calls.append((args, kwargs))
        raise BrokenPipeError("client closed the socket")

    handler._json = disconnected_response
    handler.close_connection = False
    monkeypatch.setattr(run_ui, "api_save_settings", lambda params: {"ok": True})

    handler.do_POST()

    assert len(calls) == 1
    assert handler.close_connection is True


def test_api_runtime_launch_is_bounded_to_saved_model_root(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    model_root.mkdir()
    model = model_root / "model.gguf"
    model.write_bytes(b"gguf")
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"gguf")
    store = UISettingsStore(tmp_path / "ui.json")
    store.save(
        {
            "schema": 1,
            "model_root": str(model_root),
            "environment_root": str(tmp_path / "envs"),
            "flashattention_source": "",
            "addon_roots": [],
            "form_values": {},
        }
    )
    monkeypatch.setattr(run_ui, "SETTINGS_STORE", store)
    monkeypatch.setattr(run_ui, "UI_MUTATIONS_ENABLED", True)

    with pytest.raises(ValueError, match="inside the configured model root"):
        run_ui.api_runtime_start({"model_path": str(outside)})


def test_api_runtime_start_uses_generated_argv_without_shell(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    model_root.mkdir()
    model = model_root / "model.gguf"
    model.write_bytes(b"gguf")
    store = UISettingsStore(tmp_path / "ui.json")
    store.save(
        {
            "schema": 1,
            "model_root": str(model_root),
            "environment_root": str(tmp_path / "envs"),
            "flashattention_source": "",
            "addon_roots": [],
            "form_values": {},
        }
    )
    calls = []

    class FakeProcess:
        def start(self, argv, **kwargs):
            calls.append((argv, kwargs))
            return {"running": True, "pid": 123, "log": []}

    monkeypatch.setattr(run_ui, "SETTINGS_STORE", store)
    monkeypatch.setattr(run_ui, "MODEL_PROCESS", FakeProcess())
    monkeypatch.setattr(run_ui, "UI_MUTATIONS_ENABLED", True)
    monkeypatch.setattr(
        run_ui,
        "api_generate_command",
        lambda params: {
            "command": "fake",
            "argv": ["llama-server", "--model", params["model_path"]],
            "issues": [],
        },
    )

    status = run_ui.api_runtime_start({"model_path": str(model)})

    assert status["running"] is True
    assert calls[0][0] == ["llama-server", "--model", str(model.resolve())]
    assert calls[0][1]["cwd"] == model.parent.resolve()


def test_environment_creation_requires_explicit_ui_confirmation():
    with pytest.raises(ValueError, match="explicit confirmation"):
        run_ui.api_create_environment({"profile": "fastdms", "confirm": False})


def test_environment_scan_forwards_cuda_toolkit_selection(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_ui,
        "scan_environment_profiles",
        lambda root, **kwargs: calls.append((root, kwargs)) or {"profiles": []},
    )

    result = run_ui.api_scan_environments(
        {"root": "/tmp/envs", "cuda_toolkit": "/usr/local/cuda-12.6"}
    )

    assert result == {"profiles": []}
    assert calls == [
        (
            "/tmp/envs",
            {
                "cuda_toolkit": "/usr/local/cuda-12.6",
                "local_source_profile": None,
                "local_source": None,
                "manual_dependency_override": False,
            },
        )
    ]


def test_environment_creation_forwards_cuda_toolkit_selection(monkeypatch):
    calls = []

    class FakeJobs:
        def start_create(self, profile, **kwargs):
            calls.append((profile, kwargs))
            return {"id": "job", "status": "queued"}

    monkeypatch.setattr(run_ui, "ENVIRONMENT_JOBS", FakeJobs())
    monkeypatch.setattr(run_ui, "UI_MUTATIONS_ENABLED", True)

    result = run_ui.api_create_environment(
        {
            "profile": "fastdms",
            "root": "/tmp/envs",
            "python": "3.11",
            "cuda_toolkit": "/usr/local/cuda-12.6",
            "confirm": True,
        }
    )

    assert result["status"] == "queued"
    assert calls == [
        (
            "fastdms",
            {
                "root": "/tmp/envs",
                "python": "3.11",
                "cuda_toolkit": "/usr/local/cuda-12.6",
                "local_source": None,
                "build_from_source": False,
                "max_jobs": 2,
                "recreate": False,
            },
        )
    ]


def test_environment_creation_forwards_matching_local_source(monkeypatch):
    calls = []

    class FakeJobs:
        def start_create(self, profile, **kwargs):
            calls.append((profile, kwargs))
            return {"id": "job", "status": "queued"}

    monkeypatch.setattr(run_ui, "ENVIRONMENT_JOBS", FakeJobs())
    monkeypatch.setattr(run_ui, "UI_MUTATIONS_ENABLED", True)

    run_ui.api_create_environment(
        {
            "profile": "fastdms",
            "root": "/tmp/envs",
            "local_source_profile": "fastdms",
            "local_source": "/tmp/addons/fastdms",
            "confirm": True,
        }
    )

    assert calls[0][1]["local_source"] == "/tmp/addons/fastdms"


def test_godzilla_plan_is_bounded_to_saved_roots(tmp_path, monkeypatch):
    addon_root = tmp_path / "addons"
    checkout = addon_root / "godzilla-llama.cpp"
    model_root = tmp_path / "models"
    model = model_root / "model.gguf"
    (checkout / "ggml").mkdir(parents=True)
    (checkout / "common").mkdir()
    (checkout / "scripts").mkdir()
    (checkout / "CMakeLists.txt").write_text("", encoding="utf-8")
    (checkout / "GODZILLA_KING.md").write_text("", encoding="utf-8")
    (checkout / "common" / "arg.cpp").write_text("kvarn", encoding="utf-8")
    (checkout / "scripts" / "ensure-triattention.ps1").write_text("", encoding="utf-8")
    model_root.mkdir()
    model.write_bytes(b"gguf")
    store = UISettingsStore(tmp_path / "ui.json")
    store.save(
        {
            **DEFAULT_UI_SETTINGS,
            "model_root": str(model_root),
            "addon_roots": [str(addon_root)],
        }
    )
    monkeypatch.setattr(run_ui, "SETTINGS_STORE", store)

    result = run_ui.api_plan_godzilla({"checkout": str(checkout), "gguf": str(model)})

    assert result["checkout"] == str(checkout.resolve())
    assert result["gguf"] == str(model.resolve())
    assert result["kvarn_calibration_required"] is False
    with pytest.raises(ValueError, match="saved add-on root"):
        run_ui.api_plan_godzilla({"checkout": str(tmp_path / "outside"), "gguf": str(model)})
    with pytest.raises(ValueError, match="checkout or model folder"):
        run_ui.api_plan_godzilla(
            {
                "checkout": str(checkout),
                "gguf": str(model),
                "output": str(tmp_path / "outside.triattention"),
            }
        )


def test_godzilla_official_plan_accepts_calibrator_and_text_inside_saved_roots(
    tmp_path, monkeypatch
):
    addon_root = tmp_path / "addons"
    checkout = addon_root / "godzilla-llama.cpp"
    calibrator = addon_root / "triattention" / "scripts" / "calibrate.py"
    model_root = tmp_path / "models"
    model = model_root / "model.gguf"
    calibration_input = model_root / "calibration.txt"
    python = tmp_path / "python.exe"
    (checkout / "ggml").mkdir(parents=True)
    (checkout / "common").mkdir()
    (checkout / "src").mkdir()
    (checkout / "scripts").mkdir()
    (checkout / "CMakeLists.txt").write_text("", encoding="utf-8")
    (checkout / "GODZILLA_KING.md").write_text("", encoding="utf-8")
    (checkout / "common" / "arg.cpp").write_text("kvarn", encoding="utf-8")
    calibrator.parent.mkdir(parents=True)
    calibrator.write_text(
        "AutoModelForCausalLM AutoTokenizer --max-length --attn-implementation "
        "q_mean_real q_mean_imag q_abs_mean",
        encoding="utf-8",
    )
    model_root.mkdir()
    model.write_bytes(b"gguf")
    calibration_input.write_text("coherent calibration text", encoding="utf-8")
    python.write_bytes(b"python")
    store = UISettingsStore(tmp_path / "ui.json")
    store.save(
        {
            **DEFAULT_UI_SETTINGS,
            "model_root": str(model_root),
            "addon_roots": [str(addon_root)],
        }
    )
    monkeypatch.setattr(run_ui, "SETTINGS_STORE", store)
    monkeypatch.setattr(
        "multi_turboquant.integration.godzilla_workspace.inspect_calibration_python",
        lambda path, **kwargs: _valid_calibration_python(path),
    )

    result = run_ui.api_plan_godzilla(
        {
            "checkout": str(checkout),
            "gguf": str(model),
            "python": str(python),
            "calibrator": str(calibrator),
            "calibration_input": str(calibration_input),
            "hf_model": "org/model",
            "mode": "official_python",
        }
    )

    assert result["ready"] is True
    assert result["mode"] == "official_python"
    assert result["calibration_input"] == str(calibration_input.resolve())

    outside_hf_model = tmp_path / "outside-hf"
    outside_hf_model.mkdir()
    with pytest.raises(ValueError, match="local Hugging Face model"):
        run_ui.api_plan_godzilla(
            {
                "checkout": str(checkout),
                "gguf": str(model),
                "python": str(python),
                "calibrator": str(calibrator),
                "calibration_input": str(calibration_input),
                "hf_model": str(outside_hf_model),
                "mode": "official_python",
            }
        )


def test_godzilla_plan_auto_selects_official_checkout_and_environment_python(tmp_path, monkeypatch):
    addon_root = tmp_path / "addons"
    checkout = addon_root / "godzilla-llama.cpp"
    triattention = addon_root / "triattention"
    calibrator = triattention / "scripts" / "calibrate.py"
    environment_root = tmp_path / "envs"
    calibration_python = run_ui.environment_python(environment_root / "triattention")
    model_root = tmp_path / "models"
    model = model_root / "model.gguf"
    calibration_input = model_root / "calibration.txt"
    (checkout / "ggml").mkdir(parents=True)
    (checkout / "common").mkdir()
    (checkout / "scripts").mkdir()
    (checkout / "CMakeLists.txt").write_text("", encoding="utf-8")
    (checkout / "GODZILLA_KING.md").write_text("", encoding="utf-8")
    (checkout / "common" / "arg.cpp").write_text("kvarn", encoding="utf-8")
    (triattention / "triattention").mkdir(parents=True)
    (triattention / "docs").mkdir()
    (triattention / "setup.py").write_text("", encoding="utf-8")
    (triattention / "docs" / "calibration.md").write_text("docs", encoding="utf-8")
    calibrator.parent.mkdir()
    calibrator.write_text(
        "AutoModelForCausalLM AutoTokenizer --max-length --attn-implementation "
        "q_mean_real q_mean_imag q_abs_mean",
        encoding="utf-8",
    )
    calibration_python.parent.mkdir(parents=True)
    calibration_python.write_bytes(b"python")
    model_root.mkdir()
    model.write_bytes(b"gguf")
    calibration_input.write_text("coherent calibration text", encoding="utf-8")
    store = UISettingsStore(tmp_path / "ui.json")
    store.save(
        {
            **DEFAULT_UI_SETTINGS,
            "model_root": str(model_root),
            "environment_root": str(environment_root),
            "addon_roots": [str(addon_root)],
        }
    )
    monkeypatch.setattr(run_ui, "SETTINGS_STORE", store)
    discovery = {
        "schema": 1,
        "selected": str(calibration_python.resolve()),
        "candidate_count": 1,
        "checked_count": 1,
        "probe_limit": 8,
        "truncated": False,
        "bounded": True,
        "locations": ["managed"],
        "attempts": [_valid_calibration_python(calibration_python)],
    }
    monkeypatch.setattr(
        run_ui,
        "select_compatible_calibration_python",
        lambda **kwargs: discovery,
    )
    monkeypatch.setattr(
        "multi_turboquant.integration.godzilla_workspace.inspect_calibration_python",
        lambda path, **kwargs: _valid_calibration_python(path),
    )
    monkeypatch.setattr(
        run_ui,
        "plan_environment",
        lambda *args, **kwargs: SimpleNamespace(ready=True, issues=()),
    )

    result = run_ui.api_plan_godzilla(
        {
            "checkout": str(checkout),
            "gguf": str(model),
            "calibration_input": str(calibration_input),
            "hf_model": "org/model",
            "mode": "official_python",
        }
    )

    assert result["ready"] is True
    assert result["python"] == str(calibration_python.resolve())
    assert result["calibrator"] == str(calibrator.resolve())
    assert result["dependency_repair"]["available"] is False
    assert result["dependency_repair"]["needed"] is False
    assert result["dependency_repair"]["profile"] == "triattention"
    assert result["python_discovery"] == discovery
    assert result["resource_policy"]["max_concurrent_calibrations"] == 1


def test_managed_triattention_python_preserves_linux_venv_symlink(tmp_path, monkeypatch):
    environment_root = tmp_path / "envs"
    base = tmp_path / "uv" / "python3.11"
    interpreter = environment_root / "triattention" / ".venv" / "bin" / "python"
    base.parent.mkdir()
    base.write_bytes(b"python")
    interpreter.parent.mkdir(parents=True)
    try:
        interpreter.symlink_to(base)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    store = UISettingsStore(tmp_path / "ui.json")
    store.save({**DEFAULT_UI_SETTINGS, "environment_root": str(environment_root)})
    monkeypatch.setattr(run_ui, "SETTINGS_STORE", store)
    monkeypatch.setattr(
        run_ui,
        "environment_python",
        lambda target: target / ".venv" / "bin" / "python",
    )

    selected = run_ui._default_triattention_python(store.load())

    assert selected == interpreter.absolute()
    assert selected != base.resolve()


def test_godzilla_plan_offers_managed_repair_for_incompatible_custom_python(tmp_path, monkeypatch):
    environment_root = tmp_path / "envs"
    store = UISettingsStore(tmp_path / "ui.json")
    store.save({**DEFAULT_UI_SETTINGS, "environment_root": str(environment_root)})
    plan = SimpleNamespace(
        mode="official_python",
        issues=[SimpleNamespace(code="calibration_dependencies_missing")],
        to_dict=lambda: {"ready": False},
    )
    monkeypatch.setattr(run_ui, "SETTINGS_STORE", store)
    monkeypatch.setattr(run_ui, "_godzilla_plan_from_params", lambda params: plan)
    monkeypatch.setattr(
        run_ui,
        "plan_environment",
        lambda *args, **kwargs: SimpleNamespace(ready=True, issues=()),
    )

    result = run_ui.api_plan_godzilla({"python": str(tmp_path / "custom-python")})

    assert result["dependency_repair"]["available"] is True
    assert result["dependency_repair"]["selection_reset_required"] is True


def test_godzilla_plan_preflights_managed_repair(tmp_path, monkeypatch):
    environment_root = tmp_path / "envs"
    store = UISettingsStore(tmp_path / "ui.json")
    store.save({**DEFAULT_UI_SETTINGS, "environment_root": str(environment_root)})
    plan = SimpleNamespace(
        mode="official_python",
        issues=[SimpleNamespace(code="missing_calibration_python")],
        to_dict=lambda: {"ready": False},
    )
    unavailable = SimpleNamespace(
        ready=False,
        issues=[SimpleNamespace(severity="error", message="uv is unavailable")],
    )
    monkeypatch.setattr(run_ui, "SETTINGS_STORE", store)
    monkeypatch.setattr(run_ui, "_godzilla_plan_from_params", lambda params: plan)
    monkeypatch.setattr(run_ui, "plan_environment", lambda *args, **kwargs: unavailable)

    managed_python = run_ui._managed_triattention_python(store.load())
    result = run_ui.api_plan_godzilla({"python": str(managed_python)})

    assert result["dependency_repair"]["available"] is False
    assert result["dependency_repair"]["managed_python"] == str(managed_python)
    assert result["dependency_repair"]["needed"] is True
    assert "uv is unavailable" in result["dependency_repair"]["message"]


def test_managed_triattention_repair_ignores_unreviewed_overrides(tmp_path, monkeypatch):
    environment_root = tmp_path / "envs"
    store = UISettingsStore(tmp_path / "ui.json")
    store.save({**DEFAULT_UI_SETTINGS, "environment_root": str(environment_root)})
    calls = []

    class FakeJobs:
        def start_create(self, profile, **kwargs):
            calls.append((profile, kwargs))
            return {"id": "repair-job", "status": "queued"}

    monkeypatch.setattr(run_ui, "SETTINGS_STORE", store)
    monkeypatch.setattr(run_ui, "ENVIRONMENT_JOBS", FakeJobs())
    monkeypatch.setattr(run_ui, "UI_MUTATIONS_ENABLED", True)

    result = run_ui.api_repair_triattention_environment(
        {
            "confirm": True,
            "python": "/unreviewed/python",
            "local_source": "/unreviewed/source",
            "max_jobs": 64,
        }
    )

    assert result == {"id": "repair-job", "status": "queued"}
    assert calls == [
        (
            "triattention",
            {"root": str(environment_root), "max_jobs": 2, "recreate": True},
        )
    ]


def test_godzilla_creation_requires_confirmation():
    with pytest.raises(ValueError, match="explicit confirmation"):
        run_ui.api_create_godzilla({"confirm": False})


def test_calibration_text_generation_is_confirmed_and_bounded_to_model_root(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    model_root.mkdir()
    store = UISettingsStore(tmp_path / "ui.json")
    store.save({**DEFAULT_UI_SETTINGS, "model_root": str(model_root)})
    monkeypatch.setattr(run_ui, "SETTINGS_STORE", store)
    monkeypatch.setattr(run_ui, "UI_MUTATIONS_ENABLED", True)

    with pytest.raises(ValueError, match="explicit confirmation"):
        run_ui.api_generate_calibration_text({"n_tokens": 512})

    result = run_ui.api_generate_calibration_text({"n_tokens": 512, "confirm": True})

    output = Path(result["path"])
    assert output.is_file()
    assert output.is_relative_to(model_root.resolve())
    assert result["reused"] is False

    reused = run_ui.api_generate_calibration_text({"n_tokens": 512, "confirm": True})
    assert reused["reused"] is True


def test_godzilla_creation_forwards_checked_plan(tmp_path, monkeypatch):
    plan = SimpleNamespace(
        checkout=tmp_path / "godzilla",
        gguf=tmp_path / "model.gguf",
        output=tmp_path / "model.triattention",
        python=tmp_path / "python.exe",
        calibrator=tmp_path / "calibrator.py",
        calibration_input=tmp_path / "calibration.txt",
        official_stats_input=None,
        hf_model="org/model",
        n_tokens=2048,
        device="cuda",
        mode="official_python",
        attention_implementation="sdpa",
        dependency_override=False,
        python_discovery={"selected": str(tmp_path / "python.exe")},
    )
    calls = []

    class FakeJobs:
        def start(self, checkout, gguf, **kwargs):
            calls.append((checkout, gguf, kwargs))
            return {"id": "godzilla-job", "status": "queued"}

    monkeypatch.setattr(run_ui, "_godzilla_plan_from_params", lambda params: plan)
    monkeypatch.setattr(run_ui, "GODZILLA_JOBS", FakeJobs())
    monkeypatch.setattr(run_ui, "UI_MUTATIONS_ENABLED", True)

    result = run_ui.api_create_godzilla({"confirm": True})

    assert result["status"] == "queued"
    assert calls == [
        (
            plan.checkout,
            plan.gguf,
            {
                "output": plan.output,
                "python": plan.python,
                "calibrator": plan.calibrator,
                "calibration_input": plan.calibration_input,
                "official_stats_input": None,
                "hf_model": "org/model",
                "n_tokens": 2048,
                "device": "cuda",
                "mode": "official_python",
                "attention_implementation": "sdpa",
                "tokenizer_backend": "transformers",
                "dependency_override": False,
                "python_discovery": plan.python_discovery,
            },
        )
    ]


def test_gigatoken_scan_uses_saved_environment_root(tmp_path, monkeypatch):
    store = UISettingsStore(tmp_path / "ui.json")
    environment_root = tmp_path / "envs"
    store.save({**DEFAULT_UI_SETTINGS, "environment_root": str(environment_root)})
    calls = []
    monkeypatch.setattr(run_ui, "SETTINGS_STORE", store)
    monkeypatch.setattr(
        run_ui,
        "scan_gigatoken_interpreters",
        lambda **kwargs: calls.append(kwargs) or {"interpreters": []},
    )

    result = run_ui.api_scan_gigatoken({})

    assert result == {"interpreters": []}
    assert calls == [{"environment_root": str(environment_root)}]
