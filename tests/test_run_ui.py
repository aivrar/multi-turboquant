# SPDX-License-Identifier: MIT
"""Tests for the lightweight web UI command API."""

import pytest

from multi_turboquant import CacheMethod
from run_ui import (
    _command_config,
    _command_context_extension_config,
    api_generate_command,
    api_methods,
    api_presets,
    api_scan_llamacpp,
)


def test_command_config_translates_triattention_k_method_to_flag():
    config = _command_config({"k_method": "triattention", "v_method": "f16"})

    assert config.k_method == CacheMethod.FP16
    assert config.v_method == CacheMethod.FP16
    assert config.triattention_enabled is True


def test_api_generate_command_accepts_legacy_triattention_selection():
    with pytest.warns(RuntimeWarning, match="TriAttention"):
        result = api_generate_command({
            "k_method": "triattention",
            "v_method": "f16",
            "model_path": "/opt/models/model.gguf",
            "port": 8080,
            "context": 4096,
            "parallel": 1,
        })

    assert "--cache-type-k f16" in result["command"]
    assert "--cache-type-v f16" in result["command"]
    assert any(issue["method"] == "triattention" for issue in result["issues"])


def test_api_generate_command_supports_patched_triattention_flags():
    result = api_generate_command({
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
    })

    assert "--cache-type-k turbo3" in result["command"]
    assert "--cache-type-v turbo3" in result["command"]
    assert "--triattention-stats model.triattention" in result["command"]
    assert "--triattention-budget 2048" in result["command"]
    assert "--triattention-window 256" in result["command"]
    assert "--triattention-log" in result["command"]
    assert not any(issue["method"] == "command" for issue in result["issues"])


def test_api_generate_command_returns_error_for_missing_triattention_stats():
    result = api_generate_command({
        "k_method": "turbo3",
        "v_method": "turbo3",
        "use_custom_triattention_llamacpp": True,
        "model_path": "/opt/models/model.gguf",
        "port": 8080,
        "context": 4096,
        "parallel": 1,
    })

    assert result["command"] == ""
    triattention_issues = [
        issue for issue in result["issues"] if issue["method"] == "triattention"
    ]
    assert len(triattention_issues) == 1
    assert "Stats Path" in triattention_issues[0]["message"]
    assert "--triattention-calibrate" in triattention_issues[0]["suggestion"]
    assert not any(issue["method"] == "command" for issue in result["issues"])


def test_api_generate_command_supports_cuda_weight_share_wrapper():
    result = api_generate_command({
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
    })

    assert result["command"].startswith(
        "env LD_PRELOAD=/opt/cuda-llm-weight-share.so "
    )
    assert "MODEL_SIZE=123456" in result["command"]
    assert "MODEL_SIZE_TOLERANCE=1024" in result["command"]
    assert "CUDA_VRAM_IPC_NAME=/cuda_vram_ipc_test" in result["command"]
    assert "llama-server" in result["command"]


def test_command_context_extension_config_parses_ui_values():
    assert _command_context_extension_config({"rope_scaling": "off"}) is None

    config = _command_context_extension_config({
        "rope_scale": "8",
        "yarn_orig_ctx": "4096",
        "yarn_ext_factor": "0",
        "yarn_attn_factor": "1.1",
    })

    assert config.rope_scaling is None
    assert config.rope_scale == 8.0
    assert config.yarn_orig_ctx == 4096
    assert config.yarn_ext_factor == 0.0
    assert config.yarn_attn_factor == 1.1


def test_api_generate_command_supports_context_extension_flags():
    result = api_generate_command({
        "k_method": "f16",
        "v_method": "f16",
        "model_path": "/opt/models/model.gguf",
        "port": 8080,
        "context": 32768,
        "parallel": 1,
        "rope_scale": "8",
        "yarn_orig_ctx": "4096",
    })

    assert "-c 32768" in result["command"]
    assert "--rope-scaling yarn" in result["command"]
    assert "--rope-scale 8.0" in result["command"]
    assert "--yarn-orig-ctx 4096" in result["command"]
    assert not any(issue["method"] == "command" for issue in result["issues"])


def test_api_generate_command_reports_invalid_context_extension():
    result = api_generate_command({
        "k_method": "f16",
        "v_method": "f16",
        "model_path": "/opt/models/model.gguf",
        "port": 8080,
        "context": 32768,
        "parallel": 1,
        "rope_scaling": "linear",
        "yarn_orig_ctx": "4096",
    })

    assert result["command"] == ""
    assert any(
        issue["method"] == "command" and "YaRN options require" in issue["message"]
        for issue in result["issues"]
    )


def test_api_scan_llamacpp_reports_missing_binary():
    result = api_scan_llamacpp({
        "binary": "__definitely_missing_llama_server__",
        "timeout_seconds": "0.1",
    })

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
    result = api_generate_command({
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
    })

    assert "--cache-type-k kvarn4" in result["command"]
    assert "--cache-type-v kvarn4" in result["command"]
    assert "--spec-type dflash" in result["command"]
    assert "--spec-draft-model /opt/models/draft.gguf" in result["command"]
    assert "--spec-draft-n-max 16" in result["command"]
    assert not any(issue["method"] == "command" for issue in result["issues"])


def test_api_generate_command_rejects_kvarn_without_godzilla_profile():
    result = api_generate_command({
        "k_method": "kvarn4",
        "v_method": "kvarn4",
        "model_path": "/opt/models/model.gguf",
        "port": 8080,
        "context": 8192,
        "parallel": 1,
    })

    assert result["command"] == ""
    assert any(issue["method"] == "command" for issue in result["issues"])
