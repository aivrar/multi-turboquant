# SPDX-License-Identifier: MIT
"""Tests for the lightweight web UI command API."""

import pytest

from multi_turboquant import CacheMethod
from run_ui import _command_config, api_generate_command


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
    assert any(issue["method"] == "command" for issue in result["issues"])
    assert any(issue["method"] == "triattention" for issue in result["issues"])


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
