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
