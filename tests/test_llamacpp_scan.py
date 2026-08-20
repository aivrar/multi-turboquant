# SPDX-License-Identifier: MIT
"""Tests for llama.cpp capability scanning."""

from multi_turboquant.integration import parse_llamacpp_help, scan_llamacpp_binary


def test_parse_llamacpp_help_detects_context_and_extension_features():
    help_text = """
Version: godzilla-test
  --cache-type-k TYPE        supported: f16 q8_0 turbo3_tcq kvarn2..kvarn8
  --cache-type-v TYPE        supported: f16 q8_0 turbo3_tcq kvarn2..kvarn8
  --rope-scaling {none,linear,yarn}
  --rope-scale N
  --yarn-orig-ctx N
  --triattention-stats FILE
  --spec-type TYPE           supported: dflash draft-mtp
  --spec-dflash-cross-ctx N
"""

    capabilities = parse_llamacpp_help(
        help_text,
        binary="llama-server-godzilla",
        help_returncode=0,
    )

    assert capabilities.scanned is True
    assert capabilities.build_info == "Version: godzilla-test"
    assert capabilities.supports_context_extension is True
    assert capabilities.supports_yarn is True
    assert capabilities.supports_triattention is True
    assert capabilities.supports_kvarn is True
    assert capabilities.supports_speculative is True
    assert capabilities.supports_dflash is True
    assert capabilities.supports_cache_type("kvarn4") is True
    assert capabilities.supports_cache_type("turbo3_tcq") is True
    assert capabilities.supports_speculative_type("draft-mtp") is True

    data = capabilities.to_dict()
    assert data["binary"] == "llama-server-godzilla"
    assert data["supports_yarn"] is True
    assert "kvarn4" in data["cache_types"]


def test_parse_llamacpp_help_handles_plain_upstream_binary():
    help_text = """
build: 1234
  --cache-type-k TYPE        supported: f16 q8_0 q4_0
  --cache-type-v TYPE        supported: f16 q8_0 q4_0
"""

    capabilities = parse_llamacpp_help(help_text)

    assert capabilities.supports_context_extension is False
    assert capabilities.supports_yarn is False
    assert capabilities.supports_kvarn is False
    assert capabilities.supports_triattention is False
    assert capabilities.supports_cache_type("f16") is True
    assert capabilities.supports_cache_type("kvarn4") is False


def test_parse_llamacpp_help_detects_gigatoken_self_identification():
    capabilities = parse_llamacpp_help(
        "build: llama.cpp base\nGigatoken tokenizer enabled\n  --port N"
    )

    assert capabilities.supports_gigatoken is True
    assert capabilities.build_info == "build: llama.cpp base"
    assert capabilities.to_dict()["supports_gigatoken"] is True


def test_parse_llamacpp_help_detects_pflash_kvflash_fork_without_claiming_godzilla():
    capabilities = parse_llamacpp_help(
        """
build: pflash-kvflash-test
  --pflash-mode MODE
  --pflash-keep-ratio N
  --pflash-drafter FILE
  --kvflash N
  --kvflash-policy POLICY
  --spec-type TYPE       supported: draft-dflash
  /props reports runtime configuration
"""
    )

    assert capabilities.runtime_family == "pflash_llamacpp"
    assert capabilities.supports_pflash is True
    assert capabilities.supports_kvflash is True
    assert capabilities.supports_dflash is True
    assert capabilities.supports_props_endpoint is True
    assert capabilities.supports_kvarn is False

    data = capabilities.to_dict()
    assert data["runtime_family"] == "pflash_llamacpp"
    assert data["supports_pflash"] is True
    assert data["supports_kvflash"] is True


def test_parse_llamacpp_help_detects_lucebox_runtime_capabilities():
    capabilities = parse_llamacpp_help(
        """
version: lucebox-test
  --prefill-compression MODE
  --prefill-drafter FILE
  --kvflash N
  --ddtree
  --ddtree-budget N
  --specla
"""
    )

    assert capabilities.runtime_family == "lucebox"
    assert capabilities.supports_pflash is True
    assert capabilities.supports_kvflash is True
    assert capabilities.supports_ddtree is True
    assert capabilities.supports_specla is True


def test_scan_llamacpp_binary_reports_missing_binary():
    capabilities = scan_llamacpp_binary(
        "__definitely_missing_llama_server__",
        timeout_seconds=0.1,
    )

    assert capabilities.binary == "__definitely_missing_llama_server__"
    assert capabilities.scanned is False
    assert capabilities.error
