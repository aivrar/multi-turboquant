# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from multi_turboquant.integration import godzilla_gigatoken as runtime
from multi_turboquant.integration.godzilla_gigatoken_cli import build_parser


def _available_tools(monkeypatch, names: set[str] | None = None) -> None:
    allowed = names or {"git", "cmake", "ctest", "cargo", "rustc", "rustup"}
    monkeypatch.setattr(runtime, "_supported_platform", lambda: True)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: f"/tools/{name}" if name in allowed else None)


def test_prepare_plan_is_read_only_and_pinned(tmp_path, monkeypatch):
    _available_tools(monkeypatch)
    target = tmp_path / "combined"

    plan = runtime.plan_godzilla_gigatoken(target)

    assert plan.ready
    assert plan.action == "prepare"
    assert not target.exists()
    assert plan.commands[0] == ("git", "init", str(target.resolve()))
    assert plan.source_profile.id == runtime.DEFAULT_GODZILLA_PROFILE
    assert plan.to_dict()["pins"]["godzilla"] == runtime.GODZILLA_COMMIT


def test_requested_09214b160_profile_is_exact_and_read_only(tmp_path, monkeypatch):
    _available_tools(monkeypatch)
    target = tmp_path / "combined"

    plan = runtime.plan_godzilla_gigatoken(
        target,
        godzilla_profile="09214b160",
    )

    assert plan.ready
    assert plan.source_profile.commit == runtime.GODZILLA_COMPAT_COMMIT
    assert plan.to_dict()["godzilla_profile"]["id"] == "09214b160"
    assert any(runtime.GODZILLA_COMPAT_COMMIT in command for command in plan.commands)
    assert not target.exists()


def test_prepare_plan_refuses_an_existing_target(tmp_path, monkeypatch):
    _available_tools(monkeypatch)
    target = tmp_path / "combined"
    target.mkdir()

    plan = runtime.plan_godzilla_gigatoken(target)

    assert not plan.ready
    assert any(issue.code == "target_exists" for issue in plan.issues)


def test_verify_plan_does_not_require_compilers_or_nvcc(tmp_path, monkeypatch):
    _available_tools(monkeypatch, {"git", "ctest"})
    monkeypatch.setattr(
        runtime,
        "inspect_godzilla_gigatoken",
        lambda _path, **_kwargs: {"valid": True, "issues": []},
    )

    plan = runtime.plan_godzilla_gigatoken(tmp_path, action="verify", backend="cuda")

    assert plan.ready
    assert plan.cuda_compiler is None
    assert plan.commands == ()


def test_build_plan_validates_optional_fixture_directory(tmp_path, monkeypatch):
    _available_tools(monkeypatch)
    monkeypatch.setattr(
        runtime,
        "inspect_godzilla_gigatoken",
        lambda _path, **_kwargs: {"valid": True, "issues": []},
    )

    plan = runtime.plan_godzilla_gigatoken(
        tmp_path,
        action="build",
        fixture_dir=tmp_path / "missing-fixtures",
    )

    assert not plan.ready
    assert any(issue.code == "invalid_fixture_dir" for issue in plan.issues)


def test_cuda_compiler_selection_pins_visual_studio_toolkit(tmp_path):
    compiler = tmp_path / "cuda" / "bin" / "nvcc.exe"

    assert runtime._cuda_compiler_configure_args(
        compiler, None, system_name="nt"
    ) == ("-T", f"cuda={tmp_path / 'cuda'}")
    assert runtime._cuda_compiler_configure_args(
        compiler, "Ninja", system_name="nt"
    ) == (f"-DCMAKE_CUDA_COMPILER={compiler}",)


def test_gigatoken_configure_uses_pinned_visual_studio_toolkit(tmp_path, monkeypatch):
    compiler = tmp_path / "cuda" / "bin" / "nvcc.exe"
    monkeypatch.setattr(runtime.os, "name", "nt")

    command = runtime._configure_command(
        tmp_path / "source",
        tmp_path / "build",
        backend="cuda",
        with_curl=False,
        generator=None,
        cuda_compiler=compiler,
        fixture_dir=None,
    )

    assert command[command.index("-T") + 1] == f"cuda={tmp_path / 'cuda'}"
    assert not any(item.startswith("-DCMAKE_CUDA_COMPILER=") for item in command)


def test_review_diff_selection_is_path_and_hash_bounded(monkeypatch):
    keep = b"diff --git a/keep b/keep\n--- a/keep\n+++ b/keep\n@@ -1 +1 @@\n-a\n+b\n"
    ignore = b"diff --git a/ignore b/ignore\n--- a/ignore\n+++ b/ignore\n@@ -1 +1 @@\n-x\n+y\n"
    monkeypatch.setattr(runtime, "_SELECTED_DIFF_PATHS", frozenset({"keep"}))
    monkeypatch.setattr(runtime, "SELECTED_DIFF_SHA256", hashlib.sha256(keep).hexdigest())

    assert runtime._select_reviewed_diff(keep + ignore) == keep

    with pytest.raises(RuntimeError, match="missing selected paths"):
        runtime._select_reviewed_diff(ignore)


def test_adaptation_uses_exact_anchors_and_fixes_windows_patch_handling(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "cmake").mkdir()
    (tmp_path / "CMakeLists.txt").write_text(
        'option(LLAMA_LLGUIDANCE "llama-common: include LLGuidance library for structured output in common utils" OFF)\n'
        "if (NOT TARGET ggml AND NOT LLAMA_USE_SYSTEM_GGML)\n"
        "    set(GGML_BUILD_NUMBER ${LLAMA_BUILD_NUMBER})\n"
        "    set(GGML_BUILD_COMMIT ${LLAMA_BUILD_COMMIT})\n"
        "    add_subdirectory(ggml)\n"
        "    # ... otherwise assume ggml is added by a parent CMakeLists.txt\n"
        "endif()\n\n#\n# build the library\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "CMakeLists.txt").write_text(
        "if (NOT GGML_CUDA OR GGML_BACKEND_DL)\n"
        "    target_sources(llama PRIVATE llama-triattention-gpu-stub.cpp)\n"
        "endif()\n\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "CMakeLists.txt").write_text(
        "llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         "
        "ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)\n\n",
        encoding="utf-8",
    )
    (tmp_path / "cmake" / "gigatoken.cmake").write_text(
        'message(FATAL_ERROR "GigaToken submodule is missing; run git submodule update --init vendor/gigatoken")\n'
        'file(SHA256 "${GIGATOKEN_PATCH}" GIGATOKEN_PATCH_ACTUAL_SHA256)\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "llama-gigatoken.cpp").write_text(
        'LLAMA_LOG_INFO("%s: GigaToken encode: %zu bytes -> %zu tokens\\n", __func__, text.size(), buffer.len);\n',
        encoding="utf-8",
    )

    runtime._adapt_reviewed_port(tmp_path)

    assert "LLAMA_GIGATOKEN" in (tmp_path / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "gigatoken::llama" in (tmp_path / "src" / "CMakeLists.txt").read_text(encoding="utf-8")
    test_cmake = (tmp_path / "tests" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "test-gigatoken-matrix" in test_cmake
    assert 'if (EXISTS "${LLAMA_GIGATOKEN_FIXTURE_DIR}/ggml-vocab-deepseek-v3.gguf")' in test_cmake
    cmake = (tmp_path / "cmake" / "gigatoken.cmake").read_text(encoding="utf-8")
    assert "GIGATOKEN_PATCH_CONTENT_LF" in cmake
    assert "GigaToken source is missing" in cmake
    assert "LLAMA_LOG_DEBUG" in (tmp_path / "src" / "llama-gigatoken.cpp").read_text(encoding="utf-8")


def test_inspection_detects_changed_runtime_files(tmp_path, monkeypatch):
    for relative in runtime._RUNTIME_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    vendor = tmp_path / "vendor" / "gigatoken"
    vendor.mkdir(parents=True)
    hashes = runtime._runtime_file_hashes(tmp_path)
    manifest = {
        "schema": 1,
        "godzilla": {"commit": runtime.GODZILLA_COMMIT},
        "gigatoken": {"commit": runtime.GIGATOKEN_COMMIT},
        "gigatoken_llama": {
            "commit": runtime.GIGATOKEN_LLAMA_COMMIT,
            "diff_sha256": runtime.GIGATOKEN_LLAMA_DIFF_SHA256,
        },
        "rust_toolchain": runtime.GIGATOKEN_TOOLCHAIN,
        "runtime_files_sha256": hashes,
    }
    (tmp_path / ".mtq-gigatoken-runtime.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        runtime,
        "_git_revision",
        lambda path: runtime.GIGATOKEN_COMMIT if path.name == "gigatoken" else runtime.GODZILLA_COMMIT,
    )
    def fake_run(command, **_kwargs):
        stdout = (
            ("\n".join(sorted(runtime._TRACKED_RUNTIME_FILES)) + "\n").encode()
            if tuple(command[:2]) == ("git", "diff")
            else b""
        )
        return subprocess.CompletedProcess([], 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(runtime, "_run_checked", fake_run)

    assert runtime.inspect_godzilla_gigatoken(tmp_path)["valid"]
    (tmp_path / "src" / "llama-gigatoken.cpp").write_text("changed", encoding="utf-8")
    inspection = runtime.inspect_godzilla_gigatoken(tmp_path)
    assert not inspection["valid"]
    assert "src/llama-gigatoken.cpp" in " ".join(inspection["issues"])


def test_prepare_requires_confirmation_before_network_or_files(tmp_path, monkeypatch):
    _available_tools(monkeypatch)
    plan = runtime.plan_godzilla_gigatoken(tmp_path / "combined")
    monkeypatch.setattr(
        runtime,
        "_download_reviewed_diff",
        lambda: pytest.fail("network should not be used without confirmation"),
    )

    with pytest.raises(RuntimeError, match="not confirmed"):
        runtime.prepare_godzilla_gigatoken(plan)
    assert not plan.target.exists()


def test_cli_plan_can_target_build_or_verify():
    args = build_parser().parse_args(
        [
            "plan",
            "combined",
            "--for-action",
            "build",
            "--godzilla-profile",
            "09214b160",
        ]
    )
    assert args.action == "plan"
    assert args.for_action == "build"
    assert args.godzilla_profile == "09214b160"
