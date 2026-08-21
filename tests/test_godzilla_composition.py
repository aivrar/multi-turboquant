# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess

import pytest

from multi_turboquant.integration import godzilla_composition as runtime
from multi_turboquant.integration.godzilla_composition_cli import build_parser


def _tools(monkeypatch, names=frozenset({"git", "cmake"})):
    monkeypatch.setattr(runtime, "_supported_platform", lambda: True)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: f"/tools/{name}" if name in names else None)


def test_default_plan_is_exact_read_only_and_reports_bounded_kvflash(tmp_path, monkeypatch):
    _tools(monkeypatch)
    target = tmp_path / "composed"
    plan = runtime.plan_godzilla_composition(target)

    assert plan.ready
    assert not target.exists()
    assert plan.to_dict()["godzilla_commit"] == runtime.GODZILLA_COMPAT_COMMIT
    assert any(issue.code == "kvflash_slot_tier" for issue in plan.issues)


def test_composition_rejects_known_conflicts():
    issues = runtime.validate_godzilla_composition(runtime.GodzillaComposition(
        triattention=True, kvarn=True, spec_la=True,
    ))
    assert {issue.code for issue in issues if issue.severity == "error"} == {
        "triattention_kvarn_conflict", "specla_not_available",
    }


def test_prepare_requires_new_target_and_confirmation(tmp_path, monkeypatch):
    _tools(monkeypatch)
    target = tmp_path / "composed"
    plan = runtime.plan_godzilla_composition(target)
    with pytest.raises(RuntimeError, match="not confirmed"):
        runtime.prepare_godzilla_composition(plan)
    target.mkdir()
    assert not runtime.plan_godzilla_composition(target).ready


def test_inspection_is_hash_bounded(tmp_path, monkeypatch):
    for name in runtime._RUNTIME_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    manifest = {
        "schema": runtime.COMPOSITION_SCHEMA,
        "profile": runtime.COMPOSITION_PROFILE,
        "runtime_files_sha256": runtime._file_hashes(tmp_path),
    }
    (tmp_path / runtime.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runtime, "_git_revision", lambda _root: runtime.GODZILLA_COMPAT_COMMIT)
    monkeypatch.setattr(runtime, "_EXPECTED_RUNTIME_SHA256", manifest["runtime_files_sha256"])
    monkeypatch.setattr(
        runtime,
        "_git_source_state",
        lambda _root: (set(runtime._RUNTIME_FILES), {runtime.MANIFEST_NAME}),
    )
    assert runtime.inspect_godzilla_composition(tmp_path)["valid"]
    (tmp_path / "common/common.h").write_text("changed", encoding="utf-8")
    assert not runtime.inspect_godzilla_composition(tmp_path)["valid"]


def test_inspection_allows_owned_build_tree_but_rejects_other_untracked_files(
    tmp_path, monkeypatch
):
    for name in runtime._RUNTIME_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    hashes = runtime._file_hashes(tmp_path)
    (tmp_path / runtime.MANIFEST_NAME).write_text(json.dumps({
        "schema": runtime.COMPOSITION_SCHEMA,
        "profile": runtime.COMPOSITION_PROFILE,
        "runtime_files_sha256": hashes,
    }), encoding="utf-8")
    monkeypatch.setattr(runtime, "_git_revision", lambda _root: runtime.GODZILLA_COMPAT_COMMIT)
    monkeypatch.setattr(runtime, "_EXPECTED_RUNTIME_SHA256", hashes)
    untracked = {
        runtime.MANIFEST_NAME,
        "build-mtq-composition-cuda-sm86-sm89/CMakeCache.txt",
    }
    monkeypatch.setattr(
        runtime, "_git_source_state", lambda _root: (set(runtime._RUNTIME_FILES), untracked)
    )
    assert runtime.inspect_godzilla_composition(tmp_path)["valid"]
    untracked.add("unexpected.cpp")
    assert not runtime.inspect_godzilla_composition(tmp_path)["valid"]
    untracked.remove("unexpected.cpp")
    untracked.add("build-mtq-composition-disguised.cpp")
    assert not runtime.inspect_godzilla_composition(tmp_path)["valid"]


def test_verify_requires_composed_and_godzilla_flags(tmp_path, monkeypatch):
    build = tmp_path / "build-mtq-composition-cpu" / "bin"
    build.mkdir(parents=True)
    (build / "llama-server.exe").write_bytes(b"")
    monkeypatch.setattr(runtime, "inspect_godzilla_composition", lambda _root: {
        "valid": True, "issues": [],
    })
    monkeypatch.setattr(runtime, "_run_checked", lambda *_args, **_kwargs: subprocess.CompletedProcess(
        [], 0, stdout=b"--pflash --kvflash-pages --spec-branch-budget --triattention-stats", stderr=b""
    ))
    plan = runtime.GodzillaCompositionPlan(
        "verify", tmp_path, tmp_path / "build-mtq-composition-cpu", "cpu", 2, None, None, (), (),
    )
    assert runtime.verify_godzilla_composition(plan)["valid"]


def test_cli_plan_accepts_build_target():
    args = build_parser().parse_args(["plan", "combined", "--for-action", "build"])
    assert args.action == "plan"
    assert args.for_action == "build"


def test_cuda_plan_requires_an_explicit_or_discovered_compiler(tmp_path, monkeypatch):
    _tools(monkeypatch)
    plan = runtime.plan_godzilla_composition(tmp_path / "composed", backend="cuda")
    assert not plan.ready
    assert any(issue.code == "cuda_compiler_missing" for issue in plan.issues)


def test_cuda_plan_targets_qualified_sm86_and_sm89_explicitly(tmp_path, monkeypatch):
    _tools(monkeypatch)
    monkeypatch.setattr(runtime, "_resolve_cuda_compiler", lambda _toolkit: tmp_path / "nvcc")
    target = tmp_path / "composed"
    plan = runtime.plan_godzilla_composition(
        target,
        action="all",
        backend="cuda",
        cuda_toolkit=tmp_path / "cuda",
        cuda_architectures=("sm_86", 89),
    )
    assert plan.ready
    assert plan.cuda_architectures == ("86", "89")
    assert plan.build_dir.name.endswith("cuda-sm86-sm89")
    assert "-DCMAKE_CUDA_ARCHITECTURES=86;89" in plan.commands[-2]
    assert runtime._cuda_compiler_configure_args(
        tmp_path / "cuda" / "bin" / "nvcc.exe", None, system_name="nt"
    ) == ("-T", f"cuda={tmp_path / 'cuda'}")
    assert runtime._cuda_compiler_configure_args(
        tmp_path / "cuda" / "bin" / "nvcc", "Ninja", system_name="nt"
    ) == (f"-DCMAKE_CUDA_COMPILER={tmp_path / 'cuda' / 'bin' / 'nvcc'}",)


@pytest.mark.parametrize(
    ("backend", "architectures", "code"),
    (
        ("cpu", ("86",), "cuda_architecture_without_cuda"),
        ("cuda", ("86", "86"), "duplicate_cuda_architecture"),
        ("cuda", ("90",), "unqualified_cuda_architecture"),
    ),
)
def test_cuda_architecture_selection_fails_closed(
    tmp_path, monkeypatch, backend, architectures, code
):
    _tools(monkeypatch)
    monkeypatch.setattr(runtime, "_resolve_cuda_compiler", lambda _toolkit: tmp_path / "nvcc")
    plan = runtime.plan_godzilla_composition(
        tmp_path / "composed",
        backend=backend,
        cuda_architectures=architectures,
    )
    assert not plan.ready
    assert any(issue.code == code for issue in plan.issues)


def test_cuda_verification_checks_the_cmake_cache_architectures(tmp_path, monkeypatch):
    build = tmp_path / "build-mtq-composition-cuda-sm86-sm89"
    (build / "bin").mkdir(parents=True)
    (build / "bin" / "llama-server.exe").write_bytes(b"")
    (build / "CMakeCache.txt").write_text(
        "CMAKE_CUDA_ARCHITECTURES:STRING=86;89\n"
        f"CMAKE_CUDA_COMPILER:FILEPATH={tmp_path / 'nvcc'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "inspect_godzilla_composition", lambda _root: {
        "valid": True, "issues": [],
    })
    monkeypatch.setattr(runtime, "_run_checked", lambda *_args, **_kwargs: subprocess.CompletedProcess(
        [], 0, stdout=b"--pflash --kvflash-pages --spec-branch-budget --triattention-stats", stderr=b""
    ))
    plan = runtime.GodzillaCompositionPlan(
        "verify", tmp_path, build, "cuda", 2, None, tmp_path / "nvcc", (), (), ("86", "89")
    )
    assert runtime.verify_godzilla_composition(plan)["valid"]
    (build / "CMakeCache.txt").write_text(
        "CMAKE_CUDA_ARCHITECTURES:STRING=86\n"
        f"CMAKE_CUDA_COMPILER:FILEPATH={tmp_path / 'nvcc'}\n",
        encoding="utf-8",
    )
    assert not runtime.verify_godzilla_composition(plan)["valid"]


def test_cuda_verification_accepts_visual_studio_toolkit_nvcc(tmp_path, monkeypatch):
    build = tmp_path / "build-mtq-composition-cuda-sm86-sm89"
    (build / "bin").mkdir(parents=True)
    (build / "bin" / "llama-server.exe").write_bytes(b"")
    compiler = tmp_path / "cuda" / "bin" / "nvcc.exe"
    compiler.parent.mkdir(parents=True)
    compiler.write_bytes(b"")
    (build / "CMakeCache.txt").write_text(
        "CMAKE_CUDA_ARCHITECTURES:STRING=86;89\n"
        f"CUDAToolkit_NVCC_EXECUTABLE:FILEPATH={compiler}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "inspect_godzilla_composition", lambda _root: {
        "valid": True, "issues": [],
    })
    monkeypatch.setattr(runtime, "_run_checked", lambda *_args, **_kwargs: subprocess.CompletedProcess(
        [], 0, stdout=b"--pflash --kvflash-pages --spec-branch-budget --triattention-stats", stderr=b""
    ))
    plan = runtime.GodzillaCompositionPlan(
        "verify", tmp_path, build, "cuda", 2, None, compiler, (), (), ("86", "89")
    )

    assert runtime.verify_godzilla_composition(plan)["valid"]
