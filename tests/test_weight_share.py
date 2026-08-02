from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from multi_turboquant.integration import weight_share
from multi_turboquant.integration.weight_share_cli import main as weight_share_main
from multi_turboquant.ui.discovery import scan_addon_roots


def _source_tree(tmp_path: Path) -> Path:
    source = tmp_path / "cuda-llm-weight-share"
    source.mkdir()
    (source / "README.md").write_text("CUDA weight share", encoding="utf-8")
    (source / "LICENSE").write_text("MIT", encoding="utf-8")
    (source / "cuda-llm-weight-share.c").write_text(
        "#include <cuda_runtime_api.h>\n", encoding="utf-8"
    )
    return source


def _cuda_toolkit(tmp_path: Path) -> Path:
    toolkit = tmp_path / "cuda-12.6"
    include = toolkit / "include"
    include.mkdir(parents=True)
    (include / "cuda_runtime_api.h").write_text("", encoding="utf-8")
    return toolkit


def test_source_inspection_requires_exact_reviewed_provenance(tmp_path: Path, monkeypatch):
    source = _source_tree(tmp_path)

    def git_value(_root, *arguments):
        if arguments[:3] == ("remote", "get-url", "origin"):
            return weight_share.CUDA_WEIGHT_SHARE_URL + ".git"
        return weight_share.CUDA_WEIGHT_SHARE_COMMIT

    monkeypatch.setattr(weight_share, "_git_value", git_value)

    inspection = weight_share.inspect_cuda_weight_share_source(source)

    assert inspection["valid"] is True
    assert inspection["status"] == "source_build_available"
    assert inspection["git_revision"] == weight_share.CUDA_WEIGHT_SHARE_COMMIT


def test_production_launch_requires_a_bounded_unique_ipc_name():
    config = weight_share.CudaWeightShareConfig(
        enabled=True,
        model_size_bytes=1234,
    )

    with pytest.raises(ValueError, match="unique CUDA_VRAM_IPC_NAME"):
        weight_share.get_cuda_weight_share_env(config)

    invalid = weight_share.CudaWeightShareConfig(enabled=True, ipc_name="../../unsafe")
    with pytest.raises(ValueError, match="must start with /"):
        weight_share.get_cuda_weight_share_env(invalid)


def test_build_plan_is_read_only_and_requires_linux_tools(tmp_path: Path, monkeypatch):
    source = _source_tree(tmp_path)
    toolkit = _cuda_toolkit(tmp_path)
    monkeypatch.setattr(
        weight_share,
        "inspect_cuda_weight_share_source",
        lambda _path: {"valid": True, "issues": []},
    )
    monkeypatch.setattr(weight_share.shutil, "which", lambda name: f"/usr/bin/{name}")

    plan = weight_share.plan_cuda_weight_share_build(
        source,
        cuda_toolkit=toolkit,
        system="linux",
        machine="x86_64",
    )

    assert plan.ready
    assert plan.output == source / "cuda-llm-weight-share.so"
    assert plan.command[0] == "/usr/bin/gcc"
    assert f"-I{toolkit / 'include'}" in plan.command
    assert not plan.output.exists()
    assert any(issue.code == "reconnaissance_required" for issue in plan.issues)


def test_build_requires_confirmation_before_compilation(tmp_path: Path, monkeypatch):
    source = _source_tree(tmp_path)
    toolkit = _cuda_toolkit(tmp_path)
    monkeypatch.setattr(
        weight_share,
        "inspect_cuda_weight_share_source",
        lambda _path: {"valid": True, "issues": []},
    )
    monkeypatch.setattr(weight_share.shutil, "which", lambda name: f"/usr/bin/{name}")
    plan = weight_share.plan_cuda_weight_share_build(
        source,
        cuda_toolkit=toolkit,
        system="linux",
        machine="x86_64",
    )

    with pytest.raises(RuntimeError, match="not confirmed"):
        weight_share.build_cuda_weight_share(
            plan,
            runner=lambda *_args, **_kwargs: pytest.fail("compiler should not run"),
        )


def test_build_validates_elf_hooks_and_no_libcudart_dependency(tmp_path: Path, monkeypatch):
    source = _source_tree(tmp_path)
    toolkit = _cuda_toolkit(tmp_path)
    monkeypatch.setattr(
        weight_share,
        "inspect_cuda_weight_share_source",
        lambda _path: {"valid": True, "issues": []},
    )
    monkeypatch.setattr(weight_share.shutil, "which", lambda name: name)
    plan = weight_share.plan_cuda_weight_share_build(
        source,
        cuda_toolkit=toolkit,
        system="linux",
        machine="x86_64",
    )

    def runner(argv, **_kwargs):
        if argv[0] == "gcc":
            Path(argv[argv.index("-o") + 1]).write_bytes(b"ELF")
            output = ""
        elif argv[0] == "file":
            output = "ELF 64-bit LSB shared object, x86-64"
        elif argv[0] == "nm":
            output = "0000 T cudaMalloc\n0001 T cudaFree\n"
        else:
            output = "linux-vdso.so.1\nlibc.so.6"
        return subprocess.CompletedProcess(argv, 0, stdout=output, stderr="")

    report = weight_share.build_cuda_weight_share(plan, confirmed=True, runner=runner)

    assert report["valid"] is True
    assert report["hard_libcudart_dependency"] is False
    assert report["reconnaissance_required"] is True


def test_validator_rejects_a_hard_libcudart_dependency(tmp_path: Path):
    library = tmp_path / "share.so"
    library.write_bytes(b"ELF")

    def runner(argv, **_kwargs):
        outputs = {
            "file": "ELF 64-bit LSB shared object, x86-64",
            "nm": "0000 T cudaMalloc\n0001 T cudaFree\n",
            "ldd": "libcudart.so.12 => /usr/local/cuda/lib64/libcudart.so.12",
        }
        return subprocess.CompletedProcess(argv, 0, stdout=outputs[argv[0]], stderr="")

    with pytest.raises(RuntimeError, match="hard libcudart"):
        weight_share.validate_cuda_weight_share_library(library, runner=runner)


def test_addon_scan_recognizes_weight_share_source(tmp_path: Path):
    source = _source_tree(tmp_path)

    result = scan_addon_roots([tmp_path])

    addon = next(item for item in result["addons"] if item["path"] == str(source.resolve()))
    assert addon["kind"] == "cuda_weight_share"
    assert addon["source_profile"] == "cuda_weight_share"
    assert "expected_commit" in addon["source"]


def test_cli_build_requires_yes(tmp_path: Path, monkeypatch):
    source = _source_tree(tmp_path)
    plan = weight_share.CudaWeightShareBuildPlan(
        source=source,
        output=source / "share.so",
        cuda_root=tmp_path,
        compiler="gcc",
        command=("gcc",),
        issues=(),
    )
    monkeypatch.setattr(
        "multi_turboquant.integration.weight_share_cli.plan_cuda_weight_share_build",
        lambda *args, **kwargs: plan,
    )

    assert weight_share_main(["build", str(source)]) == 2
