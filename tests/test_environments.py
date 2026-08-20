from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_turboquant.optimizations import (
    BUILTIN_ENVIRONMENT_PROFILES,
    EnvironmentContext,
    check_environment,
    diagnose_environment,
    get_environment_profile,
    inspect_profile_source,
    plan_environment,
    read_os_release,
    render_profile_project,
    synchronize_environment,
)
from multi_turboquant.optimizations import environments
from multi_turboquant.optimizations.environments import (
    environment_python,
    materialize_environment_project,
)


def linux_cuda_context(*executables: str) -> EnvironmentContext:
    return EnvironmentContext(
        os="linux",
        compute="cuda",
        available_executables=frozenset(executables or ("uv", "nvcc")),
        cuda_toolkit_version=(12, 0),
    )


def test_profiles_are_isolated_and_explicit():
    ids = [profile.id for profile in BUILTIN_ENVIRONMENT_PROFILES]
    assert ids == [
        "flashattention",
        "fastdms",
        "lmcache",
        "minference",
        "sageattention",
        "triattention",
        "jetspec",
        "proxima",
        "jetlong",
        "jetlong_fused",
        "lucebox",
        "chunkllama",
        "rabitqcache",
        "scope_pe",
        "duoattention",
        "icecache",
        "pflash_llamacpp",
        "resonance_jetlong",
        "maru",
        "speculative_prefill",
        "rocketkv",
        "lexico",
        "adadecode",
        "resonance_yarn",
    ]
    assert len(ids) == len(set(ids))
    installable = [profile for profile in BUILTIN_ENVIRONMENT_PROFILES if profile.installable]
    blocked = [profile for profile in BUILTIN_ENVIRONMENT_PROFILES if not profile.installable]
    assert all(profile.packages for profile in installable)
    assert all(profile.validation_modules for profile in installable)
    assert all(profile.blocked_reason for profile in blocked)
    assert all("linux" in profile.supported_os for profile in BUILTIN_ENVIRONMENT_PROFILES)


@pytest.mark.parametrize(
    ("profile_id", "package", "module"),
    [
        ("jetspec", "jetspec", "jetspec"),
        ("proxima", "proxima-vllm", "proxima_vllm"),
        ("jetlong", "jet-long", "jetlm"),
    ],
)
def test_issue_43_installable_profiles_are_pinned_and_accept_local_source(
    tmp_path: Path, profile_id: str, package: str, module: str
):
    profile = get_environment_profile(profile_id)
    source = tmp_path / profile_id
    source.mkdir()
    for marker in profile.local_source_markers:
        target = source / marker
        if "." in Path(marker).name:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("marker", encoding="utf-8")
        else:
            target.mkdir(parents=True, exist_ok=True)

    rendered = tomllib.loads(render_profile_project(profile, local_source=source))

    assert profile.installable
    assert any("@ git+https://github.com/" in item for item in profile.packages)
    assert module in profile.validation_modules
    assert rendered["tool"]["uv"]["sources"][package]["path"] == str(source.resolve())


def test_jetlong_default_and_fused_profiles_remain_separate():
    default = get_environment_profile("jetlong")
    fused = get_environment_profile("jetlong_fused")

    assert default.installable
    assert default.cuda_toolkit_major == 13
    assert default.no_build_isolation_packages == ("flash-attn",)
    assert not fused.installable
    assert "SM90/H100" in fused.blocked_reason


def test_blocked_profile_explains_itself_without_creating_commands(tmp_path: Path):
    root = tmp_path / "envs"
    plan = plan_environment(
        "rocketkv",
        root=root,
        context=linux_cuda_context(),
    )

    assert not plan.ready
    assert plan.commands == ()
    assert plan.project_toml == ""
    assert [issue.code for issue in plan.issues] == ["profile_blocked"]
    assert "non-commercial" in plan.issues[0].message
    assert not root.exists()


def test_blocked_profile_cannot_render_a_project():
    with pytest.raises(ValueError, match="is blocked"):
        render_profile_project(get_environment_profile("maru"))


def test_rendered_flashattention_project_is_lockable_toml():
    rendered = render_profile_project(get_environment_profile("flashattention"))
    parsed = tomllib.loads(rendered)

    assert parsed["project"]["name"] == "multi-turboquant-env-flashattention"
    assert "flash-attn>=2.7,<3" in parsed["project"]["dependencies"]
    assert parsed["tool"]["uv"]["no-build-isolation-package"] == ["flash-attn"]
    assert parsed["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cu126"}
    assert parsed["tool"]["uv"]["index"] == [
        {
            "name": "pytorch-cu126",
            "url": "https://download.pytorch.org/whl/cu126",
            "explicit": True,
        }
    ]
    assert parsed["tool"]["multi-turboquant"] == {
        "profile": "flashattention",
        "schema": 1,
    }


@pytest.mark.parametrize("profile_id", ["flashattention", "fastdms"])
def test_source_build_forces_flashattention_sdist_for_dependent_profiles(profile_id: str):
    profile = get_environment_profile(profile_id)
    default_project = tomllib.loads(render_profile_project(profile))
    source_project = tomllib.loads(render_profile_project(profile, build_from_source=True))

    assert profile.source_build_packages == ("flash-attn",)
    assert dict(profile.source_build_environment) == {
        "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
    }
    assert "no-binary-package" not in default_project["tool"]["uv"]
    assert source_project["tool"]["uv"]["no-binary-package"] == ["flash-attn"]


def test_lmcache_uses_matching_prebuilt_cuda_runtime_without_requiring_nvcc(tmp_path: Path):
    profile = get_environment_profile("lmcache")
    rendered = tomllib.loads(render_profile_project(profile))
    plan = plan_environment(
        "lmcache",
        root=tmp_path,
        context=EnvironmentContext(
            os="linux",
            compute="cuda",
            available_executables=frozenset({"uv"}),
            cuda_toolkit_version=None,
        ),
    )

    assert plan.ready
    assert profile.cuda_toolkit_major is None
    assert profile.torch_cuda_major == 13
    assert rendered["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cu130"}
    assert "lmcache==0.5.2" in rendered["project"]["dependencies"]
    assert "openai==2.46.0" in rendered["project"]["dependencies"]


def test_sageattention_is_pinned_to_a_reviewed_commit():
    profile = get_environment_profile("sageattention")
    rendered = tomllib.loads(render_profile_project(profile))

    assert profile.build_may_compile
    assert profile.no_build_isolation_packages == ("sageattention",)
    assert "numpy==2.2.6" in profile.packages
    assert profile.packages[-1].endswith("d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5")
    assert rendered["tool"]["uv"]["dependency-metadata"] == [
        {
            "name": "sageattention",
            "version": "2.2.0",
            "requires-dist": [],
        }
    ]


def test_triattention_profile_supports_the_official_local_checkout(tmp_path: Path):
    source = tmp_path / "triattention"
    (source / "triattention").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "docs").mkdir()
    (source / "setup.py").write_text("", encoding="utf-8")
    (source / "scripts" / "calibrate.py").write_text("", encoding="utf-8")
    (source / "docs" / "calibration.md").write_text("", encoding="utf-8")

    plan = plan_environment(
        "triattention",
        root=tmp_path / "envs",
        local_source=source,
        context=linux_cuda_context("uv", "git"),
    )
    project = tomllib.loads(plan.project_toml)

    assert plan.ready
    assert plan.max_jobs == 2
    assert project["tool"]["uv"]["sources"]["triattention"]["path"] == str(source.resolve())
    assert "--reinstall-package" in plan.commands[0].argv


def test_triattention_profile_pins_the_reviewed_calibration_stack():
    profile = get_environment_profile("triattention")

    assert "transformers==4.57.6" in profile.packages
    assert "accelerate==1.14.0" in profile.packages
    assert "sentencepiece==0.2.2" in profile.packages
    assert "gigatoken==0.10.0" in profile.packages


def test_read_os_release_is_data_only_and_debian_is_reported(tmp_path: Path):
    release = tmp_path / "os-release"
    release.write_text(
        'ID=debian\nVERSION_ID="13"\nBAD KEY=value\nNAME="Debian GNU/Linux"\n',
        encoding="utf-8",
    )

    assert read_os_release(release) == {
        "ID": "debian",
        "VERSION_ID": "13",
        "NAME": "Debian GNU/Linux",
    }
    plan = plan_environment(
        "triattention",
        root=tmp_path,
        context=EnvironmentContext(
            os="linux",
            compute="cuda",
            available_executables=frozenset({"uv", "git"}),
            os_release_id="debian",
            os_release_version="13",
        ),
    )

    assert plan.os_release_id == "debian"
    assert plan.to_dict()["os_release"] == {"id": "debian", "version": "13"}
    assert any(issue.code == "debian_supported" for issue in plan.issues)


def test_explicit_build_job_limit_overrides_profile_default(tmp_path: Path):
    plan = plan_environment(
        "fastdms",
        root=tmp_path,
        max_jobs=2,
        context=linux_cuda_context(),
    )
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    synchronize_environment(plan, runner=runner)

    assert plan.max_jobs == 2
    assert calls[0][1]["env"]["MAX_JOBS"] == "2"


def test_minference_is_pinned_past_the_broken_pypi_import():
    profile = get_environment_profile("minference")
    rendered = tomllib.loads(render_profile_project(profile))

    assert profile.packages[-1].endswith("d76b76e89cb59817c89e1777c4c51b1c7f233335")
    assert "transformers>=4.37,<5" in profile.packages
    assert profile.required_executables == ("git", "nvcc")
    assert dict(profile.build_environment) == {
        "MAX_JOBS": "1",
        "MINFERENCE_FORCE_BUILD": "TRUE",
    }
    assert rendered["tool"]["uv"]["dependency-metadata"] == [
        {
            "name": "minference",
            "version": "0.1.6.0",
            "requires-dist": ["transformers>=4.37.0", "torch", "triton", "einops"],
        }
    ]


def test_plan_is_read_only_and_accepts_pyenv_interpreter_path(tmp_path: Path):
    root = tmp_path / "envs"
    interpreter = "/opt/pyenv/versions/3.11.9/bin/python"
    plan = plan_environment(
        "fastdms",
        root=root,
        python=interpreter,
        context=linux_cuda_context("uv", "nvcc", "pyenv"),
    )

    assert plan.ready
    assert plan.python_request == interpreter
    assert plan.target == (root / "fastdms").resolve()
    assert not root.exists()
    assert plan.commands[0].argv[-1] == interpreter
    assert any(issue.code == "pyenv_available" for issue in plan.issues)


def test_plan_rejects_unsupported_or_incomplete_host(tmp_path: Path):
    plan = plan_environment(
        "flashattention",
        root=tmp_path,
        context=EnvironmentContext(
            os="windows",
            compute="cpu",
            available_executables=frozenset(),
            cuda_toolkit_version=None,
        ),
    )

    assert not plan.ready
    assert {issue.code for issue in plan.issues if issue.severity == "error"} == {
        "unsupported_os",
        "unsupported_compute",
        "missing_uv",
        "missing_build_tool",
        "unknown_cuda_toolkit",
    }


def test_plan_rejects_wrong_cuda_toolkit_major(tmp_path: Path):
    plan = plan_environment(
        "fastdms",
        root=tmp_path,
        context=EnvironmentContext(
            os="linux",
            compute="cuda",
            available_executables=frozenset({"uv", "nvcc"}),
            cuda_toolkit_version=(13, 0),
        ),
    )

    assert not plan.ready
    mismatch = next(issue for issue in plan.issues if issue.code == "unsupported_cuda_toolkit")
    assert "side-by-side toolkit" in mismatch.message


def test_detect_context_uses_explicit_side_by_side_cuda_toolkit(tmp_path: Path, monkeypatch):
    toolkit = tmp_path / "cuda-12.6"
    nvcc = toolkit / "bin" / ("nvcc.exe" if environments.os.name == "nt" else "nvcc")
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("test executable placeholder", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        "multi_turboquant.hardware.detect_platform",
        lambda: SimpleNamespace(os="linux", primary_compute="cuda"),
    )
    monkeypatch.setattr(
        environments.shutil,
        "which",
        lambda name: "/usr/bin/uv" if name == "uv" else None,
    )

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="Cuda compilation tools, release 12.6")

    monkeypatch.setattr(environments.subprocess, "run", fake_run)

    context = environments.detect_environment_context(cuda_toolkit=toolkit)

    assert context.cuda_toolkit_version == (12, 6)
    assert context.cuda_toolkit_root == str(toolkit.resolve())
    assert "nvcc" in context.available_executables
    assert calls[0][0] == [str(nvcc.resolve()), "--version"]


def test_plan_forces_a_cache_safe_flashattention_source_rebuild(tmp_path: Path):
    plan = plan_environment(
        "fastdms",
        root=tmp_path,
        build_from_source=True,
        context=linux_cuda_context(),
    )

    assert plan.ready
    assert plan.build_from_source
    assert plan.to_dict()["source_build_packages"] == ["flash-attn"]
    assert "--no-cache" in plan.commands[0].argv
    assert plan.commands[0].argv[-2:] == ("--reinstall-package", "flash-attn")
    assert any(issue.code == "source_build_forced" for issue in plan.issues)


def test_plan_builds_reviewed_package_from_validated_local_checkout(tmp_path: Path):
    source = tmp_path / "FastDMS local checkout"
    (source / "fastdms").mkdir(parents=True)
    (source / "pyproject.toml").write_text("[project]\nname='fastdms'\n", encoding="utf-8")

    inspection = inspect_profile_source("fastdms", source)
    plan = plan_environment(
        "fastdms",
        root=tmp_path / "envs",
        local_source=source,
        context=linux_cuda_context(),
    )
    project = tomllib.loads(plan.project_toml)

    assert inspection["valid"] is True
    assert plan.ready
    assert plan.local_source == source.resolve()
    assert "fastdms" in project["project"]["dependencies"]
    assert not any(item.startswith("fastdms>=") for item in project["project"]["dependencies"])
    assert project["tool"]["uv"]["sources"]["fastdms"]["path"] == str(source.resolve())
    assert project["tool"]["multi-turboquant"]["local-source"] == str(source.resolve())
    assert "--no-cache" in plan.commands[0].argv
    assert plan.commands[0].argv[-2:] == ("--reinstall-package", "fastdms")
    assert any(issue.code == "local_source_selected" for issue in plan.issues)
    assert any(issue.code == "local_source_cuda_constraints" for issue in plan.issues)


def test_plan_rejects_local_checkout_with_missing_reviewed_markers(tmp_path: Path):
    source = tmp_path / "not-fastdms"
    source.mkdir()

    plan = plan_environment(
        "fastdms",
        root=tmp_path / "envs",
        local_source=source,
        context=linux_cuda_context(),
    )

    assert not plan.ready
    issue = next(item for item in plan.issues if item.code == "invalid_local_source")
    assert "pyproject.toml" in issue.message
    assert "fastdms" in issue.message


def test_plan_rejects_source_build_without_a_reviewed_profile_path(tmp_path: Path):
    plan = plan_environment(
        "lmcache",
        root=tmp_path,
        build_from_source=True,
        context=linux_cuda_context("uv"),
    )

    assert not plan.ready
    assert any(issue.code == "source_build_unavailable" for issue in plan.issues)


def test_materialize_refuses_to_overwrite_foreign_project(tmp_path: Path):
    plan = plan_environment(
        "fastdms",
        root=tmp_path,
        context=linux_cuda_context(),
    )
    plan.target.mkdir(parents=True)
    project_file = plan.target / "pyproject.toml"
    project_file.write_text("[project]\nname = 'user-project'\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unmanaged environment project"):
        materialize_environment_project(plan)
    assert "user-project" in project_file.read_text(encoding="utf-8")


def test_sync_materializes_owned_project_and_uses_argv(tmp_path: Path):
    plan = plan_environment(
        "fastdms",
        root=tmp_path,
        context=linux_cuda_context(),
    )
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    synchronize_environment(plan, upgrade=True, runner=runner)

    project_file = plan.target / "pyproject.toml"
    assert project_file.is_file()
    assert calls[0][0][:2] == ["uv", "sync"]
    assert calls[0][0][-1] == "--upgrade"
    assert calls[0][1]["cwd"] == plan.target
    assert calls[0][1]["check"] is False
    assert calls[0][1]["env"]["MAX_JOBS"] == "4"
    assert "FLASH_ATTENTION_FORCE_BUILD" not in calls[0][1]["env"]


def test_recreate_preserves_the_previous_managed_venv(tmp_path: Path):
    plan = plan_environment(
        "triattention",
        root=tmp_path,
        context=linux_cuda_context("uv", "git"),
    )
    materialize_environment_project(plan)
    old_venv = plan.target / ".venv"
    old_venv.mkdir()
    (old_venv / "old.txt").write_text("old", encoding="utf-8")

    def runner(argv, **kwargs):
        replacement = Path(kwargs["cwd"]) / ".venv"
        assert not replacement.exists()
        replacement.mkdir()
        (replacement / "new.txt").write_text("new", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="synced", stderr="")

    backup = synchronize_environment(plan, recreate=True, runner=runner)

    assert backup is not None
    assert (backup / "old.txt").read_text(encoding="utf-8") == "old"
    assert (plan.target / ".venv" / "new.txt").read_text(encoding="utf-8") == "new"


def test_recreate_rolls_back_and_redacts_sync_failures(tmp_path: Path):
    plan = plan_environment(
        "triattention",
        root=tmp_path,
        context=linux_cuda_context("uv", "git"),
    )
    materialize_environment_project(plan)
    project_file = plan.target / "pyproject.toml"
    original_project = project_file.read_bytes() + b"\n# previous lock generation\n"
    project_file.write_bytes(original_project)
    lock_file = plan.target / "uv.lock"
    lock_file.write_bytes(b"version = 1\n# previous lock\n")
    old_venv = plan.target / ".venv"
    old_venv.mkdir()
    (old_venv / "old.txt").write_text("old", encoding="utf-8")

    def runner(argv, **kwargs):
        replacement = Path(kwargs["cwd"]) / ".venv"
        replacement.mkdir()
        (replacement / "partial.txt").write_text("partial", encoding="utf-8")
        project_file.write_text("replacement project", encoding="utf-8")
        lock_file.write_text("replacement lock", encoding="utf-8")
        return subprocess.CompletedProcess(
            argv,
            1,
            stdout="",
            stderr="HF_TOKEN=do-not-leak dependency conflict",
        )

    with pytest.raises(RuntimeError, match=r"HF_TOKEN=<redacted>") as error:
        synchronize_environment(plan, recreate=True, runner=runner)

    assert "do-not-leak" not in str(error.value)
    assert (plan.target / ".venv" / "old.txt").read_text(encoding="utf-8") == "old"
    assert list(plan.target.glob(".venv.mtq-failed-*"))
    assert project_file.read_bytes() == original_project
    assert lock_file.read_bytes() == b"version = 1\n# previous lock\n"


def test_recreate_refuses_unmarked_existing_venv(tmp_path: Path):
    plan = plan_environment(
        "triattention",
        root=tmp_path,
        context=linux_cuda_context("uv", "git"),
    )
    unowned = plan.target / ".venv"
    unowned.mkdir(parents=True)
    (unowned / "user.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unmanaged environment"):
        synchronize_environment(plan, recreate=True)

    assert (unowned / "user.txt").read_text(encoding="utf-8") == "keep"
    assert not (plan.target / "pyproject.toml").exists()


def test_recreate_rolls_back_when_the_sync_process_cannot_start(tmp_path: Path):
    plan = plan_environment(
        "triattention",
        root=tmp_path,
        context=linux_cuda_context("uv", "git"),
    )
    materialize_environment_project(plan)
    old_venv = plan.target / ".venv"
    old_venv.mkdir()
    (old_venv / "old.txt").write_text("old", encoding="utf-8")

    def runner(argv, **kwargs):
        raise OSError("cannot execute TOKEN=private")

    with pytest.raises(RuntimeError, match=r"TOKEN=<redacted>"):
        synchronize_environment(plan, recreate=True, runner=runner)

    assert (plan.target / ".venv" / "old.txt").is_file()


def test_diagnostics_keep_lexical_interpreter_and_parse_fail_soft_report(tmp_path: Path):
    plan = plan_environment(
        "triattention",
        root=tmp_path,
        context=linux_cuda_context("uv", "git"),
    )
    interpreter = environment_python(plan.target)
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("placeholder", encoding="utf-8")

    def runner(argv, **kwargs):
        if argv[0] == str(interpreter):
            payload = {
                "interpreter": {
                    "executable": str(interpreter),
                    "prefix": str(interpreter.parent.parent),
                    "base_prefix": "/uv/python/cpython-3.11",
                },
                "modules": {"accelerate": {"status": "error", "error": "ImportError"}},
            }
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="MTQ_DIAGNOSTICS=" + json.dumps(payload),
                stderr="",
            )
        return subprocess.CompletedProcess(argv, 0, stdout="tool version", stderr="")

    report = diagnose_environment(plan, runner=runner)

    assert report["files"]["interpreter_lexical"] == str(interpreter)
    managed = report["commands"]["managed_python"]
    assert managed["parsed"]["modules"]["accelerate"]["status"] == "error"


def test_sync_sets_flashattention_force_build_only_when_requested(tmp_path: Path):
    plan = plan_environment(
        "fastdms",
        root=tmp_path,
        build_from_source=True,
        context=linux_cuda_context(),
    )
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    synchronize_environment(plan, runner=runner)

    assert calls[0][0][-2:] == ["--reinstall-package", "flash-attn"]
    assert calls[0][1]["env"]["FLASH_ATTENTION_FORCE_BUILD"] == "TRUE"


def test_sync_exports_selected_cuda_toolkit_to_native_build(tmp_path: Path):
    toolkit = (tmp_path / "cuda-12.6").resolve()
    plan = plan_environment(
        "fastdms",
        root=tmp_path / "envs",
        context=EnvironmentContext(
            os="linux",
            compute="cuda",
            available_executables=frozenset({"uv", "nvcc"}),
            cuda_toolkit_version=(12, 6),
            cuda_toolkit_root=str(toolkit),
        ),
    )
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    synchronize_environment(plan, runner=runner)

    child_environment = calls[0][1]["env"]
    assert child_environment["CUDA_HOME"] == str(toolkit)
    assert child_environment["CUDA_PATH"] == str(toolkit)
    assert child_environment["PATH"].split(environments.os.pathsep)[0] == str(toolkit / "bin")


def test_check_uses_only_the_isolated_interpreter(tmp_path: Path):
    plan = plan_environment(
        "flashattention",
        root=tmp_path,
        context=linux_cuda_context(),
    )
    interpreter = environment_python(plan.target)
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("placeholder", encoding="utf-8")
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                "optional runtime recommendation\n"
                '{"cuda_available": true, "flash_attn": "2.8.3", '
                '"torch": "2.7.0", "torch_cuda": "12.8"}\n'
            ),
            stderr="",
        )

    report = check_environment(plan, runner=runner)

    assert report["flash_attn"] == "2.8.3"
    assert calls[0][0][0] == str(interpreter)
    assert calls[0][0][1] == "-c"


def test_check_rejects_cpu_only_torch_in_cuda_profile(tmp_path: Path):
    plan = plan_environment(
        "flashattention",
        root=tmp_path,
        context=linux_cuda_context(),
    )
    interpreter = environment_python(plan.target)
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("placeholder", encoding="utf-8")

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                '{"cuda_available": false, "flash_attn": "2.8.3", '
                '"torch": "2.7.0", "torch_cuda": null}\n'
            ),
            stderr="",
        )

    with pytest.raises(RuntimeError, match="CPU-only PyTorch"):
        check_environment(plan, runner=runner)
