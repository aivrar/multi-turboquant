from __future__ import annotations

import json
from pathlib import Path

from multi_turboquant.optimizations import EnvironmentContext, plan_environment
from multi_turboquant.optimizations import env_cli


def ready_plan(tmp_path: Path):
    return plan_environment(
        "fastdms",
        root=tmp_path,
        context=EnvironmentContext(
            os="linux",
            compute="cuda",
            available_executables=frozenset({"uv", "nvcc"}),
            cuda_toolkit_version=(12, 0),
        ),
    )


def test_create_requires_explicit_confirmation(tmp_path: Path, monkeypatch):
    plan = ready_plan(tmp_path)
    synchronized = []
    monkeypatch.setattr(env_cli, "plan_environment", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        env_cli,
        "synchronize_environment",
        lambda *args, **kwargs: synchronized.append(True),
    )

    result = env_cli.main(["create", "fastdms", "--root", str(tmp_path)])

    assert result == 2
    assert synchronized == []
    assert not plan.target.exists()


def test_confirmed_create_can_skip_validation(tmp_path: Path, monkeypatch):
    plan = ready_plan(tmp_path)
    synchronized = []
    checked = []
    monkeypatch.setattr(env_cli, "plan_environment", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        env_cli,
        "synchronize_environment",
        lambda *args, **kwargs: synchronized.append(kwargs),
    )
    monkeypatch.setattr(env_cli, "check_environment", lambda *args: checked.append(True))

    result = env_cli.main(["create", "fastdms", "--root", str(tmp_path), "--yes", "--no-check"])

    assert result == 0
    assert synchronized == [{"upgrade": False, "recreate": False}]
    assert checked == []


def test_confirmed_create_forwards_recreate(tmp_path: Path, monkeypatch):
    plan = ready_plan(tmp_path)
    synchronized = []
    monkeypatch.setattr(env_cli, "plan_environment", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        env_cli,
        "synchronize_environment",
        lambda *args, **kwargs: synchronized.append(kwargs),
    )

    result = env_cli.main(
        [
            "create",
            "fastdms",
            "--root",
            str(tmp_path),
            "--yes",
            "--no-check",
            "--recreate",
        ]
    )

    assert result == 0
    assert synchronized == [{"upgrade": False, "recreate": True}]


def test_diagnose_prints_redacted_json(tmp_path: Path, monkeypatch, capsys):
    plan = ready_plan(tmp_path)
    monkeypatch.setattr(env_cli, "plan_environment", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        env_cli,
        "diagnose_environment",
        lambda *args, **kwargs: {"schema": 1, "stderr": "HF_TOKEN=<redacted>"},
    )

    assert env_cli.main(["diagnose", "fastdms", "--root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["stderr"] == "HF_TOKEN=<redacted>"


def test_run_requires_a_command(tmp_path: Path, monkeypatch):
    plan = ready_plan(tmp_path)
    monkeypatch.setattr(env_cli, "plan_environment", lambda *args, **kwargs: plan)

    assert env_cli.main(["run", "fastdms", "--root", str(tmp_path)]) == 2


def test_list_json_distinguishes_installable_and_blocked_profiles(capsys):
    assert env_cli.main(["list", "--json"]) == 0

    profiles = {item["id"]: item for item in json.loads(capsys.readouterr().out)}
    assert profiles["lmcache"]["status"] == "installable"
    assert profiles["lmcache"]["blocked_reason"] is None
    assert profiles["rocketkv"]["status"] == "blocked"
    assert "non-commercial" in profiles["rocketkv"]["blocked_reason"]


def test_plan_forwards_source_build_request(tmp_path: Path, monkeypatch):
    plan = ready_plan(tmp_path)
    calls = []

    def capture_plan(*args, **kwargs):
        calls.append((args, kwargs))
        return plan

    monkeypatch.setattr(env_cli, "plan_environment", capture_plan)

    result = env_cli.main(
        [
            "plan",
            "fastdms",
            "--root",
            str(tmp_path),
            "--cuda-toolkit",
            str(tmp_path / "cuda-12.6"),
            "--local-source",
            str(tmp_path / "FastDMS"),
            "--build-from-source",
        ]
    )

    assert result == 0
    assert calls[0][1]["build_from_source"] is True
    assert calls[0][1]["cuda_toolkit"] == tmp_path / "cuda-12.6"
    assert calls[0][1]["local_source"] == tmp_path / "FastDMS"
