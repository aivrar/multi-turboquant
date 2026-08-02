from __future__ import annotations

import json
import subprocess
from pathlib import Path

from multi_turboquant.calibration.gigatoken_runner import build_parser, validate_tokenizer_parity
from multi_turboquant.tokenizer_backends import (
    discover_python_interpreters,
    inspect_gigatoken_python,
    scan_gigatoken_interpreters,
)


class FakeTokenizer:
    def __init__(self, ids: list[int]):
        self.ids = ids

    def encode(self, text: str, **kwargs) -> list[int]:
        assert kwargs == {
            "add_special_tokens": True,
            "truncation": True,
            "max_length": 128,
        }
        return list(self.ids)


def test_gigatoken_parity_requires_exact_ids():
    assert validate_tokenizer_parity(
        FakeTokenizer([1, 2, 3]),
        FakeTokenizer([1, 2, 3]),
        "text",
        max_length=128,
    ) == 3


def test_gigatoken_parity_reports_first_difference():
    try:
        validate_tokenizer_parity(
            FakeTokenizer([1, 2, 3]),
            FakeTokenizer([1, 9, 3]),
            "text",
            max_length=128,
        )
    except RuntimeError as exc:
        assert "index 1" in str(exc)
        assert "Hugging Face=2" in str(exc)
        assert "Gigatoken=9" in str(exc)
    else:
        raise AssertionError("parity mismatch was accepted")


def test_gigatoken_runner_accepts_domvox_without_attention_implementation():
    args = build_parser().parse_args(
        [
            "--kind",
            "domvox",
            "--calibrator",
            "calibrate.py",
            "--model",
            "org/model",
            "--input",
            "input.txt",
            "--output",
            "stats.bin",
            "--max-length",
            "512",
            "--device",
            "cpu",
        ]
    )

    assert args.kind == "domvox"
    assert args.attn_implementation is None


def test_discovery_includes_managed_and_pyenv_without_resolving_symlinks(tmp_path: Path):
    managed = tmp_path / "managed" / "triattention" / ".venv" / "bin" / "python"
    pyenv = tmp_path / "home" / ".pyenv" / "versions" / "3.11.9" / "bin" / "python"
    managed.parent.mkdir(parents=True)
    pyenv.parent.mkdir(parents=True)
    managed.write_bytes(b"python")
    pyenv.write_bytes(b"python")

    found = discover_python_interpreters(
        environment_root=tmp_path / "managed",
        home=tmp_path / "home",
        environ={},
    )

    by_path = {item["python"]: item for item in found}
    assert str(managed.absolute()) in by_path
    assert by_path[str(managed.absolute())]["sources"] == ["managed"]
    assert str(pyenv.absolute()) in by_path
    assert by_path[str(pyenv.absolute())]["sources"] == ["pyenv"]


def test_gigatoken_inspection_reports_reviewed_version(tmp_path: Path):
    python = tmp_path / "python"
    python.write_bytes(b"python")

    def runner(command, **kwargs):
        report = {
            "runtime_executable": command[0],
            "prefix": "/env",
            "base_prefix": "/base",
            "gigatoken": "0.10.0",
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(report), stderr="")

    report = inspect_gigatoken_python(python, runner=runner)

    assert report["available"] is True
    assert report["compatible"] is True
    assert report["venv"] is True


def test_scan_keeps_unreviewed_install_visible_but_incompatible(tmp_path: Path, monkeypatch):
    python = tmp_path / "managed" / "triattention" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")

    def runner(command, **kwargs):
        report = {
            "runtime_executable": command[0],
            "prefix": "/env",
            "base_prefix": "/base",
            "gigatoken": "0.11.0",
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(report), stderr="")

    monkeypatch.setattr("multi_turboquant.tokenizer_backends.sys.executable", str(tmp_path / "none"))
    monkeypatch.setattr("multi_turboquant.tokenizer_backends.shutil.which", lambda name: None)
    result = scan_gigatoken_interpreters(
        environment_root=tmp_path / "managed",
        home=tmp_path / "home",
        environ={},
        runner=runner,
    )

    assert result["count"] == 1
    assert result["compatible_count"] == 0
    assert result["interpreters"][0]["available"] is True
    assert "0.10.x" in result["interpreters"][0]["error"]
