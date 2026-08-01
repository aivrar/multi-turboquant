# SPDX-License-Identifier: MIT

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from multi_turboquant.calibration import (
    CALIBRATION_CORPUS_SCHEMA_VERSION,
    generate_calibration_text,
)
from multi_turboquant.calibration import text_corpus


def test_generated_calibration_text_is_deterministic_and_generous(tmp_path: Path):
    output = tmp_path / "calibration.txt"

    report = generate_calibration_text(output, target_tokens=512)
    text = output.read_text(encoding="utf-8")

    assert report["reused"] is False
    assert report["generic"] is True
    assert report["characters"] == len(text)
    assert report["schema"] == CALIBRATION_CORPUS_SCHEMA_VERSION
    assert len(text) >= 4096
    assert "Requested calibration tokens: 512" in text
    assert "Structured sample" in text

    reused = generate_calibration_text(output, target_tokens=512)
    assert reused["reused"] is True


def test_concurrent_generation_never_clobbers_or_reuses_partial_text(tmp_path: Path):
    output = tmp_path / "calibration.txt"

    with ThreadPoolExecutor(max_workers=4) as executor:
        reports = list(
            executor.map(
                lambda _: generate_calibration_text(output, target_tokens=512),
                range(4),
            )
        )

    text = output.read_text(encoding="utf-8")
    assert sum(report["reused"] is False for report in reports) == 1
    assert all(report["characters"] == len(text) for report in reports)
    assert text.endswith(
        "# End of Multi-TurboQuant generic calibration corpus "
        f"schema {CALIBRATION_CORPUS_SCHEMA_VERSION}\n"
    )


def test_incomplete_generated_file_is_not_reused(tmp_path: Path):
    output = tmp_path / "calibration.txt"
    output.write_text(
        "# Multi-TurboQuant generic calibration corpus\n"
        f"# Schema: {CALIBRATION_CORPUS_SCHEMA_VERSION}\n"
        "# Requested calibration tokens: 512\n",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        generate_calibration_text(output, target_tokens=512)


def test_generation_supports_filesystems_without_hardlinks(tmp_path: Path, monkeypatch):
    output = tmp_path / "calibration.txt"
    monkeypatch.setattr(
        text_corpus.os,
        "link",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("not supported")),
    )

    report = generate_calibration_text(output, target_tokens=512)

    assert report["reused"] is False
    assert output.read_text(encoding="utf-8").endswith(
        "# End of Multi-TurboQuant generic calibration corpus "
        f"schema {CALIBRATION_CORPUS_SCHEMA_VERSION}\n"
    )


def test_concurrent_generation_without_hardlinks_reuses_completed_file(
    tmp_path: Path, monkeypatch
):
    output = tmp_path / "calibration.txt"
    monkeypatch.setattr(
        text_corpus.os,
        "link",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("not supported")),
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        reports = list(
            executor.map(
                lambda _: generate_calibration_text(output, target_tokens=512),
                range(4),
            )
        )

    text = output.read_text(encoding="utf-8")
    assert sum(report["reused"] is False for report in reports) == 1
    assert all(report["characters"] == len(text) for report in reports)


def test_generated_calibration_text_refuses_unrelated_existing_file(tmp_path: Path):
    output = tmp_path / "calibration.txt"
    output.write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        generate_calibration_text(output, target_tokens=512)


@pytest.mark.parametrize("target_tokens", [127, 200_001, True, 512.5])
def test_generated_calibration_text_bounds_token_request(tmp_path: Path, target_tokens):
    with pytest.raises(ValueError, match="target_tokens"):
        generate_calibration_text(tmp_path / "calibration.txt", target_tokens=target_tokens)
