# SPDX-License-Identifier: MIT
"""Discovery and validation for optional tokenizer backends."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

from ._paths import lexical_absolute_path


GIGATOKEN_REVIEWED_SERIES = (0, 10)
MAX_PYTHON_CANDIDATES = 64


def _version_series(value: object) -> tuple[int, int] | None:
    parts = str(value or "").split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    return int(parts[0]), int(parts[1])


def gigatoken_version_is_reviewed(value: object) -> bool:
    """Return whether a version uses the reviewed 0.10 compatibility API."""
    return _version_series(value) == GIGATOKEN_REVIEWED_SERIES


def _python_entries(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return ()
    candidates: list[Path] = []
    try:
        children = sorted(root.iterdir(), key=lambda item: item.name.lower())
    except OSError:
        return ()
    for child in children[:MAX_PYTHON_CANDIDATES]:
        candidates.extend(
            (
                child / ".venv" / "bin" / "python",
                child / ".venv" / "Scripts" / "python.exe",
                child / "bin" / "python",
                child / "python.exe",
            )
        )
    return candidates


def discover_python_interpreters(
    *,
    environment_root: str | Path | None = None,
    home: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Find Python entry points in bounded, conventional environment locations.

    The scan intentionally avoids a recursive device-wide walk.  POSIX venv
    symlinks remain distinct from their base interpreters.
    """
    variables = os.environ if environ is None else environ
    home_path = Path(home).expanduser() if home is not None else Path.home()
    found: dict[str, dict[str, object]] = {}

    def add(path: str | Path | None, source: str) -> None:
        if not path:
            return
        candidate = lexical_absolute_path(path)
        if not candidate.is_file():
            return
        key = os.path.normcase(os.fspath(candidate))
        item = found.setdefault(key, {"python": str(candidate), "sources": []})
        sources = item["sources"]
        assert isinstance(sources, list)
        if source not in sources:
            sources.append(source)

    add(sys.executable, "current")
    for name in ("python", "python3"):
        add(shutil.which(name), "PATH")
    for variable in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        prefix = variables.get(variable)
        if prefix:
            add(Path(prefix) / "bin" / "python", variable.lower())
            add(Path(prefix) / "Scripts" / "python.exe", variable.lower())

    if environment_root is not None:
        for candidate in _python_entries(lexical_absolute_path(environment_root)):
            add(candidate, "managed")

    pyenv_roots: list[Path] = []
    configured_pyenv = variables.get("PYENV_ROOT")
    if configured_pyenv:
        pyenv_roots.append(Path(configured_pyenv).expanduser())
    pyenv_roots.extend((home_path / ".pyenv", home_path / ".pyenv" / "pyenv-win"))
    seen_roots: set[str] = set()
    for root in pyenv_roots:
        root_key = os.path.normcase(os.fspath(lexical_absolute_path(root)))
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)
        for candidate in _python_entries(root / "versions"):
            add(candidate, "pyenv")

    return list(found.values())[:MAX_PYTHON_CANDIDATES]


def inspect_gigatoken_python(
    python: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """Inspect one interpreter without importing it into this process."""
    interpreter = lexical_absolute_path(python)
    script = (
        "import importlib.metadata, json, sys\n"
        "report = {'runtime_executable': sys.executable, 'prefix': sys.prefix, "
        "'base_prefix': sys.base_prefix}\n"
        "try:\n"
        "    import gigatoken\n"
        "    report['gigatoken'] = importlib.metadata.version('gigatoken')\n"
        "except Exception as exc:\n"
        "    report['error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(report, sort_keys=True))\n"
    )
    try:
        result = runner(
            [str(interpreter), "-I", "-c", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "python": str(interpreter),
            "available": False,
            "compatible": False,
            "error": f"Interpreter inspection failed: {exc}",
        }
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    try:
        report = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        detail = (result.stderr or result.stdout or "").strip()
        return {
            "python": str(interpreter),
            "available": False,
            "compatible": False,
            "error": detail or f"Interpreter exited with {result.returncode}",
        }
    version = report.get("gigatoken") if isinstance(report, dict) else None
    available = result.returncode == 0 and isinstance(version, str)
    compatible = available and gigatoken_version_is_reviewed(version)
    error = report.get("error") if isinstance(report, dict) else "Unexpected inspection output"
    if available and not compatible:
        error = (
            f"Gigatoken {version} is installed, but this integration reviews the 0.10.x API."
        )
    return {
        "python": str(interpreter),
        "available": available,
        "compatible": compatible,
        "version": version,
        "runtime_executable": report.get("runtime_executable") if isinstance(report, dict) else None,
        "prefix": report.get("prefix") if isinstance(report, dict) else None,
        "base_prefix": report.get("base_prefix") if isinstance(report, dict) else None,
        "venv": (
            report.get("prefix") != report.get("base_prefix") if isinstance(report, dict) else None
        ),
        "error": error,
    }


def scan_gigatoken_interpreters(
    *,
    environment_root: str | Path | None = None,
    home: str | Path | None = None,
    environ: dict[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """Discover conventional Python environments and report Gigatoken imports."""
    candidates = discover_python_interpreters(
        environment_root=environment_root,
        home=home,
        environ=environ,
    )

    def inspect(candidate: dict[str, object]) -> dict[str, object]:
        report = inspect_gigatoken_python(str(candidate["python"]), runner=runner)
        report["sources"] = candidate["sources"]
        return report

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(candidates)))) as executor:
        interpreters = list(executor.map(inspect, candidates))
    return {
        "reviewed_series": ".".join(str(item) for item in GIGATOKEN_REVIEWED_SERIES) + ".x",
        "interpreters": interpreters,
        "count": len(interpreters),
        "compatible_count": sum(bool(item["compatible"]) for item in interpreters),
        "bounded": True,
        "locations": ["current", "PATH", "active venv/conda", "managed", "pyenv"],
    }
