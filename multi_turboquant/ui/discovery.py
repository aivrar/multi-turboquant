# SPDX-License-Identifier: MIT
"""Bounded filesystem discovery for models and optional integrations."""

from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Iterable


MODEL_FILE_SUFFIXES = frozenset({".gguf", ".safetensors", ".bin", ".pt", ".pth"})
MAX_SCAN_DIRECTORIES = 5_000


def _directory(path: str | Path, label: str) -> Path:
    if not str(path).strip():
        raise ValueError(f"{label} is not configured")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def _walk(root: Path, *, max_depth: int) -> Iterable[tuple[Path, list[str], list[str]]]:
    visited = 0
    for current, directories, files in os.walk(root, followlinks=False):
        visited += 1
        if visited > MAX_SCAN_DIRECTORIES:
            raise RuntimeError(f"Scan stopped after {MAX_SCAN_DIRECTORIES} directories")
        path = Path(current)
        depth = len(path.relative_to(root).parts)
        directories[:] = [
            item
            for item in directories
            if not item.startswith(".") and not (path / item).is_symlink()
        ]
        if depth >= max_depth:
            directories[:] = []
        yield path, directories, files


def scan_models(
    root: str | Path,
    *,
    max_depth: int = 3,
    limit: int = 500,
) -> dict[str, object]:
    """Find recognized model files and Transformers model directories."""
    if not 0 <= max_depth <= 8:
        raise ValueError("max_depth must be between 0 and 8")
    if not 1 <= limit <= 2_000:
        raise ValueError("limit must be between 1 and 2000")
    resolved = _directory(root, "Model root")
    models: list[dict[str, object]] = []
    transformer_directories: set[Path] = set()

    for directory, _, files in _walk(resolved, max_depth=max_depth):
        file_set = set(files)
        if "config.json" in file_set and any(
            Path(item).suffix.lower() in {".safetensors", ".bin"} for item in files
        ):
            transformer_directories.add(directory)
        for filename in files:
            path = directory / filename
            suffix = path.suffix.lower()
            if suffix not in MODEL_FILE_SUFFIXES:
                continue
            stat = path.stat()
            models.append(
                {
                    "name": filename,
                    "path": str(path),
                    "relative_path": str(path.relative_to(resolved)),
                    "format": suffix.removeprefix("."),
                    "size_bytes": stat.st_size,
                    "launchable": suffix == ".gguf",
                }
            )
            if len(models) >= limit:
                break
        if len(models) >= limit:
            break

    for directory in transformer_directories:
        if len(models) >= limit:
            break
        models.append(
            {
                "name": directory.name,
                "path": str(directory),
                "relative_path": str(directory.relative_to(resolved)),
                "format": "transformers",
                "size_bytes": None,
                "launchable": False,
            }
        )

    models.sort(key=lambda item: (not bool(item["launchable"]), str(item["relative_path"]).lower()))
    return {
        "root": str(resolved),
        "models": models,
        "count": len(models),
        "truncated": len(models) >= limit,
        "supported_suffixes": sorted(MODEL_FILE_SUFFIXES),
    }


def _git_remote(path: Path) -> str | None:
    config_path = path / ".git" / "config"
    if not config_path.is_file() or config_path.stat().st_size > 1_000_000:
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read(config_path, encoding="utf-8")
    except (configparser.Error, OSError):
        return None
    for section in parser.sections():
        if section.startswith('remote "') and parser.has_option(section, "url"):
            return parser.get(section, "url")
    return None


def inspect_flashattention_source(path: str | Path) -> dict[str, object]:
    """Inspect a configured FlashAttention source checkout without executing it."""
    try:
        resolved = _directory(path, "FlashAttention source")
    except ValueError as exc:
        return {"path": str(path), "valid": False, "issues": [str(exc)]}
    markers = {
        "setup.py": (resolved / "setup.py").is_file(),
        "flash_attn": (resolved / "flash_attn").is_dir(),
        "csrc": (resolved / "csrc").is_dir(),
    }
    issues = [f"Missing {name}" for name, present in markers.items() if not present]
    version = None
    version_file = resolved / "version.txt"
    if version_file.is_file() and version_file.stat().st_size < 1_024:
        version = version_file.read_text(encoding="utf-8", errors="replace").strip() or None
    return {
        "path": str(resolved),
        "valid": not issues,
        "markers": markers,
        "version": version,
        "git_remote": _git_remote(resolved),
        "issues": issues,
    }


def _classify_addon(path: Path, files: set[str], directories: set[str]) -> str | None:
    name = path.name.lower().replace("_", "-")
    directories = {item.lower() for item in directories}
    if {"setup.py"}.issubset(files) and {"flash_attn", "csrc"}.issubset(directories):
        return "flashattention"
    if "fastdms" in name or "fastdms" in directories:
        return "fastdms"
    if "lmcache" in name or "lmcache" in directories:
        return "lmcache"
    if "minference" in name or "minference" in directories:
        return "minference"
    if "sageattention" in name or "sageattention" in directories:
        return "sageattention"
    if "godzilla" in name or "GODZILLA_KING.md" in files:
        return "godzilla"
    if "llama.cpp" in name or "llama-cpp" in name:
        return "llamacpp"
    if "CMakeLists.txt" in files and "ggml" in directories:
        return "llamacpp"
    return None


def scan_addon_roots(
    roots: Iterable[str | Path],
    *,
    max_depth: int = 2,
    limit: int = 200,
) -> dict[str, object]:
    """Scan only configured roots for recognized add-on source directories."""
    if not 0 <= max_depth <= 5:
        raise ValueError("max_depth must be between 0 and 5")
    results: list[dict[str, object]] = []
    errors: list[str] = []
    resolved_roots: list[str] = []
    seen: set[Path] = set()
    for raw_root in roots:
        try:
            root = _directory(raw_root, "Add-on root")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        resolved_roots.append(str(root))
        try:
            for directory, child_directories, filenames in _walk(root, max_depth=max_depth):
                kind = _classify_addon(directory, set(filenames), set(child_directories))
                if kind is None or directory in seen:
                    continue
                seen.add(directory)
                item = {
                    "kind": kind,
                    "name": directory.name,
                    "path": str(directory),
                    "git_remote": _git_remote(directory),
                }
                if kind == "flashattention":
                    item["source"] = inspect_flashattention_source(directory)
                if kind == "godzilla":
                    from ..integration import inspect_godzilla_checkout

                    item["source"] = inspect_godzilla_checkout(directory)
                if kind in {
                    "flashattention",
                    "fastdms",
                    "lmcache",
                    "minference",
                    "sageattention",
                }:
                    from ..optimizations.environments import inspect_profile_source

                    item["environment_profile"] = kind
                    item["local_source"] = inspect_profile_source(kind, directory)
                results.append(item)
                if len(results) >= limit:
                    break
        except (OSError, RuntimeError) as exc:
            errors.append(f"{root}: {exc}")
        if len(results) >= limit:
            break
    results.sort(key=lambda item: (str(item["kind"]), str(item["path"]).lower()))
    return {
        "roots": resolved_roots,
        "addons": results,
        "count": len(results),
        "errors": errors,
        "truncated": len(results) >= limit,
    }


def scan_environment_profiles(
    root: str | Path,
    *,
    cuda_toolkit: str | Path | None = None,
    local_source_profile: str | None = None,
    local_source: str | Path | None = None,
) -> dict[str, object]:
    """Report dependency profile readiness and materialization without imports."""
    from ..optimizations.environments import (
        BUILTIN_ENVIRONMENT_PROFILES,
        detect_environment_context,
        environment_python,
        get_environment_profile,
        plan_environment,
    )

    selected_profile = local_source_profile.strip().lower() if local_source_profile else None
    if local_source is not None and not selected_profile:
        raise ValueError("Choose a local source profile before selecting its checkout")
    if selected_profile is not None:
        get_environment_profile(selected_profile)
    context = detect_environment_context(cuda_toolkit=cuda_toolkit)
    profiles: list[dict[str, object]] = []
    for profile in BUILTIN_ENVIRONMENT_PROFILES:
        selected_source = local_source if profile.id == selected_profile else None
        plan = plan_environment(
            profile.id,
            root=root,
            context=context,
            local_source=selected_source,
        )
        project_file = plan.target / "pyproject.toml"
        interpreter = environment_python(plan.target, os_name=context.os)
        if not profile.installable:
            status = "blocked"
        elif interpreter.is_file():
            status = "installed"
        elif project_file.is_file():
            status = "configured"
        elif plan.ready:
            status = "ready"
        else:
            status = "incompatible"
        profiles.append(
            {
                "id": profile.id,
                "name": profile.name,
                "status": status,
                "ready": plan.ready,
                "target": str(plan.target),
                "source_build_available": bool(profile.source_build_packages),
                "local_source_supported": profile.local_source_package is not None,
                "local_source_selected": str(plan.local_source) if plan.local_source else None,
                "issues": [issue.to_dict() for issue in plan.issues],
            }
        )
    return {
        "root": str(Path(root).expanduser().resolve()),
        "local_source_profile": selected_profile,
        "local_source": str(Path(local_source).expanduser().resolve()) if local_source else None,
        "context": {
            "os": context.os,
            "compute": context.compute,
            "cuda_toolkit_version": context.cuda_toolkit_version,
            "cuda_toolkit_root": context.cuda_toolkit_root,
        },
        "profiles": profiles,
    }
