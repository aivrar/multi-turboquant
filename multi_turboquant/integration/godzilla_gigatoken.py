# SPDX-License-Identifier: MIT
"""Pinned, opt-in Gigatoken runtime preparation for Godzilla llama.cpp.

The workflow creates a new source tree from the reviewed Godzilla v0.3.7
release.  It never patches an arbitrary checkout.  The tokenizer port is
selected from a hash-verified comparison of pinned upstream commits and then
adapted to the older Godzilla CMake layout with exact, fail-closed edits.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


GODZILLA_URL = "https://github.com/atomicmilkshake/godzilla-llama.cpp.git"
GODZILLA_TAG = "v0.3.7"
GODZILLA_COMMIT = "ea1e79925b588bc4492b7a9e492e8fa97dce548f"

GIGATOKEN_URL = "https://github.com/marcelroed/gigatoken.git"
GIGATOKEN_REF = "main"
GIGATOKEN_VERSION = "0.10.0"
GIGATOKEN_COMMIT = "34a1599f0c0ae7d7cd0d1c530e6522320158b360"
GIGATOKEN_TOOLCHAIN = "nightly-2026-07-22"

GIGATOKEN_LLAMA_URL = "https://github.com/chynggi/gigatoken-llama.cpp"
GIGATOKEN_LLAMA_BASE = "eb41d503ba9496b816ff4180a34780bc13a4f4b8"
GIGATOKEN_LLAMA_COMMIT = "b47a0fc5b5cc8ba13ab08895e99566001400e70b"
GIGATOKEN_LLAMA_DIFF_URL = (
    f"{GIGATOKEN_LLAMA_URL}/compare/"
    f"{GIGATOKEN_LLAMA_BASE}...{GIGATOKEN_LLAMA_COMMIT}.diff"
)
GIGATOKEN_LLAMA_DIFF_SHA256 = (
    "21ed4818aaab79703fa778a3d4447b83c1a81df0c1fa72ee49f03b309290ca8e"
)
SELECTED_DIFF_SHA256 = "86af3ac27eaa4ddc99954d5673b0ed988a8fd55bd489c9a95ea7a0aa8f8a9457"
GIGATOKEN_C_ABI_PATCH_SHA256 = (
    "e36f9fd2a40d896bb1aa1cd6e6b90a6132fd7ac4cb50861aae9eb3e26358021f"
)

_SELECTED_DIFF_PATHS = frozenset(
    {
        "cmake/gigatoken.cmake",
        "patches/gigatoken-llama-cpp.patch",
        "src/llama-gigatoken.cpp",
        "src/llama-gigatoken.h",
        "src/llama-vocab.cpp",
        "src/llama-vocab.h",
        "tests/test-gigatoken.cpp",
    }
)
_RUNTIME_FILES = (
    "CMakeLists.txt",
    "cmake/gigatoken.cmake",
    "patches/gigatoken-llama-cpp.patch",
    "src/CMakeLists.txt",
    "src/llama-gigatoken.cpp",
    "src/llama-gigatoken.h",
    "src/llama-vocab.cpp",
    "src/llama-vocab.h",
    "tests/CMakeLists.txt",
    "tests/test-gigatoken.cpp",
)
_TRACKED_RUNTIME_FILES = frozenset(
    {
        "CMakeLists.txt",
        "src/CMakeLists.txt",
        "src/llama-vocab.cpp",
        "src/llama-vocab.h",
        "tests/CMakeLists.txt",
    }
)

_GIGATOKEN_TEST_REGEX = (
    r"^test-gigatoken-(matrix|gpt-2|llama-bpe|llama-spm|mpt|qwen2|"
    r"qwen35|gemma-4|bert-fallback)$"
)


@dataclass(frozen=True)
class RuntimeIssue:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class GodzillaGigatokenPlan:
    action: str
    target: Path
    build_dir: Path
    backend: str
    max_jobs: int
    with_curl: bool
    generator: str | None
    cuda_compiler: Path | None
    fixture_dir: Path | None
    commands: tuple[tuple[str, ...], ...]
    issues: tuple[RuntimeIssue, ...]

    @property
    def ready(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "target": str(self.target),
            "build_dir": str(self.build_dir),
            "backend": self.backend,
            "max_jobs": self.max_jobs,
            "with_curl": self.with_curl,
            "generator": self.generator,
            "cuda_compiler": str(self.cuda_compiler) if self.cuda_compiler else None,
            "fixture_dir": str(self.fixture_dir) if self.fixture_dir else None,
            "commands": [list(command) for command in self.commands],
            "issues": [issue.to_dict() for issue in self.issues],
            "ready": self.ready,
            "pins": {
                "godzilla": GODZILLA_COMMIT,
                "gigatoken_llama": GIGATOKEN_LLAMA_COMMIT,
                "gigatoken": GIGATOKEN_COMMIT,
                "rust": GIGATOKEN_TOOLCHAIN,
            },
        }


def _supported_platform() -> bool:
    system = platform.system().lower()
    machine = platform.machine().lower()
    return system in {"windows", "linux"} and machine in {"amd64", "x86_64"}


def _resolve_cuda_compiler(value: str | Path | None) -> Path | None:
    if value:
        candidate = Path(value).expanduser().resolve()
        if candidate.is_dir():
            candidate = candidate / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
        return candidate if candidate.is_file() else None
    found = shutil.which("nvcc")
    return Path(found).resolve() if found else None


def _configure_command(
    target: Path,
    build_dir: Path,
    *,
    backend: str,
    with_curl: bool,
    generator: str | None,
    cuda_compiler: Path | None,
    fixture_dir: Path | None,
) -> tuple[str, ...]:
    command = ["cmake", "-S", str(target), "-B", str(build_dir)]
    if generator:
        command.extend(("-G", generator))
    command.extend(
        (
            "-DLLAMA_GIGATOKEN=ON",
            "-DLLAMA_BUILD_TESTS=ON",
            "-DLLAMA_BUILD_SERVER=ON",
            "-DLLAMA_BUILD_UI=OFF",
            f"-DLLAMA_CURL={'ON' if with_curl else 'OFF'}",
            f"-DGGML_CUDA={'ON' if backend == 'cuda' else 'OFF'}",
            "-DGGML_NATIVE=OFF",
            "-DGGML_CCACHE=OFF",
            "-DBUILD_SHARED_LIBS=OFF",
        )
    )
    if cuda_compiler is not None:
        command.append(f"-DCMAKE_CUDA_COMPILER={cuda_compiler}")
    if fixture_dir is not None:
        command.append(f"-DLLAMA_GIGATOKEN_FIXTURE_DIR={fixture_dir}")
    return tuple(command)


def plan_godzilla_gigatoken(
    target: str | Path,
    *,
    action: str = "prepare",
    backend: str = "cpu",
    max_jobs: int = 2,
    with_curl: bool = False,
    generator: str | None = None,
    cuda_toolkit: str | Path | None = None,
    fixture_dir: str | Path | None = None,
) -> GodzillaGigatokenPlan:
    """Create a read-only plan for a pinned preparation or build."""
    target_path = Path(target).expanduser().resolve()
    normalized_action = action.strip().lower()
    normalized_backend = backend.strip().lower()
    build_dir = target_path / f"build-gigatoken-{normalized_backend}"
    resolved_fixtures = Path(fixture_dir).expanduser().resolve() if fixture_dir else None
    issues: list[RuntimeIssue] = []

    if normalized_action not in {"prepare", "build", "all", "verify"}:
        issues.append(RuntimeIssue("error", "invalid_action", "Action is not supported."))
    if normalized_backend not in {"cpu", "cuda"}:
        issues.append(RuntimeIssue("error", "invalid_backend", "Backend must be cpu or cuda."))
    if not 1 <= max_jobs <= 64:
        issues.append(RuntimeIssue("error", "invalid_max_jobs", "max_jobs must be 1 to 64."))
    if not _supported_platform():
        issues.append(
            RuntimeIssue(
                "error",
                "unsupported_platform",
                "The reviewed runtime port supports Windows x64 and Linux x86_64 only.",
            )
        )

    preparing = normalized_action in {"prepare", "all"}
    compiling = normalized_action in {"build", "all"}
    verifying = normalized_action in {"build", "all", "verify"}
    if preparing and target_path.exists():
        issues.append(
            RuntimeIssue(
                "error",
                "target_exists",
                f"The output must be a new path; it will not overwrite {target_path}.",
            )
        )
    if verifying and normalized_action != "all":
        inspection = inspect_godzilla_gigatoken(target_path)
        issues.extend(
            RuntimeIssue("error", "invalid_prepared_tree", str(message))
            for message in inspection["issues"]
        )

    required_tools = ["git"]
    if compiling:
        required_tools.extend(("cmake", "ctest", "cargo", "rustc", "rustup"))
    elif verifying:
        required_tools.append("ctest")
    for tool in required_tools:
        if shutil.which(tool) is None:
            issues.append(RuntimeIssue("error", f"missing_{tool}", f"Required tool not found: {tool}"))

    cuda_compiler = (
        _resolve_cuda_compiler(cuda_toolkit)
        if normalized_backend == "cuda" and compiling
        else None
    )
    if normalized_backend == "cuda" and compiling and cuda_compiler is None:
        issues.append(
            RuntimeIssue(
                "error",
                "missing_nvcc",
                "CUDA builds require nvcc on PATH or --cuda-toolkit pointing to its toolkit root.",
            )
        )
    if resolved_fixtures is not None and not resolved_fixtures.is_dir():
        issues.append(
            RuntimeIssue(
                "error",
                "invalid_fixture_dir",
                f"Optional GGUF fixture directory not found: {resolved_fixtures}",
            )
        )

    clone = (
        "git",
        "clone",
        "--config",
        "core.autocrlf=false",
        "--depth",
        "1",
        "--branch",
        GODZILLA_TAG,
        "--single-branch",
        GODZILLA_URL,
        str(target_path),
    )
    configure = _configure_command(
        target_path,
        build_dir,
        backend=normalized_backend,
        with_curl=with_curl,
        generator=generator,
        cuda_compiler=cuda_compiler,
        fixture_dir=resolved_fixtures,
    )
    build = (
        "cmake",
        "--build",
        str(build_dir),
        "--config",
        "Release",
        "--target",
        "llama-server",
        "test-gigatoken",
        "test-tokenizer-0",
        "--parallel",
        str(max_jobs),
    )
    commands: list[tuple[str, ...]] = []
    if preparing:
        commands.append(clone)
    if compiling:
        commands.extend(
            (
                ("rustup", "toolchain", "install", GIGATOKEN_TOOLCHAIN, "--profile", "minimal"),
                configure,
                build,
            )
        )
    issues.append(
        RuntimeIssue(
            "info",
            "pinned_source",
            "Preparation uses Godzilla v0.3.7 and exact Gigatoken/Gigatoken-llama revisions; arbitrary checkouts are not patched.",
        )
    )
    if normalized_backend == "cuda":
        issues.append(
            RuntimeIssue(
                "warning",
                "cuda_qualification",
                "The tokenizer port was fully compiled and tested in a CPU build; run the same differential suite after this CUDA build before serving.",
            )
        )
    return GodzillaGigatokenPlan(
        action=normalized_action,
        target=target_path,
        build_dir=build_dir,
        backend=normalized_backend,
        max_jobs=max_jobs,
        with_curl=with_curl,
        generator=generator,
        cuda_compiler=cuda_compiler,
        fixture_dir=resolved_fixtures,
        commands=tuple(commands),
        issues=tuple(issues),
    )


def _runtime_file_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in _RUNTIME_FILES:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"Prepared runtime file is missing: {path}")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _download_reviewed_diff(
    *,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> bytes:
    request = urllib.request.Request(
        GIGATOKEN_LLAMA_DIFF_URL,
        headers={"User-Agent": "multi-turboquant-godzilla-gigatoken"},
    )
    with opener(request, timeout=60) as response:  # type: ignore[attr-defined]
        data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != GIGATOKEN_LLAMA_DIFF_SHA256:
        raise RuntimeError(
            "Pinned Gigatoken-llama review diff hash mismatch: "
            f"expected {GIGATOKEN_LLAMA_DIFF_SHA256}, found {actual}"
        )
    return data


def _select_reviewed_diff(data: bytes) -> bytes:
    sections = re.split(br"(?=^diff --git )", data, flags=re.MULTILINE)
    selected: list[bytes] = []
    found: set[str] = set()
    for section in sections:
        match = re.match(br"diff --git a/(\S+) b/", section)
        if not match:
            continue
        path = match.group(1).decode("utf-8")
        if path in _SELECTED_DIFF_PATHS:
            found.add(path)
            selected.append(section)
    if found != _SELECTED_DIFF_PATHS:
        missing = ", ".join(sorted(_SELECTED_DIFF_PATHS - found))
        raise RuntimeError(f"Pinned review diff is missing selected paths: {missing}")
    result = b"".join(selected)
    actual = hashlib.sha256(result).hexdigest()
    if actual != SELECTED_DIFF_SHA256:
        raise RuntimeError(
            f"Selected Gigatoken runtime diff hash mismatch: expected {SELECTED_DIFF_SHA256}, found {actual}"
        )
    return result


def _run_checked(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    input_data: bytes | None = None,
    env: Mapping[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        input=input_data,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )
    if result.returncode:
        detail = ""
        if capture:
            detail = (result.stderr or result.stdout or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{detail}".rstrip())
    return result


def _remove_created_tree(path: Path) -> None:
    """Remove only a workflow-owned temporary tree, including read-only Git packs."""
    def make_writable_and_retry(function, value, _error) -> None:
        os.chmod(value, stat.S_IWRITE | stat.S_IREAD)
        function(value)

    shutil.rmtree(path, onerror=make_writable_and_retry)


def _git_revision(root: Path) -> str | None:
    try:
        result = _run_checked(("git", "rev-parse", "HEAD"), cwd=root, capture=True)
    except (OSError, RuntimeError):
        return None
    return result.stdout.decode("utf-8", errors="replace").strip()


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one reviewed edit anchor in {path}; found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


def _adapt_reviewed_port(root: Path) -> None:
    root_option = (
        'option(LLAMA_LLGUIDANCE "llama-common: include LLGuidance library for structured output in common utils" OFF)\n'
    )
    _replace_once(
        root / "CMakeLists.txt",
        root_option,
        root_option
        + 'option(LLAMA_GIGATOKEN  "llama: use the experimental GigaToken tokenizer backend" OFF)\n',
    )
    root_build_anchor = (
        "if (NOT TARGET ggml AND NOT LLAMA_USE_SYSTEM_GGML)\n"
        "    set(GGML_BUILD_NUMBER ${LLAMA_BUILD_NUMBER})\n"
        "    set(GGML_BUILD_COMMIT ${LLAMA_BUILD_COMMIT})\n"
        "    add_subdirectory(ggml)\n"
        "    # ... otherwise assume ggml is added by a parent CMakeLists.txt\n"
        "endif()\n\n"
        "#\n# build the library\n"
    )
    _replace_once(
        root / "CMakeLists.txt",
        root_build_anchor,
        root_build_anchor.replace(
            "\n#\n# build the library\n",
            '\nif (LLAMA_GIGATOKEN)\n    include("cmake/gigatoken.cmake")\nendif()\n\n#\n# build the library\n',
        ),
    )

    source_anchor = (
        "if (NOT GGML_CUDA OR GGML_BACKEND_DL)\n"
        "    target_sources(llama PRIVATE llama-triattention-gpu-stub.cpp)\n"
        "endif()\n\n"
    )
    source_block = (
        "if (LLAMA_GIGATOKEN)\n"
        "    target_sources(llama PRIVATE llama-gigatoken.cpp llama-gigatoken.h)\n"
        "    target_compile_definitions(llama PRIVATE LLAMA_GIGATOKEN=1)\n"
        "    if (LLAMA_BUILD_TESTS)\n"
        "        target_compile_definitions(llama PRIVATE LLAMA_GIGATOKEN_TESTS=1)\n"
        "    endif()\n"
        "    target_link_libraries(llama PRIVATE gigatoken::llama)\n"
        "endif()\n\n"
    )
    _replace_once(root / "src" / "CMakeLists.txt", source_anchor, source_anchor + source_block)

    test_anchor = (
        "llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         "
        "ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)\n\n"
    )
    test_block = """if (LLAMA_GIGATOKEN)
    set(LLAMA_GIGATOKEN_FIXTURE_DIR "${PROJECT_SOURCE_DIR}/../fixtures/gguf" CACHE PATH
        "Directory containing optional real-model vocab-only GGUF fixtures")

    llama_build(test-gigatoken.cpp)
    target_include_directories(test-gigatoken PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_compile_definitions(test-gigatoken PRIVATE LLAMA_GIGATOKEN_TESTS=1)

    llama_test(test-gigatoken NAME test-gigatoken-matrix)
    llama_test(test-gigatoken NAME test-gigatoken-gpt-2
        ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf enabled long)
    llama_test(test-gigatoken NAME test-gigatoken-llama-bpe
        ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf enabled)
    llama_test(test-gigatoken NAME test-gigatoken-llama-spm
        ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf enabled long)
    llama_test(test-gigatoken NAME test-gigatoken-mpt
        ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf enabled)
    llama_test(test-gigatoken NAME test-gigatoken-qwen2
        ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf enabled)
    llama_test(test-gigatoken NAME test-gigatoken-qwen35
        ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf enabled)
    if (EXISTS "${LLAMA_GIGATOKEN_FIXTURE_DIR}/ggml-vocab-deepseek-v3.gguf")
        llama_test(test-gigatoken NAME test-gigatoken-deepseek-v3
            ARGS ${LLAMA_GIGATOKEN_FIXTURE_DIR}/ggml-vocab-deepseek-v3.gguf enabled)
    endif()
    if (EXISTS "${LLAMA_GIGATOKEN_FIXTURE_DIR}/ggml-vocab-gpt-oss-120b.gguf")
        llama_test(test-gigatoken NAME test-gigatoken-gpt-oss
            ARGS ${LLAMA_GIGATOKEN_FIXTURE_DIR}/ggml-vocab-gpt-oss-120b.gguf enabled)
    endif()
    if (EXISTS "${LLAMA_GIGATOKEN_FIXTURE_DIR}/ggml-vocab-kimi-k2.7-code.gguf")
        llama_test(test-gigatoken NAME test-gigatoken-kimi-k2.7-code
            ARGS ${LLAMA_GIGATOKEN_FIXTURE_DIR}/ggml-vocab-kimi-k2.7-code.gguf enabled)
    endif()
    llama_test(test-gigatoken NAME test-gigatoken-gemma-4
        ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf enabled)
    llama_test(test-gigatoken NAME test-gigatoken-bert-fallback
        ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf disabled)
endif()

"""
    _replace_once(root / "tests" / "CMakeLists.txt", test_anchor, test_anchor + test_block)

    cmake_path = root / "cmake" / "gigatoken.cmake"
    _replace_once(
        cmake_path,
        'message(FATAL_ERROR "GigaToken submodule is missing; run git submodule update --init vendor/gigatoken")',
        'message(FATAL_ERROR "GigaToken source is missing; prepare the pinned vendor/gigatoken checkout")',
    )
    _replace_once(
        cmake_path,
        'file(SHA256 "${GIGATOKEN_PATCH}" GIGATOKEN_PATCH_ACTUAL_SHA256)',
        'file(READ "${GIGATOKEN_PATCH}" GIGATOKEN_PATCH_CONTENT)\n'
        'string(REPLACE "\\r\\n" "\\n" GIGATOKEN_PATCH_CONTENT_LF "${GIGATOKEN_PATCH_CONTENT}")\n'
        'string(SHA256 GIGATOKEN_PATCH_ACTUAL_SHA256 "${GIGATOKEN_PATCH_CONTENT_LF}")',
    )
    _replace_once(
        root / "src" / "llama-gigatoken.cpp",
        'LLAMA_LOG_INFO("%s: GigaToken encode: %zu bytes -> %zu tokens\\n", __func__, text.size(), buffer.len);',
        'LLAMA_LOG_DEBUG("%s: GigaToken encode: %zu bytes -> %zu tokens\\n", __func__, text.size(), buffer.len);',
    )


def prepare_godzilla_gigatoken(
    plan: GodzillaGigatokenPlan,
    *,
    confirmed: bool = False,
) -> dict[str, object]:
    """Create a new pinned combined source tree after explicit confirmation."""
    if plan.action not in {"prepare", "all"}:
        raise ValueError("Preparation requires a prepare or all plan")
    if not plan.ready:
        raise RuntimeError("Preparation plan contains errors")
    if not confirmed:
        raise RuntimeError("Preparation was not confirmed")

    reviewed_diff = _select_reviewed_diff(_download_reviewed_diff())
    target = plan.target
    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{target.name}.mtq-", dir=target.parent)).resolve()
    try:
        _run_checked(
            (
                "git",
                "clone",
                "--config",
                "core.autocrlf=false",
                "--depth",
                "1",
                "--branch",
                GODZILLA_TAG,
                "--single-branch",
                GODZILLA_URL,
                str(work),
            )
        )
        revision = _git_revision(work)
        if revision != GODZILLA_COMMIT:
            raise RuntimeError(f"Godzilla revision mismatch: expected {GODZILLA_COMMIT}, found {revision}")
        _run_checked(("git", "apply", "--check", "--whitespace=nowarn", "-"), cwd=work, input_data=reviewed_diff)
        _run_checked(("git", "apply", "--whitespace=nowarn", "-"), cwd=work, input_data=reviewed_diff)
        _adapt_reviewed_port(work)
        c_abi_patch = work / "patches" / "gigatoken-llama-cpp.patch"
        c_abi_patch.write_bytes(c_abi_patch.read_bytes().replace(b"\r\n", b"\n"))

        vendor = work / "vendor" / "gigatoken"
        vendor.parent.mkdir(parents=True, exist_ok=True)
        _run_checked(
            (
                "git",
                "clone",
                "--config",
                "core.autocrlf=false",
                "--filter=blob:none",
                "--no-checkout",
                GIGATOKEN_URL,
                str(vendor),
            )
        )
        _run_checked(("git", "checkout", "--detach", GIGATOKEN_COMMIT), cwd=vendor)
        vendor_revision = _git_revision(vendor)
        if vendor_revision != GIGATOKEN_COMMIT:
            raise RuntimeError(
                f"Gigatoken revision mismatch: expected {GIGATOKEN_COMMIT}, found {vendor_revision}"
            )
        patch = c_abi_patch.read_bytes()
        normalized_patch = patch.replace(b"\r\n", b"\n")
        patch_hash = hashlib.sha256(normalized_patch).hexdigest()
        if patch_hash != GIGATOKEN_C_ABI_PATCH_SHA256:
            raise RuntimeError(
                f"Gigatoken C ABI patch mismatch: expected {GIGATOKEN_C_ABI_PATCH_SHA256}, found {patch_hash}"
            )

        manifest = {
            "schema": 1,
            "godzilla": {"url": GODZILLA_URL, "tag": GODZILLA_TAG, "commit": GODZILLA_COMMIT},
            "gigatoken_llama": {
                "url": GIGATOKEN_LLAMA_URL,
                "base": GIGATOKEN_LLAMA_BASE,
                "commit": GIGATOKEN_LLAMA_COMMIT,
                "diff_sha256": GIGATOKEN_LLAMA_DIFF_SHA256,
                "selected_diff_sha256": SELECTED_DIFF_SHA256,
            },
            "gigatoken": {
                "url": GIGATOKEN_URL,
                "ref": GIGATOKEN_REF,
                "version": GIGATOKEN_VERSION,
                "commit": GIGATOKEN_COMMIT,
                "c_abi_patch_sha256": patch_hash,
            },
            "rust_toolchain": GIGATOKEN_TOOLCHAIN,
            "runtime_files_sha256": _runtime_file_hashes(work),
            "notes": [
                "Prepared by Multi-TurboQuant from pinned MIT-licensed upstream sources.",
                "Unsupported vocabularies retain llama.cpp's C++ tokenizer fallback.",
                "Optional external real-model fixtures are registered only when present.",
            ],
        }
        (work / ".mtq-gigatoken-runtime.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if target.exists():
            raise RuntimeError(f"Target appeared during preparation and will not be overwritten: {target}")
        os.replace(work, target)
    except Exception:
        if work.exists():
            _remove_created_tree(work)
        raise
    return inspect_godzilla_gigatoken(target)


def _load_manifest(root: Path) -> dict[str, object] | None:
    path = root / ".mtq-gigatoken-runtime.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _find_binary(build_dir: Path, name: str) -> Path | None:
    names = (f"{name}.exe", name)
    for directory in (build_dir / "bin" / "Release", build_dir / "bin", build_dir / "Release"):
        for filename in names:
            candidate = directory / filename
            if candidate.is_file():
                return candidate.resolve()
    return None


def inspect_godzilla_gigatoken(path: str | Path) -> dict[str, object]:
    """Validate provenance and markers in a prepared source tree."""
    root = Path(path).expanduser().resolve()
    issues: list[str] = []
    manifest = _load_manifest(root) if root.is_dir() else None
    if not root.is_dir():
        issues.append(f"Prepared source directory not found: {root}")
    if manifest is None:
        issues.append("Multi-TurboQuant Gigatoken runtime manifest is missing or invalid.")
    elif manifest.get("schema") != 1:
        issues.append("Unsupported Gigatoken runtime manifest schema.")
    elif not (
        isinstance(manifest.get("godzilla"), dict)
        and manifest["godzilla"].get("commit") == GODZILLA_COMMIT
        and isinstance(manifest.get("gigatoken"), dict)
        and manifest["gigatoken"].get("commit") == GIGATOKEN_COMMIT
        and isinstance(manifest.get("gigatoken_llama"), dict)
        and manifest["gigatoken_llama"].get("commit") == GIGATOKEN_LLAMA_COMMIT
        and manifest["gigatoken_llama"].get("diff_sha256") == GIGATOKEN_LLAMA_DIFF_SHA256
        and manifest.get("rust_toolchain") == GIGATOKEN_TOOLCHAIN
    ):
        issues.append("Prepared runtime manifest does not match the reviewed source pins.")
    revision = _git_revision(root) if root.is_dir() else None
    vendor = root / "vendor" / "gigatoken"
    vendor_revision = _git_revision(vendor) if vendor.is_dir() else None
    if revision != GODZILLA_COMMIT:
        issues.append(f"Godzilla revision must be {GODZILLA_COMMIT}; found {revision}.")
    if revision == GODZILLA_COMMIT:
        try:
            changed_output = _run_checked(
                ("git", "diff", "--name-only", "HEAD"),
                cwd=root,
                capture=True,
            ).stdout.decode("utf-8", errors="replace")
            changed = {line.strip().replace("\\", "/") for line in changed_output.splitlines() if line.strip()}
        except (OSError, RuntimeError) as exc:
            issues.append(f"Could not validate Godzilla source status: {exc}")
        else:
            if changed != _TRACKED_RUNTIME_FILES:
                issues.append(
                    "Godzilla tracked changes differ from the reviewed adapter set: "
                    + ", ".join(sorted(changed ^ _TRACKED_RUNTIME_FILES))
                )
    if vendor_revision != GIGATOKEN_COMMIT:
        issues.append(f"Gigatoken revision must be {GIGATOKEN_COMMIT}; found {vendor_revision}.")
    if vendor_revision == GIGATOKEN_COMMIT:
        try:
            status = _run_checked(
                ("git", "status", "--porcelain", "--untracked-files=no"),
                cwd=vendor,
                capture=True,
            ).stdout.decode("utf-8", errors="replace").strip()
        except (OSError, RuntimeError) as exc:
            issues.append(f"Could not validate Gigatoken source status: {exc}")
        else:
            if status:
                issues.append("Gigatoken source contains tracked modifications.")
    markers = (
        root / "cmake" / "gigatoken.cmake",
        root / "patches" / "gigatoken-llama-cpp.patch",
        root / "src" / "llama-gigatoken.cpp",
        root / "tests" / "test-gigatoken.cpp",
    )
    issues.extend(f"Prepared runtime marker missing: {marker}" for marker in markers if not marker.is_file())
    expected_hashes = manifest.get("runtime_files_sha256") if manifest else None
    if not isinstance(expected_hashes, dict):
        issues.append("Prepared runtime file hashes are missing from the manifest.")
    elif root.is_dir():
        try:
            actual_hashes = _runtime_file_hashes(root)
        except RuntimeError as exc:
            issues.append(str(exc))
        else:
            changed = [
                relative
                for relative, actual in actual_hashes.items()
                if expected_hashes.get(relative) != actual
            ]
            if changed:
                issues.append("Prepared runtime files changed: " + ", ".join(changed))
    return {
        "path": str(root),
        "valid": not issues,
        "issues": issues,
        "manifest": manifest,
        "godzilla_revision": revision,
        "gigatoken_revision": vendor_revision,
        "features": {"godzilla": revision == GODZILLA_COMMIT, "gigatoken": all(item.is_file() for item in markers)},
    }


def verify_godzilla_gigatoken(plan: GodzillaGigatokenPlan) -> dict[str, object]:
    """Run the differential tokenizer suites and verify the server binary."""
    inspection = inspect_godzilla_gigatoken(plan.target)
    if not inspection["valid"]:
        raise RuntimeError("Prepared source validation failed: " + "; ".join(inspection["issues"]))
    gigatoken_test = _find_binary(plan.build_dir, "test-gigatoken")
    tokenizer_test = _find_binary(plan.build_dir, "test-tokenizer-0")
    server = _find_binary(plan.build_dir, "llama-server")
    missing = [
        name
        for name, value in (
            ("test-gigatoken", gigatoken_test),
            ("test-tokenizer-0", tokenizer_test),
            ("llama-server", server),
        )
        if value is None
    ]
    if missing:
        raise RuntimeError(f"Build outputs are missing: {', '.join(missing)}")
    common = ("ctest", "--test-dir", str(plan.build_dir), "-C", "Release", "--output-on-failure")
    _run_checked((*common, "-R", _GIGATOKEN_TEST_REGEX))
    _run_checked((*common, "-R", r"^test-tokenizer-0-"))
    version = _run_checked((str(server), "--version"), capture=True)
    return {
        "valid": True,
        "backend": plan.backend,
        "build_dir": str(plan.build_dir),
        "server": str(server),
        "server_version": (version.stdout or version.stderr or b"").decode("utf-8", errors="replace").strip(),
        "differential_tests": 9,
        "legacy_tokenizer_suite": "passed",
        "optional_external_fixtures": "run only when supplied at configure time",
    }


def build_godzilla_gigatoken(
    plan: GodzillaGigatokenPlan,
    *,
    confirmed: bool = False,
) -> dict[str, object]:
    """Configure, build, and verify the prepared runtime."""
    if plan.action not in {"build", "all"}:
        raise ValueError("Build requires a build or all plan")
    if not plan.ready:
        raise RuntimeError("Build plan contains errors")
    if not confirmed:
        raise RuntimeError("Build was not confirmed")
    if plan.action == "all":
        inspection = inspect_godzilla_gigatoken(plan.target)
        if not inspection["valid"]:
            raise RuntimeError("Prepare the combined source tree before building it")
    env = os.environ.copy()
    if plan.cuda_compiler is not None:
        env["CUDACXX"] = str(plan.cuda_compiler)
        env["PATH"] = str(plan.cuda_compiler.parent) + os.pathsep + env.get("PATH", "")
    _run_checked(("rustup", "toolchain", "install", GIGATOKEN_TOOLCHAIN, "--profile", "minimal"), env=env)
    configure = _configure_command(
        plan.target,
        plan.build_dir,
        backend=plan.backend,
        with_curl=plan.with_curl,
        generator=plan.generator,
        cuda_compiler=plan.cuda_compiler,
        fixture_dir=plan.fixture_dir,
    )
    _run_checked(configure, env=env)
    _run_checked(
        (
            "cmake",
            "--build",
            str(plan.build_dir),
            "--config",
            "Release",
            "--target",
            "llama-server",
            "test-gigatoken",
            "test-tokenizer-0",
            "--parallel",
            str(plan.max_jobs),
        ),
        env=env,
    )
    return verify_godzilla_gigatoken(plan)
