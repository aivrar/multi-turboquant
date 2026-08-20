# SPDX-License-Identifier: MIT
"""Bounded filesystem discovery for models and optional integrations."""

from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Iterable


MODEL_FILE_SUFFIXES = frozenset({".gguf", ".safetensors", ".bin", ".pt", ".pth"})
MAX_SCAN_DIRECTORIES = 5_000


# These source projects are not installable Python profiles. Marker groups let
# the UI inspect them without executing checkout code; each setup contract says
# whether a separate reviewed build workflow exists.
_SOURCE_ADDON_SPECS: dict[str, dict[str, object]] = {
    "cuda_weight_share": {
        "profile": "cuda_weight_share",
        "name": "CUDA LLM Weight Share",
        "source_url": "https://github.com/pontostroy/cuda-llm-weight-share",
        "marker_groups": (
            ("README.md",),
            ("LICENSE",),
            ("cuda-llm-weight-share.c",),
        ),
        "summary": "Pinned Linux CUDA LD_PRELOAD helper for sharing one model-weight allocation.",
        "setup": {
            "mode": "reviewed_source_build",
            "automatic": True,
            "requirements": (
                "Linux x86_64, CUDA toolkit headers, GCC, nm, ldd, and file",
                "The exact reviewed source commit",
                "A reconnaissance run before any production MODEL_SIZE is accepted",
            ),
            "next_steps": (
                "Run mtq-weight-share plan, then build with explicit confirmation.",
                "Use MODEL_SIZE=0 once to identify the exact weight allocation.",
                "Use a unique IPC name per model, GPU, allocation, and runtime build.",
            ),
        },
    },
    "gigatoken_llamacpp": {
        "profile": "gigatoken_llamacpp",
        "name": "Gigatoken llama.cpp",
        "source_url": "https://github.com/chynggi/gigatoken-llama.cpp",
        "marker_groups": (
            ("docs/gigatoken.md",),
            ("cmake/gigatoken.cmake",),
            ("src/llama-gigatoken.cpp", "src/llama-gigatoken.h"),
            ("patches/gigatoken-llama-cpp.patch",),
        ),
        "summary": "Experimental Windows x64/Linux x86_64 llama.cpp fork; Multi-TurboQuant can port its reviewed tokenizer changes onto exact Godzilla v0.3.7 or 09214b160 through mtq-godzilla-gigatoken.",
        "setup": {
            "mode": "separate_runtime_fork",
            "automatic": False,
            "requirements": (
                "Windows x64 or Linux x86_64, CMake, a C++ toolchain, and the pinned nightly Rust toolchain",
                "The pinned Gigatoken submodule and reviewed upstream patch",
                "A separate build tree with LLAMA_GIGATOKEN=ON",
            ),
            "next_steps": (
                "Run the fork's differential tokenizer tests against its preserved C++ path.",
                "Use mtq-godzilla-gigatoken to prepare a separate exact Godzilla v0.3.7 or 09214b160 tree; arbitrary checkouts are not patched.",
                "Keep the normal tokenizer fallback for unsupported vocabularies.",
            ),
        },
    },
    "maru": {
        "profile": "maru",
        "name": "Maru",
        "source_url": "https://github.com/xcena-dev/maru",
        "marker_groups": (
            ("README.md",),
            ("pyproject.toml", "setup.py"),
            ("maru_resource_manager", "maru_server", "maru"),
        ),
        "summary": "CXL shared-memory KV-cache project; host services and CXL hardware remain required.",
        "setup": {
            "mode": "guided_host_setup",
            "automatic": False,
            "requirements": (
                "Ubuntu 24.04 or newer with Python 3.12+, GCC, and CMake",
                "A configured CXL DAX device (/dev/dax*) or upstream's documented emulation",
                "Permission to install and operate the Maru resource-manager and server services",
            ),
            "next_steps": (
                "Review the checkout's README.md and install.sh on the target CXL host.",
                "Complete upstream host/device setup before running its installer.",
                "Verify maru-resource-manager, maru-server, and maru_lmcache before enabling the catalog entry.",
            ),
        },
    },
    "speculative_prefill": {
        "profile": "speculative_prefill",
        "name": "Speculative Prefill",
        "source_url": "https://github.com/Jingyu6/speculative_prefill",
        "marker_groups": (
            ("README.md",),
            ("requirements.txt", "environment.yml", "setup.py", "pyproject.toml"),
            ("speculative_prefill", "vllm", "scripts", "src"),
        ),
        "summary": "Experimental vLLM monkey patch; version-pinned runtime integration is not maintained here.",
        "setup": {
            "mode": "research_manual",
            "automatic": False,
            "requirements": (
                "The upstream pinned Torch 2.4.0 and vLLM 0.6.3.post1 stack",
                "A disposable environment for its source monkey patches",
            ),
            "next_steps": (
                "Reproduce the exact upstream environment independently.",
                "Validate the patch against a compatible vLLM checkout before serving.",
            ),
        },
    },
    "rocketkv": {
        "profile": "rocketkv",
        "name": "RocketKV",
        "source_url": "https://github.com/NVlabs/RocketKV",
        "marker_groups": (
            ("README.md",),
            ("requirements.txt", "environment.yml", "setup.py", "pyproject.toml"),
            ("rocketkv", "src", "evaluation", "eval"),
        ),
        "summary": "Research KV-compression snapshot; packaging is limited by its upstream license and stack.",
        "setup": {
            "mode": "research_manual",
            "automatic": False,
            "requirements": (
                "Acceptance of the upstream non-commercial licensing boundary",
                "The repository's research dependency stack and supported model artifacts",
            ),
            "next_steps": (
                "Review the license before use.",
                "Run the upstream evaluation workflow outside production serving.",
            ),
        },
    },
    "lexico": {
        "profile": "lexico",
        "name": "Lexico",
        "source_url": "https://github.com/krafton-ai/lexico",
        "marker_groups": (
            ("README.md",),
            ("setup.py", "pyproject.toml"),
            ("lexico", "src"),
        ),
        "summary": "WIP sparse-coding project; a model-specific trained dictionary is still required.",
        "setup": {
            "mode": "artifact_required",
            "automatic": False,
            "requirements": (
                "A compatible model-specific trained dictionary",
                "The upstream Python environment and evaluation configuration",
            ),
            "next_steps": (
                "Train or obtain the matching dictionary.",
                "Validate reconstruction quality before runtime integration.",
            ),
        },
    },
    "adadecode": {
        "profile": "adadecode",
        "name": "AdaDecode",
        "source_url": "https://github.com/weizhepei/AdaDecode",
        "marker_groups": (
            ("README.md",),
            ("requirements.txt", "environment.yml", "setup.py", "pyproject.toml"),
            ("adadecode", "src", "scripts", "eval", "evaluation"),
        ),
        "summary": "Research speculative-decoding code; task-specific prediction heads are not a generic add-on.",
        "setup": {
            "mode": "artifact_required",
            "automatic": False,
            "requirements": (
                "Compatible trained prediction heads",
                "A reviewed license and matching research runtime",
            ),
            "next_steps": (
                "Resolve the licensing and artifact requirements upstream.",
                "Validate the prediction heads for the selected model and task.",
            ),
        },
    },
    "resonance_yarn": {
        "profile": "resonance_yarn",
        "name": "Resonance YaRN",
        "source_url": "https://github.com/sheryc/resonance_rope",
        "marker_groups": (
            ("README.md",),
            ("requirements.txt", "environment.yml", "setup.py", "pyproject.toml"),
            ("src", "resonance_rope", "llama", "scripts"),
        ),
        "summary": "RoPE training/fine-tuning fork; it is not a drop-in serving-runtime plugin.",
        "setup": {
            "mode": "backend_work_required",
            "automatic": False,
            "requirements": (
                "A maintained native serving implementation for the selected backend",
                "Model training or fine-tuning artifacts compatible with the RoPE method",
            ),
            "next_steps": (
                "Validate the method in its upstream training workflow.",
                "Implement and test the required backend support before exposing it at runtime.",
            ),
        },
    },
    "jetspec": {
        "profile": "jetspec",
        "name": "JetSpec",
        "source_url": "https://github.com/hao-ai-lab/JetSpec",
        "marker_groups": (("README.md",), ("pyproject.toml",), ("jetspec",)),
        "summary": "Tree-speculative decoding source with a separate optimized CUDA/Triton engine.",
        "setup": {
            "mode": "isolated_environment",
            "automatic": True,
            "requirements": (
                "Linux CUDA and the pinned isolated JetSpec environment",
                "A draft head matching the exact target model",
            ),
            "next_steps": (
                "Create the jetspec environment only after reviewing the plan.",
                "Validate greedy output parity and acceptance length before benchmarking.",
            ),
        },
    },
    "lucebox": {
        "profile": "lucebox",
        "name": "Lucebox",
        "source_url": "https://github.com/Luce-Org/lucebox",
        "marker_groups": (
            ("README.md",),
            ("server/CMakeLists.txt",),
            ("optimizations/pflash", "server/src"),
        ),
        "summary": "Separate native/Docker runtime with model- and hardware-specific inference paths.",
        "setup": {
            "mode": "separate_runtime_fork",
            "automatic": False,
            "requirements": (
                "A supported Linux CUDA or ROCm host, or a matching upstream container image",
                "Supported target and optional drafter/scorer GGUF artifacts",
            ),
            "next_steps": (
                "Build or pull Lucebox independently of other llama.cpp runtimes.",
                "Capability-scan the server and establish a direct-path output baseline.",
            ),
        },
    },
    "proxima": {
        "profile": "proxima",
        "name": "Proxima / STAR-KV",
        "source_url": "https://github.com/Tenosra/Proxima",
        "marker_groups": (("README.md",), ("pyproject.toml",), ("proxima_vllm",)),
        "summary": "Source-only vLLM 0.10.1.1 plugin for calibrated low-rank KV compression.",
        "setup": {
            "mode": "isolated_environment",
            "automatic": True,
            "requirements": (
                "Linux CUDA and the exact pinned vLLM plugin environment",
                "A STAR-KV checkpoint calibrated from the selected base model",
            ),
            "next_steps": (
                "Create the proxima environment only after reviewing the source pin.",
                "Compare quality and capacity against stock vLLM before serving.",
            ),
        },
    },
    "jetlong": {
        "profile": "jetlong",
        "name": "Jet-Long",
        "source_url": "https://github.com/jet-ai-projects/jet-long",
        "marker_groups": (
            ("README.md",),
            ("pyproject.toml",),
            ("jetlm",),
            ("model_configs",),
        ),
        "summary": "Training-free Qwen3 context extension with separately gated fused kernels.",
        "setup": {
            "mode": "isolated_environment",
            "automatic": True,
            "requirements": (
                "Linux CUDA 13, Python 3.12, and the exact reviewed Torch/Transformers stack",
                "A supported Qwen3 base checkpoint and generated method configuration",
            ),
            "next_steps": (
                "Use the non-fused jetlong profile unless an H100 fused stack is separately qualified.",
                "Run base-window, perplexity, RULER, and retrieval validation.",
            ),
        },
    },
    "chunkllama": {
        "profile": "chunkllama",
        "name": "ChunkLlama",
        "source_url": "https://github.com/HKUNLP/ChunkLlama",
        "marker_groups": (
            ("README.md",),
            ("requirements.txt",),
            ("chunkllama_attn_replace.py", "vllm"),
        ),
        "summary": "Older DCA reference used to cross-check Jet-Long's maintained DCA path.",
        "setup": {
            "mode": "research_reference",
            "automatic": False,
            "requirements": (
                "A disposable copy of the upstream legacy environment",
                "Separate review of code versus non-commercial data/weight licensing",
            ),
            "next_steps": (
                "Use the source as a DCA reference rather than a production installer.",
                "Prefer Jet-Long's DCA implementation for the reviewed Qwen3 workflow.",
            ),
        },
    },
    "rabitqcache": {
        "profile": "rabitqcache",
        "name": "RaBitQCache",
        "source_url": "https://github.com/Sakuraaa0/RaBitQCache",
        "marker_groups": (("README.md",), ("setup.py",), ("rabitqcache",)),
        "summary": "Unlicensed research source for adaptive binary-quantized KV retrieval.",
        "setup": {
            "mode": "license_blocked",
            "automatic": False,
            "requirements": ("A repository software license supplied by upstream",),
            "next_steps": (
                "Do not install, redistribute, or port source code while licensing is absent.",
                "Retain only catalog metadata and public research references.",
            ),
        },
    },
    "scope_pe": {
        "profile": "scope_pe",
        "name": "ScoPE",
        "source_url": "https://github.com/oncemoe/ScoPE",
        "marker_groups": (("README.md",), ("pyproject.toml",), ("torchtitan",)),
        "summary": "TorchTitan positional-encoding training system requiring compatible trained weights.",
        "setup": {
            "mode": "artifact_required",
            "automatic": False,
            "requirements": (
                "A model trained with ScoPE-compatible architecture and per-head masks",
                "A recent FlexAttention/context-parallel Torch stack",
            ),
            "next_steps": (
                "Use upstream's training and evaluation workflow in a research environment.",
                "Do not expose ScoPE as a retrofit option for ordinary checkpoints.",
            ),
        },
    },
    "duoattention": {
        "profile": "duoattention",
        "name": "DuoAttention",
        "source_url": "https://github.com/mit-han-lab/duo-attention",
        "marker_groups": (("README.md",), ("setup.py",), ("duo_attn",), ("attn_patterns",)),
        "summary": "Research per-head retrieval/streaming policy with model-specific attention patterns.",
        "setup": {
            "mode": "guided_native_build",
            "automatic": False,
            "requirements": (
                "Linux CUDA and the documented Torch, FlashAttention, FlashInfer, and block-sparse stack",
                "A retrieval-head pattern calibrated for the exact model",
            ),
            "next_steps": (
                "Reproduce the pinned upstream environment on disposable target hardware.",
                "Validate native kernels and long-context retrieval quality before integration.",
            ),
        },
    },
    "icecache": {
        "profile": "icecache",
        "name": "IceCache",
        "source_url": "https://github.com/yuzhenmao/IceCache",
        "marker_groups": (
            ("README.md",),
            ("IceCache/requirements.txt",),
            ("IceCache/source/setup.py",),
        ),
        "summary": "CPU KV offload research runtime requiring a second native M-DCI source project.",
        "setup": {
            "mode": "guided_multi_source_build",
            "automatic": False,
            "requirements": (
                "Linux CUDA, GCC, CMake, OpenBLAS, OpenMP, and matching M-DCI source",
                "High CPU parallelism; upstream recommends at least 64 threads",
            ),
            "next_steps": (
                "Inspect IceCache and M-DCI independently before any build.",
                "Benchmark TTFT and retrieval quality at the selected CPU/page budget.",
            ),
        },
    },
    "pflash_llamacpp": {
        "profile": "pflash_llamacpp",
        "name": "llama.cpp PFlash/KVFlash fork",
        "source_url": "https://github.com/HawgAuto/llama.cpp-dflash-pflash-kvflash",
        "marker_groups": (
            ("README.md",),
            ("CMakeLists.txt",),
            ("tools/server/qwen36-smart-router.py",),
        ),
        "summary": "Separate experimental llama.cpp runtime with PFlash, KVFlash, DFlash, and routing.",
        "setup": {
            "mode": "separate_runtime_fork",
            "automatic": False,
            "requirements": (
                "A separately built capability-scanned server binary",
                "Exact target, drafter, and scorer GGUFs for the selected path",
            ),
            "next_steps": (
                "Establish a direct server baseline before enabling PFlash or KVFlash.",
                "Use direct verification or bypass for exactness-sensitive requests.",
            ),
        },
    },
    "domvox_triattention": {
        "profile": "domvox_triattention",
        "name": "domvox TriAttention",
        "source_url": "https://github.com/domvox/triattention-ggml",
        "marker_groups": (
            ("triattention_calibrate.py",),
            ("triattention_common.py",),
            ("TRIA_FORMAT.md",),
        ),
        "summary": "Experimental TRIA v2 calibration source; conversion to Godzilla v1 is explicitly lossy.",
        "setup": {
            "mode": "supported_calibration_adapter",
            "automatic": False,
            "requirements": (
                "The managed TriAttention Python dependency environment",
                "The matching Hugging Face checkpoint and explicit lossy-conversion acknowledgement",
            ),
            "next_steps": (
                "Select the detected calibrator in Godzilla Source Setup.",
                "Check the preparation plan, then run and validate the converted artifact.",
            ),
        },
    },
}


def _marker_group_present(root: Path, group: tuple[str, ...]) -> bool:
    return any((root / marker).exists() for marker in group)


def inspect_addon_source(profile_id: str, path: str | Path) -> dict[str, object]:
    """Inspect a reviewed add-on checkout without importing or executing it.

    Blocked profiles are deliberately accepted here for source selection and
    documentation.  They remain blocked for environment creation.
    """
    normalized = str(profile_id).strip().lower()
    if normalized == "cuda_weight_share":
        from ..integration import inspect_cuda_weight_share_source

        return inspect_cuda_weight_share_source(path)
    spec = _SOURCE_ADDON_SPECS.get(normalized)
    if spec is None:
        raise ValueError(f"Unknown informational add-on profile: {profile_id}")
    raw_path = str(path).strip()
    if not raw_path:
        return {
            "profile": normalized,
            "name": spec["name"],
            "path": "",
            "source_url": spec["source_url"],
            "status": "not_selected",
            "valid": False,
            "marker_groups": {},
            "issues": ["Select a local source folder to inspect."],
            "summary": spec["summary"],
            "setup": spec["setup"],
        }
    resolved = Path(raw_path).expanduser().resolve()
    marker_groups = {
        " or ".join(group): _marker_group_present(resolved, group)
        for group in spec["marker_groups"]
    }
    issues: list[str] = []
    if not resolved.is_dir():
        issues.append(f"Source folder is not a directory: {resolved}")
    else:
        issues.extend(
            f"Missing reviewed source marker (one of: {', '.join(group)})"
            for group, present in zip(spec["marker_groups"], marker_groups.values())
            if not present
        )
    valid = resolved.is_dir() and not issues
    calibrator = None
    if valid and normalized == "domvox_triattention":
        from ..calibration import inspect_domvox_triattention_checkout

        checkout = inspect_domvox_triattention_checkout(resolved)
        if not checkout["valid"]:
            issues.extend(str(item) for item in checkout.get("issues", []))
            valid = False
        calibrator = checkout.get("calibrator")
    return {
        "profile": normalized,
        "name": spec["name"],
        "path": str(resolved),
        "source_url": spec["source_url"],
        "status": "informational_only" if valid else "invalid_source",
        "valid": valid,
        "marker_groups": marker_groups,
        "issues": issues,
        "summary": spec["summary"],
        "setup": spec["setup"],
        "git_remote": _git_remote(resolved) if resolved.is_dir() else None,
        "calibrator": calibrator,
    }


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
    if (path / "scripts" / "calibrate.py").is_file() and "triattention" in directories:
        return "triattention"
    if "fastdms" in directories and "pyproject.toml" in files:
        return "fastdms"
    if "lmcache" in directories and {"pyproject.toml", "setup.py"}.issubset(files):
        return "lmcache"
    if "minference" in directories and "setup.py" in files and "csrc" in directories:
        return "minference"
    if "sageattention" in directories and "setup.py" in files and "csrc" in directories:
        return "sageattention"
    llama_markers = "CMakeLists.txt" in files and "ggml" in directories
    if (
        llama_markers
        and (path / "tools" / "server" / "qwen36-smart-router.py").is_file()
        and (path / "README.md").is_file()
    ):
        return "pflash_llamacpp"
    if (
        "GODZILLA_KING.md" in files
        or (path / "scripts" / "godzilla-paths.ps1").is_file()
        or ("godzilla" in name and llama_markers)
    ):
        return "godzilla"
    if (
        (path / "docs" / "gigatoken.md").is_file()
        and (path / "cmake" / "gigatoken.cmake").is_file()
        and (path / "src" / "llama-gigatoken.cpp").is_file()
        and (path / "patches" / "gigatoken-llama-cpp.patch").is_file()
    ):
        return "gigatoken_llamacpp"
    if llama_markers:
        return "llamacpp"
    if {
        "triattention_calibrate.py",
        "triattention_common.py",
        "TRIA_FORMAT.md",
    }.issubset(files):
        return "domvox_triattention"
    if "cuda-llm-weight-share.c" in files and {"README.md", "LICENSE"}.issubset(files):
        return "cuda_weight_share"
    # Non-profile source projects are recognized conservatively. A direct
    # source-folder inspection is available when a checkout is renamed.
    source_name = name.replace("-", "_")
    source_aliases = {
        "maru": "maru",
        "speculative_prefill": "speculative_prefill",
        "speculativeprefill": "speculative_prefill",
        "rocketkv": "rocketkv",
        "lexico": "lexico",
        "adadecode": "adadecode",
        "resonance_rope": "resonance_yarn",
        "resonance_yarn": "resonance_yarn",
        "gigatoken_llama.cpp": "gigatoken_llamacpp",
        "gigatoken_llamacpp": "gigatoken_llamacpp",
        "cuda_llm_weight_share": "cuda_weight_share",
        "cuda_weight_share": "cuda_weight_share",
        "jetspec": "jetspec",
        "lucebox": "lucebox",
        "lucebox_hub": "lucebox",
        "proxima": "proxima",
        "jet_long": "jetlong",
        "jetlong": "jetlong",
        "chunkllama": "chunkllama",
        "rabitqcache": "rabitqcache",
        "scope": "scope_pe",
        "scope_pe": "scope_pe",
        "duo_attention": "duoattention",
        "duoattention": "duoattention",
        "icecache": "icecache",
        "llama.cpp_dflash_pflash_kvflash": "pflash_llamacpp",
        "pflash_llamacpp": "pflash_llamacpp",
    }
    source_kind = source_aliases.get(source_name)
    if source_kind is not None and all(
        _marker_group_present(path, group)
        for group in _SOURCE_ADDON_SPECS[source_kind]["marker_groups"]
    ):
        return source_kind
    return None


def scan_addon_roots(
    roots: Iterable[str | Path],
    *,
    max_depth: int = 3,
    limit: int = 200,
) -> dict[str, object]:
    """Scan only configured roots for recognized add-on source directories."""
    if not 0 <= max_depth <= 5:
        raise ValueError("max_depth must be between 0 and 5")
    results: list[dict[str, object]] = []
    errors: list[str] = []
    warnings: list[str] = []
    resolved_roots: list[str] = []
    seen: set[Path] = set()
    scanned_directories = 0
    configured_roots = list(roots)
    if not configured_roots:
        warnings.append("No add-on roots are configured. Add a checkout folder or its parent directory.")
    for raw_root in configured_roots:
        try:
            root = _directory(raw_root, "Add-on root")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        resolved_roots.append(str(root))
        try:
            for directory, child_directories, filenames in _walk(root, max_depth=max_depth):
                scanned_directories += 1
                kind = _classify_addon(directory, set(filenames), set(child_directories))
                if kind is None or directory in seen:
                    continue
                # A recognized checkout is the useful unit. Do not descend into
                # its package/source subdirectories and report them as duplicate
                # add-ons merely because they repeat the project name.
                child_directories[:] = []
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
                if kind == "triattention":
                    from ..calibration import inspect_official_triattention_checkout

                    item["source"] = inspect_official_triattention_checkout(directory)
                if kind == "domvox_triattention":
                    from ..calibration import inspect_domvox_triattention_checkout

                    item["source"] = inspect_domvox_triattention_checkout(directory)
                    item["source_profile"] = kind
                if kind == "cuda_weight_share":
                    from ..integration import inspect_cuda_weight_share_source

                    item["source"] = inspect_cuda_weight_share_source(directory)
                    item["source_profile"] = kind
                elif kind in _SOURCE_ADDON_SPECS:
                    item["source"] = inspect_addon_source(kind, directory)
                    item["source_profile"] = kind
                if kind in {
                    "flashattention",
                    "fastdms",
                    "lmcache",
                    "minference",
                    "sageattention",
                    "triattention",
                    "jetspec",
                    "proxima",
                    "jetlong",
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
    if resolved_roots and not results and not errors:
        warnings.append(
            f"Scanned {scanned_directories} directories to depth {max_depth}, but found no "
            "recognized checkout. Select the repository folder itself or a parent directory; "
            "renamed Godzilla checkouts must include scripts/godzilla-paths.ps1."
        )
    if len(results) >= limit:
        warnings.append(f"Add-on results were limited to {limit} entries.")
    results.sort(key=lambda item: (str(item["kind"]), str(item["path"]).lower()))
    return {
        "roots": resolved_roots,
        "addons": results,
        "count": len(results),
        "errors": errors,
        "warnings": warnings,
        "scanned_directories": scanned_directories,
        "max_depth": max_depth,
        "truncated": len(results) >= limit,
    }


def scan_environment_profiles(
    root: str | Path,
    *,
    cuda_toolkit: str | Path | None = None,
    local_source_profile: str | None = None,
    local_source: str | Path | None = None,
    manual_dependency_override: bool = False,
) -> dict[str, object]:
    """Report dependency readiness, validating materialized environments by import."""
    from ..optimizations.environments import (
        BUILTIN_ENVIRONMENT_PROFILES,
        check_environment,
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
        issues = [issue.to_dict() for issue in plan.issues]
        validation: dict[str, object] | None = None
        if not profile.installable:
            status = "blocked"
        elif interpreter.is_file():
            try:
                validation = dict(check_environment(plan))
            except RuntimeError as exc:
                if manual_dependency_override:
                    status = "manual"
                    issues.append(
                        {
                            "severity": "warning",
                            "code": "manual_dependency_override",
                            "message": (
                                "Automatic dependency validation failed, but the manual override "
                                f"is active: {exc}. Runtime failures remain possible."
                            ),
                        }
                    )
                else:
                    status = "broken"
                    issues.append(
                        {
                            "severity": "error",
                            "code": "environment_validation_failed",
                            "message": (
                                f"The environment exists but its dependency check failed: {exc}. "
                                "Repair it or use the manual override only if the scan is wrong."
                            ),
                        }
                    )
            else:
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
                "validation": validation,
                "issues": issues,
            }
        )
    return {
        "root": str(Path(root).expanduser().resolve()),
        "local_source_profile": selected_profile,
        "local_source": str(Path(local_source).expanduser().resolve()) if local_source else None,
        "manual_dependency_override": manual_dependency_override,
        "context": {
            "os": context.os,
            "compute": context.compute,
            "os_release_id": context.os_release_id,
            "os_release_version": context.os_release_version,
            "cuda_toolkit_version": context.cuda_toolkit_version,
            "cuda_toolkit_root": context.cuda_toolkit_root,
        },
        "profiles": profiles,
    }
