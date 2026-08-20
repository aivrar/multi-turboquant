# SPDX-License-Identifier: MIT
"""Pinned PFlash/KVFlash composition overlay for the reviewed Godzilla tree.

The overlay is intentionally conservative.  PFlash is a request-gated,
deterministic prompt thinning policy.  KVFlash is a server-level LRU residency
budget for complete idle slots; it does not claim the unfinished arbitrary-page
restore path from the research fork.  Neither feature changes KVarN's cache
representation or TriAttention's intra-sequence eviction policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .godzilla_gigatoken import (
    GODZILLA_COMPAT_COMMIT,
    GODZILLA_URL,
    RuntimeIssue,
    _checkout_godzilla_source,
    _git_revision,
    _replace_once,
    _resolve_cuda_compiler,
    _run_checked,
    get_godzilla_source_profile,
)


COMPOSITION_SCHEMA = 1
COMPOSITION_PROFILE = "godzilla-09214b160-pflash-kvflash-v1"
MANIFEST_NAME = ".mtq-godzilla-composition.json"
_RUNTIME_FILES = (
    "common/common.h",
    "common/arg.cpp",
    "tools/server/server-task.h",
    "tools/server/server-task.cpp",
    "tools/server/server-context.cpp",
)
_EXPECTED_RUNTIME_SHA256 = {
    "common/common.h": "1a7ab495602bd662a58d96afe90187c2d81ac5815c8fc818456226093d2b2695",
    "common/arg.cpp": "9aa5ee48029fdece1e40dac129065bac2ae7d460f432dc6e9161ad36e0f72886",
    "tools/server/server-task.h": "c2c65b8b65740eab74df7d16c6f0379a61c14184cab029984ea80db5f85bf6f2",
    "tools/server/server-task.cpp": "334c6f8e5963bf400eeadba81a53caadf006b5096f09a963d26ccebd235f18fc",
    "tools/server/server-context.cpp": "46009558c7a3a1c29b056cb1c287d61408abc429207fd9fe55bfb8a00a5a23ce",
}


@dataclass(frozen=True)
class GodzillaComposition:
    pflash: bool = True
    kvflash: bool = True
    dflash: bool = True
    ddtree: bool = True
    triattention: bool = False
    kvarn: bool = False
    spec_la: bool = False


def validate_godzilla_composition(config: GodzillaComposition) -> tuple[RuntimeIssue, ...]:
    """Return the fail-closed compatibility contract for the pinned overlay."""
    issues: list[RuntimeIssue] = []
    if config.triattention and config.kvarn:
        issues.append(RuntimeIssue(
            "error",
            "triattention_kvarn_conflict",
            "Godzilla 09214b160 rejects TriAttention with KVarN; KVarN-aware pruning is not implemented.",
        ))
    if config.spec_la:
        issues.append(RuntimeIssue(
            "error",
            "specla_not_available",
            "SpecLA is a specialized linear-attention runtime, not a stackable add-on for this Godzilla tree.",
        ))
    if config.pflash and config.triattention:
        issues.append(RuntimeIssue(
            "warning",
            "two_stage_token_selection",
            "PFlash changes the prompt before TriAttention scores retained KV entries; qualify quality for the chosen model and workload.",
        ))
    if config.kvflash:
        issues.append(RuntimeIssue(
            "info",
            "kvflash_slot_tier",
            "KVFlash is limited to LRU residency of complete idle slots; arbitrary KV pages, prefill skipping, and disk restore are not claimed.",
        ))
    return tuple(issues)


@dataclass(frozen=True)
class GodzillaCompositionPlan:
    action: str
    target: Path
    build_dir: Path
    backend: str
    max_jobs: int
    generator: str | None
    cuda_compiler: Path | None
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
            "generator": self.generator,
            "cuda_compiler": str(self.cuda_compiler) if self.cuda_compiler else None,
            "commands": [list(command) for command in self.commands],
            "issues": [issue.to_dict() for issue in self.issues],
            "ready": self.ready,
            "profile": COMPOSITION_PROFILE,
            "godzilla_commit": GODZILLA_COMPAT_COMMIT,
        }


def _supported_platform() -> bool:
    return platform.system().lower() in {"windows", "linux"} and platform.machine().lower() in {
        "amd64", "x86_64"
    }


def plan_godzilla_composition(
    target: str | Path,
    *,
    action: str = "prepare",
    backend: str = "cpu",
    max_jobs: int = 2,
    generator: str | None = None,
    cuda_toolkit: str | Path | None = None,
) -> GodzillaCompositionPlan:
    target_path = Path(target).expanduser().resolve()
    normalized_action = action.strip().lower()
    normalized_backend = backend.strip().lower()
    build_dir = target_path / f"build-mtq-composition-{normalized_backend}"
    cuda_compiler = _resolve_cuda_compiler(cuda_toolkit) if normalized_backend == "cuda" else None
    issues = list(validate_godzilla_composition(GodzillaComposition()))
    commands: list[tuple[str, ...]] = []

    if normalized_action not in {"prepare", "build", "all", "verify"}:
        issues.append(RuntimeIssue("error", "invalid_action", "Action must be prepare, build, all, or verify."))
    if normalized_backend not in {"cpu", "cuda"}:
        issues.append(RuntimeIssue("error", "invalid_backend", "Backend must be cpu or cuda."))
    if max_jobs < 1:
        issues.append(RuntimeIssue("error", "invalid_max_jobs", "max_jobs must be at least 1."))
    if not _supported_platform():
        issues.append(RuntimeIssue("error", "unsupported_platform", "Only Windows/Linux x86-64 is qualified."))
    if normalized_backend == "cuda" and cuda_compiler is None:
        issues.append(RuntimeIssue("error", "cuda_compiler_missing", "CUDA builds require nvcc on PATH or --cuda-toolkit."))

    if normalized_action in {"prepare", "all"}:
        if target_path.exists():
            issues.append(RuntimeIssue("error", "target_exists", "Preparation requires a new destination."))
        if shutil.which("git") is None:
            issues.append(RuntimeIssue("error", "git_missing", "git is required."))
        commands.extend((
            ("git", "init", str(target_path)),
            ("git", "fetch", "--depth", "1", "origin", GODZILLA_COMPAT_COMMIT),
            ("git", "checkout", "--detach", "FETCH_HEAD"),
        ))
    if normalized_action in {"build", "verify"}:
        inspection = inspect_godzilla_composition(target_path)
        if not inspection["valid"]:
            issues.append(RuntimeIssue("error", "invalid_source", "; ".join(inspection["issues"])))
    if normalized_action in {"build", "all"}:
        if shutil.which("cmake") is None:
            issues.append(RuntimeIssue("error", "cmake_missing", "CMake is required."))
        configure = [
            "cmake", "-S", str(target_path), "-B", str(build_dir),
            "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_UI=OFF",
            "-DLLAMA_CURL=OFF", f"-DGGML_CUDA={'ON' if normalized_backend == 'cuda' else 'OFF'}",
            "-DGGML_NATIVE=OFF", "-DGGML_CCACHE=OFF",
        ]
        if generator:
            configure[5:5] = ["-G", generator]
        if cuda_compiler is not None:
            configure.append(f"-DCMAKE_CUDA_COMPILER={cuda_compiler}")
        commands.extend((tuple(configure), (
            "cmake", "--build", str(build_dir), "--config", "Release",
            "--target", "llama-server", "-j", str(max_jobs),
        )))
    return GodzillaCompositionPlan(
        normalized_action, target_path, build_dir, normalized_backend, max_jobs, generator, cuda_compiler,
        tuple(commands), tuple(issues),
    )


def _apply_overlay(root: Path) -> None:
    header_anchor = "    int32_t cache_ram_mib       = 8192;  // -1 = no limit, 0 - disable, 1 = 1 MiB, etc.\n"
    _replace_once(root / "common/common.h", header_anchor, header_anchor + """

    // Multi-TurboQuant composition overlay (off by default)
    bool    pflash_enabled       = false; // allow request-gated prompt thinning
    float   pflash_keep_ratio    = 0.75f; // fraction retained when a request sets pflash=true
    int32_t pflash_min_tokens    = 4096;  // do not thin shorter prompts
    int32_t pflash_prefix_tokens = 256;   // protected leading tokens
    int32_t pflash_suffix_tokens = 128;   // protected trailing tokens
    int32_t kvflash_pages        = 0;     // complete-idle-slot residency budget in pages
    int32_t kvflash_page_tokens  = 256;   // accounting granularity; not arbitrary-page restore
""")

    decl_anchor = "static void common_params_speculative_normalize(common_params & params);\n"
    _replace_once(root / "common/arg.cpp", decl_anchor, decl_anchor + "static void common_params_composition_normalize(common_params & params);\n")
    call_anchor = "    common_params_speculative_normalize(params);\n"
    _replace_once(root / "common/arg.cpp", call_anchor, call_anchor + "    common_params_composition_normalize(params);\n")

    arg_anchor = """    add_opt(common_arg(
        {"--cache-idle-slots"},
        {"--no-cache-idle-slots"},
        "save and clear idle slots on new task (default: enabled, requires unified KV and cache-ram)",
        [](common_params & params, bool value) {
            params.cache_idle_slots = value;
        }
    ).set_env("LLAMA_ARG_CACHE_IDLE_SLOTS").set_examples({LLAMA_EXAMPLE_SERVER}));
"""
    arg_block = """
    add_opt(common_arg(
        {"--pflash"},
        "enable request-gated PFlash prompt thinning (requests must also set pflash=true)",
        [](common_params & params) { params.pflash_enabled = true; }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--pflash-keep-ratio"}, "RATIO",
        "fraction of prompt tokens retained by PFlash (default: 0.75)",
        [](common_params & params, const std::string & value) { params.pflash_keep_ratio = std::stof(value); }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--pflash-min-tokens"}, "N",
        "minimum prompt length eligible for PFlash (default: 4096)",
        [](common_params & params, int value) { params.pflash_min_tokens = value; }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--kvflash-pages"}, "N",
        "retain up to N pages of complete idle-slot KV state using LRU (0 disables)",
        [](common_params & params, int value) { params.kvflash_pages = value; }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--kvflash-page-tokens"}, "N",
        "KVFlash accounting page size in tokens (default: 256)",
        [](common_params & params, int value) { params.kvflash_page_tokens = value; }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
"""
    _replace_once(root / "common/arg.cpp", arg_anchor, arg_anchor + arg_block)

    normalize_anchor = "static void common_params_speculative_normalize(common_params & params) {\n"
    normalize_block = """static void common_params_composition_normalize(common_params & params) {
    if (params.pflash_keep_ratio <= 0.0f || params.pflash_keep_ratio > 1.0f) {
        throw std::invalid_argument("--pflash-keep-ratio must be in (0, 1]");
    }
    if (params.pflash_min_tokens < 1 || params.pflash_prefix_tokens < 0 || params.pflash_suffix_tokens < 0) {
        throw std::invalid_argument("PFlash token limits must be non-negative and min-tokens must be positive");
    }
    if (params.kvflash_pages < 0 || params.kvflash_page_tokens < 1) {
        throw std::invalid_argument("--kvflash-pages must be >= 0 and --kvflash-page-tokens must be >= 1");
    }
    if (params.kvflash_pages > 0 && params.cache_idle_slots) {
        LOG_WRN("warning: KVFlash residency supersedes --cache-idle-slots; retaining bounded idle slots instead\\n");
        params.cache_idle_slots = false;
    }
}

"""
    _replace_once(root / "common/arg.cpp", normalize_anchor, normalize_block + normalize_anchor)

    task_header_anchor = "    bool return_progress = false;\n"
    _replace_once(root / "tools/server/server-task.h", task_header_anchor, task_header_anchor + "    bool pflash         = false; // explicit per-request opt-in\n")
    task_parse_anchor = "    params.post_sampling_probs         = json_value(data, \"post_sampling_probs\", defaults.post_sampling_probs);\n"
    _replace_once(root / "tools/server/server-task.cpp", task_parse_anchor, task_parse_anchor + "    params.pflash                     = json_value(data, \"pflash\", false);\n")

    process_anchor = "    void process_single_task(server_task && task) {\n"
    helpers = r'''    bool maybe_apply_pflash(server_task & task) {
        if (!params_base.pflash_enabled || !task.params.pflash || task.is_parent() ||
                (task.type != SERVER_TASK_TYPE_COMPLETION && task.type != SERVER_TASK_TYPE_INFILL) ||
                (task.params.res_type != TASK_RESPONSE_TYPE_NONE && task.params.res_type != TASK_RESPONSE_TYPE_OAI_CMPL) ||
                task.tokens.has_media() || task.n_tokens() < params_base.pflash_min_tokens) {
            return false;
        }
        const size_t before = task.tokens.size();
        const size_t prefix = std::min(before, (size_t) params_base.pflash_prefix_tokens);
        const size_t suffix = std::min(before - prefix, (size_t) params_base.pflash_suffix_tokens);
        const size_t target = std::max(prefix + suffix,
                (size_t) std::ceil(before * params_base.pflash_keep_ratio));
        if (target >= before) {
            return false;
        }
        const size_t middle_source = before - prefix - suffix;
        const size_t middle_keep = target - prefix - suffix;
        llama_tokens selected;
        selected.reserve(target);
        for (size_t i = 0; i < prefix; ++i) {
            selected.push_back(task.tokens[i]);
        }
        for (size_t i = 0; i < middle_keep; ++i) {
            selected.push_back(task.tokens[prefix + (i * middle_source) / middle_keep]);
        }
        for (size_t i = before - suffix; i < before; ++i) {
            selected.push_back(task.tokens[i]);
        }
        task.tokens = server_tokens(selected, false);
        task.params.n_keep = std::min(task.params.n_keep, (int32_t) selected.size());
        SRV_INF("PFlash request %d: retained %zu/%zu tokens (explicit opt-in)\n",
                task.id, selected.size(), before);
        return true;
    }

    void enforce_kvflash_residency() {
        if (params_base.kvflash_pages <= 0) {
            return;
        }
        const int64_t budget = (int64_t) params_base.kvflash_pages * params_base.kvflash_page_tokens;
        int64_t resident = 0;
        for (const auto & slot : slots) {
            if (!slot.is_processing()) {
                resident += slot.prompt.n_tokens();
            }
        }
        while (resident > budget) {
            server_slot * victim = nullptr;
            for (auto & slot : slots) {
                if (!slot.is_processing() && slot.prompt.n_tokens() > 0 &&
                        (!victim || slot.t_last_used < victim->t_last_used)) {
                    victim = &slot;
                }
            }
            if (!victim) {
                break;
            }
            const int64_t removed = victim->prompt.n_tokens();
            slot_clear_hybrid_safe(*victim, "KVFlash LRU residency eviction");
            resident -= removed;
            SRV_INF("KVFlash LRU eviction: slot=%d removed=%" PRId64 " resident=%" PRId64 "/%" PRId64 " tokens\n",
                    victim->id, removed, resident, budget);
        }
    }

'''
    _replace_once(root / "tools/server/server-context.cpp", process_anchor, helpers + process_anchor)

    tokenize_anchor = """                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_slot = task.id_slot;
"""
    _replace_once(root / "tools/server/server-context.cpp", tokenize_anchor, tokenize_anchor.replace(
        "\n                    const int id_slot", "\n                    maybe_apply_pflash(task);\n\n                    const int id_slot"
    ))

    idle_anchor = """                    if (params_base.cache_idle_slots) {
                        for (auto & s : slots) {
                            if (!s.is_processing()) {
                                if (params_base.cache_ram_mib != 0 && prompt_cache) {
                                    slot_save_and_clear(s);
                                } else {
                                    slot_clear_hybrid_safe(s, "before new task");
                                }
                            }
                        }
                    }
"""
    idle_new = """                    if (params_base.kvflash_pages > 0) {
                        enforce_kvflash_residency();
                    } else if (params_base.cache_idle_slots) {
                        for (auto & s : slots) {
                            if (!s.is_processing()) {
                                if (params_base.cache_ram_mib != 0 && prompt_cache) {
                                    slot_save_and_clear(s);
                                } else {
                                    slot_clear_hybrid_safe(s, "before new task");
                                }
                            }
                        }
                    }
"""
    _replace_once(root / "tools/server/server-context.cpp", idle_anchor, idle_new)

    props_anchor = "            { \"cors_proxy_enabled\",          params.ui_mcp_proxy || params.webui_mcp_proxy },\n"
    props_block = """            { "mtq_composition",               json {
                {"profile", "godzilla-09214b160-pflash-kvflash-v1"},
                {"pflash", json {
                    {"enabled", params.pflash_enabled},
                    {"request_gated", true},
                    {"telemetry", "server_log"},
                }},
                {"kvflash", json {
                    {"tier", "complete_idle_slots"},
                    {"pages", params.kvflash_pages},
                    {"page_tokens", params.kvflash_page_tokens},
                    {"telemetry", "server_log"},
                    {"arbitrary_page_restore", false},
                    {"prefill_skip", false},
                }},
            } },
"""
    _replace_once(root / "tools/server/server-context.cpp", props_anchor, props_anchor + props_block)


def _file_hashes(root: Path) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _RUNTIME_FILES}


def _git_source_state(root: Path) -> tuple[set[str], set[str]] | None:
    try:
        changed_result = _run_checked(
            ("git", "diff", "--name-only", "HEAD", "--"), cwd=root, capture=True,
        )
        untracked_result = _run_checked(
            ("git", "ls-files", "--others", "--exclude-standard"), cwd=root, capture=True,
        )
    except (OSError, RuntimeError):
        return None
    changed = set(changed_result.stdout.decode("utf-8", errors="replace").splitlines())
    untracked = set(untracked_result.stdout.decode("utf-8", errors="replace").splitlines())
    return changed, untracked


def _remove_tree(path: Path) -> None:
    def retry(function, value, _error):
        os.chmod(value, stat.S_IWRITE | stat.S_IREAD)
        function(value)
    shutil.rmtree(path, onerror=retry)


def prepare_godzilla_composition(plan: GodzillaCompositionPlan, *, confirmed: bool = False) -> dict[str, object]:
    if plan.action not in {"prepare", "all"} or not plan.ready:
        raise RuntimeError("A ready prepare/all plan is required")
    if not confirmed:
        raise RuntimeError("Preparation was not confirmed")
    plan.target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{plan.target.name}.mtq-", dir=plan.target.parent)).resolve()
    try:
        _checkout_godzilla_source(work, get_godzilla_source_profile("09214b160"))
        if _git_revision(work) != GODZILLA_COMPAT_COMMIT:
            raise RuntimeError("Pinned Godzilla revision did not match after checkout")
        _apply_overlay(work)
        runtime_hashes = _file_hashes(work)
        if runtime_hashes != _EXPECTED_RUNTIME_SHA256:
            raise RuntimeError("Generated overlay hashes do not match the reviewed exact-commit profile")
        manifest = {
            "schema": COMPOSITION_SCHEMA,
            "profile": COMPOSITION_PROFILE,
            "godzilla": {"url": GODZILLA_URL, "commit": GODZILLA_COMPAT_COMMIT},
            "capabilities": {
                "pflash": "request-gated deterministic prompt thinning",
                "kvflash": "complete idle-slot LRU residency",
                "arbitrary_kv_page_restore": False,
                "prefill_skip": False,
            },
            "runtime_files_sha256": runtime_hashes,
        }
        (work / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        os.replace(work, plan.target)
    except BaseException:
        if work.exists():
            _remove_tree(work)
        raise
    return inspect_godzilla_composition(plan.target)


def inspect_godzilla_composition(target: str | Path) -> dict[str, object]:
    root = Path(target).expanduser().resolve()
    issues: list[str] = []
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return {"valid": False, "target": str(root), "issues": [f"missing {MANIFEST_NAME}"]}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"valid": False, "target": str(root), "issues": [f"invalid manifest: {exc}"]}
    if manifest.get("schema") != COMPOSITION_SCHEMA or manifest.get("profile") != COMPOSITION_PROFILE:
        issues.append("manifest schema/profile mismatch")
    if _git_revision(root) != GODZILLA_COMPAT_COMMIT:
        issues.append("Godzilla revision mismatch")
    expected = manifest.get("runtime_files_sha256", {})
    if not isinstance(expected, dict):
        issues.append("manifest runtime hash table is invalid")
        expected = {}
    try:
        actual = _file_hashes(root)
    except OSError as exc:
        issues.append(str(exc))
        actual = {}
    for name in _RUNTIME_FILES:
        if expected.get(name) != actual.get(name):
            issues.append(f"runtime file changed: {name}")
        if _EXPECTED_RUNTIME_SHA256.get(name) != actual.get(name):
            issues.append(f"runtime file does not match reviewed overlay: {name}")
    source_state = _git_source_state(root)
    if source_state is None:
        issues.append("unable to inspect the prepared Git source state")
    else:
        changed, untracked = source_state
        if changed != set(_RUNTIME_FILES):
            issues.append("tracked source changes do not match the reviewed overlay allowlist")
        unexpected = untracked - {MANIFEST_NAME}
        if unexpected:
            issues.append("unexpected untracked source files: " + ", ".join(sorted(unexpected)))
    return {"valid": not issues, "target": str(root), "profile": COMPOSITION_PROFILE, "issues": issues}


def build_godzilla_composition(plan: GodzillaCompositionPlan, *, confirmed: bool = False) -> dict[str, object]:
    if plan.action not in {"build", "all"} or not plan.ready:
        raise RuntimeError("A ready build/all plan is required")
    if not confirmed:
        raise RuntimeError("Build was not confirmed")
    for command in plan.commands[-2:]:
        _run_checked(command)
    return verify_godzilla_composition(plan)


def _find_server(build_dir: Path) -> Path | None:
    candidates = sorted(build_dir.glob("**/llama-server.exe")) + sorted(build_dir.glob("**/llama-server"))
    return next((path for path in candidates if path.is_file()), None)


def verify_godzilla_composition(plan: GodzillaCompositionPlan) -> dict[str, object]:
    inspection = inspect_godzilla_composition(plan.target)
    server = _find_server(plan.build_dir)
    issues = list(inspection["issues"])
    help_text = ""
    if server is None:
        issues.append(f"llama-server was not found under {plan.build_dir}")
    else:
        result = _run_checked((str(server), "--help"), capture=True)
        help_text = ((result.stdout or b"") + (result.stderr or b"")).decode("utf-8", errors="replace")
        for flag in ("--pflash", "--kvflash-pages", "--spec-branch-budget", "--triattention-stats"):
            if flag not in help_text:
                issues.append(f"built server help is missing {flag}")
    return {
        "valid": not issues,
        "target": str(plan.target),
        "build_dir": str(plan.build_dir),
        "server": str(server) if server else None,
        "issues": issues,
    }
