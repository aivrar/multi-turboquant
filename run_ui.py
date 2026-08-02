#!/usr/bin/env python3
"""Multi-TurboQuant Web UI — lightweight browser interface for the library.

No WSL, no Linux, no bridge. Just the library with a web face.

Usage:
    python run_ui.py              # opens http://localhost:9092
    python run_ui.py --port 8080  # custom port
"""

import argparse
import http.server
import json
import os
import shlex
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from multi_turboquant import (
    __version__, CacheConfig, CacheMethod,
    get_method, get_preset, list_presets, registered_methods,
    plan_agents,
)
from multi_turboquant.calibration import (
    CALIBRATION_CORPUS_SCHEMA_VERSION,
    generate_calibration_text,
)
from multi_turboquant._paths import lexical_absolute_path, same_lexical_path
from multi_turboquant.hardware import detect_platform, detect_gpus
from multi_turboquant.optimizations.environments import environment_python, plan_environment
from multi_turboquant.tokenizer_backends import scan_gigatoken_interpreters
from multi_turboquant.compatibility import check_config
from multi_turboquant.config import CALIBRATION_REQUIRED, METHOD_BITS, METHOD_FAMILIES
from multi_turboquant.integration import (
    CudaWeightShareConfig,
    LlamaCppContextExtensionConfig,
    inspect_godzilla_checkout,
    plan_godzilla_triattention,
    scan_llamacpp_binary,
)
from multi_turboquant.ui import (
    EnvironmentJobManager,
    GodzillaCalibrationJobManager,
    ManagedProcess,
    inspect_addon_source,
    UISettingsStore,
    inspect_flashattention_source,
    scan_addon_roots,
    scan_environment_profiles,
    scan_models,
)


BACKEND_ONLY_METHODS = (
    CacheMethod.KVARN2,
    CacheMethod.KVARN3,
    CacheMethod.KVARN4,
    CacheMethod.KVARN5,
    CacheMethod.KVARN6,
    CacheMethod.KVARN8,
)

SETTINGS_STORE = UISettingsStore()
MODEL_PROCESS = ManagedProcess()
ENVIRONMENT_JOBS = EnvironmentJobManager()
GODZILLA_JOBS = GodzillaCalibrationJobManager()
UI_MUTATIONS_ENABLED = True


# ─── API Handlers ───────────────────────────────────────────────────────────────

def api_status():
    plat = detect_platform()
    gpus = [{"name": g.name, "vram_mb": g.vram_total_mb, "vram_used_mb": g.vram_used_mb,
             "vendor": g.vendor, "compute": g.compute} for g in plat.gpus]
    return {
        "version": __version__,
        "platform": plat.os,
        "arch": plat.arch,
        "gpus": gpus,
        "gpu_count": plat.gpu_count,
        "total_vram_gb": round(plat.total_vram_gb, 1),
        "available_vram_gb": round(plat.available_vram_gb, 1),
        "system_ram_gb": round(plat.system_memory_gb, 1),
        "available_system_ram_gb": round(plat.available_system_memory_gb, 1),
        "combined_memory_gb": round(plat.combined_memory_gb, 1),
        "unified_memory": plat.unified_memory,
        "cuda": plat.cuda_available,
        "torch_version": torch.__version__,
        "torch_cuda": torch.cuda.is_available(),
        "methods": len(registered_methods()) + len(BACKEND_ONLY_METHODS),
        "presets": len(list_presets()),
    }


def api_methods():
    methods = []
    for m in registered_methods():
        inst = get_method(m)
        info = inst.info()
        methods.append({
            "value": m.value,
            "family": info.family.value,
            "bits": info.bits,
            "compression": round(16.0 / info.bits, 1),
            "requires_calibration": info.requires_calibration,
            "supports_asymmetric": info.supports_asymmetric,
            "transform": info.transform_name,
            "description": info.description,
            "fma_count": info.fma_count,
            "backend_only": False,
        })
    for m in BACKEND_ONLY_METHODS:
        bits = METHOD_BITS[m]
        methods.append({
            "value": m.value,
            "family": METHOD_FAMILIES[m].value,
            "bits": bits,
            "compression": round(16.0 / bits, 1),
            "requires_calibration": m in CALIBRATION_REQUIRED,
            "supports_asymmetric": True,
            "transform": "KVarN",
            "description": (
                "Godzilla llama.cpp target-cache alias; requires "
                "fork_profile=godzilla and 128-slice-compatible heads."
            ),
            "fma_count": 0,
            "backend_only": True,
        })
    return methods


def api_presets():
    descs = list_presets()
    results = []
    for name, desc in descs.items():
        preset = get_preset(name)
        results.append({
            "name": name,
            "description": desc,
            "k_method": preset.k_method.value,
            "v_method": preset.v_method.value,
            "triattention": preset.triattention_enabled,
            "k_compression": round(16.0 / METHOD_BITS[preset.k_method], 1),
            "v_compression": round(16.0 / METHOD_BITS[preset.v_method], 1),
        })
    return results


def api_plan(params):
    gpus_raw = params.get("gpus", [])
    if not gpus_raw:
        detected = detect_gpus()
        gpus_raw = [g.to_planner_dict() for g in detected]
    result = plan_agents(
        gpus=gpus_raw,
        model_params_b=float(params.get("model_params_b", 7)),
        model_quant=params.get("model_quant", "Q4_K_M"),
        desired_agents=int(params.get("agents", 4)),
        desired_context=int(params.get("context", 8192)),
        compute=params.get("compute", "cuda"),
    )
    return result.to_dict()


def api_benchmark(params):
    """Run a quick encode/decode benchmark on all methods."""
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    head_dim = int(params.get("head_dim", 128))
    seq_len = int(params.get("seq_len", 64))
    num_heads = int(params.get("num_heads", 8))

    x = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float32,
                     device=torch.device(device))
    results = []
    for m in registered_methods():
        try:
            inst = get_method(m)
            t0 = time.perf_counter()
            compressed = inst.encode(x)
            t_enc = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            decoded = inst.decode(compressed, dtype=torch.float32)
            t_dec = (time.perf_counter() - t0) * 1000
            if device == "cuda":
                torch.cuda.synchronize()
            cos = torch.nn.functional.cosine_similarity(
                x.reshape(-1, head_dim), decoded.reshape(-1, head_dim), dim=-1
            ).mean().item()
            mse = (x - decoded).square().mean().item()
            packed_bytes = compressed.data.numel() * compressed.data.element_size()
            orig_bytes = seq_len * num_heads * head_dim * 2
            results.append({
                "method": m.value,
                "bits": METHOD_BITS[m],
                "encode_ms": round(t_enc, 2),
                "decode_ms": round(t_dec, 2),
                "cosine": round(cos, 4),
                "mse": round(mse, 6),
                "compression": round(orig_bytes / max(packed_bytes, 1), 1),
                "status": "ok",
            })
        except Exception as e:
            results.append({"method": m.value, "status": f"error: {e}"})
    return {"device": device, "head_dim": head_dim, "seq_len": seq_len, "results": results}


def _truthy(value):
    return value is True or (
        isinstance(value, str) and value.lower() in {"1", "true", "yes", "on"}
    )


def _optional_int(value, default=None):
    if value is None or value == "":
        return default
    return int(value)


def _optional_float(value, default=None):
    if value is None or value == "":
        return default
    return float(value)


def _optional_text(value):
    if value is None or value == "":
        return None
    return str(value)


def _command_config(params):
    """Build a CacheConfig from command-generator params.

    Older UI payloads can send triattention as a K/V method. TriAttention is a
    token eviction mode, so translate that selection into the explicit flag and
    leave the affected cache side at FP16 for llama.cpp command generation.
    """
    k = params.get("k_method", "turbo4")
    v = params.get("v_method", k)
    triattention_enabled = _truthy(params.get("triattention"))
    use_custom_triattention_llamacpp = _truthy(
        params.get("use_custom_triattention_llamacpp")
        or params.get("custom_triattention_llamacpp")
    )
    triattention_stats_path = params.get("triattention_stats_path") or None

    if k == CacheMethod.TRIATTENTION.value:
        triattention_enabled = True
        k = CacheMethod.FP16.value
    if v == CacheMethod.TRIATTENTION.value:
        triattention_enabled = True
        v = CacheMethod.FP16.value
    if use_custom_triattention_llamacpp:
        triattention_enabled = True

    return CacheConfig(
        k_method=CacheMethod(k),
        v_method=CacheMethod(v),
        triattention_enabled=triattention_enabled,
        triattention_budget=_optional_int(
            params.get("triattention_budget"), 4096,
        ),
        triattention_window=_optional_int(
            params.get("triattention_window"), 512,
        ),
        triattention_stats_path=triattention_stats_path,
        use_custom_triattention_llamacpp=use_custom_triattention_llamacpp,
        triattention_log=_truthy(params.get("triattention_log")),
    )


def _cuda_weight_share_config(params):
    if not _truthy(params.get("cuda_weight_share")):
        return None
    return CudaWeightShareConfig(
        enabled=True,
        library_path=params.get("cuda_weight_share_library")
        or "./cuda-llm-weight-share.so",
        model_size_bytes=_optional_int(params.get("cuda_weight_share_model_size")),
        model_size_tolerance=_optional_int(
            params.get("cuda_weight_share_tolerance"), 0,
        ),
        ipc_name=params.get("cuda_weight_share_ipc_name")
        or "/cuda_vram_ipc_auto",
        shm_wait_sec=_optional_int(params.get("cuda_weight_share_shm_wait_sec")),
        suppress_master_free=_truthy(params.get("cuda_weight_share_suppress_master_free")),
        trace_callers=_truthy(params.get("cuda_weight_share_trace")),
        trace_depth=_optional_int(params.get("cuda_weight_share_trace_depth")),
        trace_normal_allocs=_truthy(params.get("cuda_weight_share_trace_normal_allocs")),
    )


def _command_speculative_config(params):
    spec_type = params.get("spec_type")
    if _truthy(params.get("spec_dflash")):
        spec_type = "dflash"
    elif _truthy(params.get("spec_mtp")):
        spec_type = "draft-mtp"
    if not spec_type or spec_type == "none":
        return None

    from multi_turboquant.integration import LlamaCppSpeculativeConfig

    return LlamaCppSpeculativeConfig(
        spec_type=spec_type,
        draft_model=_optional_text(params.get("spec_draft_model")),
        draft_hf=_optional_text(params.get("spec_draft_hf")),
        draft_context_size=_optional_int(params.get("spec_draft_context")),
        draft_gpu_layers=_optional_text(params.get("spec_draft_gpu_layers")),
        draft_device=_optional_text(params.get("spec_draft_device")),
        draft_cache_type_k=_optional_text(params.get("spec_draft_cache_k")),
        draft_cache_type_v=_optional_text(params.get("spec_draft_cache_v")),
        draft_n_max=_optional_int(params.get("spec_draft_n_max")),
        draft_n_min=_optional_int(params.get("spec_draft_n_min")),
        branch_budget=_optional_int(params.get("spec_branch_budget")),
        draft_top_k=_optional_int(params.get("spec_draft_top_k")),
        draft_p_split=_optional_float(params.get("spec_draft_p_split")),
        draft_p_min=_optional_float(params.get("spec_draft_p_min")),
        draft_temp=_optional_text(params.get("spec_draft_temp")),
        dflash_cross_ctx=_optional_int(params.get("spec_dflash_cross_ctx")),
        dflash_max_slots=_optional_int(params.get("spec_dflash_max_slots")),
    )


def _command_context_extension_config(params):
    rope_scaling = _optional_text(params.get("rope_scaling"))
    if rope_scaling is not None:
        rope_scaling = rope_scaling.strip().lower()
        if rope_scaling in {"", "off"}:
            rope_scaling = None

    values = {
        "rope_scaling": rope_scaling,
        "rope_scale": _optional_float(params.get("rope_scale")),
        "rope_freq_base": _optional_float(params.get("rope_freq_base")),
        "rope_freq_scale": _optional_float(params.get("rope_freq_scale")),
        "yarn_orig_ctx": _optional_int(params.get("yarn_orig_ctx")),
        "yarn_ext_factor": _optional_float(params.get("yarn_ext_factor")),
        "yarn_attn_factor": _optional_float(params.get("yarn_attn_factor")),
        "yarn_beta_slow": _optional_float(params.get("yarn_beta_slow")),
        "yarn_beta_fast": _optional_float(params.get("yarn_beta_fast")),
    }
    if not any(value is not None for value in values.values()):
        return None
    return LlamaCppContextExtensionConfig(**values)


def api_scan_llamacpp(params):
    binary = _optional_text(params.get("binary")) or "llama-server"
    timeout = _optional_float(params.get("timeout_seconds"), 10.0)
    return scan_llamacpp_binary(binary, timeout_seconds=timeout).to_dict()


def api_generate_command(params):
    """Generate a llama.cpp or vLLM launch command."""
    from multi_turboquant.integration import get_llamacpp_command
    config = _command_config(params)
    cuda_weight_share = _cuda_weight_share_config(params)
    fork_profile = params.get("fork_profile") or params.get("llamacpp_profile") or "upstream"
    speculative = _command_speculative_config(params)
    context_extension = _command_context_extension_config(params)
    issues = []
    command = ""
    command_argv = []
    missing_patched_triattention_stats = (
        config.triattention_enabled
        and config.use_custom_triattention_llamacpp
        and not config.triattention_stats_path
    )
    if not missing_patched_triattention_stats:
        try:
            cmd = get_llamacpp_command(
                config,
                binary=params.get("binary") or "llama-server",
                model_path=params.get("model_path", "/opt/models/model.gguf"),
                host=params.get("host") or "127.0.0.1",
                port=int(params.get("port", 8080)),
                context_size=int(params.get("context", 4096)),
                gpu_layers=_optional_int(params.get("gpu_layers"), 99),
                tensor_split=params.get("tensor_split"),
                parallel_slots=int(params["parallel"]) if params.get("parallel") else None,
                cuda_weight_share=cuda_weight_share,
                fork_profile=fork_profile,
                context_extension=context_extension,
                speculative=speculative,
            )
            command_argv = cmd
            command = shlex.join(cmd)
        except ValueError as e:
            issues.append({"severity": "error", "method": "command",
                           "message": str(e), "suggestion": "Fix the command inputs."})

    plat = detect_platform()
    for issue in check_config(config, plat, fork_profile=fork_profile):
        issues.append({"severity": issue.severity, "method": issue.method,
                        "message": issue.message, "suggestion": issue.suggestion})
    if cuda_weight_share is not None:
        for warning in cuda_weight_share.validate():
            issues.append({"severity": "error", "method": "cuda_weight_share",
                           "message": warning, "suggestion": "Fix weight-share inputs."})
    return {"command": command, "argv": command_argv, "issues": issues}


def api_settings():
    return SETTINGS_STORE.public_state()


def api_save_settings(params):
    if not UI_MUTATIONS_ENABLED:
        raise PermissionError("The UI is running in read-only mode")
    return {
        "path": str(SETTINGS_STORE.path.resolve()),
        "settings": SETTINGS_STORE.save(params),
    }


def api_reset_settings():
    if not UI_MUTATIONS_ENABLED:
        raise PermissionError("The UI is running in read-only mode")
    return {
        "path": str(SETTINGS_STORE.path.resolve()),
        "settings": SETTINGS_STORE.reset(),
    }


def _saved_settings():
    return SETTINGS_STORE.load()


def api_scan_models(params):
    settings = _saved_settings()
    root = params.get("root") or settings["model_root"]
    return scan_models(
        root,
        max_depth=int(params.get("max_depth", 3)),
        limit=int(params.get("limit", 500)),
    )


def api_scan_addons(params):
    settings = _saved_settings()
    roots = params["roots"] if "roots" in params else settings["addon_roots"]
    if not isinstance(roots, list):
        raise ValueError("roots must be a list")
    return scan_addon_roots(
        roots,
        max_depth=int(params.get("max_depth", 3)),
        limit=int(params.get("limit", 200)),
    )


def api_scan_flashattention(params):
    settings = _saved_settings()
    path = params.get("path") or settings["flashattention_source"]
    return inspect_flashattention_source(path)


def api_inspect_addon_source(params):
    profile = _optional_text(params.get("profile"))
    path = _optional_text(params.get("path"))
    if not profile:
        raise ValueError("An add-on source profile is required")
    if not path:
        raise ValueError("An add-on source folder is required")
    return inspect_addon_source(profile, path)


def api_scan_environments(params):
    settings = _saved_settings()
    root = params.get("root") or settings["environment_root"]
    return scan_environment_profiles(
        root,
        cuda_toolkit=_optional_text(params.get("cuda_toolkit")),
        local_source_profile=_optional_text(params.get("local_source_profile")),
        local_source=_optional_text(params.get("local_source")),
        manual_dependency_override=_truthy(params.get("manual_dependency_override")),
    )


def api_create_environment(params):
    if not UI_MUTATIONS_ENABLED:
        raise PermissionError("The UI is running in read-only mode")
    if params.get("confirm") is not True:
        raise ValueError("Environment creation requires explicit confirmation")
    settings = _saved_settings()
    profile = _optional_text(params.get("profile"))
    if profile is None:
        raise ValueError("An environment profile is required")
    local_source_profile = _optional_text(params.get("local_source_profile"))
    local_source = _optional_text(params.get("local_source"))
    if local_source and local_source_profile != profile:
        raise ValueError("The selected local checkout does not match this environment profile")
    return ENVIRONMENT_JOBS.start_create(
        profile,
        root=params.get("root") or settings["environment_root"],
        python=_optional_text(params.get("python")),
        cuda_toolkit=_optional_text(params.get("cuda_toolkit")),
        local_source=local_source,
        build_from_source=_truthy(params.get("build_from_source")),
        max_jobs=_optional_int(params.get("max_jobs"), 2),
    )


def api_environment_jobs():
    return {"jobs": ENVIRONMENT_JOBS.list()}


def api_scan_gigatoken(params):
    settings = _saved_settings()
    return scan_gigatoken_interpreters(environment_root=settings["environment_root"])


def _validated_godzilla_checkout(value: str) -> Path:
    settings = _saved_settings()
    roots = [Path(item).expanduser().resolve() for item in settings["addon_roots"]]
    checkout = Path(value).expanduser().resolve()
    if not any(checkout == root or checkout.is_relative_to(root) for root in roots):
        raise ValueError("The Godzilla checkout must be inside a saved add-on root")
    inspection = inspect_godzilla_checkout(checkout)
    if not inspection["valid"]:
        raise ValueError("; ".join(str(item) for item in inspection.get("issues", [])))
    return checkout


def _validated_godzilla_output(value, *, checkout: Path, model: Path) -> Path | None:
    text_value = _optional_text(value)
    if text_value is None:
        return None
    output = Path(text_value).expanduser().resolve()
    model_root = model.parent.resolve()
    if not (output.is_relative_to(checkout) or output.is_relative_to(model_root)):
        raise ValueError("Calibration output must be inside the Godzilla checkout or model folder")
    return output


def _validated_calibration_file(value, *, label: str) -> Path | None:
    text_value = _optional_text(value)
    if text_value is None:
        return None
    path = Path(text_value).expanduser().resolve()
    settings = _saved_settings()
    roots = [Path(item).expanduser().resolve() for item in settings["addon_roots"]]
    if settings.get("model_root"):
        roots.append(Path(settings["model_root"]).expanduser().resolve())
    if not any(path == root or path.is_relative_to(root) for root in roots):
        raise ValueError(f"{label} must be inside the saved model or add-on roots")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def _validated_hf_model(value) -> str | None:
    model = _optional_text(value)
    if model is None:
        return None
    candidate = Path(model).expanduser()
    if not candidate.exists():
        return model
    resolved = candidate.resolve()
    settings = _saved_settings()
    roots = [Path(item).expanduser().resolve() for item in settings["addon_roots"]]
    if settings.get("model_root"):
        roots.append(Path(settings["model_root"]).expanduser().resolve())
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ValueError("A local Hugging Face model must be inside a saved model or add-on root")
    if not resolved.is_dir():
        raise ValueError(f"Local Hugging Face model is not a directory: {resolved}")
    return str(resolved)


def _managed_triattention_python(settings) -> Path:
    target = Path(settings["environment_root"]).expanduser().resolve() / "triattention"
    return lexical_absolute_path(environment_python(target))


def _default_triattention_python(settings) -> Path | None:
    interpreter = _managed_triattention_python(settings)
    return interpreter if interpreter.is_file() else None


def _default_triattention_calibrator(settings) -> Path | None:
    scan = scan_addon_roots(settings["addon_roots"])
    candidates: list[Path] = []
    for addon in scan["addons"]:
        if addon.get("kind") != "triattention":
            continue
        source = addon.get("source")
        if not isinstance(source, dict) or not source.get("valid") or not source.get("calibrator"):
            continue
        candidates.append(Path(str(source["calibrator"])).resolve())
    return candidates[0] if len(candidates) == 1 else None


def _godzilla_plan_from_params(params):
    settings = _saved_settings()
    checkout = _validated_godzilla_checkout(str(params.get("checkout", "")))
    model = _validated_launch_model(str(params.get("gguf", "")))
    output = _validated_godzilla_output(params.get("output"), checkout=checkout, model=model)
    mode = _optional_text(params.get("mode")) or "official_python"
    python = _optional_text(params.get("python"))
    if python is None and mode in {"official_python", "official_convert", "domvox"}:
        discovered_python = _default_triattention_python(settings)
        python = str(discovered_python) if discovered_python is not None else None
    calibrator = _validated_calibration_file(params.get("calibrator"), label="Calibrator")
    if calibrator is None and mode == "official_python":
        calibrator = _default_triattention_calibrator(settings)
    domvox_calibrator = _validated_calibration_file(
        params.get("domvox_calibrator"), label="domvox calibrator"
    )
    if domvox_calibrator is None and mode == "domvox":
        domvox_calibrator = calibrator
    return plan_godzilla_triattention(
        checkout,
        model,
        output=output,
        python=python,
        calibrator=calibrator,
        calibration_input=_validated_calibration_file(
            params.get("calibration_input"), label="Calibration input"
        ),
        official_stats_input=_validated_calibration_file(
            params.get("official_stats_input"), label="Official statistics"
        ),
        domvox_calibrator=domvox_calibrator,
        domvox_accept_lossy=_truthy(params.get("domvox_accept_lossy")),
        allow_long_calibration=_truthy(params.get("allow_long_calibration")),
        hf_model=_validated_hf_model(params.get("hf_model")),
        n_tokens=_optional_int(params.get("n_tokens"), 2048),
        device=_optional_text(params.get("device")) or "cuda",
        mode=mode,
        attention_implementation=(
            _optional_text(params.get("attention_implementation")) or "sdpa"
        ),
        tokenizer_backend=_optional_text(params.get("tokenizer_backend")) or "transformers",
        verify_dependencies=True,
        dependency_override=_truthy(params.get("dependency_override")),
    )


def api_plan_godzilla(params):
    plan = _godzilla_plan_from_params(params)
    result = plan.to_dict()
    issue_codes = {issue.code for issue in plan.issues}
    settings = _saved_settings()
    requested_python = _optional_text(params.get("python"))
    managed_python = _managed_triattention_python(settings)
    requested_managed_python = (
        requested_python is not None
        and same_lexical_path(requested_python, managed_python)
    )
    managed_python_requested = (
        (requested_python is None or requested_managed_python)
        and plan.mode in {"official_python", "official_convert", "domvox"}
    )
    repairable_codes = {
        "missing_calibration_python",
        "calibration_dependencies_missing",
        "calibration_dependency_override",
    }
    repair_requested = managed_python_requested and bool(issue_codes & repairable_codes)
    repair_plan = None
    repair_errors: list[str] = []
    if repair_requested:
        repair_plan = plan_environment(
            "triattention",
            root=settings["environment_root"],
            max_jobs=2,
        )
        repair_errors = [
            issue.message for issue in repair_plan.issues if issue.severity == "error"
        ]
    result["dependency_repair"] = {
        "needed": repair_requested,
        "available": repair_requested and repair_plan is not None and repair_plan.ready,
        "profile": "triattention",
        "managed_python": str(managed_python),
        "message": (
            "Synchronize the pinned managed TriAttention environment and validate torch, "
            "transformers, accelerate, Gigatoken, and triattention before automatically rechecking "
            "the plan."
            if not repair_errors
            else "Managed repair is unavailable: " + "; ".join(repair_errors)
        ),
    }
    result["resource_policy"] = {
        "max_concurrent_calibrations": GODZILLA_JOBS.max_concurrent_jobs,
        "message": (
            "The local UI permits one calibration job at a time to prevent overlapping "
            "model loads. This does not reduce the memory used by one long sequence."
        ),
    }
    return result


def api_repair_triattention_environment(params):
    """Repair only the reviewed managed profile, ignoring unrelated UI overrides."""
    if not UI_MUTATIONS_ENABLED:
        raise PermissionError("The UI is running in read-only mode")
    if params.get("confirm") is not True:
        raise ValueError("TriAttention dependency repair requires explicit confirmation")
    settings = _saved_settings()
    return ENVIRONMENT_JOBS.start_create(
        "triattention",
        root=settings["environment_root"],
        max_jobs=2,
    )


def api_generate_calibration_text(params):
    if not UI_MUTATIONS_ENABLED:
        raise PermissionError("The UI is running in read-only mode")
    if params.get("confirm") is not True:
        raise ValueError("Calibration text generation requires explicit confirmation")
    settings = _saved_settings()
    model_root_value = str(settings.get("model_root", "")).strip()
    if not model_root_value:
        raise ValueError("Configure and save a model root before generating calibration text")
    model_root = Path(model_root_value).expanduser().resolve()
    if not model_root.is_dir():
        raise ValueError(f"Configured model root is not available: {model_root}")
    target_tokens = _optional_int(params.get("n_tokens"), 2048)
    output = (
        model_root
        / ".mtq"
        / "calibration"
        / f"generic-calibration-v{CALIBRATION_CORPUS_SCHEMA_VERSION}-{target_tokens}.txt"
    ).resolve()
    if not output.is_relative_to(model_root):
        raise ValueError("Generated calibration text must remain inside the saved model root")
    return generate_calibration_text(output, target_tokens=target_tokens)


def api_create_godzilla(params):
    if not UI_MUTATIONS_ENABLED:
        raise PermissionError("The UI is running in read-only mode")
    if params.get("confirm") is not True:
        raise ValueError("Godzilla calibration requires explicit confirmation")
    plan = _godzilla_plan_from_params(params)
    job_kwargs = {
        "output": plan.output,
        "python": plan.python,
        "calibrator": plan.calibrator,
        "calibration_input": plan.calibration_input,
        "official_stats_input": plan.official_stats_input,
        "hf_model": plan.hf_model,
        "n_tokens": plan.n_tokens,
        "device": plan.device,
        "mode": plan.mode,
        "attention_implementation": plan.attention_implementation,
        "tokenizer_backend": getattr(plan, "tokenizer_backend", "transformers"),
        "dependency_override": plan.dependency_override,
    }
    if plan.mode == "domvox":
        job_kwargs["domvox_calibrator"] = plan.domvox_calibrator
        job_kwargs["domvox_accept_lossy"] = plan.domvox_accept_lossy
    if getattr(plan, "allow_long_calibration", False):
        job_kwargs["allow_long_calibration"] = True
    return GODZILLA_JOBS.start(
        plan.checkout,
        plan.gguf,
        **job_kwargs,
    )


def api_godzilla_jobs():
    return {"jobs": GODZILLA_JOBS.list()}


def _validated_launch_model(model_path: str) -> Path:
    settings = _saved_settings()
    model_root = str(settings["model_root"])
    if not model_root:
        raise ValueError("Configure and save a model root before launching")
    root = Path(model_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Configured model root is not available: {root}")
    model = Path(model_path).expanduser().resolve()
    if not model.is_file() or model.suffix.lower() != ".gguf":
        raise ValueError("Quick Run launches existing .gguf files only")
    if not model.is_relative_to(root):
        raise ValueError("The selected model must be inside the configured model root")
    return model


def api_runtime_start(params):
    if not UI_MUTATIONS_ENABLED:
        raise PermissionError("The UI is running in read-only mode")
    model = _validated_launch_model(str(params.get("model_path", "")))
    payload = dict(params)
    payload["model_path"] = str(model)
    payload["host"] = payload.get("host") or "127.0.0.1"
    result = api_generate_command(payload)
    errors = [issue for issue in result["issues"] if issue["severity"] == "error"]
    if errors:
        raise ValueError("; ".join(str(issue["message"]) for issue in errors))
    argv = result["argv"]
    if not argv:
        raise ValueError("The current settings did not produce a launch command")
    return MODEL_PROCESS.start(argv, cwd=model.parent)


def api_runtime_stop():
    if not UI_MUTATIONS_ENABLED:
        raise PermissionError("The UI is running in read-only mode")
    return MODEL_PROCESS.stop()


def api_runtime_status():
    return MODEL_PROCESS.status()


# ─── HTML UI ────────────────────────────────────────────────────────────────────

UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-TurboQuant</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0d1117; color: #c9d1d9; line-height: 1.5; }
.container { max-width: 1100px; margin: 0 auto; padding: 20px; }
h1 { color: #58a6ff; font-size: 24px; margin-bottom: 4px; }
.subtitle { color: #8b949e; font-size: 13px; margin-bottom: 10px; }
.topbar { display:flex; justify-content:space-between; gap:12px; align-items:center;
          margin-bottom:16px; flex-wrap:wrap; }
.view-tabs { display:flex; gap:6px; padding:4px; border:1px solid #30363d;
             background:#161b22; border-radius:8px; }
.view-tabs button { background:transparent; color:#8b949e; padding:7px 12px; }
.view-tabs button.active { background:#1f6feb; color:#fff; }
.save-state { font-size:12px; color:#8b949e; }
.save-state.ok { color:#3fb950; }
.save-state.error { color:#ff7b72; }
.view { display:none; }
.view.active { display:block; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
.card h2 { color: #58a6ff; font-size: 14px; margin-bottom: 12px; text-transform: uppercase;
           letter-spacing: 0.5px; }
.card.full { grid-column: 1 / -1; }
details.card { padding: 0; }
details.card > summary { padding: 16px; }
details > summary { cursor: pointer; list-style: none; display:flex; justify-content:space-between;
                    gap:12px; align-items:center; color:#58a6ff; font-size:14px; font-weight:700;
                    text-transform:uppercase; letter-spacing:0.5px; }
details > summary::-webkit-details-marker { display:none; }
details > summary::before { content:'▸'; color:#8b949e; margin-right:8px; }
details[open] > summary::before { content:'▾'; }
.section-summary { display:flex; align-items:center; gap:8px; }
.section-summary .muted { text-transform:none; letter-spacing:0; font-weight:400; }
.details-body { padding:0 16px 16px; }
.inline-disclosure { border:1px solid #30363d; border-radius:6px; padding:0 10px; margin:10px 0 12px; }
.inline-disclosure > summary { padding:9px 0; font-size:12px; text-transform:none; letter-spacing:0; }
.inline-disclosure .details-body { padding:0 0 2px; }
.gpu-badge { display: inline-block; background: #1f6feb22; border: 1px solid #1f6feb;
             border-radius: 4px; padding: 4px 10px; margin: 2px; font-size: 12px; }
.method-row { display: flex; justify-content: space-between; padding: 6px 0;
              border-bottom: 1px solid #21262d; font-size: 13px; }
.method-row:last-child { border-bottom: none; }
.tag { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px;
       font-weight: 600; margin-left: 6px; }
.tag-turbo { background: #1f6feb33; color: #58a6ff; }
.tag-iso { background: #23863633; color: #3fb950; }
.tag-planar { background: #a371f733; color: #bc8cff; }
.tag-rotor { background: #ff7b7233; color: #ff7b72; }
.tag-tri { background: #f0883e33; color: #f0883e; }
.tag-kvarn { background: #2f81f733; color: #79c0ff; }
.tag-tcq { background: #1f6feb33; color: #79c0ff; }
.tag-free { background: #23863633; color: #3fb950; font-size: 10px; }
.capability-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 12px; }
.capability { border: 1px solid #30363d; border-radius: 4px; padding: 3px 8px;
              font-size: 11px; color: #8b949e; }
.cap-ok { border-color: #238636; color: #3fb950; }
.cap-warn { border-color: #d29922; color: #d29922; }
.cap-bad { border-color: #f85149; color: #ff7b72; }
.mini-label { color: #8b949e; font-size: 11px; text-transform: uppercase;
              letter-spacing: 0.5px; margin: 8px 0; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; color: #8b949e; font-weight: 600; padding: 8px 6px;
     border-bottom: 2px solid #30363d; }
td { padding: 6px; border-bottom: 1px solid #21262d; }
.num { text-align: right; font-family: 'SF Mono', Monaco, monospace; }
input, select, textarea { background: #0d1117; border: 1px solid #30363d; color: #c9d1d9;
                 padding: 6px 10px; border-radius: 4px; font-size: 13px; width: 100%; }
textarea { resize:vertical; font-family:inherit; }
label { font-size: 12px; color: #8b949e; display: block; margin-bottom: 4px; }
.form-row { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px;
            margin-bottom: 12px; }
button { background: #238636; color: #fff; border: none; padding: 8px 16px;
         border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; }
button:hover { background: #2ea043; }
button.secondary { background: #30363d; }
button.secondary:hover { background: #3d444d; }
button.danger { background:#8e2424; }
button.danger:hover { background:#a83232; }
button:disabled { opacity:0.55; cursor:not-allowed; }
.button-row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:10px; }
.muted { color:#8b949e; font-size:12px; }
.setup-note { border-left:3px solid #1f6feb; background:#0d1117; padding:10px 12px;
              color:#8b949e; font-size:12px; margin-bottom:12px; }
.item-list { display:grid; gap:8px; margin-top:10px; }
.item { border:1px solid #30363d; background:#0d1117; border-radius:6px; padding:10px; }
.item-title { display:flex; justify-content:space-between; gap:10px; font-weight:600; }
.status-pill { display:inline-block; border:1px solid #30363d; border-radius:999px;
               padding:2px 8px; font-size:10px; text-transform:uppercase; }
.status-ready,.status-installed,.status-completed { color:#3fb950; border-color:#238636; }
.status-configured,.status-running,.status-queued,.status-manual { color:#d29922; border-color:#d29922; }
.status-blocked,.status-incompatible,.status-failed,.status-broken { color:#ff7b72; border-color:#f85149; }
.runtime-actions { display:grid; grid-template-columns:1fr auto auto; gap:8px; align-items:end; }
.runtime-log { min-height:70px; }
.result-box { background: #0d1117; border: 1px solid #30363d; border-radius: 4px;
              padding: 12px; font-family: 'SF Mono', Monaco, monospace; font-size: 12px;
              margin-top: 10px; white-space: pre-wrap; overflow-x: auto; max-height: 300px;
              overflow-y: auto; }
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
              margin-right: 6px; }
.dot-green { background: #3fb950; }
.dot-yellow { background: #d29922; }
.dot-red { background: #f85149; }
.cos-bar { display: inline-block; height: 6px; border-radius: 3px; background: #238636; }
#loading { text-align: center; padding: 40px; color: #8b949e; }
@media (max-width: 760px) {
  .grid,.form-row { grid-template-columns:1fr; }
  .card.full { grid-column:1; }
  .runtime-actions { grid-template-columns:1fr; }
  [style*="grid-column"] { grid-column:1 !important; }
}
</style>
</head>
<body>
<div class="container">
  <h1>Multi-TurboQuant</h1>
  <div class="subtitle" id="version-info">Loading...</div>
  <div class="topbar">
    <div class="view-tabs" role="tablist" aria-label="Workspace views">
      <button id="tab-quick" class="active" onclick="showView('quick')">Quick Run</button>
      <button id="tab-setup" onclick="showView('setup')">Setup &amp; Add-ons</button>
    </div>
    <div id="save-state" class="save-state">Loading saved settings...</div>
  </div>

  <section id="view-quick" class="view active">
  <div class="grid">
    <!-- GPU Status -->
    <div class="card">
      <h2>Hardware</h2>
      <div id="gpu-status">Detecting...</div>
      <div id="memory-status" class="muted" style="margin-top:10px"></div>
      <label title="Capacity inventory only. System RAM does not replace discrete GPU VRAM during calibration."><input type="checkbox" id="memory-include-vram" checked onchange="renderMemoryStatus()"> Include VRAM in combined capacity</label>
    </div>

    <!-- Quick Stats -->
    <div class="card">
      <h2>Library</h2>
      <div id="lib-status">Loading...</div>
    </div>

    <!-- Methods -->
    <div class="card full">
      <h2>Compression Methods</h2>
      <div id="methods-list">Loading...</div>
    </div>

    <!-- Presets -->
    <div class="card">
      <h2>Presets</h2>
      <div id="presets-list" style="max-height:300px;overflow-y:auto;">Loading...</div>
    </div>

    <!-- Capacity Planner -->
    <div class="card">
      <h2>Capacity Planner</h2>
      <div class="form-row">
        <div><label>Model (B params)</label><input type="number" id="plan-model" value="32"></div>
        <div><label>Quant</label><select id="plan-quant">
          <option value="Q4_K_M" selected>Q4_K_M</option>
          <option value="Q3_K_M">Q3_K_M</option>
          <option value="Q6_K">Q6_K</option>
          <option value="Q8_0">Q8_0</option>
        </select></div>
        <div><label>Agents</label><input type="number" id="plan-agents" value="4"></div>
        <div><label>Context</label><input type="number" id="plan-context" value="8192"></div>
      </div>
      <button onclick="runPlanner()">Plan</button>
      <div class="result-box" id="plan-result" style="display:none;"></div>
    </div>

    <!-- Benchmark -->
    <div class="card full">
      <h2>Benchmark</h2>
      <div class="form-row">
        <div><label>Device</label><select id="bench-device">
          <option value="cuda">CUDA (GPU)</option>
          <option value="cpu">CPU</option>
        </select></div>
        <div><label>Head Dim</label><input type="number" id="bench-headdim" value="128"></div>
        <div><label>Seq Length</label><input type="number" id="bench-seqlen" value="64"></div>
        <div><label>&nbsp;</label><button onclick="runBenchmark()">Run Benchmark</button></div>
      </div>
      <div id="bench-result"></div>
    </div>

    <!-- Command Generator -->
    <div class="card full">
      <h2>Command Generator</h2>
      <div class="form-row">
        <div style="grid-column:1/4"><label title="Path to the llama.cpp-compatible server binary used for both scanning and command generation.">llama-server Binary</label>
          <input type="text" id="cmd-binary" value="llama-server" onchange="generateCommand()">
        </div>
        <div><label>&nbsp;</label><button class="secondary" onclick="scanLlamaCpp()">Scan</button></div>
      </div>
      <div class="capability-row" id="cmd-capabilities">
        <span class="capability">Not scanned</span>
      </div>
      <div class="form-row">
        <div><label>Profile</label><select id="cmd-profile" onchange="generateCommand()">
          <option value="upstream" selected>upstream</option>
          <option value="patched_triattention">patched_triattention</option>
          <option value="godzilla">godzilla</option>
        </select></div>
        <div><label>K Method</label><select id="cmd-k" onchange="generateCommand()">
          <option value="f16">f16</option>
        </select></div>
        <div><label>V Method</label><select id="cmd-v" onchange="generateCommand()">
          <option value="f16">f16</option>
        </select></div>
        <div><label>Context</label><input type="number" id="cmd-ctx" value="4096" onchange="generateCommand()"></div>
      </div>
      <details class="inline-disclosure">
        <summary>Context extension (RoPE / YaRN)</summary>
        <div class="details-body">
      <div class="form-row">
        <div><label title="Select the RoPE scaling mode passed to llama.cpp. Leave model default unless extending beyond the model's trained context.">RoPE Mode</label><select id="cmd-rope-scaling" onchange="generateCommand()">
          <option value="" selected>model default</option>
          <option value="linear">linear</option>
          <option value="yarn">YaRN</option>
          <option value="none">none</option>
        </select></div>
        <div><label title="Context scaling factor. For example, 8 extends a 4K-trained model toward a 32K target.">RoPE Scale</label><input type="number" step="0.0001" id="cmd-rope-scale" value="" placeholder="8" onchange="generateCommand()"></div>
        <div><label title="Original trained context for YaRN. Use the model's training context, such as 4096 or 8192.">YaRN Orig Ctx</label><input type="number" id="cmd-yarn-orig-ctx" value="" placeholder="4096" onchange="generateCommand()"></div>
        <div><label title="RoPE base frequency override for models or experiments that need a custom base.">Freq Base</label><input type="number" step="0.0001" id="cmd-rope-freq-base" value="" placeholder="10000" onchange="generateCommand()"></div>
      </div>
      <div class="form-row">
        <div><label title="RoPE frequency scale. This conflicts with RoPE Scale because llama.cpp maps both to frequency scaling.">Freq Scale</label><input type="number" step="0.0001" id="cmd-rope-freq-scale" value="" placeholder="0.125" onchange="generateCommand()"></div>
        <div><label title="YaRN extrapolation mix factor. Zero means full interpolation; leave blank for llama.cpp default.">YaRN Ext</label><input type="number" step="0.0001" id="cmd-yarn-ext-factor" value="" placeholder="0" onchange="generateCommand()"></div>
        <div><label title="YaRN attention magnitude factor. Leave blank unless a model card specifies it.">YaRN Attn</label><input type="number" step="0.0001" id="cmd-yarn-attn-factor" value="" placeholder="1" onchange="generateCommand()"></div>
        <div><label title="YaRN correction range. Use slow,fast such as 1,32 if a model card or experiment specifies it.">YaRN Beta Slow/Fast</label><div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
          <input type="number" step="0.0001" id="cmd-yarn-beta-slow" value="" placeholder="1" onchange="generateCommand()">
          <input type="number" step="0.0001" id="cmd-yarn-beta-fast" value="" placeholder="32" onchange="generateCommand()">
        </div></div>
      </div>
        </div>
      </details>
      <div class="form-row">
        <div style="grid-column:1/3"><label>Discovered Model</label>
          <select id="cmd-model-select" onchange="selectDiscoveredModel()">
            <option value="">Configure a model root in Setup</option>
          </select>
        </div>
        <div><label>&nbsp;</label><button class="secondary" onclick="scanModels()">Refresh Models</button></div>
        <div><label>GPU Layers</label><input type="number" id="cmd-gpu-layers" value="99" onchange="generateCommand()"></div>
      </div>
      <div class="form-row">
        <div style="grid-column:1/4"><label>Model Path</label>
          <input type="text" id="cmd-model" value="/opt/models/model.gguf" onchange="generateCommand()">
        </div>
        <div><label>Parallel Slots</label><input type="number" id="cmd-parallel" value="1" onchange="generateCommand()"></div>
      </div>
      <div class="form-row">
        <div style="grid-column:1/3"><label>Server Host</label><input type="text" id="cmd-host" value="127.0.0.1" onchange="generateCommand()"></div>
        <div><label>Server Port</label><input type="number" id="cmd-port" value="8080" onchange="generateCommand()"></div>
        <div><label>Tensor Split</label><input type="text" id="cmd-tensor-split" value="" placeholder="0.7,0.3" onchange="generateCommand()"></div>
      </div>
      <details class="inline-disclosure">
        <summary>Speculative decoding (DFlash)</summary>
        <div class="details-body">
      <div class="form-row">
        <div><label><input type="checkbox" id="cmd-spec-dflash" onchange="generateCommand()"> DFlash</label></div>
        <div><label>Draft Model</label><input type="text" id="cmd-spec-draft-model" value="" placeholder="draft.gguf" onchange="generateCommand()"></div>
        <div><label>Draft N Max</label><input type="number" id="cmd-spec-draft-n-max" value="16" onchange="generateCommand()"></div>
        <div><label>DFlash Cross Ctx</label><input type="number" id="cmd-spec-cross-ctx" value="512" onchange="generateCommand()"></div>
      </div>
      <div class="form-row">
        <div><label>Draft GPU Layers</label><input type="text" id="cmd-spec-draft-ngl" value="all" onchange="generateCommand()"></div>
        <div><label>Branch Budget</label><input type="number" id="cmd-spec-branch-budget" value="0" onchange="generateCommand()"></div>
        <div><label>Draft K Cache</label><input type="text" id="cmd-spec-cache-k" value="" placeholder="f16" onchange="generateCommand()"></div>
        <div><label>Draft V Cache</label><input type="text" id="cmd-spec-cache-v" value="" placeholder="f16" onchange="generateCommand()"></div>
      </div>
        </div>
      </details>
      <details class="inline-disclosure">
        <summary>TriAttention (experimental)</summary>
        <div class="details-body">
      <div class="form-row">
        <div><label><input type="checkbox" id="cmd-tri" onchange="generateCommand()"> TriAttention</label></div>
        <div><label><input type="checkbox" id="cmd-tri-custom" onchange="generateCommand()"> Patched llama.cpp (stats required)</label></div>
        <div><label>TriAttn Budget</label><input type="number" id="cmd-tri-budget" value="4096" onchange="generateCommand()"></div>
        <div><label>TriAttn Window</label><input type="number" id="cmd-tri-window" value="512" onchange="generateCommand()"></div>
      </div>
      <div class="form-row">
        <div style="grid-column:1/-1"><label>TriAttention Stats Path</label>
          <input type="text" id="cmd-tri-stats" value="" placeholder="model.triattention" onchange="generateCommand()">
        </div>
        <div class="setup-note" style="grid-column:1/-1">Godzilla's current policy treats TriAttention as experimental, opt-in, and manually calibrated. Its checkout may not include a calibrator. A GGUF file alone cannot supply the required pre-RoPE query statistics; Multi-TurboQuant's Python <code>.pt</code> stats are a different format.</div>
      </div>
        </div>
      </details>
      <details class="inline-disclosure">
        <summary>CUDA weight sharing (advanced)</summary>
        <div class="details-body">
      <div class="form-row">
        <div><label><input type="checkbox" id="cmd-tri-log" onchange="generateCommand()"> TriAttn Log</label></div>
        <div><label><input type="checkbox" id="cmd-weight-share" onchange="generateCommand()"> CUDA Weight Share</label></div>
        <div><label title="Exact model-weight CUDA allocation size in bytes. Use 0 or blank only for a discovery run; this is not the GGUF file size.">MODEL_SIZE</label><input type="number" id="cmd-ws-model-size" value="" onchange="generateCommand()"></div>
        <div><label title="Allowed byte difference when matching MODEL_SIZE. Keep 0 unless allocator variation requires a small tolerance.">MODEL_SIZE_TOLERANCE</label><input type="number" id="cmd-ws-tolerance" value="0" onchange="generateCommand()"></div>
      </div>
      <div class="form-row">
        <div><label title="LD_PRELOAD loads the external Linux helper before llama.cpp so it can intercept CUDA allocations.">LD_PRELOAD library</label><input type="text" id="cmd-ws-library" value="./cuda-llm-weight-share.so" onchange="generateCommand()"></div>
        <div><label title="Shared-memory namespace. Every process sharing one model must use the same unique name; unrelated groups should use different names.">CUDA_VRAM_IPC_NAME</label><input type="text" id="cmd-ws-ipc" value="/cuda_vram_ipc_auto" onchange="generateCommand()"></div>
        <div><label title="Seconds a worker waits for the master process to publish shared allocation metadata.">SHM wait seconds</label><input type="number" id="cmd-ws-wait" value="" onchange="generateCommand()"></div>
        <div><label title="Diagnostic caller tracing adds log volume and overhead; leave disabled for normal serving."><input type="checkbox" id="cmd-ws-trace" onchange="generateCommand()"> Trace callers</label></div>
      </div>
      <div class="form-row">
        <div><label title="Maximum captured call-stack depth when caller tracing is enabled.">Trace depth</label><input type="number" id="cmd-ws-trace-depth" value="" min="1" onchange="generateCommand()"></div>
        <div><label title="Also trace CUDA allocations that are not classified as model weights; useful only for diagnostics."><input type="checkbox" id="cmd-ws-trace-normal" onchange="generateCommand()"> Trace normal allocations</label></div>
        <div><label title="Specialized helper option that prevents the master from freeing the shared backing allocation prematurely; leave disabled unless the helper's workflow requires it."><input type="checkbox" id="cmd-ws-suppress-free" onchange="generateCommand()"> Suppress master free</label></div>
        <div></div>
      </div>
      <div class="setup-note">CUDA weight sharing is a Linux + CUDA feature supplied by an external preload helper. First run one trusted process with <code>MODEL_SIZE=0</code> to discover the model-weight allocation, then reuse the reported byte count for the master and workers. It shares model weights only—not the KV cache or context—and all participating processes must use the same model/build/device setup.</div>
        </div>
      </details>
      <div class="result-box" id="cmd-result">Select methods above...</div>
      <div class="mini-label">Managed llama-server</div>
      <div class="runtime-actions">
        <div id="runtime-summary" class="muted">Not running</div>
        <button id="runtime-start" onclick="startRuntime()">Start</button>
        <button id="runtime-stop" class="danger" onclick="stopRuntime()" disabled>Stop</button>
      </div>
      <div class="result-box runtime-log" id="runtime-log">No process output.</div>
    </div>
  </div>
  </section>

  <section id="view-setup" class="view">
    <div class="grid">
      <div class="card full">
        <h2>Persistent Workspace</h2>
        <div class="setup-note">Settings are stored in a versioned JSON file under your home folder by default, outside the Git checkout, so normal pulls do not remove them. Scanners only inspect the roots configured here; they never search entire disks.</div>
        <div class="form-row">
          <div style="grid-column:1/3"><label>Default Model Root</label><input type="text" id="setup-model-root" placeholder="D:\\models or /opt/models"></div>
          <div style="grid-column:3/5"><label>Dependency Environment Root</label><input type="text" id="setup-environment-root" value=".mtq/environments"></div>
        </div>
        <div class="form-row">
          <div style="grid-column:1/3"><label>FlashAttention Source Folder</label><input type="text" id="setup-flash-source" placeholder="/path/to/flash-attention"></div>
          <div style="grid-column:3/5"><label>Add-on Roots (one per line)</label><textarea id="setup-addon-roots" rows="3" placeholder="/opt/addons"></textarea></div>
        </div>
        <div class="button-row">
          <button onclick="saveSettingsNow()">Save Settings</button>
          <button class="secondary" onclick="resetSettings()">Reset</button>
          <button class="secondary" onclick="exportSettings()">Export</button>
          <button class="secondary" onclick="document.getElementById('settings-import').click()">Import</button>
          <input id="settings-import" type="file" accept="application/json" style="display:none" onchange="importSettings(this)">
          <span class="muted" id="settings-path"></span>
        </div>
      </div>

      <div class="card">
        <h2>Model Library</h2>
        <p class="muted">Finds GGUF files for Quick Run and reports other recognized model formats.</p>
        <div class="button-row"><button onclick="scanModels()">Scan Models</button></div>
        <div id="model-scan-result" class="item-list"><div class="muted">Not scanned.</div></div>
      </div>

      <div class="card">
        <h2>Add-on Sources</h2>
        <p class="muted">Automatically scans configured roots for reviewed Python package checkouts and Godzilla/llama.cpp source trees. A recognized checkout can be selected below without executing scanner-discovered code.</p>
        <div class="button-row">
          <button onclick="scanAddons()">Scan Add-ons</button>
          <button class="secondary" onclick="scanFlashAttention()">Check FlashAttention</button>
        </div>
        <div id="addon-scan-result" class="item-list"><div class="muted">Not scanned.</div></div>
        <div class="mini-label">Inspect a local source folder</div>
        <div class="setup-note">Source inspection is read-only. A recognized source is recorded as informational until its dependencies, runtime integration, and licensing are separately reviewed.</div>
        <div class="form-row">
          <div><label>Source profile</label><select id="addon-source-profile">
            <option value="maru">Maru</option>
            <option value="speculative_prefill">Speculative Prefill</option>
            <option value="rocketkv">RocketKV</option>
            <option value="lexico">Lexico</option>
            <option value="adadecode">AdaDecode</option>
            <option value="resonance_yarn">Resonance YaRN</option>
            <option value="domvox_triattention">domvox TriAttention</option>
            <option value="gigatoken_llamacpp">Gigatoken llama.cpp</option>
          </select></div>
          <div style="grid-column:span 3"><label>Local source folder</label><input type="text" id="addon-source-path" placeholder="/path/to/checkout"></div>
        </div>
        <div class="button-row"><button class="secondary" onclick="inspectSelectedAddonSource()">Inspect source</button></div>
        <div id="addon-source-result" class="item-list"><div class="muted">No source selected.</div></div>
      </div>

      <div class="card full">
        <h2>Dependency Environments</h2>
        <div class="setup-note">Creation reuses the reviewed <code>mtq-env</code> profiles. A local checkout changes the package source, not its CUDA ABI. A newer NVIDIA driver can run older CUDA applications, but native extensions must be compiled with a toolkit matching the profile's PyTorch CUDA major. Select a side-by-side toolkit here; downloads and builds still require confirmation.</div>
        <div class="form-row">
          <div style="grid-column:span 2"><label>Python override</label><input type="text" id="env-python" placeholder="3.11 or interpreter path"></div>
          <div style="grid-column:span 2"><label>CUDA toolkit override</label><input type="text" id="env-cuda-toolkit" placeholder="/usr/local/cuda-12.6 or /path/to/nvcc"></div>
        </div>
        <div class="form-row">
          <div><label>Local source profile</label><select id="env-source-profile">
            <option value="">Use pinned/default source</option>
            <option value="flashattention">FlashAttention</option><option value="fastdms">FastDMS</option>
            <option value="lmcache">LMCache</option><option value="minference">MInference</option>
            <option value="sageattention">SageAttention</option><option value="triattention">TriAttention calibration</option>
          </select></div>
          <div style="grid-column:span 3"><label>Reviewed local checkout</label><input type="text" id="env-local-source" placeholder="Select a recognized add-on checkout above"></div>
        </div>
        <div class="button-row">
          <label><input type="checkbox" id="env-build-source"> Force reviewed source build</label>
          <label>Build jobs <input type="number" id="env-max-jobs" value="2" min="1" max="64" style="width:6em"></label>
          <label><input type="checkbox" id="env-manual-override"> Treat failed dependency checks as installed</label>
          <label><input type="checkbox" id="env-confirm"> Confirm downloads/builds</label>
          <button onclick="scanEnvironments()">Refresh Profiles</button>
        </div>
        <div class="setup-note">Existing environments are validated before another build is suggested. The manual override suppresses a rebuild recommendation when automatic imports fail, but it cannot prove the dependencies are usable and may move the failure to calibration or launch time.</div>
        <div class="setup-note">Blocked profiles are informational entries, not failed installations. They remain unavailable when upstream hardware, licensing, artifacts, or a maintained serving integration is missing.</div>
        <div id="environment-result" class="item-list"><div class="muted">Not scanned.</div></div>
      </div>

      <details class="card full setup-section">
        <summary><span class="section-summary">Godzilla Source Setup <span class="muted">Calibration and conversion</span></span></summary>
        <div class="details-body">
        <h2>Godzilla Source Setup</h2>
        <div class="setup-note">The recommended mode runs the official WeianMao/triattention Python calibrator, converts its <code>.pt</code> statistics to Godzilla v1, and validates the finished file. Select the recognized TriAttention checkout above and its isolated dependency environment is used automatically after validation. Existing official <code>.pt</code> statistics can be converted without another model forward pass. Neither route needs <code>llama-cli</code>.</div>
        <div class="form-row">
          <div style="grid-column:span 2"><label>Godzilla checkout</label><input type="text" id="godzilla-checkout" placeholder="Select a recognized Godzilla checkout above"></div>
          <div style="grid-column:span 2"><label>GGUF model</label><input type="text" id="godzilla-gguf" placeholder="Model inside the saved model root"></div>
        </div>
        <div class="form-row">
          <div><label>Calibration mode</label><select id="godzilla-mode"><option value="official_python" selected>Generate stats + convert</option><option value="official_convert">Convert existing official .pt</option><option value="godzilla_script">Godzilla checkout script</option><option value="domvox">domvox TRIA v2 (experimental)</option></select></div>
          <div><label>Attention implementation</label><select id="godzilla-attention"><option value="sdpa" selected>SDPA</option><option value="eager">Eager</option><option value="flash_attention_2">FlashAttention 2</option></select></div>
          <div><label>Tokenizer backend</label><select id="godzilla-tokenizer"><option value="transformers" selected>Hugging Face (default)</option><option value="gigatoken">Gigatoken (parity required)</option></select></div>
          <div><label>Calibration Python (optional)</label><input type="text" id="godzilla-python" placeholder="Auto: .mtq/environments/triattention"></div>
        </div>
        <div class="setup-note">Gigatoken is opt-in for the recommended official calibration mode. Before model loading, Multi-TurboQuant compares every token ID it produces with Hugging Face and stops on any mismatch. For inference, <code>mtq-godzilla-gigatoken</code> can prepare and qualify a separate pinned Godzilla v0.3.7 + Gigatoken runtime tree; this view never patches an arbitrary checkout.</div>
        <div class="button-row"><button class="secondary" onclick="scanGigatoken()">Scan Python and pyenv environments for Gigatoken</button></div>
        <div id="gigatoken-scan" class="item-list"><div class="muted">Not scanned.</div></div>
        <div class="form-row">
          <div style="grid-column:span 2"><label>TriAttention calibrator</label><input type="text" id="godzilla-calibrator" placeholder="Official scripts/calibrate.py or domvox/triattention_calibrate.py"></div>
          <div style="grid-column:span 2"><label>Calibration text</label><input type="text" id="godzilla-input" placeholder="Non-empty plain-text file inside a saved root"><div class="button-row"><button class="secondary" onclick="generateCalibrationInput()">Generate generic starter text</button></div><div class="muted">Offline and deterministic. Use representative domain text for final quality qualification.</div></div>
        </div>
        <div class="form-row">
          <div style="grid-column:span 2"><label>Existing official .pt statistics</label><input type="text" id="godzilla-official-stats" placeholder="Used only by Convert existing official .pt"></div>
          <div style="grid-column:span 2"><label><input type="checkbox" id="godzilla-dependency-override"> Dependencies are installed (override failed automatic check)</label><div class="muted">Use only when the dependency scanner is wrong; this does not bypass artifact validation.</div></div>
        </div>
        <div class="form-row">
          <div style="grid-column:span 2"><label>Matching Hugging Face model</label><input type="text" id="godzilla-hf-model" placeholder="org/model or local Transformers directory"></div>
          <div style="grid-column:span 2"><label>Output .triattention (optional)</label><input type="text" id="godzilla-output" placeholder="Defaults to checkout/calibrations/model.triattention"></div>
        </div>
        <div class="form-row">
          <div><label>Calibration tokens</label><input type="number" id="godzilla-tokens" value="2048" min="128" max="200000"></div>
          <div><label>Device</label><select id="godzilla-device"><option value="cuda">CUDA</option><option value="cpu">CPU</option></select></div>
          <div style="grid-column:span 2" class="button-row">
            <button class="secondary" onclick="planGodzilla()">Check Plan</button>
            <label><input type="checkbox" id="godzilla-confirm"> Confirm model load/download and preparation</label>
            <button onclick="startGodzilla()">Prepare TriAttention</button>
          </div>
        </div>
        <div class="setup-note"><label><input type="checkbox" id="godzilla-long-calibration"> Allow one-shot calibration above 32,768 tokens (up to 200,000; high memory/runtime risk)</label></div>
        <div class="setup-note"><label><input type="checkbox" id="godzilla-domvox-lossy"> I understand domvox TRIA v2 to Godzilla v1 conversion is experimental and drops fields not represented by Godzilla v1.</label></div>
        <div id="godzilla-plan" class="item-list"><div class="muted">No preparation plan checked.</div></div>
        </div>
      </details>

      <details class="card full setup-section">
        <summary><span class="section-summary">Background Jobs <span class="muted">Environment and calibration activity</span></span></summary>
        <div class="details-body">
        <h2>Background Jobs</h2>
        <div id="environment-jobs" class="item-list"><div class="muted">No environment jobs.</div></div>
        <div class="mini-label">Godzilla TriAttention</div>
        <div id="godzilla-jobs" class="item-list"><div class="muted">No Godzilla jobs.</div></div>
        </div>
      </details>
    </div>
  </section>
</div>

<script>
const API = '';
let llamaCppCapabilities = null;
let currentSettings = null;
let discoveredModels = [];
let saveTimer = null;
let pendingTriattentionRepairJob = null;
let addonScanTimer = null;
let statusTimer = null;
let hardwareStatus = null;

async function api(path, body) {
  const opts = body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {};
  const r = await fetch(API + path, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || `Request failed (${r.status})`);
  return data;
}

function showView(name) {
  document.querySelectorAll('.view').forEach(el => el.classList.toggle('active', el.id === `view-${name}`));
  document.querySelectorAll('.view-tabs button').forEach(el => el.classList.toggle('active', el.id === `tab-${name}`));
  if (currentSettings) {
    currentSettings.form_values.active_view = name;
    scheduleSave();
  }
}

function setSaveState(message, kind='') {
  const el = document.getElementById('save-state');
  el.textContent = message;
  el.className = `save-state ${kind}`;
}

function persistentControls() {
  return [...document.querySelectorAll('input[id], select[id], textarea[id]')].filter(el =>
    !['settings-import', 'env-confirm', 'godzilla-confirm', 'setup-model-root', 'setup-environment-root',
      'setup-flash-source', 'setup-addon-roots'].includes(el.id)
  );
}

function collectFormValues() {
  const values = {};
  persistentControls().forEach(el => {
    values[el.id] = el.type === 'checkbox' ? el.checked : el.value;
  });
  values.active_view = document.getElementById('view-setup').classList.contains('active') ? 'setup' : 'quick';
  return values;
}

function applyFormValues(values={}) {
  Object.entries(values).forEach(([id, value]) => {
    if (id === 'active_view') return;
    const el = document.getElementById(id);
    if (!el) return;
    if (el.type === 'checkbox') el.checked = Boolean(value);
    else el.value = value ?? '';
  });
  if (values.active_view === 'setup') showView('setup');
}

function settingsPayload() {
  return {
    schema: 1,
    model_root: document.getElementById('setup-model-root').value.trim(),
    environment_root: document.getElementById('setup-environment-root').value.trim() || '.mtq/environments',
    flashattention_source: document.getElementById('setup-flash-source').value.trim(),
    addon_roots: document.getElementById('setup-addon-roots').value.split(/\\r?\\n/).map(v => v.trim()).filter(Boolean),
    form_values: collectFormValues(),
  };
}

async function loadSettings() {
  const state = await api('/api/settings');
  currentSettings = state.settings;
  document.getElementById('settings-path').textContent = state.path;
  document.getElementById('setup-model-root').value = currentSettings.model_root || '';
  document.getElementById('setup-environment-root').value = currentSettings.environment_root || '.mtq/environments';
  document.getElementById('setup-flash-source').value = currentSettings.flashattention_source || '';
  document.getElementById('setup-addon-roots').value = (currentSettings.addon_roots || []).join('\\n');
  setSaveState('Settings loaded', 'ok');
  return currentSettings;
}

async function saveSettingsNow() {
  clearTimeout(saveTimer);
  setSaveState('Saving...');
  try {
    const state = await api('/api/settings', settingsPayload());
    currentSettings = state.settings;
    document.getElementById('settings-path').textContent = state.path;
    setSaveState('Settings saved', 'ok');
  } catch (error) {
    setSaveState(error.message, 'error');
    throw error;
  }
}

function scheduleSave() {
  if (!currentSettings) return;
  setSaveState('Unsaved changes');
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => saveSettingsNow().catch(() => {}), 500);
}

async function resetSettings() {
  if (!confirm('Reset all remembered UI settings?')) return;
  const state = await api('/api/settings/reset', {});
  currentSettings = state.settings;
  location.reload();
}

function exportSettings() {
  const payload = JSON.stringify(settingsPayload(), null, 2);
  const url = URL.createObjectURL(new Blob([payload], {type:'application/json'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'multi-turboquant-ui-settings.json';
  link.click();
  URL.revokeObjectURL(url);
}

async function importSettings(input) {
  const file = input.files?.[0];
  if (!file) return;
  try {
    const imported = JSON.parse(await file.text());
    await api('/api/settings', imported);
    location.reload();
  } catch (error) {
    setSaveState(`Import failed: ${error.message}`, 'error');
  } finally {
    input.value = '';
  }
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[ch]));
}

function familyTag(family) {
  const tags = {turboquant:'tag-turbo',tcq:'tag-tcq',isoquant:'tag-iso',planarquant:'tag-planar',rotorquant:'tag-rotor',kvarn:'tag-kvarn',triattention:'tag-tri'};
  return `<span class="tag ${tags[family]||''}">${family}</span>`;
}

function capabilityTag(label, ok, title='') {
  const cls = ok ? 'cap-ok' : 'cap-warn';
  return `<span class="capability ${cls}" title="${escapeHtml(title)}">${escapeHtml(label)} ${ok ? 'ok' : 'missing'}</span>`;
}

function renderCapabilities(cap) {
  const el = document.getElementById('cmd-capabilities');
  if (!cap) {
    el.innerHTML = '<span class="capability">Not scanned</span>';
    return;
  }
  if (cap.error) {
    el.innerHTML = `<span class="capability cap-bad" title="${escapeHtml(cap.error)}">Scan failed</span>`;
    return;
  }
  el.innerHTML = [
    capabilityTag('Context', cap.supports_context_extension, 'RoPE scaling flags'),
    capabilityTag('YaRN', cap.supports_yarn, 'YaRN context-extension flags'),
    capabilityTag('TriAttn', cap.supports_triattention, 'Patched TriAttention flags'),
    capabilityTag('KVarN', cap.supports_kvarn, 'Godzilla KVarN cache aliases'),
    capabilityTag('DFlash', cap.supports_dflash, 'Godzilla DFlash speculative flags'),
    capabilityTag('/props', cap.supports_props_endpoint, 'llama.cpp server props endpoint'),
    capabilityTag('Gigatoken', cap.supports_gigatoken, 'Binary self-identification; source inspection is more reliable'),
  ].join('');
}

async function scanLlamaCpp() {
  const el = document.getElementById('cmd-capabilities');
  el.innerHTML = '<span class="capability">Scanning...</span>';
  llamaCppCapabilities = await api('/api/llamacpp/scan', {
    binary: document.getElementById('cmd-binary').value,
  });
  renderCapabilities(llamaCppCapabilities);
  generateCommand();
}

function renderMemoryStatus() {
  if (!hardwareStatus) return;
  const includeVram = document.getElementById('memory-include-vram').checked;
  const combined = includeVram
    ? hardwareStatus.combined_memory_gb
    : hardwareStatus.system_ram_gb;
  const label = hardwareStatus.unified_memory
    ? 'Unified capacity (not double-counted)'
    : (includeVram ? 'Combined capacity inventory' : 'System RAM capacity');
  document.getElementById('memory-status').innerHTML =
    `<div>System RAM: ${hardwareStatus.system_ram_gb} GB total, ${hardwareStatus.available_system_ram_gb} GB available</div>` +
    `<div>GPU VRAM: ${hardwareStatus.total_vram_gb} GB total, ${hardwareStatus.available_vram_gb} GB free</div>` +
    `<div>${label}: ${combined} GB</div>` +
    '<div>RAM and VRAM remain separate limits; this total is informational.</div>';
}

async function init() {
  const saved = await loadSettings();
  const [status, methods, presets] = await Promise.all([
    api('/api/status'), api('/api/methods'), api('/api/presets')
  ]);

  document.getElementById('version-info').textContent =
    `v${status.version} | ${status.platform} ${status.arch} | torch ${status.torch_version} | ${status.methods} methods | ${status.presets} presets`;
  hardwareStatus = status;

  // GPUs
  let gpuHtml = '';
  if (status.gpus.length === 0) {
    gpuHtml = '<span class="status-dot dot-red"></span>No GPUs detected';
  } else {
    status.gpus.forEach(g => {
      const freeGb = Math.max(0, g.vram_mb - g.vram_used_mb) / 1024;
      gpuHtml += `<div class="gpu-badge">${g.name} &mdash; ${(g.vram_mb/1024).toFixed(1)} GB total, ${freeGb.toFixed(1)} GB free (${g.vendor}/${g.compute})</div> `;
    });
  }
  document.getElementById('gpu-status').innerHTML = gpuHtml;

  // Library stats
  document.getElementById('lib-status').innerHTML =
    `<div style="font-size:13px;">
      <div><span class="status-dot dot-green"></span>CUDA: ${status.torch_cuda ? 'Available' : 'No'}</div>
      <div><span class="status-dot dot-green"></span>${status.gpu_count} GPU(s) &mdash; ${status.total_vram_gb} GB total</div>
      <div><span class="status-dot dot-green"></span>${status.methods} compression methods</div>
      <div><span class="status-dot dot-green"></span>${status.presets} presets</div>
    </div>`;

  // Methods table
  let mHtml = '<table><tr><th>Method</th><th>Family</th><th>Bits</th><th class="num">Ratio</th><th>Calibration</th><th>Description</th></tr>';
  methods.forEach(m => {
    const cal = m.requires_calibration ? 'Required' : '<span class="tag tag-free">Free</span>';
    const backend = m.backend_only ? '<span class="tag tag-free">Backend</span>' : '';
    mHtml += `<tr><td><b>${m.value}</b> ${backend}</td><td>${familyTag(m.family)}</td>
      <td class="num">${m.bits}</td><td class="num">${m.compression}x</td>
      <td>${cal}</td><td style="color:#8b949e">${m.description}</td></tr>`;
  });
  mHtml += '</table>';
  document.getElementById('methods-list').innerHTML = mHtml;

  // Presets
  let pHtml = '';
  presets.forEach(p => {
    pHtml += `<div class="method-row"><span><b>${p.name}</b></span><span style="color:#8b949e;font-size:12px">${p.k_method}/${p.v_method} ${p.triattention?'+ TriAttn':''}</span></div>`;
  });
  document.getElementById('presets-list').innerHTML = pHtml;

  // Populate command generator dropdowns
  const cmdK = document.getElementById('cmd-k');
  const cmdV = document.getElementById('cmd-v');
  cmdK.innerHTML = '<option value="f16">f16 (no compression)</option>';
  cmdV.innerHTML = '<option value="f16">f16 (no compression)</option>';
  methods.forEach(m => {
    if (m.family === 'triattention' || m.value === 'f16') return;
    cmdK.innerHTML += `<option value="${m.value}">${m.value} (${m.compression}x)</option>`;
    cmdV.innerHTML += `<option value="${m.value}">${m.value} (${m.compression}x)</option>`;
  });
  applyFormValues(saved.form_values || {});
  renderMemoryStatus();
  persistentControls().forEach(el => {
    el.addEventListener('input', scheduleSave);
    el.addEventListener('change', scheduleSave);
  });
  ['setup-model-root', 'setup-environment-root', 'setup-flash-source']
    .forEach(id => document.getElementById(id).addEventListener('input', scheduleSave));
  document.getElementById('setup-addon-roots').addEventListener('input', () => {
    scheduleSave();
    clearTimeout(addonScanTimer);
    addonScanTimer = setTimeout(() => scanAddons().catch(() => {}), 700);
  });
  await Promise.allSettled([
    scanModels(), scanAddons(), scanEnvironments(), refreshRuntime(), refreshJobs(), refreshGodzillaJobs()
  ]);
  await generateCommand();
  statusTimer = setInterval(() => {
    refreshRuntime().catch(() => {});
    refreshJobs().catch(() => {});
    refreshGodzillaJobs().catch(() => {});
  }, 2000);
}

function renderFailure(target, error) {
  document.getElementById(target).innerHTML = `<div class="item"><span class="cap-bad">${escapeHtml(error.message)}</span></div>`;
}

function selectDiscoveredModel() {
  const selected = document.getElementById('cmd-model-select').value;
  if (!selected) return;
  document.getElementById('cmd-model').value = selected;
  document.getElementById('godzilla-gguf').value = selected;
  scheduleSave();
  generateCommand();
}

async function scanModels() {
  const target = document.getElementById('model-scan-result');
  target.innerHTML = '<div class="muted">Scanning configured model root...</div>';
  try {
    const result = await api('/api/discovery/models', {
      root: document.getElementById('setup-model-root').value,
    });
    discoveredModels = result.models;
    const selector = document.getElementById('cmd-model-select');
    const current = document.getElementById('cmd-model').value;
    selector.innerHTML = '<option value="">Select a discovered GGUF model</option>';
    result.models.filter(model => model.launchable).forEach(model => {
      const option = document.createElement('option');
      option.value = model.path;
      option.textContent = model.relative_path;
      option.selected = model.path === current;
      selector.appendChild(option);
    });
    if (!result.models.length) {
      target.innerHTML = '<div class="muted">No recognized model files found.</div>';
      return;
    }
    target.innerHTML = result.models.slice(0, 100).map(model =>
      `<div class="item"><div class="item-title"><span>${escapeHtml(model.relative_path)}</span>` +
      `<span class="status-pill ${model.launchable ? 'status-ready' : 'status-configured'}">${escapeHtml(model.format)}</span></div>` +
      `<div class="muted">${model.launchable ? 'Available in Quick Run' : 'Discovered; not launchable by llama-server'}</div></div>`
    ).join('');
  } catch (error) {
    discoveredModels = [];
    renderFailure('model-scan-result', error);
  }
}

async function scanAddons() {
  const target = document.getElementById('addon-scan-result');
  target.innerHTML = '<div class="muted">Scanning configured add-on roots...</div>';
  try {
    const roots = document.getElementById('setup-addon-roots').value
      .split(/\\r?\\n/).map(value => value.trim()).filter(Boolean);
    const result = await api('/api/discovery/addons', {roots});
    let html = '<div class="item"><div class="muted">Scanned ' +
      escapeHtml(result.scanned_directories) + ' directories under ' +
      escapeHtml(result.roots.length) + ' configured root(s), up to depth ' +
      escapeHtml(result.max_depth) + '.</div></div>';
    html += result.addons.map(addon => {
      const encodedPath = encodeURIComponent(addon.path).replace(/'/g, '%27');
      const encodedBinary = encodeURIComponent(addon.source?.preferred_binary || '').replace(/'/g, '%27');
      const encodedCalibrator = encodeURIComponent(addon.source?.calibrator || '').replace(/'/g, '%27');
      let action = '';
      if (addon.environment_profile && addon.local_source?.valid) {
        action += `<button onclick="useAddonSource('${escapeHtml(addon.environment_profile)}','${encodedPath}')">Use for ${escapeHtml(addon.environment_profile)}</button>`;
      }
      if (addon.source_profile && addon.source?.valid) {
        action += `<button class="secondary" onclick="useInformationalAddonSource('${escapeHtml(addon.source_profile)}','${encodedPath}')">Inspect source</button>`;
      }
      if (addon.kind === 'godzilla' && addon.source?.valid) {
        action += `<button onclick="useGodzillaSource('${encodedPath}','${encodedBinary}')">Use Godzilla checkout</button>`;
      }
      if (addon.kind === 'triattention' && addon.source?.valid) {
        action += `<button onclick="useTriAttentionSource('${encodedCalibrator}','${encodedPath}')">Use calibrator & dependencies</button>`;
      }
      if (addon.kind === 'domvox_triattention' && addon.source?.valid) {
        action += `<button onclick="useDomvoxSource('${encodedCalibrator}','${encodedPath}')">Use domvox calibrator</button>`;
      }
      let features = addon.kind === 'godzilla' && addon.source?.features
        ? `<div class="muted">KVarN ${addon.source.features.kvarn ? 'found' : 'missing'}; TriAttention ${addon.source.features.triattention ? 'found' : 'missing'}; Gigatoken runtime ${addon.source.features.gigatoken ? 'found' : 'missing'}; preparation script ${addon.source.features.triattention_prepare ? 'found' : 'missing'}</div>`
        : '';
      if (addon.kind === 'godzilla' && addon.source?.features) {
        features += '<div class="muted">Calibrator ' +
          (addon.source.features.bundled_calibrator ? 'bundled' : 'not bundled; external manual tool required') +
          '.</div>';
      }
      if (addon.kind === 'triattention' && addon.source?.valid) {
        features += '<div class="muted">Official scripts/calibrate.py found; output will be converted and validated for Godzilla.</div>';
      }
      if (addon.environment_profile && addon.local_source?.valid) {
        features += '<div class="muted">Managed source setup is available: the reviewed profile resolves dependencies, builds the selected checkout when needed, limits build jobs, and validates imports.</div>';
      }
      if (addon.source_profile && addon.source) {
        features += `<div class="muted">${escapeHtml(addon.source.summary || 'Informational source only; no environment is created.')}</div>`;
        const setup = addon.source.setup || {};
        features += `<div class="cap-warn">Setup mode: ${escapeHtml(setup.mode || 'informational_only')}; automatic repository execution is ${setup.automatic ? 'available' : 'disabled'}.</div>`;
      }
      const inspection = addon.source || addon.local_source;
      (inspection?.issues || []).forEach(issue => {
        features += '<div class="cap-bad">' + escapeHtml(issue) + '</div>';
      });
      const statusClass = inspection && !inspection.valid ? 'status-failed' :
        (addon.source_profile ? 'status-configured' : 'status-ready');
      const statusText = addon.source_profile
        ? (inspection?.valid ? 'informational' : 'source check failed')
        : addon.kind;
      return `<div class="item"><div class="item-title"><span>${escapeHtml(addon.name)}</span>` +
        `<span class="status-pill ${statusClass}">${escapeHtml(statusText)}</span></div>` +
        `<div class="muted">${escapeHtml(addon.path)}</div>` +
        `${addon.git_remote ? `<div class="muted">${escapeHtml(addon.git_remote)}</div>` : ''}` +
        features + (action ? `<div class="button-row">${action}</div>` : '') + '</div>';
    }).join('');
    (result.warnings || []).forEach(warning => {
      html += '<div class="item cap-warn">' + escapeHtml(warning) + '</div>';
    });
    result.errors.forEach(error => { html += `<div class="item cap-bad">${escapeHtml(error)}</div>`; });
    target.innerHTML = html;
  } catch (error) {
    renderFailure('addon-scan-result', error);
  }
}

function useAddonSource(profile, encodedPath) {
  document.getElementById('env-source-profile').value = profile;
  document.getElementById('env-local-source').value = decodeURIComponent(encodedPath);
  scheduleSave();
  scanEnvironments();
}

function useInformationalAddonSource(profile, encodedPath) {
  document.getElementById('addon-source-profile').value = profile;
  document.getElementById('addon-source-path').value = decodeURIComponent(encodedPath);
  scheduleSave();
  inspectSelectedAddonSource();
}

async function inspectSelectedAddonSource() {
  const target = document.getElementById('addon-source-result');
  target.innerHTML = '<div class="muted">Inspecting source markers...</div>';
  try {
    const result = await api('/api/discovery/addon-source', {
      profile: document.getElementById('addon-source-profile').value,
      path: document.getElementById('addon-source-path').value,
    });
    const statusClass = result.valid ? 'status-configured' : 'status-failed';
    const setup = result.setup || {};
    target.innerHTML = `<div class="item"><div class="item-title"><span>${escapeHtml(result.name)}</span>` +
      `<span class="status-pill ${statusClass}">${escapeHtml(result.status)}</span></div>` +
      `<div class="muted">${escapeHtml(result.path)}</div>` +
      `<div class="muted">${escapeHtml(result.summary)}</div>` +
      `${result.source_url ? `<div class="muted"><a href="${escapeHtml(result.source_url)}" target="_blank" rel="noreferrer">Open upstream</a></div>` : ''}` +
      `${result.git_remote ? `<div class="muted">${escapeHtml(result.git_remote)}</div>` : ''}` +
      `<div class="cap-warn">Setup mode: ${escapeHtml(setup.mode || 'informational_only')}; automatic repository execution is ${setup.automatic ? 'available' : 'disabled'}.</div>` +
      `${(setup.requirements || []).map(item => `<div class="muted">Requirement: ${escapeHtml(item)}</div>`).join('')}` +
      `${(setup.next_steps || []).map(item => `<div class="muted">Next: ${escapeHtml(item)}</div>`).join('')}` +
      `${Object.entries(result.marker_groups || {}).map(([marker, present]) =>
        `<div class="${present ? 'muted' : 'cap-bad'}">${present ? 'Found' : 'Missing'}: ${escapeHtml(marker)}</div>`
      ).join('')}` +
      `${(result.issues || []).map(issue => `<div class="cap-bad">${escapeHtml(issue)}</div>`).join('')}</div>`;
  } catch (error) {
    renderFailure('addon-source-result', error);
  }
}

function useGodzillaSource(encodedPath, encodedBinary) {
  const checkout = decodeURIComponent(encodedPath);
  const binary = decodeURIComponent(encodedBinary);
  document.getElementById('godzilla-checkout').value = checkout;
  document.getElementById('cmd-profile').value = 'godzilla';
  if (binary) document.getElementById('cmd-binary').value = binary;
  const quickModel = document.getElementById('cmd-model').value;
  if (quickModel && !document.getElementById('godzilla-gguf').value) {
    document.getElementById('godzilla-gguf').value = quickModel;
  }
  scheduleSave();
  generateCommand();
}

function useTriAttentionSource(encodedCalibrator, encodedPath) {
  document.getElementById('godzilla-mode').value = 'official_python';
  document.getElementById('godzilla-calibrator').value = decodeURIComponent(encodedCalibrator);
  document.getElementById('env-source-profile').value = 'triattention';
  document.getElementById('env-local-source').value = decodeURIComponent(encodedPath);
  scheduleSave();
  scanEnvironments();
}

function useDomvoxSource(encodedCalibrator, encodedPath) {
  document.getElementById('godzilla-mode').value = 'domvox';
  document.getElementById('godzilla-tokenizer').value = 'transformers';
  document.getElementById('godzilla-calibrator').value = decodeURIComponent(encodedCalibrator);
  document.getElementById('addon-source-profile').value = 'domvox_triattention';
  document.getElementById('addon-source-path').value = decodeURIComponent(encodedPath);
  scheduleSave();
  inspectSelectedAddonSource();
}

async function scanFlashAttention() {
  const target = document.getElementById('addon-scan-result');
  target.innerHTML = '<div class="muted">Inspecting FlashAttention source...</div>';
  try {
    const result = await api('/api/discovery/flashattention', {
      path: document.getElementById('setup-flash-source').value,
    });
    const status = result.valid ? 'status-ready' : 'status-failed';
    target.innerHTML = `<div class="item"><div class="item-title"><span>FlashAttention source</span>` +
      `<span class="status-pill ${status}">${result.valid ? 'valid' : 'invalid'}</span></div>` +
      `<div class="muted">${escapeHtml(result.path)}</div>` +
      `${result.version ? `<div class="muted">Version ${escapeHtml(result.version)}</div>` : ''}` +
      `${(result.issues || []).map(issue => `<div class="cap-bad">${escapeHtml(issue)}</div>`).join('')}</div>`;
  } catch (error) {
    renderFailure('addon-scan-result', error);
  }
}

async function scanEnvironments() {
  const target = document.getElementById('environment-result');
  target.innerHTML = '<div class="muted">Inspecting dependency profiles...</div>';
  try {
    const result = await api('/api/environments/scan', {
      root: document.getElementById('setup-environment-root').value,
      cuda_toolkit: document.getElementById('env-cuda-toolkit').value,
      local_source_profile: document.getElementById('env-source-profile').value,
      local_source: document.getElementById('env-local-source').value,
      manual_dependency_override: document.getElementById('env-manual-override').checked,
    });
    const toolkitVersion = result.context.cuda_toolkit_version?.join('.') || 'not detected';
    const toolkitRoot = result.context.cuda_toolkit_root || 'PATH/default';
    const contextHtml = `<div class="item"><div class="item-title"><span>Selected build context</span>` +
      `<span class="status-pill status-configured">CUDA ${escapeHtml(toolkitVersion)}</span></div>` +
      `<div class="muted">${escapeHtml(toolkitRoot)}</div></div>`;
    target.innerHTML = contextHtml + result.profiles.map(profile => {
      const severityRank = {error: 0, warning: 1, info: 2};
      const issueHtml = [...profile.issues]
        .sort((left, right) => severityRank[left.severity] - severityRank[right.severity])
        .map(issue => {
          const cls = issue.severity === 'error'
            ? 'cap-bad'
            : (issue.severity === 'warning' ? 'cap-warn' : 'muted');
          return '<div class="' + cls + '">' + escapeHtml(issue.message) + '</div>';
        }).join('');
      const canCreate = profile.ready && !['installed', 'manual', 'blocked'].includes(profile.status);
      const action = profile.status === 'blocked'
        ? '<span class="muted">Informational only; automatic installation is intentionally unavailable.</span>'
        : `<button ${canCreate ? '' : 'disabled'} onclick="createEnvironment('${escapeHtml(profile.id)}')">${profile.status === 'broken' ? 'Repair' : 'Create'}</button>`;
      return `<div class="item"><div class="item-title"><span>${escapeHtml(profile.name)}</span>` +
        `<span class="status-pill status-${escapeHtml(profile.status)}">${escapeHtml(profile.status)}</span></div>` +
        `<div class="muted">${escapeHtml(profile.target)}</div>` +
        `${issueHtml}` +
        `<div class="button-row">${action}` +
        `${profile.local_source_selected ? `<span class="muted">Local checkout: ${escapeHtml(profile.local_source_selected)}</span>` : ''}` +
        `${profile.source_build_available ? '<span class="muted">Reviewed source build available</span>' : ''}</div></div>`;
    }).join('');
  } catch (error) {
    renderFailure('environment-result', error);
  }
}

async function createEnvironment(profile) {
  if (!document.getElementById('env-confirm').checked) {
    alert('Confirm downloads/builds before creating an environment.');
    return;
  }
  try {
    const sourceProfile = document.getElementById('env-source-profile').value;
    await api('/api/environments/create', {
      profile,
      root: document.getElementById('setup-environment-root').value,
      python: document.getElementById('env-python').value,
      cuda_toolkit: document.getElementById('env-cuda-toolkit').value,
      local_source_profile: sourceProfile === profile ? sourceProfile : '',
      local_source: sourceProfile === profile ? document.getElementById('env-local-source').value : '',
      build_from_source: document.getElementById('env-build-source').checked,
      max_jobs: document.getElementById('env-max-jobs').value,
      confirm: true,
    });
    document.getElementById('env-confirm').checked = false;
    await refreshJobs();
  } catch (error) {
    alert(`Environment creation failed: ${error.message}`);
  }
}

function godzillaPayload() {
  return {
    checkout: document.getElementById('godzilla-checkout').value,
    gguf: document.getElementById('godzilla-gguf').value,
    mode: document.getElementById('godzilla-mode').value,
    attention_implementation: document.getElementById('godzilla-attention').value,
    tokenizer_backend: document.getElementById('godzilla-tokenizer').value,
    python: document.getElementById('godzilla-python').value,
    calibrator: document.getElementById('godzilla-calibrator').value,
    calibration_input: document.getElementById('godzilla-input').value,
    official_stats_input: document.getElementById('godzilla-official-stats').value,
    hf_model: document.getElementById('godzilla-hf-model').value,
    output: document.getElementById('godzilla-output').value,
    n_tokens: document.getElementById('godzilla-tokens').value,
    device: document.getElementById('godzilla-device').value,
    dependency_override: document.getElementById('godzilla-dependency-override').checked,
    allow_long_calibration: document.getElementById('godzilla-long-calibration').checked,
    domvox_accept_lossy: document.getElementById('godzilla-domvox-lossy').checked,
  };
}

function renderGodzillaPlan(plan) {
  const target = document.getElementById('godzilla-plan');
  const status = plan.ready ? 'ready' : 'blocked';
  const repairAction = plan.dependency_repair?.available
    ? `<div class="button-row"><button onclick="repairTriAttentionEnvironment()">Repair TriAttention dependencies</button>` +
      `<span class="muted">${escapeHtml(plan.dependency_repair.message)}</span></div>`
    : (plan.dependency_repair?.needed
      ? `<div class="cap-warn">${escapeHtml(plan.dependency_repair.message)}</div>`
      : '');
  target.innerHTML = `<div class="item"><div class="item-title"><span>TriAttention preparation</span>` +
    `<span class="status-pill status-${status}">${status}</span></div>` +
    `<div class="muted">Mode: ${escapeHtml(plan.mode)}</div>` +
    `<div class="muted">Tokenizer: ${escapeHtml(plan.tokenizer_backend || 'transformers')}</div>` +
    `<div class="muted">Python: ${escapeHtml(plan.python || 'not found')}</div>` +
    `<div class="muted">Output: ${escapeHtml(plan.output)}</div>` +
    (plan.resource_policy ? `<div class="muted">${escapeHtml(plan.resource_policy.message)}</div>` : '') +
    (plan.official_stats ? `<div class="muted">Official .pt stats: ${escapeHtml(plan.official_stats)}</div>` : '') +
    (plan.issues || []).map(issue =>
      `<div class="${issue.severity === 'error' ? 'cap-bad' : 'muted'}">${escapeHtml(issue.message)}</div>`
    ).join('') +
    repairAction +
    `${plan.command?.length ? `<div class="result-box">${escapeHtml(plan.command.join(' '))}</div>` : ''}</div>`;
}

function useGigatokenPython(encodedPath) {
  document.getElementById('godzilla-mode').value = 'official_python';
  document.getElementById('godzilla-python').value = decodeURIComponent(encodedPath);
  document.getElementById('godzilla-tokenizer').value = 'gigatoken';
  scheduleSave();
  planGodzilla();
}

async function scanGigatoken() {
  const target = document.getElementById('gigatoken-scan');
  target.innerHTML = '<div class="muted">Inspecting bounded Python, managed, and pyenv locations...</div>';
  try {
    const result = await api('/api/tokenizers/gigatoken/scan', {});
    if (!result.interpreters.length) {
      target.innerHTML = '<div class="muted">No Python interpreters found in the bounded locations.</div>';
      return;
    }
    target.innerHTML = result.interpreters.map(item => {
      const status = item.compatible ? 'ready' : (item.available ? 'configured' : 'blocked');
      const encoded = encodeURIComponent(item.python).replace(/'/g, '%27');
      const action = item.compatible
        ? `<button onclick="useGigatokenPython('${encoded}')">Use for calibration</button>`
        : '';
      return `<div class="item"><div class="item-title"><span>${escapeHtml(item.python)}</span>` +
        `<span class="status-pill status-${status}">${item.compatible ? 'compatible' : (item.available ? 'unreviewed' : 'not installed')}</span></div>` +
        `<div class="muted">Sources: ${escapeHtml((item.sources || []).join(', '))}; version: ${escapeHtml(item.version || 'none')}</div>` +
        `${item.error ? `<div class="${item.available ? 'cap-warn' : 'muted'}">${escapeHtml(item.error)}</div>` : ''}` +
        `<div class="button-row">${action}</div></div>`;
    }).join('');
  } catch (error) {
    renderFailure('gigatoken-scan', error);
  }
}

async function repairTriAttentionEnvironment() {
  if (!confirm('Repair the managed TriAttention environment? This may download packages and run reviewed package builds with the configured job limit.')) return;
  try {
    const job = await api('/api/environments/repair-triattention', {
      confirm: true,
    });
    pendingTriattentionRepairJob = job.id;
    await refreshJobs();
    alert('TriAttention dependency repair started. The plan will be checked again automatically when it finishes.');
  } catch (error) {
    alert(`TriAttention dependency repair failed: ${error.message}`);
  }
}

async function generateCalibrationInput() {
  if (!confirm('Generate deterministic generic calibration text inside the saved model root? Existing unrelated files will not be overwritten.')) return;
  try {
    const result = await api('/api/godzilla/calibration-text', {
      n_tokens: document.getElementById('godzilla-tokens').value,
      confirm: true,
    });
    document.getElementById('godzilla-input').value = result.path;
    scheduleSave();
    alert(result.reused
      ? 'Existing generated calibration text selected.'
      : `Generated ${result.characters} characters of generic calibration text.`);
  } catch (error) {
    alert(`Calibration text generation failed: ${error.message}`);
  }
}

async function planGodzilla() {
  const target = document.getElementById('godzilla-plan');
  target.innerHTML = '<div class="muted">Checking checkout and prerequisites...</div>';
  try {
    renderGodzillaPlan(await api('/api/godzilla/plan', godzillaPayload()));
  } catch (error) {
    renderFailure('godzilla-plan', error);
  }
}

async function startGodzilla() {
  if (!document.getElementById('godzilla-confirm').checked) {
    alert('Confirm model load/download and TriAttention preparation first.');
    return;
  }
  try {
    const payload = godzillaPayload();
    payload.confirm = true;
    await api('/api/godzilla/create', payload);
    document.getElementById('godzilla-confirm').checked = false;
    await refreshGodzillaJobs();
  } catch (error) {
    alert(`Godzilla preparation failed: ${error.message}`);
  }
}

async function refreshGodzillaJobs() {
  const result = await api('/api/godzilla/jobs');
  const target = document.getElementById('godzilla-jobs');
  if (!result.jobs.length) {
    target.innerHTML = '<div class="muted">No Godzilla jobs.</div>';
    return;
  }
  target.innerHTML = result.jobs.map(job =>
    `<div class="item"><div class="item-title"><span>TriAttention: ${escapeHtml(job.model)}</span>` +
    `<span class="status-pill status-${escapeHtml(job.status)}">${escapeHtml(job.status)}</span></div>` +
    `<div class="muted">${escapeHtml(job.output)}</div>` +
    `${job.error ? `<div class="cap-bad">${escapeHtml(job.error)}</div>` : ''}` +
    `${job.report ? `<pre class="muted">${escapeHtml(JSON.stringify(job.report, null, 2))}</pre>` : ''}` +
    `${job.log?.length ? `<div class="result-box">${escapeHtml(job.log.slice(-40).join('\\n'))}</div>` : ''}</div>`
  ).join('');
}

async function refreshJobs() {
  const result = await api('/api/environments/jobs');
  const target = document.getElementById('environment-jobs');
  if (!result.jobs.length) {
    target.innerHTML = '<div class="muted">No environment jobs.</div>';
    return;
  }
  target.innerHTML = result.jobs.map(job =>
    `<div class="item"><div class="item-title"><span>${escapeHtml(job.profile)}</span>` +
    `<span class="status-pill status-${escapeHtml(job.status)}">${escapeHtml(job.status)}</span></div>` +
    `<div class="muted">${escapeHtml(job.target)}</div>` +
    `${job.error ? `<div class="cap-bad">${escapeHtml(job.error)}</div>` : ''}` +
    `${job.report ? `<pre class="muted">${escapeHtml(JSON.stringify(job.report, null, 2))}</pre>` : ''}` +
    `${job.log?.length ? `<div class="result-box">${escapeHtml(job.log.slice(-40).join('\\n'))}</div>` : ''}</div>`
  ).join('');
  if (pendingTriattentionRepairJob) {
    const repairJob = result.jobs.find(job => job.id === pendingTriattentionRepairJob);
    if (repairJob && ['completed', 'failed'].includes(repairJob.status)) {
      pendingTriattentionRepairJob = null;
      await planGodzilla();
      alert(repairJob.status === 'completed'
        ? 'TriAttention dependency repair completed and the plan was checked again.'
        : `TriAttention dependency repair failed: ${repairJob.error || 'see the job log for details'}`);
    }
  }
}

function renderRuntime(status) {
  const summary = document.getElementById('runtime-summary');
  const start = document.getElementById('runtime-start');
  const stop = document.getElementById('runtime-stop');
  summary.textContent = status.running ? `Running as PID ${status.pid}` :
    (status.returncode === null ? 'Not running' : `Stopped with exit code ${status.returncode}`);
  summary.className = `muted ${status.running ? 'status-ready' : ''}`;
  start.disabled = status.running;
  stop.disabled = !status.running;
  document.getElementById('runtime-log').textContent = status.log?.length ? status.log.join('\\n') : 'No process output.';
}

async function refreshRuntime() {
  renderRuntime(await api('/api/runtime/status'));
}

async function startRuntime() {
  if (!confirm('Start llama-server with the current Quick Run settings?')) return;
  try {
    renderRuntime(await api('/api/runtime/start', commandPayload()));
  } catch (error) {
    alert(`Launch failed: ${error.message}`);
  }
}

async function stopRuntime() {
  if (!confirm('Stop the llama-server process started by this UI?')) return;
  try {
    renderRuntime(await api('/api/runtime/stop', {}));
  } catch (error) {
    alert(`Stop failed: ${error.message}`);
  }
}

async function runPlanner() {
  const el = document.getElementById('plan-result');
  el.style.display = 'block';
  el.textContent = 'Planning...';
  const result = await api('/api/plan', {
    model_params_b: document.getElementById('plan-model').value,
    model_quant: document.getElementById('plan-quant').value,
    agents: document.getElementById('plan-agents').value,
    context: document.getElementById('plan-context').value,
  });
  let txt = result.feasible ? 'FEASIBLE' : 'INFEASIBLE';
  txt += `\\nPreset: ${result.preset}`;
  txt += `\\nAgents: ${result.max_agents} at ${result.context_per_agent} ctx`;
  txt += `\\nKV/agent: ${result.kv_per_agent_mb?.toFixed(1)} MB`;
  txt += `\\nTotal KV: ${result.total_kv_mb?.toFixed(1)} MB`;
  txt += `\\nHeadroom: ${result.vram_headroom_mb?.toFixed(0)} MB`;
  if (result.tensor_split) txt += `\\nTensor split: ${result.tensor_split}`;
  if (result.bottleneck) txt += `\\n${result.bottleneck}`;
  el.textContent = txt;
}

async function runBenchmark() {
  const el = document.getElementById('bench-result');
  el.innerHTML = '<div style="color:#8b949e;padding:10px;">Running benchmark...</div>';
  const result = await api('/api/benchmark', {
    device: document.getElementById('bench-device').value,
    head_dim: document.getElementById('bench-headdim').value,
    seq_len: document.getElementById('bench-seqlen').value,
  });
  let html = `<div style="font-size:12px;color:#8b949e;margin-bottom:8px;">Device: ${result.device} | head_dim: ${result.head_dim} | seq_len: ${result.seq_len}</div>`;
  html += '<table><tr><th>Method</th><th class="num">Bits</th><th class="num">Ratio</th><th class="num">Encode</th><th class="num">Decode</th><th class="num">Cosine</th><th>Quality</th></tr>';
  result.results.forEach(r => {
    if (r.status !== 'ok') { html += `<tr><td>${r.method}</td><td colspan="6" style="color:#f85149">${r.status}</td></tr>`; return; }
    const barW = Math.max(0, Math.min(100, (r.cosine - 0.9) * 1000));
    html += `<tr><td><b>${r.method}</b></td><td class="num">${r.bits}</td><td class="num">${r.compression}x</td>
      <td class="num">${r.encode_ms}ms</td><td class="num">${r.decode_ms}ms</td>
      <td class="num">${r.cosine}</td><td><span class="cos-bar" style="width:${barW}px"></span></td></tr>`;
  });
  html += '</table>';
  el.innerHTML = html;
}

function commandPayload() {
  return {
    binary: document.getElementById('cmd-binary').value,
    fork_profile: document.getElementById('cmd-profile').value,
    k_method: document.getElementById('cmd-k').value,
    v_method: document.getElementById('cmd-v').value,
    model_path: document.getElementById('cmd-model').value,
    host: document.getElementById('cmd-host').value,
    port: document.getElementById('cmd-port').value,
    context: document.getElementById('cmd-ctx').value,
    gpu_layers: document.getElementById('cmd-gpu-layers').value,
    tensor_split: document.getElementById('cmd-tensor-split').value,
    parallel: document.getElementById('cmd-parallel').value,
    rope_scaling: document.getElementById('cmd-rope-scaling').value,
    rope_scale: document.getElementById('cmd-rope-scale').value,
    rope_freq_base: document.getElementById('cmd-rope-freq-base').value,
    rope_freq_scale: document.getElementById('cmd-rope-freq-scale').value,
    yarn_orig_ctx: document.getElementById('cmd-yarn-orig-ctx').value,
    yarn_ext_factor: document.getElementById('cmd-yarn-ext-factor').value,
    yarn_attn_factor: document.getElementById('cmd-yarn-attn-factor').value,
    yarn_beta_slow: document.getElementById('cmd-yarn-beta-slow').value,
    yarn_beta_fast: document.getElementById('cmd-yarn-beta-fast').value,
    spec_dflash: document.getElementById('cmd-spec-dflash').checked,
    spec_draft_model: document.getElementById('cmd-spec-draft-model').value,
    spec_draft_n_max: document.getElementById('cmd-spec-draft-n-max').value,
    spec_dflash_cross_ctx: document.getElementById('cmd-spec-cross-ctx').value,
    spec_draft_gpu_layers: document.getElementById('cmd-spec-draft-ngl').value,
    spec_branch_budget: document.getElementById('cmd-spec-branch-budget').value,
    spec_draft_cache_k: document.getElementById('cmd-spec-cache-k').value,
    spec_draft_cache_v: document.getElementById('cmd-spec-cache-v').value,
    triattention: document.getElementById('cmd-tri').checked,
    use_custom_triattention_llamacpp: document.getElementById('cmd-tri-custom').checked,
    triattention_stats_path: document.getElementById('cmd-tri-stats').value,
    triattention_budget: document.getElementById('cmd-tri-budget').value,
    triattention_window: document.getElementById('cmd-tri-window').value,
    triattention_log: document.getElementById('cmd-tri-log').checked,
    cuda_weight_share: document.getElementById('cmd-weight-share').checked,
    cuda_weight_share_library: document.getElementById('cmd-ws-library').value,
    cuda_weight_share_model_size: document.getElementById('cmd-ws-model-size').value,
    cuda_weight_share_tolerance: document.getElementById('cmd-ws-tolerance').value,
    cuda_weight_share_ipc_name: document.getElementById('cmd-ws-ipc').value,
    cuda_weight_share_shm_wait_sec: document.getElementById('cmd-ws-wait').value,
    cuda_weight_share_trace: document.getElementById('cmd-ws-trace').checked,
    cuda_weight_share_trace_depth: document.getElementById('cmd-ws-trace-depth').value,
    cuda_weight_share_trace_normal_allocs: document.getElementById('cmd-ws-trace-normal').checked,
    cuda_weight_share_suppress_master_free: document.getElementById('cmd-ws-suppress-free').checked,
  };
}

function scannerWarnings(payload) {
  if (!llamaCppCapabilities || !llamaCppCapabilities.scanned || llamaCppCapabilities.error) {
    return [];
  }
  const warnings = [];
  const wantsContext = payload.rope_scaling || payload.rope_scale ||
    payload.rope_freq_base || payload.rope_freq_scale || payload.yarn_orig_ctx ||
    payload.yarn_ext_factor || payload.yarn_attn_factor ||
    payload.yarn_beta_slow || payload.yarn_beta_fast;
  const wantsYarn = payload.rope_scaling === 'yarn' || payload.yarn_orig_ctx ||
    payload.yarn_ext_factor || payload.yarn_attn_factor ||
    payload.yarn_beta_slow || payload.yarn_beta_fast;
  if (wantsContext && !llamaCppCapabilities.supports_context_extension) {
    warnings.push('The scanned binary does not advertise RoPE context-extension flags.');
  }
  if (wantsYarn && !llamaCppCapabilities.supports_yarn) {
    warnings.push('The scanned binary does not advertise YaRN flags.');
  }
  if (payload.use_custom_triattention_llamacpp && !llamaCppCapabilities.supports_triattention) {
    warnings.push('The scanned binary does not advertise patched TriAttention flags.');
  }
  if ((payload.k_method || '').startsWith('kvarn') || (payload.v_method || '').startsWith('kvarn')) {
    if (payload.triattention || payload.use_custom_triattention_llamacpp) {
      warnings.push('Godzilla rejects KVarN with TriAttention until KVarN-aware prune is implemented.');
    }
    if (!llamaCppCapabilities.supports_kvarn) {
      warnings.push('The scanned binary does not advertise Godzilla KVarN cache aliases.');
    }
  }
  if (payload.spec_dflash && !llamaCppCapabilities.supports_dflash) {
    warnings.push('The scanned binary does not advertise DFlash speculative flags.');
  }
  return warnings;
}

async function generateCommand() {
  try {
    const payload = commandPayload();
    const result = await api('/api/command', payload);
    let txt = result.command || 'Fix errors below to generate a command.';
    if (result.issues?.length) {
      txt += '\\n\\nIssues:';
      result.issues.forEach(i => {
        txt += `\\n  [${i.severity}] ${i.message}`;
        if (i.suggestion) txt += `\\n    Fix: ${i.suggestion}`;
      });
    }
    const warnings = scannerWarnings(payload);
    if (warnings.length) {
      txt += '\\n\\nScanner warnings:';
      warnings.forEach(w => { txt += `\\n  [warning] ${w}`; });
    }
    document.getElementById('cmd-result').textContent = txt;
  } catch (error) {
    document.getElementById('cmd-result').textContent = `Could not generate command: ${error.message}`;
  }
}

init().catch(error => setSaveState(`UI initialization failed: ${error.message}`, 'error'));
</script>
</body>
</html>"""


# ─── HTTP Server ────────────────────────────────────────────────────────────────

class UIHandler(http.server.BaseHTTPRequestHandler):
    MAX_REQUEST_BYTES = 2 * 1024 * 1024

    def log_message(self, fmt, *args):
        pass

    def _json(self, data, status=200):
        payload = json.dumps(data).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)
        except ConnectionError:
            # Browsers can cancel an in-flight request when a view is refreshed,
            # navigated away from, or superseded by another scan.  The response
            # can no longer be delivered, so do not try to write a second error
            # response to the same closed socket.
            self.close_connection = True
            return False
        return True

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length < 0 or length > self.MAX_REQUEST_BYTES:
            raise ValueError("Request body is too large")
        if length == 0:
            return {}
        body = json.loads(self.rfile.read(length))
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object")
        return body

    def do_GET(self):
        try:
            path = urlparse(self.path).path
            if path in ("/", "/ui"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                    "script-src 'self' 'unsafe-inline'",
                )
                self.end_headers()
                self.wfile.write(UI_HTML.encode())
            elif path == "/api/status":
                self._json(api_status())
            elif path == "/api/methods":
                self._json(api_methods())
            elif path == "/api/presets":
                self._json(api_presets())
            elif path == "/api/settings":
                self._json(api_settings())
            elif path == "/api/runtime/status":
                self._json(api_runtime_status())
            elif path == "/api/environments/jobs":
                self._json(api_environment_jobs())
            elif path == "/api/godzilla/jobs":
                self._json(api_godzilla_jobs())
            else:
                self.send_error(404)
        except ConnectionError:
            self.close_connection = True
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            self._json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._json({"error": f"Unexpected server error: {exc}"}, status=500)

    def do_POST(self):
        try:
            body = self._read_json()
            path = urlparse(self.path).path
            routes = {
                "/api/plan": lambda: api_plan(body),
                "/api/benchmark": lambda: api_benchmark(body),
                "/api/llamacpp/scan": lambda: api_scan_llamacpp(body),
                "/api/command": lambda: api_generate_command(body),
                "/api/settings": lambda: api_save_settings(body),
                "/api/settings/reset": api_reset_settings,
                "/api/discovery/models": lambda: api_scan_models(body),
                "/api/discovery/addons": lambda: api_scan_addons(body),
                "/api/discovery/flashattention": lambda: api_scan_flashattention(body),
                "/api/discovery/addon-source": lambda: api_inspect_addon_source(body),
                "/api/environments/scan": lambda: api_scan_environments(body),
                "/api/environments/create": lambda: api_create_environment(body),
                "/api/tokenizers/gigatoken/scan": lambda: api_scan_gigatoken(body),
                "/api/environments/repair-triattention": lambda: api_repair_triattention_environment(body),
                "/api/godzilla/plan": lambda: api_plan_godzilla(body),
                "/api/godzilla/calibration-text": lambda: api_generate_calibration_text(body),
                "/api/godzilla/create": lambda: api_create_godzilla(body),
                "/api/runtime/start": lambda: api_runtime_start(body),
                "/api/runtime/stop": api_runtime_stop,
            }
            action = routes.get(path)
            if action is None:
                self.send_error(404)
                return
            self._json(action())
        except ConnectionError:
            self.close_connection = True
        except PermissionError as exc:
            self._json({"error": str(exc)}, status=403)
        except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._json({"error": f"Unexpected server error: {exc}"}, status=500)

    def do_OPTIONS(self):
        self.send_error(403)


def main():
    global SETTINGS_STORE, UI_MUTATIONS_ENABLED
    parser = argparse.ArgumentParser(description="Multi-TurboQuant Web UI")
    parser.add_argument("--port", type=int, default=9092)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--settings-file", type=Path)
    parser.add_argument(
        "--read-only",
        action="store_true",
        help=(
            "Disable settings writes, environment/calibration jobs, "
            "and model process controls"
        ),
    )
    args = parser.parse_args()
    SETTINGS_STORE = UISettingsStore(args.settings_file)
    UI_MUTATIONS_ENABLED = not args.read_only

    print(f"Multi-TurboQuant v{__version__}")
    print(f"Starting UI on http://localhost:{args.port}")
    print(f"Settings: {SETTINGS_STORE.path.resolve()}")
    if args.read_only:
        print("Mode: read-only")

    # Detect hardware
    plat = detect_platform()
    print(f"Platform: {plat.os} {plat.arch}")
    for g in plat.gpus:
        print(f"  GPU: {g.name} ({g.vram_total_mb} MB, {g.vendor}/{g.compute})")
    if not plat.gpus:
        print("  No GPUs detected (benchmarks will use CPU)")
    print()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), UIHandler)

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        MODEL_PROCESS.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
