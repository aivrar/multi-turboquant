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
import sys
import threading
import time
import webbrowser
from urllib.parse import urlparse, parse_qs

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from multi_turboquant import (
    __version__, CacheConfig, CacheMethod,
    get_method, get_preset, list_presets, registered_methods,
    plan_agents, plan_scenarios,
)
from multi_turboquant.hardware import detect_platform, detect_gpus
from multi_turboquant.compatibility import (
    check_config, get_available_methods, get_recommended_config,
)
from multi_turboquant.config import CALIBRATION_REQUIRED, METHOD_BITS, METHOD_FAMILIES
from multi_turboquant.integration import CudaWeightShareConfig


BACKEND_ONLY_METHODS = (
    CacheMethod.KVARN2,
    CacheMethod.KVARN3,
    CacheMethod.KVARN4,
    CacheMethod.KVARN5,
    CacheMethod.KVARN6,
    CacheMethod.KVARN8,
)


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
        trace_callers=_truthy(params.get("cuda_weight_share_trace")),
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


def api_generate_command(params):
    """Generate a llama.cpp or vLLM launch command."""
    from multi_turboquant.integration import get_llamacpp_command
    config = _command_config(params)
    cuda_weight_share = _cuda_weight_share_config(params)
    fork_profile = params.get("fork_profile") or params.get("llamacpp_profile") or "upstream"
    speculative = _command_speculative_config(params)
    issues = []
    command = ""
    missing_patched_triattention_stats = (
        config.triattention_enabled
        and config.use_custom_triattention_llamacpp
        and not config.triattention_stats_path
    )
    if not missing_patched_triattention_stats:
        try:
            cmd = get_llamacpp_command(
                config,
                model_path=params.get("model_path", "/opt/models/model.gguf"),
                port=int(params.get("port", 8080)),
                context_size=int(params.get("context", 4096)),
                tensor_split=params.get("tensor_split"),
                parallel_slots=int(params["parallel"]) if params.get("parallel") else None,
                cuda_weight_share=cuda_weight_share,
                fork_profile=fork_profile,
                speculative=speculative,
            )
            command = " ".join(cmd)
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
    return {"command": command, "issues": issues}


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
.subtitle { color: #8b949e; font-size: 13px; margin-bottom: 20px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
.card h2 { color: #58a6ff; font-size: 14px; margin-bottom: 12px; text-transform: uppercase;
           letter-spacing: 0.5px; }
.card.full { grid-column: 1 / -1; }
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
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; color: #8b949e; font-weight: 600; padding: 8px 6px;
     border-bottom: 2px solid #30363d; }
td { padding: 6px; border-bottom: 1px solid #21262d; }
.num { text-align: right; font-family: 'SF Mono', Monaco, monospace; }
input, select { background: #0d1117; border: 1px solid #30363d; color: #c9d1d9;
                padding: 6px 10px; border-radius: 4px; font-size: 13px; width: 100%; }
label { font-size: 12px; color: #8b949e; display: block; margin-bottom: 4px; }
.form-row { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px;
            margin-bottom: 12px; }
button { background: #238636; color: #fff; border: none; padding: 8px 16px;
         border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; }
button:hover { background: #2ea043; }
button.secondary { background: #30363d; }
button.secondary:hover { background: #3d444d; }
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
</style>
</head>
<body>
<div class="container">
  <h1>Multi-TurboQuant</h1>
  <div class="subtitle" id="version-info">Loading...</div>

  <div class="grid">
    <!-- GPU Status -->
    <div class="card">
      <h2>Hardware</h2>
      <div id="gpu-status">Detecting...</div>
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
      <div class="form-row">
        <div style="grid-column:1/4"><label>Model Path</label>
          <input type="text" id="cmd-model" value="/opt/models/model.gguf" onchange="generateCommand()">
        </div>
        <div><label>Parallel Slots</label><input type="number" id="cmd-parallel" value="1" onchange="generateCommand()"></div>
      </div>
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
      </div>
      <div class="form-row">
        <div><label><input type="checkbox" id="cmd-tri-log" onchange="generateCommand()"> TriAttn Log</label></div>
        <div><label><input type="checkbox" id="cmd-weight-share" onchange="generateCommand()"> CUDA Weight Share</label></div>
        <div><label>MODEL_SIZE</label><input type="number" id="cmd-ws-model-size" value="" onchange="generateCommand()"></div>
        <div><label>Tolerance</label><input type="number" id="cmd-ws-tolerance" value="0" onchange="generateCommand()"></div>
      </div>
      <div class="form-row">
        <div><label>Preload Library</label><input type="text" id="cmd-ws-library" value="./cuda-llm-weight-share.so" onchange="generateCommand()"></div>
        <div><label>IPC Name</label><input type="text" id="cmd-ws-ipc" value="/cuda_vram_ipc_auto" onchange="generateCommand()"></div>
        <div><label>SHM Wait</label><input type="number" id="cmd-ws-wait" value="" onchange="generateCommand()"></div>
        <div><label><input type="checkbox" id="cmd-ws-trace" onchange="generateCommand()"> Trace</label></div>
      </div>
      <div class="result-box" id="cmd-result">Select methods above...</div>
    </div>
  </div>
</div>

<script>
const API = '';

async function api(path, body) {
  const opts = body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {};
  const r = await fetch(API + path, opts);
  return r.json();
}

function familyTag(family) {
  const tags = {turboquant:'tag-turbo',tcq:'tag-tcq',isoquant:'tag-iso',planarquant:'tag-planar',rotorquant:'tag-rotor',kvarn:'tag-kvarn',triattention:'tag-tri'};
  return `<span class="tag ${tags[family]||''}">${family}</span>`;
}

async function init() {
  const [status, methods, presets] = await Promise.all([
    api('/api/status'), api('/api/methods'), api('/api/presets')
  ]);

  document.getElementById('version-info').textContent =
    `v${status.version} | ${status.platform} ${status.arch} | torch ${status.torch_version} | ${status.methods} methods | ${status.presets} presets`;

  // GPUs
  let gpuHtml = '';
  if (status.gpus.length === 0) {
    gpuHtml = '<span class="status-dot dot-red"></span>No GPUs detected';
  } else {
    status.gpus.forEach(g => {
      gpuHtml += `<div class="gpu-badge">${g.name} &mdash; ${(g.vram_mb/1024).toFixed(0)} GB (${g.vendor}/${g.compute})</div> `;
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

async function generateCommand() {
  const result = await api('/api/command', {
    fork_profile: document.getElementById('cmd-profile').value,
    k_method: document.getElementById('cmd-k').value,
    v_method: document.getElementById('cmd-v').value,
    model_path: document.getElementById('cmd-model').value,
    context: document.getElementById('cmd-ctx').value,
    parallel: document.getElementById('cmd-parallel').value,
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
  });
  let txt = result.command || 'Fix errors below to generate a command.';
  if (result.issues?.length) {
    txt += '\\n\\nIssues:';
    result.issues.forEach(i => {
      txt += `\\n  [${i.severity}] ${i.message}`;
      if (i.suggestion) txt += `\\n    Fix: ${i.suggestion}`;
    });
  }
  document.getElementById('cmd-result').textContent = txt;
}

init();
</script>
</body>
</html>"""


# ─── HTTP Server ────────────────────────────────────────────────────────────────

class UIHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, data, status=200):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/ui"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(UI_HTML.encode())
        elif path == "/api/status":
            self._json(api_status())
        elif path == "/api/methods":
            self._json(api_methods())
        elif path == "/api/presets":
            self._json(api_presets())
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length > 0 else {}
        path = urlparse(self.path).path
        if path == "/api/plan":
            self._json(api_plan(body))
        elif path == "/api/benchmark":
            self._json(api_benchmark(body))
        elif path == "/api/command":
            self._json(api_generate_command(body))
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Multi-TurboQuant Web UI")
    parser.add_argument("--port", type=int, default=9092)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print(f"Multi-TurboQuant v{__version__}")
    print(f"Starting UI on http://localhost:{args.port}")

    # Detect hardware
    plat = detect_platform()
    print(f"Platform: {plat.os} {plat.arch}")
    for g in plat.gpus:
        print(f"  GPU: {g.name} ({g.vram_total_mb} MB, {g.vendor}/{g.compute})")
    if not plat.gpus:
        print("  No GPUs detected (benchmarks will use CPU)")
    print()

    server = http.server.HTTPServer(("127.0.0.1", args.port), UIHandler)

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
