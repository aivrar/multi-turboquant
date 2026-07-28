<h1 align="center">Multi-TurboQuant</h1>

<p align="center">
  <strong>Unified KV cache compression toolkit for LLM inference</strong><br>
  <em>12 Python-native methods plus Godzilla KVarN backend aliases. 19 presets. One API.</em>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white" alt="Windows"></a>
  <a href="#"><img src="https://img.shields.io/badge/Platform-Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black" alt="Linux"></a>
  <a href="#"><img src="https://img.shields.io/badge/Platform-macOS-000000?style=for-the-badge&logo=apple&logoColor=white" alt="macOS"></a>
</p>
<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/NVIDIA-CUDA%20GPU-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="NVIDIA"></a>
  <a href="#"><img src="https://img.shields.io/badge/AMD-ROCm-ED1C24?style=flat-square&logo=amd&logoColor=white" alt="AMD"></a>
  <a href="#"><img src="https://img.shields.io/badge/Apple-Metal-000000?style=flat-square&logo=apple&logoColor=white" alt="Metal"></a>
  <a href="#gpu-validated-results"><img src="https://img.shields.io/badge/TurboQuant-WHT%20KV%20Cache-ff6b6b?style=flat-square" alt="TurboQuant"></a>
  <a href="#what-it-does"><img src="https://img.shields.io/badge/Methods-12%20compression-blueviolet?style=flat-square" alt="12 Methods"></a>
  <a href="#tests"><img src="https://img.shields.io/badge/Tests-182%20passing-brightgreen?style=flat-square" alt="Tests"></a>
  <a href="#isolated-add-on-environments"><img src="https://img.shields.io/badge/Add--ons-Isolated%20uv%20environments-6f42c1?style=flat-square" alt="Isolated add-on environments"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT"></a>
</p>

---

## What Is This

A Python toolkit that compresses the KV cache in large language models. The KV cache is the #1 memory bottleneck during inference — a 32B model at 32K context uses 8+ GB just for the cache. This library gives you 12 different ways to compress it, all under one API.

Install it, pick a preset, and get the exact launch command for llama.cpp or vLLM with optimal compression. Or use it directly in your own inference code.

Optional runtimes such as FastDMS, FlashAttention, LMCache, MInference, and
SageAttention are managed through `mtq-env`. Each receives its own reviewed,
locked `uv` project and virtual environment, so experimenting with an add-on
does not replace packages in the core Multi-TurboQuant environment.

```bash
git clone https://github.com/aivrar/multi-turboquant
cd multi-turboquant
pip install -e .
python run_ui.py
```

Four lines. Opens a browser dashboard. See your GPUs, benchmark methods, plan deployments, generate commands.

## Methods

| Method | Family | Transform | Bits | Compression | Calibration | Speed Impact |
|--------|--------|-----------|------|:-----------:|:-----------:|:------------:|
| `turbo2` | TurboQuant | Walsh-Hadamard 128-d | 2.25 | 7.1x | Required | -3% |
| `turbo3` | TurboQuant | Walsh-Hadamard 128-d | 3.25 | 4.9x | Required | -5% |
| `turbo4` | TurboQuant | Walsh-Hadamard 128-d | 4.25 | 3.8x | Required | -4% |
| `turbo2_tcq` | TCQ | WHT + Viterbi trellis | 2.25 | 7.1x | Required | -3% |
| `turbo3_tcq` | TCQ | WHT + Viterbi trellis | 3.25 | 4.9x | Required | -5% |
| `iso3` | IsoQuant | Quaternion 4D rotation | 3.25 | 4.9x | **No** | ~0% |
| `iso4` | IsoQuant | Quaternion 4D rotation | 4.25 | 3.8x | **No** | ~0% |
| `planar3` | PlanarQuant | Givens 2D rotation | 3.25 | 4.9x | **No** | -1% |
| `planar4` | PlanarQuant | Givens 2D rotation | 4.25 | 3.8x | **No** | ~0% |
| `rotor3` | RotorQuant | Cl(3,0) SO(3) sandwich | 3.25 | 4.7x | **No** | Python-API only |
| `rotor4` | RotorQuant | Cl(3,0) SO(3) sandwich | 4.25 | 3.6x | **No** | ⚠️ Experimental |
| `triattention` | TriAttention | DFT token eviction | 16 | 10-16x | Required | Varies |
| `kvarn2`..`kvarn8` | KVarN | Godzilla llama.cpp backend | 2-8 | 2-8x | **No** | Godzilla profile only |

**Combined mode** (unique to this repo): Token eviction + quantization together. Evict unimportant tokens, compress the survivors. ~80x total KV reduction.

The 12 Python-native methods run through this library. KVarN is exposed as a Godzilla llama.cpp command/profile extension for target KV cache types.

**Note on rotor3/rotor4:** These use a Cl(3,0) Clifford-algebra rotor sandwich product on groups of 3 dimensions (head_dim is padded to a multiple of 3 internally). They work end-to-end through the Python API on CPU or GPU, but llama.cpp and vLLM do not yet register these cache types upstream — for inference-backend use, pick `iso3`/`iso4`/`planar3`/`planar4` instead. `rotor4` is gated experimental with a runtime warning: upstream's 4-bit rotor path has known dispatch crashes, and our pure-torch implementation is untested at production scale.

## GPU-Validated Results

Every method tested on RTX 3090, real CUDA tensors, our code:

| Method | Cosine Similarity | Compression | GPU Verified |
|--------|:-----------------:|:-----------:|:------------:|
| turbo2 | 0.9420 | 5.8x | ✅ |
| turbo3 | 0.9817 | 4.0x | ✅ |
| turbo4 | 0.9947 | 3.2x | ✅ |
| turbo3_tcq | 0.9817 | 4.0x | ✅ |
| iso3 | 0.9783 | 4.7x | ✅ |
| iso4 | 0.9951 | 3.7x | ✅ |
| planar3 | 0.9783 | 4.7x | ✅ |
| planar4 | 0.9952 | 3.7x | ✅ |
| rotor3 | 0.9780 | 4.7x | ✅ |
| rotor4 | 0.9951 | 3.6x | ✅ |
| TriAttn + iso3 | 0.9782 | 9.5x | ✅ |

## Tests

193 automated test cases: 182 pass in the current Windows CPU environment and
11 hardware-specific GPU/Metal cases skip when their devices are unavailable.

| Suite | Tests | What It Proves |
|-------|:-----:|----------------|
| `test_methods.py` | 64 | All 12 methods encode/decode, config, presets, integration |
| `test_integration.py` | 34 | Vectorized kernels, paged KV cache, dispatch, TriAttention composition |
| `test_lmcache.py` | 15 | LMCache connector payloads, commands, version and input validation |
| `test_optimizations.py` | 11 | Catalog isolation, dependency checks, conflicts, platform/KV validation |
| `test_environments.py` + `test_env_cli.py` | 26 | Locked profile rendering, side-by-side CUDA selection, read-only plans, overwrite safety, opt-in creation, isolated validation |
| `test_run_ui.py` + `test_ui_workspace.py` | 29 | Command generation, rendered JavaScript, absent/default settings, bounded discovery, managed processes, and explicit environment creation |
| `test_llamacpp_scan.py` | 3 | Capability discovery and failure reporting without executing inference |
| `test_gpu.py` | 9 | Real GPU inference, calibration generation, hardware detection |
| `test_metal_fused.py` | 2 | Fused Metal path when Apple hardware is available |

```bash
pytest tests/                              # full suite
pytest tests/ --ignore=tests/test_gpu.py   # skip real-GPU validation
```

## Quick Start

### Pick a preset

```python
from multi_turboquant import get_preset

config = get_preset("balanced")       # turbo3_tcq symmetric, 5x
config = get_preset("k_only_iso")     # ISO3 K-only, zero speed cost, no calibration
config = get_preset("extreme")        # TriAttention + turbo3_tcq, ~80x
config = get_preset("godzilla_kvarn4")  # Godzilla llama.cpp KVarN extension
config = get_preset("agents_8x16k")   # 8 agents at 16K context
```

### Generate a llama.cpp command

```python
from multi_turboquant.integration import get_llamacpp_command

cmd = get_llamacpp_command(
    config,
    model_path="/opt/models/model.gguf",
    port=8080,
    tensor_split="24,12",    # dual GPU
    parallel_slots=8,        # 8 concurrent agents
)
# llama-server --model ... --cache-type-k turbo3_tcq --cache-type-v turbo3_tcq
#   -fa on -c 131072 --tensor-split 24,12 --parallel 8
```

### llama.cpp context extension

Multi-TurboQuant can generate llama.cpp launch-time RoPE and YaRN flags while
keeping KV-cache compression separate from context scaling:

```python
from multi_turboquant.integration import (
    LlamaCppContextExtensionConfig,
    get_llamacpp_command,
)

cmd = get_llamacpp_command(
    config,
    model_path="/opt/models/model.gguf",
    context_size=32768,
    context_extension=LlamaCppContextExtensionConfig(
        rope_scaling="yarn",
        rope_scale=8,
        yarn_orig_ctx=4096,
    ),
)
# ... -c 32768 --rope-scaling yarn --rope-scale 8 --yarn-orig-ctx 4096
```

These are startup flags for `llama-server`, not runtime `/props` mutations. Use
the values recommended by the model card or your own evals. The web UI includes
a binary scanner that checks whether the selected `llama-server` advertises
RoPE, YaRN, KVarN, TriAttention, speculative, and DFlash flags before you run it.

### Patched llama.cpp TriAttention

TriAttention is token eviction, not a K/V cache dtype. Upstream llama.cpp ignores
it, but patched forks such as `atomicmilkshake/llama-cpp-turboquant` expose
runtime flags:

```python
from multi_turboquant import CacheConfig, CacheMethod
from multi_turboquant.integration import get_llamacpp_command

config = CacheConfig(
    k_method=CacheMethod.TURBO3,
    v_method=CacheMethod.TURBO3,
    triattention_enabled=True,
    use_custom_triattention_llamacpp=True,
    triattention_stats_path="model.triattention",
    triattention_budget=4096,
    triattention_window=256,
)

cmd = get_llamacpp_command(config, model_path="/opt/models/model.gguf")
# ... --cache-type-k turbo3 --cache-type-v turbo3
#     --triattention-stats model.triattention --triattention-budget 4096
```

The stats file is required by the patched llama.cpp binary. Godzilla calibration
runs offline against the matching Hugging Face checkpoint using a calibrator
compatible with that fork's binary `.triattention` format, for example:

```bash
python /path/to/calibrate-triattention.py \
  --model organization/original-model \
  --n-tokens 2048 \
  --output model.triattention
```

This cannot be inferred reliably for every GGUF: calibration needs the original
model (or an exact compatible source), and the GGUF does not contain the needed
pre-RoPE query statistics. Configure the calibrator provided for your Godzilla
build and verify its supported architectures. The `mtq-triattention-stats`
command instead writes PyTorch `.pt` data for Multi-TurboQuant's Python/vLLM
path; it is not a Godzilla `.triattention` file.

### Godzilla KVarN and DFlash

KVarN is available through `atomicmilkshake/godzilla-llama.cpp`, not upstream
llama.cpp. Select the Godzilla profile explicitly:

```python
from multi_turboquant import get_preset
from multi_turboquant.integration import (
    LlamaCppProfile,
    LlamaCppSpeculativeConfig,
    get_llamacpp_command,
)

config = get_preset("godzilla_kvarn4")
cmd = get_llamacpp_command(
    config,
    model_path="/opt/models/model.gguf",
    fork_profile=LlamaCppProfile.GODZILLA,
    speculative=LlamaCppSpeculativeConfig(
        spec_type="dflash",
        draft_model="/opt/models/draft-dflash.gguf",
        draft_n_max=16,
        branch_budget=0,
        dflash_cross_ctx=512,
        draft_gpu_layers="all",
    ),
)
```

The wrapper rejects KVarN unless both K and V use KVarN, the profile is
`godzilla`, TriAttention is disabled, and `head_dim` is one of 128, 256, 384,
or 512. Draft-cache KVarN is rejected because Godzilla accepts KVarN aliases
for target cache types only.

### CUDA weight-share launcher

For multi-process serving on Linux + CUDA, the launcher can wrap commands for
`pontostroy/cuda-llm-weight-share` without vendoring the preload library:

```python
from multi_turboquant.integration import CudaWeightShareConfig, get_llamacpp_command

cmd = get_llamacpp_command(
    config,
    model_path="/opt/models/model.gguf",
    cuda_weight_share=CudaWeightShareConfig(
        enabled=True,
        library_path="/opt/cuda-llm-weight-share.so",
        model_size_bytes=32060375552,
        ipc_name="/cuda_vram_ipc_qwen3_gpu0",
    ),
)
# env LD_PRELOAD=/opt/cuda-llm-weight-share.so MODEL_SIZE=32060375552 ...
```

### Optional optimization planner and LMCache

External inference optimizations are cataloged separately from the compression
methods and remain disabled unless explicitly selected. Inspect requirements,
platform support, KV-format validation, and conflicts without importing the
third-party projects:

```bash
mtq-optimizations --engine vllm --kv-format fp16 --select lmcache
```

The LMCache integration generates its documented vLLM connector configuration
and optional multiprocess server command without launching processes or changing
the current environment:

```python
from multi_turboquant.integration import (
    LMCacheIntegrationConfig,
    build_lmcache_launch_plan,
)

plan = build_lmcache_launch_plan(LMCacheIntegrationConfig(server_l1_size_gb=20))
server_command = plan.server_command
vllm_command = plan.extend_vllm_command(["vllm", "serve", "Qwen/Qwen3-8B"])
```

This integration is currently limited to validated standard FP16, BF16, and
FP8 cache layouts. It does not claim that LMCache can serialize custom
Multi-TurboQuant or KVarN layouts. See [the optimization integration notes](docs/optimizations.md).

### Isolated add-on environments

FastDMS, FlashAttention, LMCache, MInference, and SageAttention have stricter or
mutually incompatible runtime stacks. Their dependencies remain completely
optional and are managed in separate, locked environments. Reviewed research
projects also appear in the list with an explicit reason when automatic
installation would be unsafe or incomplete:

```bash
# Read-only: shows requirements, compatibility errors, and build warnings
mtq-env list
mtq-env plan fastdms
mtq-env plan flashattention
mtq-env plan lmcache
mtq-env plan minference
mtq-env plan rocketkv  # reports its research/license block; changes nothing

# Explicitly create .mtq/environments/fastdms/{pyproject.toml,uv.lock,.venv}
mtq-env create fastdms --yes
mtq-env check fastdms

# Preview and force a local FlashAttention build when a wheel is unsuitable
mtq-env plan fastdms --build-from-source
mtq-env create fastdms --build-from-source --yes

# Run the standalone engine without activating or modifying the current environment
mtq-env run fastdms -- python -c "import fastdms; print(fastdms.__version__)"
```

`uv` must be installed to create an environment, but it is not required to
install or use Multi-TurboQuant normally. `pyenv` is optional: select one of its
interpreters with `--python /path/from/pyenv`. Profiles never install drivers,
modify the system CUDA toolkit, clone an unreviewed moving branch, or perform
privileged host installation. Native builds are announced in the read-only plan,
and creation always requires the explicit `--yes` flag. The optional
`--build-from-source` switch is available for the `flashattention` and `fastdms`
profiles. It forces a fresh local FlashAttention compilation without changing
the normal wheel-first behavior of either profile. See
[the dependency-profile table and validation record](docs/optimizations.md#built-in-profiles).

Native extensions must be compiled with the same CUDA major used by the
profile's PyTorch build. A newer NVIDIA driver may remain installed while a
matching toolkit is selected side by side:

```bash
mtq-env plan fastdms --cuda-toolkit /usr/local/cuda-12.6
mtq-env create fastdms --cuda-toolkit /usr/local/cuda-12.6 --yes
```

The Setup & Add-ons view exposes the same override. CUDA 13 `nvcc` is not used
to compile extensions for the CUDA 12.6 PyTorch profiles.

### Plan a multi-agent deployment

```python
from multi_turboquant import plan_agents

result = plan_agents(
    gpus=[{"name": "RTX 3090", "vram_gb": 24}, {"name": "RTX 3060", "vram_gb": 12}],
    model_params_b=32,
    model_quant="Q4_K_M",
    desired_agents=8,
    desired_context=16384,
)
result.print_report()
# Preset: turbo4 | 8 agents at 16K | KV: 8.5 GB | Headroom: 9 GB
```

### Compress tensors directly

```python
import torch
from multi_turboquant import compress, decompress, CacheConfig, CacheMethod

config = CacheConfig(k_method=CacheMethod.ISO3, v_method=CacheMethod.FP16)
keys = torch.randn(32, 8, 128, device="cuda")
compressed = compress(keys, config, which="k")
reconstructed = decompress(compressed)
# cosine similarity > 0.97
```

### Detect hardware

```python
from multi_turboquant.hardware import detect_platform
from multi_turboquant.compatibility import check_config, get_recommended_config

platform = detect_platform()
print(platform.summary())
# NVIDIA: all 10 methods | AMD: iso/planar only | Mac: iso/planar only

config = get_recommended_config(platform)
issues = check_config(config, platform)
```

## Presets

| Preset | Config | Use Case |
|--------|--------|----------|
| `k_only_iso` | K=iso3, V=f16 | Zero speed cost, no calibration |
| `balanced` | turbo3_tcq symmetric | Best quality at 5x |
| `speed` | turbo3 symmetric | Fastest on Ampere |
| `quality` | turbo4 symmetric | Near-lossless 3.8x |
| `max_compression` | turbo2_tcq symmetric | Maximum 7x |
| `extreme` | turbo3_tcq + TriAttention | ~80x total reduction |
| `godzilla_kvarn4` | kvarn4 symmetric | Godzilla llama.cpp extension |
| `godzilla_kvarn2_max` | kvarn2 symmetric | Aggressive Godzilla KVarN |
| `godzilla_kvarn8_quality` | kvarn8 symmetric | Quality-focused Godzilla KVarN |
| `agents_8x16k` | turbo4 symmetric | 8 agents at 16K context |
| `agents_4x8k_70b` | turbo4 symmetric | 4 agents on 70B model |
| `no_calibration_symmetric` | iso3 symmetric | No setup needed |

[Full list: 19 presets](docs/manual.md#7-presets)

## Capacity Planner

```bash
python scripts/plan_and_launch.py --model 32 --agents 8 --context 16384 --gpus 24 12
```

Works with any number of GPUs. Auto-detects NVIDIA, AMD, Apple Silicon. Generates the exact launch command with tensor-split and parallel flags.

## Calibration

TurboQuant/TCQ methods need a one-time calibration from the model's safetensors weights:

```bash
mtq-calibrate /path/to/model-safetensors --recipe turbo3
# Generates turboquant_kv.json (~200 KB, ~30 seconds)
```

IsoQuant and PlanarQuant need **no calibration** — just works.

## Platform Support

| Platform | Methods Available | Engine |
|----------|:-----------------:|--------|
| Linux + NVIDIA | Python-native methods + Godzilla KVarN profile | llama.cpp + vLLM |
| Windows + NVIDIA | Python-native methods + Godzilla KVarN profile | llama.cpp + vLLM |
| Linux + AMD (ROCm) | iso/planar (4) + rotor (Python) | llama.cpp |
| macOS + Apple Silicon | iso/planar (4) + rotor (Python) + fused MLX kernels (ForgeAttention) | llama.cpp (Metal) + MLX |
| Any (CPU) | All 12 | Library only |

## Local UI Workspace

```bash
python run_ui.py
```

The browser UI now has two focused views:

- **Quick Run** keeps hardware detection, cache-method benchmarking, capacity
  planning, presets, and llama.cpp command generation together. It can discover
  models under a configured folder and start or stop a selected GGUF model with
  the generated argument list.
- **Setup & Add-ons** stores the model, environment, add-on, and optional
  FlashAttention source folders; scans only those configured folders; and can
  create the reviewed isolated `mtq-env` dependency profiles after explicit
  confirmation.

Settings and form defaults persist in
`~/.multi-turboquant/ui-settings.json`. The server remains bound to localhost,
and neither the UI nor its settings require an npm or frontend build. See the
**[UI workspace guide](docs/ui-workspace.md)** for model formats, launch safety,
settings import/export, and command-line options.

## Architecture

```
multi_turboquant/
  config.py              CacheConfig, CacheMethod, cache type metadata
  registry.py            Method registration and discovery
  presets.py             19 named presets + auto-recommend
  planner.py             Multi-agent capacity planning, any GPU count
  hardware.py            GPU auto-detection (NVIDIA, AMD, Metal)
  compatibility.py       Method/platform compatibility checks
  optimizations/         Optional manifests, conflict planner, isolated env manager
  methods/               5 method families, all with encode/decode
  kernels/triton/        Attention backend, vectorized encode, dispatch
  calibration/           Weight-norm analysis, frequency stats, auto-calibrate
  integration/           llama.cpp flags, CUDA weight-share wrapper, vLLM patch
  benchmark/             Head-to-head comparison, perplexity, VRAM profiling
```

## Documentation

Full manual with 23 chapters: **[docs/manual.md](docs/manual.md)**

Context extension research and implementation notes:
**[docs/context-extension.md](docs/context-extension.md)**

Optional optimization catalog, isolated add-on environments, compatibility
planner, and LMCache integration:
**[docs/optimizations.md](docs/optimizations.md)**

Persistent local UI, model discovery and launching, and add-on setup:
**[docs/ui-workspace.md](docs/ui-workspace.md)**

## Attribution

This project reimplements algorithms from published research. All original repos are MIT or Apache-2.0 licensed:

| Contribution | Source |
|-------------|--------|
| Walsh-Hadamard KV compression | TheTom/llama-cpp-turboquant |
| Trellis Coded Quantization | spiritbuun/buun-llama-cpp |
| IsoQuant / PlanarQuant / RotorQuant | scrya-com/rotorquant (ParaMind2025) |
| CUDA + Metal kernels | johndpope/llama-cpp-turboquant |
| TriAttention token eviction | WeianMao/triattention |
| Godzilla llama.cpp profile, KVarN alias surface, DFlash flags | [atomicmilkshake/godzilla-llama.cpp](https://github.com/atomicmilkshake/godzilla-llama.cpp) |
| BeeLlama / DFlash lineage | [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp) |
| KVarN research and reference implementation | [huawei-csl/KVarN](https://github.com/huawei-csl/KVarN) |
| Context-extension research notes: Position Interpolation, YaRN, Resonance RoPE, LongRoPE | [llama.cpp](https://github.com/ggml-org/llama.cpp), [sheryc/resonance_rope](https://github.com/sheryc/resonance_rope), published papers |

We reimplemented the Python-native algorithms in Python. Godzilla/KVarN support
is a command-generation and compatibility integration only; context-extension
support is a llama.cpp command-generation and capability-scanning integration
only. Multi-TurboQuant does not vendor Godzilla, BeeLlama, KVarN, Resonance
RoPE, LongRoPE, or llama.cpp code.

## Community Contributors

| Contribution | Contributor | Reference |
|--------------|-------------|-----------|
| Suggested the Godzilla llama.cpp + KVarN integration and provided the issue context that shaped the backend-only profile design | [@jawadala](https://github.com/jawadala) | Issue [#9](https://github.com/aivrar/multi-turboquant/issues/9) |
| Suggested context-extension support, Resonance RoPE research, UI capability scanning, and the KVarN/TriAttention compatibility review | [@jawadala](https://github.com/jawadala) | Issue [#11](https://github.com/aivrar/multi-turboquant/issues/11) |
| Reported CUDA toolkit/profile friction, prompting side-by-side toolkit selection and clearer Godzilla calibration guidance | [@jawadala](https://github.com/jawadala) | Issue [#23](https://github.com/aivrar/multi-turboquant/issues/23) |
| Suggested the modular optimization catalog, LMCache/Maru investigation, attention alternatives, and compatibility planning | [@jawadala](https://github.com/jawadala) | Issue [#13](https://github.com/aivrar/multi-turboquant/issues/13) |
| Suggested isolated dependency handling for FastDMS, FlashAttention, and the other optional add-ons, including pyenv-compatible interpreter selection | [@jawadala](https://github.com/jawadala) | Issue [#15](https://github.com/aivrar/multi-turboquant/issues/15) |
| Suggested an explicit local source-build option for projects that depend on FlashAttention | [@jawadala](https://github.com/jawadala) | Issue [#17](https://github.com/aivrar/multi-turboquant/issues/17) |
| Suggested separating quick-run controls from advanced setup, persisting defaults, discovering models and add-ons, and exposing the recent KV, weight-sharing, and RoPE/YaRN options in the UI | [@jawadala](https://github.com/jawadala) | Issue [#19](https://github.com/aivrar/multi-turboquant/issues/19) |
| ForgeAttention — fused MLX kernels for Apple Silicon (`multi_turboquant/kernels/metal/`): packed-3-bit fused QK, tiled SV, flash decode, sparse SV with phase-1/2 early exit, per-head attention budget calibration | [@user-23xyz](https://github.com/user-23xyz) | PR [#1](https://github.com/aivrar/multi-turboquant/pull/1) · sibling project [user-23xyz/forgeattention](https://github.com/user-23xyz/forgeattention) |

Thanks to [@jawadala](https://github.com/jawadala) for the sustained issue
reports and feature suggestions that informed the Godzilla/KVarN support,
context-extension tooling, optimization catalog, and isolated dependency
system, including the practical UI workflow that brings those additions
together.

The Metal path is community-maintained — the maintainer does not have Apple Silicon hardware, so issues specific to MLX/Metal should tag the contributor for context.

## License

MIT
