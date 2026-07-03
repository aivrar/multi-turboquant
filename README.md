<h1 align="center">Multi-TurboQuant</h1>

<p align="center">
  <strong>Unified KV cache compression toolkit for LLM inference</strong><br>
  <em>12 methods. 16 presets. GPU-validated. One API.</em>
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
  <a href="#tests"><img src="https://img.shields.io/badge/Tests-87%20passing-brightgreen?style=flat-square" alt="Tests"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT"></a>
</p>

---

## What Is This

A Python toolkit that compresses the KV cache in large language models. The KV cache is the #1 memory bottleneck during inference — a 32B model at 32K context uses 8+ GB just for the cache. This library gives you 12 different ways to compress it, all under one API.

Install it, pick a preset, and get the exact launch command for llama.cpp or vLLM with optimal compression. Or use it directly in your own inference code.

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

**Combined mode** (unique to this repo): Token eviction + quantization together. Evict unimportant tokens, compress the survivors. ~80x total KV reduction.

All 12 methods run on GPU through our code. No upstream forks needed.

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

87 automated tests: 78 CPU + 9 GPU.

| Suite | Tests | What It Proves |
|-------|:-----:|----------------|
| `test_methods.py` | 50 | All 12 methods encode/decode, config, presets, integration |
| `test_integration.py` | 31 | Vectorized kernels, paged KV cache, dispatch, TriAttention composition |
| `test_gpu.py` | 9 | Real GPU inference, calibration generation, hardware detection |

```bash
pytest tests/                              # all 77 tests
pytest tests/ --ignore=tests/test_gpu.py   # CPU only (68 tests)
```

## Quick Start

### Pick a preset

```python
from multi_turboquant import get_preset

config = get_preset("balanced")       # turbo3_tcq symmetric, 5x
config = get_preset("k_only_iso")     # ISO3 K-only, zero speed cost, no calibration
config = get_preset("extreme")        # TriAttention + turbo3_tcq, ~80x
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

The stats file is required by the patched llama.cpp binary. Generate it with the
patched fork, for example:

```bash
llama-cli -m /opt/models/model.gguf -ngl 99 \
  --triattention-calibrate corpus.txt \
  --triattention-calibrate-out model.triattention
```

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
| `agents_8x16k` | turbo4 symmetric | 8 agents at 16K context |
| `agents_4x8k_70b` | turbo4 symmetric | 4 agents on 70B model |
| `no_calibration_symmetric` | iso3 symmetric | No setup needed |

[Full list: 16 presets](docs/manual.md#7-presets)

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
| Linux + NVIDIA | All 10 (+rotor via Python API) | llama.cpp + vLLM |
| Windows + NVIDIA | All 10 (+rotor via Python API) | llama.cpp + vLLM |
| Linux + AMD (ROCm) | iso/planar (4) + rotor (Python) | llama.cpp |
| macOS + Apple Silicon | iso/planar (4) + rotor (Python) + fused MLX kernels (ForgeAttention) | llama.cpp (Metal) + MLX |
| Any (CPU) | All 12 | Library only |

## Web Dashboard

```bash
python run_ui.py
```

Browser-based UI for exploring methods, running benchmarks, planning deployments, and generating commands. No dependencies beyond the library itself.

## Architecture

```
multi_turboquant/
  config.py              CacheConfig, CacheMethod, 12 cache types
  registry.py            Method registration and discovery
  presets.py             16 named presets + auto-recommend
  planner.py             Multi-agent capacity planning, any GPU count
  hardware.py            GPU auto-detection (NVIDIA, AMD, Metal)
  compatibility.py       Method/platform compatibility checks
  methods/               5 method families, all with encode/decode
  kernels/triton/        Attention backend, vectorized encode, dispatch
  calibration/           Weight-norm analysis, frequency stats, auto-calibrate
  integration/           llama.cpp flags, CUDA weight-share wrapper, vLLM patch
  benchmark/             Head-to-head comparison, perplexity, VRAM profiling
```

## Documentation

Full manual with 23 chapters: **[docs/manual.md](docs/manual.md)**

## Attribution

This project reimplements algorithms from published research. All original repos are MIT or Apache-2.0 licensed:

| Contribution | Source |
|-------------|--------|
| Walsh-Hadamard KV compression | TheTom/llama-cpp-turboquant |
| Trellis Coded Quantization | spiritbuun/buun-llama-cpp |
| IsoQuant / PlanarQuant / RotorQuant | scrya-com/rotorquant (ParaMind2025) |
| CUDA + Metal kernels | johndpope/llama-cpp-turboquant |
| TriAttention token eviction | WeianMao/triattention |

We reimplemented the algorithms in Python. Credit goes to these authors for the mathematical ideas.

## Community Contributors

| Contribution | Contributor | Reference |
|--------------|-------------|-----------|
| ForgeAttention — fused MLX kernels for Apple Silicon (`multi_turboquant/kernels/metal/`): packed-3-bit fused QK, tiled SV, flash decode, sparse SV with phase-1/2 early exit, per-head attention budget calibration | [@user-23xyz](https://github.com/user-23xyz) | PR [#1](https://github.com/aivrar/multi-turboquant/pull/1) · sibling project [user-23xyz/forgeattention](https://github.com/user-23xyz/forgeattention) |

The Metal path is community-maintained — the maintainer does not have Apple Silicon hardware, so issues specific to MLX/Metal should tag the contributor for context.

## License

MIT
