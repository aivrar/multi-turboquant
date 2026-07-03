# Multi-TurboQuant Manual

## Table of Contents

1. [What Is Multi-TurboQuant](#1-what-is-multi-turboquant)
2. [How KV Cache Compression Works](#2-how-kv-cache-compression-works)
3. [Compression Methods](#3-compression-methods)
4. [Installation](#4-installation)
5. [Quick Start](#5-quick-start)
6. [Configuration](#6-configuration)
7. [Presets](#7-presets)
8. [Capacity Planner](#8-capacity-planner)
9. [Calibration](#9-calibration)
10. [Integration with llama.cpp](#10-integration-with-llamacpp)
11. [Integration with vLLM](#11-integration-with-vllm)
12. [Multi-GPU Setup](#12-multi-gpu-setup)
13. [Multi-Agent Deployments](#13-multi-agent-deployments)
14. [Platform Guide](#14-platform-guide)
15. [Web Dashboard](#15-web-dashboard)
16. [Benchmarking](#16-benchmarking)
17. [GPU Benchmark Results](#17-gpu-benchmark-results)
18. [Using as a Library in Your Own App](#18-using-as-a-library-in-your-own-app)
19. [Hardware Detection](#19-hardware-detection)
20. [Troubleshooting](#20-troubleshooting)
21. [Architecture Reference](#21-architecture-reference)
22. [API Reference](#22-api-reference)
23. [Attribution](#23-attribution)

---

## 1. What Is Multi-TurboQuant

Multi-TurboQuant is a unified KV cache compression toolkit for LLM inference. It combines five different compression method families under a single Python API, with a capacity planner, multi-GPU support, and cross-platform compatibility.

### The problem it solves

When a large language model generates text, it stores intermediate computations called the KV (Key-Value) cache. This cache grows linearly with context length and consumes enormous amounts of GPU memory. For a 32B parameter model at 32K context, the KV cache alone can require 8+ GB of VRAM -- often more than the model weights themselves.

This creates three practical bottlenecks:

- **VRAM limits**: You can't fit the model + context on your GPU
- **Throughput limits**: You can't serve multiple users simultaneously
- **Context limits**: You can't use the full context window the model supports

### What it does differently

Every existing KV cache compression method lives in its own repository with its own API, its own build system, and no interoperability. Multi-TurboQuant unifies them:

- **One config, any method**: Switch compression methods with a single line of code
- **Combined modes**: Apply token eviction AND quantization together for multiplicative savings (something nobody else ships)
- **Capacity planning**: Tell it your GPUs, model, and desired agents -- it calculates the optimal configuration
- **Platform-aware**: Auto-detects NVIDIA, AMD, and Apple Silicon GPUs and recommends compatible methods

### What it is NOT

- Not a fork of llama.cpp or vLLM. It is an original Python library that reimplements published compression algorithms.
- Not a model runner. It generates configuration and flags for inference engines (llama.cpp, vLLM) that do the actual GPU work.
- Not a training tool. It operates at inference time only.

---

## 2. How KV Cache Compression Works

### The KV cache

During text generation, a transformer model computes attention over all previous tokens. The Key and Value projections for each token are cached so they don't need to be recomputed. This is the KV cache.

For each token, the KV cache stores:

```
K cache: [num_layers x num_kv_heads x head_dim] values (float16 = 2 bytes each)
V cache: [num_layers x num_kv_heads x head_dim] values (float16 = 2 bytes each)
```

For a typical 32B model (64 layers, 8 KV heads, 128 head_dim):

```
Per token: 2 x 64 x 8 x 128 x 2 bytes = 262 KB
At 8K context: 262 KB x 8192 = 2.0 GB
At 32K context: 262 KB x 32768 = 8.2 GB
With 8 agents at 8K each: 2.0 GB x 8 = 16.2 GB
```

### Two approaches to compression

**Quantization** (TurboQuant, TCQ, IsoQuant, PlanarQuant): Reduce the precision of each cached value. Instead of 16 bits per value, store 2-4 bits. Every token is kept, but at lower precision.

**Token eviction** (TriAttention): Keep fewer tokens in the cache. Instead of 32K tokens at full precision, keep only the 4K most important ones. The precision stays high, but unimportant tokens are discarded.

**Combined mode**: Apply both. Evict unimportant tokens, then compress the survivors. This multiplies the savings.

### The math behind each method

**Walsh-Hadamard Transform (TurboQuant)**: Rotates each head vector using a 128-dimensional butterfly network, separates high-variance dimensions from low-variance ones, and quantizes each group at different precision. O(d log d) complexity.

**Quaternion Rotation (IsoQuant)**: Applies 4D isoclinic rotations (from Clifford algebra) to groups of 4 dimensions, then uniform-quantizes. Simpler than WHT with O(d) complexity and no calibration needed.

**Givens Rotation (PlanarQuant)**: Applies 2D rotations to pairs of dimensions using golden-angle-spaced angles, then uniform-quantizes. The simplest method, with O(d) complexity.

**Rotor Sandwich (RotorQuant)**: Groups dimensions in threes and applies an SO(3) rotation via the Clifford Cl(3,0) rotor sandwich product R·v·R̃. We materialize the sandwich as a precomputed 3×3 rotation matrix about the (1,1,1)/√3 axis at the golden angle — equivalent to the upstream rotor sandwich on pure 3-vectors. Head dims that aren't multiples of 3 (e.g. 64 and 128) are zero-padded to the next multiple and truncated on decode.

**Trellis Coded Quantization (TCQ)**: Replaces TurboQuant's nearest-centroid assignment with Viterbi-optimal path selection through a trellis. Same compression ratio, better quality.

**Trigonometric Frequency Scoring (TriAttention)**: Scores token importance by analyzing the frequency spectrum of pre-RoPE key vectors via DFT. Low-scoring tokens are evicted from the cache entirely.

---

## 3. Compression Methods

### Method comparison

| Method | Family | Bits | Compression | Calibration | Speed Impact | Best For |
|--------|--------|------|-------------|-------------|-------------|----------|
| turbo2 | TurboQuant | 2.25 | 7.1x | Required | -16% decode | Maximum VRAM savings |
| turbo3 | TurboQuant | 3.25 | 4.9x | Required | -5% decode | Balanced compression |
| turbo4 | TurboQuant | 4.25 | 3.8x | Required | -4% decode | Near-lossless quality |
| turbo2_tcq | TCQ | 2.25 | 7.1x | Required | -16% decode | Max savings + better quality |
| turbo3_tcq | TCQ | 3.25 | 4.9x | Required | -5% decode | Best quality at 5x |
| iso3 | IsoQuant | 3.25 | 4.9x | No | ~0% decode | K-only, zero speed cost |
| iso4 | IsoQuant | 4.25 | 3.8x | No | ~0% decode | Higher quality K-only |
| planar3 | PlanarQuant | 3.25 | 4.9x | No | -1% decode | Simplest, Metal support |
| planar4 | PlanarQuant | 4.25 | 3.8x | No | ~0% decode | Quality K-only |
| rotor3 | RotorQuant | 3.25 | 4.7x | No | Python-API only | Research / Cl(3,0) sibling |
| rotor4 | RotorQuant | 4.25 | 3.6x | No | ⚠️ Experimental | Research only, quality gap vs iso4 |
| triattention | TriAttention | 16 | 10-16x | Required | Varies | Long reasoning, compose with above |

### Choosing a method

**If you want zero setup**: Use `iso3` or `planar3`. No calibration files needed, no speed penalty in K-only mode.

**If you want maximum quality**: Use `turbo4` symmetric. Near-lossless at 3.8x compression. Requires calibration.

**If you want maximum VRAM savings**: Use `turbo2_tcq` symmetric (7.1x) or combine any method with TriAttention for 40-80x total.

**If you're on AMD or Mac**: Use `iso3` or `planar3`. TurboQuant requires CUDA flash attention kernels.

**If you want speed**: Use `iso3` K-only. Your benchmarks showed it can actually beat FP16 decode speed because the reduced memory bandwidth outweighs the rotation cost.

### Asymmetric configurations

You can use different methods for K and V caches. This is useful because:

- K cache benefits more from compression (attention score computation is bandwidth-bound)
- V cache quality matters more for output quality (weighted sum of values)

Common asymmetric configs:

```python
# K compressed, V full precision -- zero speed cost
CacheConfig(k_method=CacheMethod.ISO3, v_method=CacheMethod.FP16)

# K at higher compression, V at lower
CacheConfig(k_method=CacheMethod.TURBO3, v_method=CacheMethod.TURBO4)
```

---

## 4. Installation

### Requirements

- Python 3.10 or later
- PyTorch 2.1.0 or later
- An NVIDIA, AMD, or Apple Silicon GPU (for inference; the library itself runs on CPU)

### Install the library

```bash
# Clone the repository
git clone https://github.com/aivrar/multi-turboquant
cd portable-turboquant-server

# Core install (PyTorch only)
pip install -e .

# With calibration support (needed for TurboQuant/TCQ)
pip install -e ".[calibration]"

# Full install (all optional dependencies)
pip install -e ".[all]"
```

### Build llama.cpp (GPU inference)

The library generates configuration for llama.cpp, but you need to build the binary separately:

If you want to use llama.cpp for inference (optional — the library works standalone), build it from the official repo or a TurboQuant-enabled fork:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON    # NVIDIA
cmake --build build --config Release -j$(nproc)
```

### Verify installation

```python
import multi_turboquant
print(f"Version: {multi_turboquant.__version__}")
print(f"Methods: {[m.value for m in multi_turboquant.registered_methods()]}")
print(f"Presets: {len(multi_turboquant.list_presets())}")
```

---

## 5. Quick Start

### Compress and decompress KV vectors

```python
from multi_turboquant import CacheConfig, CacheMethod, compress, decompress

# Create a configuration
config = CacheConfig(k_method=CacheMethod.ISO3, v_method=CacheMethod.FP16)

# Compress key vectors
import torch
keys = torch.randn(32, 8, 128)  # [seq_len, num_heads, head_dim]
compressed = compress(keys, config, which="k")

# Decompress
reconstructed = decompress(compressed)
print(f"Shape: {reconstructed.shape}")  # [32, 8, 128]
```

### Use a named preset

```python
from multi_turboquant import get_preset

config = get_preset("balanced")       # turbo3_tcq symmetric, 5x compression
config = get_preset("k_only_iso")     # ISO3 K-only, zero speed cost
config = get_preset("extreme")        # TriAttention + turbo3_tcq, ~80x
config = get_preset("agents_8x16k")   # 8 agents at 16K context
```

### Generate a llama.cpp launch command

```python
from multi_turboquant import get_preset
from multi_turboquant.integration import get_llamacpp_command

config = get_preset("balanced")
cmd = get_llamacpp_command(
    config,
    model_path="/opt/models/Qwen2.5-32B-Q4_K_M.gguf",
    port=8080,
    tensor_split="24,12",  # dual GPU
    parallel_slots=8,      # 8 concurrent agents
)
print(" ".join(cmd))
# llama-server --model /opt/models/... --cache-type-k turbo3_tcq
#   --cache-type-v turbo3_tcq -fa on -c 131072 -ngl 99
#   --tensor-split 24,12 --parallel 8 --host 0.0.0.0 --port 8080
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
```

---

## 6. Configuration

### CacheConfig

The central configuration object:

```python
from multi_turboquant import CacheConfig, CacheMethod

config = CacheConfig(
    # Compression methods
    k_method=CacheMethod.TURBO3,      # key cache compression
    v_method=CacheMethod.TURBO3,      # value cache compression

    # TriAttention (composable with any method)
    triattention_enabled=False,
    triattention_budget=4096,          # max tokens to keep
    triattention_window=512,           # recent tokens never evicted

    # Calibration file paths
    turboquant_metadata_path=None,     # path to turboquant_kv.json
    triattention_stats_path=None,      # path to triattention_stats.pt

    # Model info (for VRAM estimation)
    head_dim=128,
    num_kv_heads=8,
    num_layers=32,
)
```

### Available cache methods

```python
from multi_turboquant import CacheMethod

# TurboQuant (Walsh-Hadamard, requires calibration)
CacheMethod.TURBO2      # 2.25-bit, 7.1x compression
CacheMethod.TURBO3      # 3.25-bit, 4.9x compression
CacheMethod.TURBO4      # 4.25-bit, 3.8x compression

# TCQ (Trellis Coded, requires calibration)
CacheMethod.TURBO2_TCQ  # 2.25-bit, better quality than turbo2
CacheMethod.TURBO3_TCQ  # 3.25-bit, better quality than turbo3

# IsoQuant (quaternion rotation, NO calibration)
CacheMethod.ISO3        # 3.25-bit
CacheMethod.ISO4        # 4.25-bit

# PlanarQuant (Givens rotation, NO calibration)
CacheMethod.PLANAR3     # 3.25-bit
CacheMethod.PLANAR4     # 4.25-bit

# RotorQuant (Cl(3,0) sandwich, NO calibration — Python API only)
CacheMethod.ROTOR3      # 3.25-bit
CacheMethod.ROTOR4      # 4.25-bit  (experimental — emits UserWarning)

# TriAttention (token eviction)
CacheMethod.TRIATTENTION

# Baselines
CacheMethod.FP16        # no compression
CacheMethod.Q8_0        # 8-bit
```

### Config properties

```python
config.is_symmetric       # True if K and V use the same method
config.is_k_only          # True if only K is compressed
config.needs_calibration  # True if any method needs calibration files
config.k_compression      # compression ratio for K cache
config.v_compression      # compression ratio for V cache
config.estimate_kv_bytes(context_length=8192)  # total KV bytes estimate
config.validate()         # returns list of warnings
```

---

## 7. Presets

Named configurations for common use cases:

### K-only presets (zero speed cost)

| Preset | Config | Use Case |
|--------|--------|----------|
| `k_only_iso` | K=iso3, V=f16 | Free 5x K compression, no calibration |
| `k_only_planar` | K=planar3, V=f16 | Simplest transform, Metal support |

### Symmetric presets (full K+V compression)

| Preset | Config | Use Case |
|--------|--------|----------|
| `balanced` | turbo3_tcq symmetric | Best quality at 5x compression |
| `speed` | turbo3 symmetric | Fastest symmetric on Ampere |
| `quality` | turbo4 symmetric | Near-lossless 3.8x |
| `max_compression` | turbo2_tcq symmetric | Maximum 7x savings |

### Combined presets (TriAttention + quantization)

| Preset | Config | Use Case |
|--------|--------|----------|
| `extreme` | turbo3_tcq + TriAttention (2K budget) | ~80x total, throughput-critical |
| `long_context` | iso3 K-only + TriAttention (4K budget) | Quality-preserving long context |

### Agent presets (multi-slot)

| Preset | Config | Use Case |
|--------|--------|----------|
| `agents_8x16k` | turbo4 symmetric | 8 agents at 16K on 32B+dual GPU |
| `agents_4x8k_70b` | turbo4 symmetric | 4 agents at 8K on 70B+dual GPU |
| `agents_8x8k_70b` | turbo2_tcq symmetric | 8 agents at 8K on 70B+dual GPU |
| `agents_16x4k` | turbo2_tcq symmetric | 16 agents, maximum parallelism |

### Calibration-free presets

| Preset | Config | Use Case |
|--------|--------|----------|
| `no_calibration_symmetric` | iso3 symmetric | No setup required |
| `no_calibration_quality` | iso4 symmetric | Higher quality, no setup |

### Auto-recommendation

```python
from multi_turboquant import recommend_preset

# Recommends based on hardware constraints
preset = recommend_preset(
    vram_gb=24,
    model_size_b=32,
    context_length=16384,
    has_calibration=True,
    priority="balanced",   # or "speed" or "quality"
)
```

---

## 8. Capacity Planner

The planner calculates exactly what fits on your hardware for a given model and agent configuration.

### Basic usage

```python
from multi_turboquant import plan_agents

result = plan_agents(
    gpus=[{"name": "RTX 3090", "vram_gb": 24}],
    model_params_b=32,
    model_quant="Q4_K_M",
    desired_agents=4,
    desired_context=8192,
)
result.print_report()
```

### Multi-GPU

```python
result = plan_agents(
    gpus=[
        {"name": "RTX 3090", "vram_gb": 24},
        {"name": "RTX 3060", "vram_gb": 12},
    ],
    model_params_b=70,
    model_quant="Q3_K_M",
    desired_agents=4,
    desired_context=8192,
)
# Automatically calculates tensor-split ratio and per-GPU VRAM allocation
```

### Platform-aware planning

```python
result = plan_agents(
    gpus=[{"name": "RX 7900 XTX", "vram_gb": 24}],
    model_params_b=32,
    model_quant="Q4_K_M",
    desired_agents=8,
    desired_context=8192,
    compute="rocm",  # AMD GPU -- only iso/planar methods available
)
```

### Scenario comparison

```python
from multi_turboquant import plan_scenarios

results = plan_scenarios(
    gpus=[{"name": "RTX 3090", "vram_gb": 24}],
    model_params_b=32,
    model_quant="Q4_K_M",
)
# Returns plans for 1, 2, 4, 8, 16 agents at various context lengths
```

### CLI planner

```bash
python scripts/plan_and_launch.py \
    --model 32 --agents 8 --context 16384 \
    --gpus 24 12 \
    --model-path /opt/models/model.gguf

# Shows capacity report and prints the exact launch command
```

### Understanding the output

The planner reports:

- **Preset**: The least aggressive compression that fits your requirements
- **KV/agent**: Estimated KV cache memory per concurrent agent
- **Total KV**: Total KV cache across all agents
- **Headroom**: Free VRAM after model weights + KV cache
- **Tensor split**: How to distribute layers across GPUs
- **Launch command**: The exact llama-server command with all flags

Note: All VRAM estimates are approximate. Actual usage depends on model architecture, batch size, and runtime buffers. The planner adds a 0.5 GB overhead buffer for runtime allocations.

---

## 9. Calibration

### Which methods need calibration

| Method Family | Calibration File | How to Generate |
|--------------|-----------------|-----------------|
| TurboQuant (turbo2/3/4) | `turboquant_kv.json` | Weight-norm analysis of safetensors |
| TCQ (turbo2_tcq/turbo3_tcq) | `turboquant_kv.json` | Same as TurboQuant |
| IsoQuant (iso3/iso4) | **None** | Parameters are baked into the algorithm |
| PlanarQuant (planar3/planar4) | **None** | Parameters are baked into the algorithm |
| TriAttention | `triattention_stats.pt` | Frequency analysis from model forward passes |

### Generating TurboQuant calibration

Requires the model's safetensors weights (not GGUF):

```bash
# CLI
mtq-calibrate /path/to/model-safetensors-dir --recipe turbo3

# Python
from multi_turboquant.calibration import generate_turboquant_metadata

metadata = generate_turboquant_metadata(
    "/path/to/model-safetensors-dir",
    recipe="turbo3",
)
# Saves turboquant_kv.json alongside the model
```

The calibration analyzes each layer's K/V projection weight matrices, computes per-dimension L2 norms, and selects the highest-variance dimensions for the high-precision group. This takes about 30 seconds for a 7B model.

### Auto-calibration

```python
from multi_turboquant import get_preset
from multi_turboquant.calibration import auto_calibrate

config = get_preset("balanced")
results = auto_calibrate(config, "/path/to/model")
# Detects which methods need calibration, generates all needed files
```

### Where calibration files are stored

By convention, calibration files are stored alongside the model:

```
/opt/models/Qwen2.5-7B-Instruct/
  config.json
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  ...
  turboquant_kv.json           <-- generated calibration
```

For GGUF models, the calibration file goes in the same directory as the GGUF file.

---

## 10. Integration with llama.cpp

### Generating flags

```python
from multi_turboquant import CacheConfig, CacheMethod
from multi_turboquant.integration import get_llamacpp_args, get_llamacpp_command

config = CacheConfig(k_method=CacheMethod.TURBO4, v_method=CacheMethod.TURBO4)

# Just the cache-related flags
args = get_llamacpp_args(config)
# ['--cache-type-k', 'turbo4', '--cache-type-v', 'turbo4', '-fa', 'on']

# Full launch command
cmd = get_llamacpp_command(
    config,
    model_path="/opt/models/model.gguf",
    port=8080,
    context_size=8192,
    tensor_split="24,12",
    parallel_slots=4,
)
```

### Supported cache types per llama.cpp fork

The available cache types depend on which llama.cpp fork you built:

| Fork | Cache Types |
|------|------------|
| atomicmilkshake/llama-cpp-turboquant | turbo2, turbo3, turbo4 + patched TriAttention flags |
| spiritbuun/buun-llama-cpp | turbo2, turbo3, turbo4, turbo2_tcq, turbo3_tcq |
| johndpope/llama-cpp-turboquant | turbo2, turbo3, turbo4, iso3, iso4, planar3, planar4 |
| ggml-org/llama.cpp (upstream) | f16, q8_0, q4_0, q5_0 (no TurboQuant) |

### Patched llama.cpp TriAttention

TriAttention is token eviction, not a K/V cache type. Keep K/V as normal cache
methods and enable patched llama.cpp mode only when using a fork that exposes
TriAttention runtime flags:

```python
config = CacheConfig(
    k_method=CacheMethod.TURBO3,
    v_method=CacheMethod.TURBO3,
    triattention_enabled=True,
    use_custom_triattention_llamacpp=True,
    triattention_stats_path="model.triattention",
    triattention_budget=4096,
    triattention_window=256,
    triattention_log=True,
)

cmd = get_llamacpp_command(config, model_path="/opt/models/model.gguf")
# ... --cache-type-k turbo3 --cache-type-v turbo3
#     --triattention-stats model.triattention
#     --triattention-budget 4096 --triattention-window 256 --triattention-log
```

For upstream llama.cpp, leave `use_custom_triattention_llamacpp=False`; the
command generator will warn that TriAttention is ignored in that backend.

The stats file is required before patched llama.cpp TriAttention can run. With
`atomicmilkshake/llama-cpp-turboquant`, generate it from representative text:

```bash
llama-cli -m /opt/models/model.gguf -ngl 99 \
  --triattention-calibrate corpus.txt \
  --triattention-calibrate-out model.triattention
```

Then put the generated `model.triattention` path into the web UI's
`TriAttention Stats Path` field or `CacheConfig.triattention_stats_path`.

### CUDA weight-share launch wrapper

For Linux + CUDA multi-process serving, you can wrap the launch command for an
external `LD_PRELOAD` helper such as `pontostroy/cuda-llm-weight-share`.
Multi-TurboQuant only generates the environment prefix; build and provide the
preload library separately.

```python
from multi_turboquant.integration import CudaWeightShareConfig

cmd = get_llamacpp_command(
    config,
    model_path="/opt/models/model.gguf",
    cuda_weight_share=CudaWeightShareConfig(
        enabled=True,
        library_path="/opt/cuda-llm-weight-share.so",
        model_size_bytes=32060375552,
        model_size_tolerance=16777216,
        ipc_name="/cuda_vram_ipc_qwen3_gpu0",
    ),
)
# env LD_PRELOAD=/opt/cuda-llm-weight-share.so MODEL_SIZE=32060375552 ...
```

Run the external helper once in reconnaissance mode (`MODEL_SIZE=0`) to find the
model weight allocation size, then reuse that value for the master and worker
processes.

### Build flags

```python
from multi_turboquant.integration.llamacpp_args import get_cmake_flags

flags = get_cmake_flags(config)
# ['-DGGML_CUDA=ON', '-DGGML_CUDA_FA=ON', '-DGGML_CUDA_FA_ALL_QUANTS=ON',
#  '-DCMAKE_CUDA_ARCHITECTURES=native']
```

---

## 11. Integration with vLLM

### Runtime monkeypatch

```python
from multi_turboquant import CacheConfig, CacheMethod
from multi_turboquant.integration import patch_vllm

config = CacheConfig(k_method=CacheMethod.TURBO3_TCQ, v_method=CacheMethod.TURBO3_TCQ)
patch_vllm(config)  # Must be called BEFORE importing vLLM attention modules

# Then start vLLM normally
# python -m vllm.entrypoints.openai.api_server --model ... --kv-cache-dtype turboquant35
```

### vLLM dtype mapping

```python
from multi_turboquant.integration.vllm_patch import get_vllm_kv_cache_dtype

dtype = get_vllm_kv_cache_dtype(config)
# "turboquant35" for turbo3/turbo3_tcq
# "isoquant3" for iso3
# "auto" for FP16
```

### Known limitations

- vLLM integration requires Linux + CUDA
- The monkeypatch modifies vLLM internals and may break with vLLM updates
- TriAttention scheduler integration is EXPERIMENTAL (scoring works, scheduler hook is manual only)
- Asymmetric K/V methods are not supported through vLLM (would need a custom attention backend)

---

## 12. Multi-GPU Setup

### How tensor splitting works

llama.cpp's `--tensor-split` distributes model layers across GPUs proportionally by VRAM. Each GPU holds the weights and KV cache for its assigned layers.

```
RTX 3090 (24 GB) + RTX 3060 (12 GB) = 36 GB total
  3090 gets 67% of layers (24/36)
  3060 gets 33% of layers (12/36)
  KV cache splits the same way
```

### Generating tensor-split flags

```python
from multi_turboquant import plan_agents

result = plan_agents(
    gpus=[{"name": "RTX 3090", "vram_gb": 24}, {"name": "RTX 3060", "vram_gb": 12}],
    model_params_b=32,
    model_quant="Q4_K_M",
    desired_agents=8,
    desired_context=16384,
)

print(result.tensor_split)  # "24,12"
# Use in: llama-server ... --tensor-split 24,12
```

### GPU ordering

nvidia-smi and CUDA sometimes enumerate GPUs in different orders. The hardware module detects this:

```python
from multi_turboquant.hardware import detect_platform

plat = detect_platform()
if plat.cuda_device_order_warning:
    print(plat.cuda_device_order_warning)
    # "GPU ordering mismatch: set CUDA_VISIBLE_DEVICES=1,0 to force largest GPU as primary"
```

### Any number of GPUs

The planner handles 1, 2, 3, or more GPUs:

```python
plan_agents(
    gpus=[
        {"name": "A100 #0", "vram_gb": 80},
        {"name": "A100 #1", "vram_gb": 80},
        {"name": "A100 #2", "vram_gb": 80},
        {"name": "A100 #3", "vram_gb": 80},
    ],
    model_params_b=70,
    desired_agents=32,
    desired_context=16384,
)
```

---

## 13. Multi-Agent Deployments

### The concept

Multiple agents (or users) sharing a single model server, each with their own KV cache and context window. llama.cpp's `--parallel N` flag enables this.

### Planning

```python
from multi_turboquant import plan_agents

# "Can I run 8 agents on a 32B model with 16K context each?"
result = plan_agents(
    gpus=[{"name": "RTX 3090", "vram_gb": 24}, {"name": "RTX 3060", "vram_gb": 12}],
    model_params_b=32,
    model_quant="Q4_K_M",
    desired_agents=8,
    desired_context=16384,
)

if result.feasible:
    print(f"Yes! Using {result.preset_name}")
    print(f"KV per agent: {result.kv_per_agent_mb:.0f} MB")
    print(f"Headroom: {result.vram_headroom_mb:.0f} MB")
```

### What's achievable on consumer hardware

Real capacity plans for RTX 3090 (24 GB) + RTX 3060 (12 GB):

| Model | Agents | Context | Preset | KV Total |
|-------|--------|---------|--------|----------|
| 32B Q4 | 8 | 16K | turbo4 | 8.5 GB |
| 32B Q4 | 16 | 4K | turbo2_tcq | 2.8 GB |
| 70B Q3 | 4 | 8K | turbo4 | 2.7 GB |
| 70B Q3 | 8 | 8K | turbo2_tcq | 2.9 GB |

---

## 14. Platform Guide

### NVIDIA (CUDA)

Full support. All 10 compression methods available. If using llama.cpp for inference:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release -j$(nproc)
```

### AMD (ROCm)

IsoQuant and PlanarQuant only (TurboQuant requires CUDA flash attention kernels).

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_HIP=ON
cmake --build build --config Release -j$(nproc)
```

The planner and compatibility checker automatically filter to available methods:

```python
from multi_turboquant.compatibility import get_recommended_config

config = get_recommended_config(platform)
# Returns iso3 K-only for AMD (best available without CUDA)
```

### Apple Silicon (Metal)

IsoQuant and PlanarQuant only. PlanarQuant has Metal kernels in johndpope's llama.cpp fork.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

### Compatibility check

```python
from multi_turboquant.compatibility import check_config
from multi_turboquant.hardware import detect_platform

platform = detect_platform()
issues = check_config(config, platform)
for issue in issues:
    print(f"[{issue.severity}] {issue.method}: {issue.message}")
    print(f"  Suggestion: {issue.suggestion}")
```

---

## 15. Web Dashboard

The included `run_ui.py` provides a browser-based dashboard for exploring the library without writing code.

### Starting the dashboard

```bash
python run_ui.py
```

Opens `http://localhost:9092` in your browser automatically. Use `--port 8080` for a different port, or `--no-browser` to suppress auto-open.

### What the dashboard shows

- **Hardware panel**: Auto-detected GPUs with VRAM, vendor, compute backend
- **Methods table**: All 10 compression methods with bits, ratios, calibration requirements
- **Presets list**: All 16 named presets
- **Capacity planner**: Enter model size, agents, context -- get instant feasibility report
- **Benchmark runner**: Run encode/decode on all methods (CPU or GPU), see cosine similarity and timing
- **Command generator**: Pick K/V methods, model path, context -- get the exact llama.cpp command to copy-paste

### How it works

The dashboard is a single Python file using stdlib `http.server`. No Flask, no npm, no build step. The HTML/CSS/JS is embedded in the Python file. It calls the same `multi_turboquant` API functions that your code would call -- `detect_gpus()`, `plan_agents()`, `get_method()`, etc.

The dashboard does not run models or serve inference. It is a configuration and planning tool.

---

## 16. Benchmarking

### CPU benchmark (synthetic)

```bash
mtq-benchmark --head-dim 128 --seq-len 1024 --device cpu
```

### GPU benchmark

```bash
bash scripts/validate_gpu.sh /opt/models/model.gguf 4096
```

### Programmatic benchmarking

```python
from multi_turboquant.benchmark import run_benchmark

results = run_benchmark(
    head_dim=128,
    seq_len=1024,
    num_heads=8,
    num_runs=3,
    device="cpu",
)
for r in results:
    print(f"{r.method}: {r.encode_time_ms:.1f}ms enc, {r.decode_time_ms:.1f}ms dec, "
          f"MSE={r.mse:.6f}, cos={r.cosine_sim:.4f}")
```

---

## 17. GPU Benchmark Results

All results from real inference on RTX 3090 + RTX 3060, using llama-server with actual model outputs.

### Qwen2.5-1.5B Q4_K_M

| Config | Decode tok/s | vs FP16 |
|--------|-------------|---------|
| FP16 baseline | 162.7 | -- |
| turbo3 K-only | 152.8 | -6.1% |
| turbo4 symmetric | 135.9 | -16.5% |
| turbo2 symmetric | 157.6 | -3.1% |

### Qwen3.5-9B Q4_K_M

| Config | Decode tok/s | vs FP16 |
|--------|-------------|---------|
| FP16 baseline | 66.1 | -- |
| turbo3 K-only | 65.6 | -0.8% |
| turbo4 symmetric | 63.3 | -4.2% |
| turbo3 symmetric | 62.9 | -4.8% |

### Cross-architecture

| Model | Architecture | FP16 tok/s | Status |
|-------|-------------|-----------|--------|
| Qwen2.5-1.5B Q4 | qwen2.5 | 162.7 | Proven |
| Qwen3.5-4B Q4 | qwen3.5 | 88.0 | Proven |
| Qwen3.5-9B Q4 | qwen3.5 | 66.1 | Proven |
| Llama-2-7B Q4 | llama | 146.3 | Proven |

### Key findings

- turbo3 K-only costs less than 1% speed on the 9B model
- turbo4 symmetric costs 4-5% speed for 3.8x compression
- Tensor split across dual GPUs works with minimal overhead
- Multi-slot (parallel 4) works for concurrent agent serving
- Calibration generation from safetensors completes in ~30 seconds

---

## 18. Using as a Library in Your Own App

### Basic integration pattern

```python
from multi_turboquant import CacheConfig, CacheMethod, get_preset
from multi_turboquant.integration import get_llamacpp_command
from multi_turboquant.hardware import detect_gpus

# Detect hardware
gpus = detect_gpus()

# Get a preset (or let the user choose)
config = get_preset("balanced")

# Generate the launch command
cmd = get_llamacpp_command(
    config,
    model_path="/opt/models/model.gguf",
    port=8080,
)

# Start the server with subprocess
import subprocess
proc = subprocess.Popen(cmd)
```

### UI dropdown population

```python
from multi_turboquant.integration import BridgeAdapter

adapter = BridgeAdapter(config)
options = adapter.get_ui_options()
# Returns list of dicts with value, label, group, needs_calibration, description
# Ready to populate a <select> element
```

### Real-time capacity display

```python
from multi_turboquant import plan_agents
from multi_turboquant.hardware import detect_gpus

def on_config_change(model_size, agents, context):
    """Called when user adjusts sliders in the UI."""
    gpus = [g.to_planner_dict() for g in detect_gpus()]
    result = plan_agents(
        gpus=gpus,
        model_params_b=model_size,
        desired_agents=agents,
        desired_context=context,
    )
    return result.to_dict()  # JSON-serializable for the frontend
```

### Auto-calibration before service start

```python
from multi_turboquant.calibration import auto_calibrate

# Call before starting inference with TurboQuant
results = auto_calibrate(config, "/opt/models/MyModel")
if results:
    print(f"Generated: {results}")
```

### Full example

See `multi_turboquant/integration/example_app_integration.py` for a complete integration example showing UI dropdown, capacity planning, auto-calibration, and command generation.

---

## 19. Hardware Detection

### Auto-detection

```python
from multi_turboquant.hardware import detect_platform, detect_gpus

# Full platform info
platform = detect_platform()
print(platform.summary())
# Platform: windows x86_64
#   WSL: available
#   GPUs: 2 (36.0 GB total)
#     [0] NVIDIA GeForce RTX 3090 -- 24576 MB (nvidia/cuda)
#     [1] NVIDIA GeForce RTX 3060 -- 12288 MB (nvidia/cuda)
#   Compute: cuda
#   llama.cpp: yes

# Just GPUs
gpus = detect_gpus()
for g in gpus:
    print(f"{g.name}: {g.vram_gb:.0f} GB ({g.vendor}/{g.compute})")
```

### GPU ordering fix

```python
from multi_turboquant.hardware import get_cuda_visible_devices_for_primary, detect_gpus

gpus = detect_gpus()
cuda_order = get_cuda_visible_devices_for_primary(gpus)
if cuda_order:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_order
    # Ensures the largest GPU is CUDA device 0
```

### Build instructions

```python
from multi_turboquant.hardware import detect_platform, get_build_instructions

platform = detect_platform()
print(get_build_instructions(platform))
# Outputs the correct git clone + cmake commands for your GPU vendor
```

---

## 20. Troubleshooting

### "unknown cache type" error from llama-server

Your llama.cpp binary doesn't support the requested cache type. Check which fork you built:

```bash
llama-server --help 2>&1 | grep cache-type
```

TurboQuant types (turbo2/3/4, turbo2_tcq/turbo3_tcq) require the spiritbuun or TheTom fork.
IsoQuant/PlanarQuant types (iso3/4, planar3/4) require johndpope's fork.

### Model loads slowly on Windows

Models on Windows drives (`/mnt/c/`, `/mnt/e/`) load 10-20x slower than models on the WSL native filesystem (`/opt/models/`). Copy models to `/opt/models/` inside WSL for production use. See `MODELS.md` for details.

### GPU ordering mismatch

nvidia-smi and CUDA sometimes enumerate GPUs differently. If the wrong GPU is being used:

```bash
export CUDA_VISIBLE_DEVICES=1,0  # swap order
```

The hardware module detects this automatically and warns you.

### Calibration fails with "no safetensors found"

TurboQuant calibration requires the original safetensors model weights, not a GGUF file. Download the safetensors version from HuggingFace, run calibration, then use the generated `turboquant_kv.json` with your GGUF file.

### vLLM patch breaks after update

The vLLM monkeypatch modifies internal APIs that can change between versions. If vLLM updates break the patch:

1. Check if `setup_vllm.sh` has been updated for the new version
2. Try the runtime monkeypatch (`patch_vllm()`) which is more resilient
3. Pin your vLLM version until compatibility is verified

### Tests skip GPU tests

GPU tests require:

- NVIDIA GPU with nvidia-smi accessible
- WSL2 with a configured distro (linbox-Multi_TQ or linbox-Llama_TQ)
- llama-server built at `/opt/llama.cpp/build/bin/llama-server`
- Test model at `/opt/models/qwen2.5-1.5b-instruct-q4_k_m.gguf`

Run `pytest tests/test_gpu.py -v` to see which prerequisites are missing.

---

## 21. Architecture Reference

### Package structure

```
multi_turboquant/
  __init__.py              Public API: compress, decompress, plan_agents, etc.
  config.py                CacheConfig, CacheMethod enums, constants
  registry.py              @register_method decorator, get_method()
  presets.py               16 named presets, recommend_preset()
  planner.py               Capacity planning for any GPU/model/agent config
  hardware.py              GPU auto-detection (NVIDIA, AMD, Metal)
  compatibility.py         Method/platform compatibility matrix

  methods/
    base.py                CompressionMethod ABC, CompressedKV, MethodInfo
    turboquant.py           WHT butterfly encode/decode (turbo2/3/4)
    tcq.py                 Viterbi trellis coding (turbo2_tcq/turbo3_tcq)
    isoquant.py            Quaternion 4D rotation (iso3/iso4)
    planarquant.py         Givens 2D rotation (planar3/planar4)
    triattention.py        DFT token eviction

  kernels/triton/
    attention_backend.py    Unified dispatch for all methods
    multi_tq_attention.py   MultiTQKVCacheManager for paged cache ops
    iso_encode_kernel.py    Vectorized iso encode/decode (no Python loops)
    planar_encode_kernel.py Vectorized planar encode/decode
    turboquant_decode.py    TurboQuant fused decode (PyTorch reference)
    turboquant_update.py    TurboQuant fused update (PyTorch reference)
    isoquant_ops.py         IsoQuant Triton kernel templates
    planarquant_ops.py      PlanarQuant Triton kernel templates
    triattention_evict.py   TriAttention scoring kernel

  calibration/
    generate_metadata.py    TurboQuant weight-norm calibration
    generate_stats.py       TriAttention frequency stats
    auto_calibrate.py       Unified calibration dispatcher

  integration/
    llamacpp_args.py        Generate llama.cpp CLI flags
    weight_share.py         CUDA LD_PRELOAD launch wrapper
    vllm_patch.py           Monkeypatch vLLM for all methods
    bridge_adapter.py       Adapter for Llama_TQ bridge apps
    example_app_integration.py  Usage examples

  benchmark/
    run_benchmark.py        Head-to-head method comparison
    perplexity.py           Quality evaluation via API
    vram_profile.py         Memory usage profiling
```

### Method registration

Methods self-register at import time via the `@register_method` decorator:

```python
@register_method(CacheMethod.ISO3)
class IsoQuant3(CompressionMethod):
    def encode(self, x, **kwargs): ...
    def decode(self, compressed, **kwargs): ...
    def packed_dim(self, head_dim): ...
    def info(self): ...
```

### Data flow

```
User config
  -> CacheConfig(k_method=..., v_method=...)
  -> get_llamacpp_command() or patch_vllm()
  -> llama-server or vLLM runs inference with compressed KV cache
  -> OpenAI-compatible API on configured port
```

---

## 22. API Reference

### Top-level functions

```python
multi_turboquant.compress(x, config, which="k", layer_idx=0)
multi_turboquant.decompress(compressed, dtype=torch.float16)
multi_turboquant.get_method(CacheMethod.TURBO3)
multi_turboquant.get_preset("balanced")
multi_turboquant.list_presets()
multi_turboquant.recommend_preset(vram_gb, model_size_b, context_length)
multi_turboquant.registered_methods()
multi_turboquant.plan_agents(gpus, model_params_b, desired_agents, desired_context)
multi_turboquant.plan_scenarios(gpus, model_params_b, model_quant)
```

### Integration functions

```python
multi_turboquant.integration.get_llamacpp_args(config)
multi_turboquant.integration.get_llamacpp_command(config, model_path, port, ...)
multi_turboquant.integration.patch_vllm(config)
multi_turboquant.integration.is_vllm_patched()
multi_turboquant.integration.BridgeAdapter(config)
```

### Hardware functions

```python
multi_turboquant.hardware.detect_gpus()
multi_turboquant.hardware.detect_platform()
multi_turboquant.hardware.get_cuda_visible_devices_for_primary(gpus)
multi_turboquant.hardware.get_build_instructions(platform)
multi_turboquant.hardware.get_recommended_engine(platform)
```

### Calibration functions

```python
multi_turboquant.calibration.generate_turboquant_metadata(model_path, recipe)
multi_turboquant.calibration.generate_triattention_stats(model, tokenizer, prompts)
multi_turboquant.calibration.auto_calibrate(config, model_path)
```

### Compatibility functions

```python
multi_turboquant.compatibility.check_config(config, platform)
multi_turboquant.compatibility.get_available_methods(platform)
multi_turboquant.compatibility.get_recommended_config(platform)
multi_turboquant.compatibility.get_cmake_flags(platform)
```

---

## 23. Attribution

Multi-TurboQuant reimplements algorithms from these repositories. All are MIT or Apache-2.0 licensed. Credit goes to these authors for the mathematical ideas and research:

| Contribution | Author / Repo | License |
|-------------|---------------|---------|
| Walsh-Hadamard Transform KV compression | TheTom / llama-cpp-turboquant | MIT |
| Trellis Coded Quantization variant | spiritbuun / buun-llama-cpp | MIT |
| IsoQuant (quaternion rotation) | scrya-com / rotorquant (ParaMind2025) | MIT |
| PlanarQuant (Givens rotation) | scrya-com / rotorquant (ParaMind2025) | MIT |
| RotorQuant (Cl(3,0) sandwich) | scrya-com / rotorquant (ParaMind2025) | MIT |
| llama.cpp iso/planar CUDA+Metal kernels | johndpope / llama-cpp-turboquant | MIT |
| TriAttention (trigonometric token eviction) | WeianMao / triattention | Apache-2.0 |

We reimplemented these algorithms in Python under a unified API. We did not fork or copy code from these repositories. This project is an original work that stands on the mathematical foundations they established.

---

*Multi-TurboQuant v0.1.0 -- MIT License*
