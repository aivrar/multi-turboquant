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
  <a href="#tests"><img src="https://img.shields.io/badge/Tests-pytest%20suite-brightgreen?style=flat-square" alt="Tests"></a>
  <a href="#isolated-add-on-environments"><img src="https://img.shields.io/badge/Add--ons-Isolated%20uv%20environments-6f42c1?style=flat-square" alt="Isolated add-on environments"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT"></a>
</p>

---

## What Is This

A Python toolkit that compresses the KV cache in large language models. The KV cache is the #1 memory bottleneck during inference — a 32B model at 32K context uses 8+ GB just for the cache. This library gives you 12 different ways to compress it, all under one API.

Install it, pick a preset, and get the exact launch command for llama.cpp or vLLM with optimal compression. Or use it directly in your own inference code.

Optional runtimes such as FastDMS, FlashAttention, LMCache, MInference,
SageAttention, and the official TriAttention calibrator are managed through
`mtq-env`. Each receives its own reviewed,
locked `uv` project and virtual environment, so experimenting with an add-on
does not replace packages in the core Multi-TurboQuant environment.
`mtq-compose` adds a side-effect-free composition layer over the full reviewed
catalog: a complete pairwise compatibility matrix, guarded execution profiles,
deterministic workload routing with baseline fallback, benchmark provenance,
and analytical KV-capacity estimates. It never installs or launches an add-on.
The separate `mtq-godzilla-gigatoken` workflow can also prepare, build, and
qualify a revision-pinned Godzilla llama.cpp runtime with native Gigatoken
tokenization without modifying an existing checkout.
For the exact Godzilla `09214b160` compatibility baseline,
`mtq-godzilla-compose` can create a separate hash-tracked source tree with an
off-by-default, request-gated PFlash policy and bounded whole-idle-slot
KVFlash residency. It does not claim arbitrary KV-page restore or remove
Godzilla's existing TriAttention/KVarN conflict.

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

The suite is exercised on Windows and in CI containers for Debian 12 with
Python 3.11 and Debian 13 with Python 3.13. Hardware-specific CUDA, ROCm, and
Metal cases skip when their required device is unavailable.

| Suite | What It Proves |
|-------|----------------|
| `test_methods.py` | All 12 methods encode/decode, config, presets, integration |
| `test_integration.py` | Vectorized kernels, paged KV cache, dispatch, TriAttention composition |
| `test_lmcache.py` | LMCache connector payloads, commands, version and input validation |
| `test_optimizations.py` | Catalog isolation, add-on dependencies, conflicts, and platform/KV/architecture validation |
| `test_environments.py` + `test_env_cli.py` | Locked profiles, Debian detection, clean repair/rollback, redacted diagnostics, local builds, CUDA selection, overwrite safety, and isolated validation |
| `test_godzilla_workspace.py` + `test_godzilla_triattention.py` | Official and domvox calibration/conversion, fail-closed tokenizer parity, bounded compatible-interpreter discovery, final preflight, redacted failure diagnostics, length guardrails, reuse, and artifact validation |
| `test_godzilla_gigatoken.py` | Exact Godzilla source profiles, source pins, confirmation, reviewed-diff filtering, optional fixtures, build/verify boundaries, and tree integrity |
| `test_weight_share.py` | Exact source provenance, Linux build planning, ELF/symbol/dependency validation, reconnaissance rules, and safe launch configuration |
| `test_run_ui.py` + `test_ui_workspace.py` | Command generation, settings, dependency repair, discovery, source-specific setup, bounded background work, managed process cleanup, and confirmed jobs |
| `test_tokenizer_backends.py` | Exact token-ID parity, mismatch refusal, reviewed-version checks, and bounded managed/pyenv interpreter discovery |
| `test_calibration_text.py` | Deterministic offline corpus generation, bounds, concurrent publication, and incomplete-file refusal |
| Hardware suites | Host/GPU detection, real-GPU inference, and fused Metal behavior when available |

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

The stats file is required by the patched llama.cpp binary. The recommended
workflow runs the official `WeianMao/triattention` calibrator and then converts
its `.pt` payload into Godzilla's distinct v1 binary format:

```bash
mtq-godzilla-triattention calibrate \
  --calibrator /path/to/triattention/scripts/calibrate.py \
  --model organization/original-model \
  --input calibration.txt \
  --output model.triattention \
  --max-length 2048 \
  --device cuda:1 \
  --attn-implementation sdpa
```

Hugging Face tokenization remains the default. For an opt-in CPU tokenization
accelerator, add `--tokenizer-backend gigatoken`. Multi-TurboQuant supports the
reviewed Gigatoken 0.10.x API only and compares every token ID for the complete
selected calibration text against the Hugging Face tokenizer before the model
is loaded. Any mismatch stops calibration; it never silently substitutes a
different token sequence. The same guard is used by the reviewed official and
domvox Python calibrators; conversion of existing statistics does not tokenize.

The command retains the official stats as `model.official.pt`, loads the matching
Hugging Face config for layer/head/RoPE metadata, writes the Godzilla artifact
atomically, and reads it back with strict shape, index, finite-value, and file-size
validation. An existing official payload can be converted separately with
`mtq-godzilla-triattention convert`, and a finished artifact can be checked with
`mtq-godzilla-triattention inspect`.

The Setup & Add-ons view recognizes the official checkout, can build its
isolated calibration environment from that directory with `MAX_JOBS=2`, and
automatically uses a validated interpreter. When no Python is selected, it
probes at most eight candidates in a deterministic order: the owned
TriAttention environment, an active virtual/Conda environment, the current
interpreter, other managed environments, `PATH`, and conventional pyenv roots.
Each candidate is launched in isolation and must import Torch, Transformers,
Accelerate, NumPy, Safetensors, Hugging Face Hub, Tokenizers, and SentencePiece;
Gigatoken or FlashAttention is additionally required only when that option is
actually selected. Packages are never borrowed through cross-environment
`sys.path` or `site-packages` injection.

Its second official mode converts
an existing `.pt` payload without repeating the model forward pass. If the
managed interpreter is incomplete—for example, it cannot import `accelerate`—
the preparation plan offers a confirmed **Repair TriAttention dependencies**
action. It first checks that the host can create the reviewed environment, then
re-synchronizes the pinned owned profile with a conservative two-job limit,
validates every declared module, and automatically checks the preparation plan
again. The managed repair ignores unrelated Python and local-source overrides;
a manually selected Python is never modified. A known missing import cannot be
bypassed by a dependency override, and the exact selected interpreter is
checked again immediately before execution so a changed or removed environment
cannot start calibration.

If a background calibration still fails, the job view shows the full redacted
diagnostic bundle and writes an atomic JSON copy next to the requested output as
`<output>.<job-id>.diagnostics.json`. It includes the exact command and working
directory, selected and host Python paths/prefixes, per-module import results and
tracebacks, source revisions, relevant input/output path state, bounded
interpreter discovery, CUDA/VRAM and toolchain state, disk space, OS details,
log tails, and recovery guidance. Credential assignments, bearer tokens,
Hugging Face tokens, and credentials embedded in URLs are redacted.

On Linux and macOS, the managed `.venv/bin/python` entry is intentionally kept
as a lexical path. It is commonly a symlink; resolving it to the base `uv` or
system interpreter would discard the virtual environment and its installed
packages. The UI's Gigatoken scan checks a bounded set of current, `PATH`, active
virtual/Conda, managed `.mtq`, and conventional pyenv interpreter locations and
lets you select a compatible 0.10.x environment without scanning entire drives.

This route does not use `llama-cli`. A native `llama-cli` calibration choice is
not offered because the current Godzilla binary does not expose a real calibration
command. Calibration still needs the exact Hugging Face model or a compatible
source; a GGUF alone does not contain the pre-RoPE query statistics. The official
script uses `trust_remote_code=True`, so only calibrate model sources you trust.
`IQ4_XS`, `Q4_K_M`, and similar labels describe the selected GGUF's weight
encoding; they do not make the Transformers calibration load quantized. Before
weights are downloaded, the workflow now reads the matching model config,
prefers authoritative nested `rope_parameters.rope_theta` over conflicting
legacy defaults, and rejects a requested sequence longer than the model's
declared context. `--device` accepts `cuda:N`, and the UI lists each GPU and
defaults to the one with the most free VRAM. A model-config Transformers version
different from the managed runtime is reported for qualification rather than
silently treated as compatible.
The older Godzilla checkout-owned PowerShell workflow remains available as an
explicit fallback for checkouts that provide it. `mtq-triattention-stats` writes
a different `.pt` schema for Multi-TurboQuant's Python/vLLM path and cannot be
passed directly to Godzilla.

#### Optional domvox TRIA v2 adapter

The Setup view and CLI also recognize a reviewed `domvox/triattention-ggml`
checkout. Its `triattention_calibrate.py` output is a distinct TRIA v2 binary,
not a Godzilla artifact. The experimental adapter validates the TRIA header,
the required sibling `triattention_common.py`, model dimensions, RoPE metadata,
finite values, and exact file size before writing a Godzilla v1
`.triattention` file. Both calibration and conversion run under the exact
preflighted Python rather than returning to the UI's host interpreter:

```bash
mtq-godzilla-triattention domvox \
  --calibrator /path/to/triattention-ggml/triattention_calibrate.py \
  --python /path/to/calibration/python \
  --model organization/original-model \
  --input calibration.txt \
  --output model.triattention \
  --max-length 32768 \
  --device cuda \
  --tokenizer-backend gigatoken \
  --accept-lossy
```

The conversion is deliberately opt-in and lossy: Godzilla v1 has no fields
for domvox layer-budget scales or attention scale, so those fields are reported
as dropped. The upstream guide targets enough coherent text to approach its
32,768-token default; it does not establish 200,000 tokens as an optimal
calibration. Multi-TurboQuant retains 200,000 only as a global input ceiling.
The effective limit is the smaller of that ceiling and the matching model's
declared context, and anything above 32,768 still requires
`--allow-long-calibration`. The official path also estimates its retained Q
tensors, BF16 weights, and transient state and fails before model download when
that conservative floor exceeds the selected GPU's free VRAM. The estimate is
a lower bound, not a promise that a run will fit. System RAM and GPU VRAM remain
separate capacity domains. The UI can also create deterministic
offline starter text inside the saved model root without overwriting unrelated
files. Corpus files carry a schema and completion marker, and simultaneous
requests cannot clobber one another or reuse a partial file. Use representative
domain text for final quality qualification, the matching Hugging Face checkpoint
for shape and RoPE metadata, and validate retrieval quality on the target model
before relying on the result.

### Godzilla + Gigatoken runtime

`mtq-godzilla-gigatoken` now performs the reviewed runtime port requested in
issue #39. It creates a **new** combined source tree from one exact reviewed
Godzilla profile: `v0.3.7` (`ea1e799`) by default, or the issue #40
compatibility baseline `09214b160` (`09214b160b402011359f0ef9d5fa8f8be1112e85`),
selects only the tokenizer-related changes from the pinned
[`chynggi/gigatoken-llama.cpp`](https://github.com/chynggi/gigatoken-llama.cpp)
revision, and vendors a separate checkout of Gigatoken 0.10.0 at its exact
commit. The complete upstream diff, selected diff, dependency patch, Git
revisions, and adapted runtime files are hash-verified. It refuses an existing
target and never applies this port to an arbitrary Godzilla checkout.

```bash
# Read-only: checks platform, tools, output safety, pins, and planned commands
mtq-godzilla-gigatoken plan /opt/godzilla-gigatoken

# Select the older reviewed compatibility baseline explicitly
mtq-godzilla-gigatoken plan /opt/godzilla-gigatoken \
  --godzilla-profile 09214b160

# Prepare, compile the CPU runtime, and run both tokenizer suites
mtq-godzilla-gigatoken all /opt/godzilla-gigatoken --backend cpu --max-jobs 2 --yes

# Or qualify a CUDA build with a matching side-by-side toolkit
mtq-godzilla-gigatoken plan /opt/godzilla-gigatoken --for-action build \
  --backend cuda --cuda-toolkit /usr/local/cuda-12.6
mtq-godzilla-gigatoken build /opt/godzilla-gigatoken \
  --backend cuda --cuda-toolkit /usr/local/cuda-12.6 --max-jobs 2 --yes
```

On Windows, use a fresh destination such as `D:\src\godzilla-gigatoken` and a
CUDA toolkit root or `nvcc.exe` path. The reviewed port supports Windows x64
and Linux x86_64, pins Rust `nightly-2026-07-22`, defaults to local-model server
support (`LLAMA_CURL=OFF`), and accepts `--with-curl` when URL downloads are
needed. CPU is the conservative default; CUDA remains explicit and runs the
same tests after building. A clean two-worker Windows CPU build took about
eight minutes in validation, dominated by the first Rust dependency build.

For supported BPE and SentencePiece pre-tokenizers, model loading creates the
Gigatoken backend and normal `/completion` and `/v1/chat/completions` requests
use it transparently. llama.cpp still handles special-token partitioning,
BOS/EOS behavior, and detokenization. Unsupported vocabulary families retain
the original C++ tokenizer; malformed supported vocabularies and runtime ABI
errors fail closed rather than silently changing token IDs.

Every build runs 9 differential cases (including GPT-2, Llama BPE/SPM, MPT,
Qwen2, Qwen3.5, Gemma 4, long input, invalid UTF-8, concurrency, and fallback)
plus Godzilla's 15 existing tokenizer fixtures. Optional DeepSeek V3, GPT-OSS,
and Kimi K2.7 vocab-only GGUF fixtures can be supplied with `--fixture-dir` and
are registered only when their files exist. `verify` reruns qualification for
an existing build:

```bash
mtq-godzilla-gigatoken verify /opt/godzilla-gigatoken --backend cpu
```

The validated server is under `build-gigatoken-<backend>/bin` (and
`bin/Release` for multi-config Windows builds). The default build may serve the
API without an embedded browser UI; launch it with a local GGUF model path.

### Exact Godzilla PFlash/KVFlash composition

`mtq-godzilla-compose` creates a new source tree pinned to Godzilla commit
`09214b160b402011359f0ef9d5fa8f8be1112e85`, applies exact fail-closed source
edits, records hashes for every changed runtime file, and builds only
`llama-server`. It refuses existing destinations and arbitrary Godzilla
checkouts.

```bash
# Read-only plan
mtq-godzilla-compose plan /opt/godzilla-composed

# Prepare, build, and verify the CPU server
mtq-godzilla-compose all /opt/godzilla-composed \
  --backend cpu --max-jobs 2 --yes

# Separate CUDA build with the matching toolkit
mtq-godzilla-compose build /opt/godzilla-composed \
  --backend cuda --cuda-toolkit /usr/local/cuda-12.6 \
  --cuda-architectures 86 89 --max-jobs 2 --yes
```

SM86 and SM89 are the explicitly reviewed CUDA targets. Verification checks
that CMake retained both the selected architecture list and requested toolkit;
the Windows Visual Studio path uses CMake's CUDA toolset selection so another
installed toolkit is not silently substituted.

Both additions are disabled by default. Start with `--pflash` to permit the
feature, then set `"pflash": true` only on a plain completion request that has
passed workload-specific quality checks. Chat, multimodal, embedding, rerank,
and parallel-parent requests are bypassed. Prefix and suffix tokens are
protected, while the eligible middle is thinned deterministically.

`--kvflash-pages N --kvflash-page-tokens 256` sets a token-accounted LRU budget
for complete idle-slot KV state. This is a useful, composable server-residency
tier; it is not the research fork's unfinished arbitrary-page restore path and
does not claim disk restore or prefill skipping. `/props` reports the exact
composition profile and tier, while application and eviction events are sent
to the server log.

The overlay leaves DFlash/DDTree, KVarN, and TriAttention internals unchanged.
PFlash may be used before either KVarN or TriAttention, and whole-slot KVFlash
does not change their cache representation. Godzilla still rejects
TriAttention plus KVarN, and SpecLA remains outside this profile because it is
a specialized linear-attention runtime rather than a stackable add-on.

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

For multi-process serving on Linux x86-64 + CUDA, Multi-TurboQuant recognizes
the exact reviewed `pontostroy/cuda-llm-weight-share` source revision, plans a
GCC build without writing, and validates the resulting ELF library before it is
used:

```bash
git clone https://github.com/pontostroy/cuda-llm-weight-share.git
git -C cuda-llm-weight-share checkout 15bcecaebdbcec479f13df1c4396d5318b5bb85d
mtq-weight-share inspect cuda-llm-weight-share
mtq-weight-share plan cuda-llm-weight-share --cuda-toolkit /usr/local/cuda
mtq-weight-share build cuda-llm-weight-share --cuda-toolkit /usr/local/cuda --yes
mtq-weight-share validate cuda-llm-weight-share/cuda-llm-weight-share.so
```

The launcher then wraps a checked command with the preload environment:

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

The wrapper exposes the external helper's environment contract:

| Variable | Purpose |
|----------|---------|
| `LD_PRELOAD` | Loads the helper `.so` before llama.cpp so it can intercept CUDA allocations. Linux only. |
| `MODEL_SIZE` | Expected model-weight allocation in bytes. Use `0` for one discovery run, then reuse the reported allocation size; do not substitute the GGUF file size. |
| `MODEL_SIZE_TOLERANCE` | Permitted byte difference when matching the weight allocation. Keep `0` unless a small allocator variation requires it; a broad tolerance can match the wrong allocation. |
| `CUDA_VRAM_IPC_NAME` | Shared IPC namespace. Processes sharing one model must use the same unique name; unrelated groups should use different names. |
| `CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC` | How long a worker waits for the master to publish shared-memory metadata. Useful for staggered startup. |
| `CUDA_VRAM_IPC_SUPPRESS_MASTER_FREE` | Specialized option that keeps the master from freeing the shared backing allocation prematurely. Leave off unless the helper workflow requires it. |
| `CUDA_VRAM_IPC_TRACE_CALLERS` | Enables diagnostic allocation caller tracing. |
| `CUDA_VRAM_IPC_TRACE_DEPTH` | Maximum captured call-stack depth when tracing callers. |
| `CUDA_VRAM_IPC_TRACE_NORMAL_ALLOCS` | Also traces allocations not classified as model weights. This adds diagnostic noise and overhead. |

Weight sharing shares model weights between matching Linux/CUDA processes. It
does not share KV caches or contexts or reduce a single process's VRAM use.
Use the same model, build, device configuration,
`MODEL_SIZE`, and IPC name for every process in one sharing group.

### Optional optimization planner and LMCache

External inference optimizations are cataloged separately from the compression
methods and remain disabled unless explicitly selected. Inspect requirements,
platform support, KV-format validation, required artifacts, validation gates,
quality risk, and conflicts without importing third-party projects:

```bash
mtq-optimizations --engine vllm --kv-format fp16 --select lmcache
mtq-optimizations --engine godzilla --select triattention --select gigatoken
mtq-optimizations --engine godzilla --active-feature kvarn --select triattention
mtq-optimizations --engine transformers --select jetlong --select resonance_jetlong
```

The planner is fail-closed for unreviewed composition: two methods that alter
the same attention, KV-cache, position-encoding, or speculative-decoding domain
are rejected unless both catalog entries explicitly allow the pairing. The
reviewed Resonance profile is limited to JetLong's `yarn`/`jetlong_freq` mode;
it does not authorize plain JetLong or claim a production Qwen3 checkpoint.

The Godzilla plan models the exact boundaries: Gigatoken calibration requires
TriAttention, KVarN conflicts with TriAttention in the reviewed source profiles,
and CUDA weight sharing is limited to Linux/CUDA/x86-64 with a validated source
build. Add-ons for different engines are not forced into one process.

The newest reviewed additions are RestoreKV (pinned KVPress quality-repair
profile), ARCHead (pinned Transformers output-head compression profile), and
guarded research records for NOVA-KV and DSpark. See the
[August 2026 inference research review](docs/recent-inference-research.md) for
the newest-first evidence, duplicate analysis, side-effect controls, and the
paper-only directions that are deliberately not presented as implemented.

### Guarded composition and workload routing

The composition layer covers every reviewed optimization in the catalog and
classifies every pair as compatible, conditional, conflicting, or separate
runtime. Thirteen executable profiles describe combinations that have a
concrete activation path, including FastDMS/FlashAttention, LMCache,
MInference, TriAttention, Proxima, JetSpec, Jet-Long, FlashAttention or
SageAttention, two LuceBox routes, and two Godzilla routes. Missing model
artifacts, unsupported hosts, conflicting active features, and exact-output
requirements fail closed. When no candidate qualifies, routing explicitly
returns the unmodified baseline instead of guessing.

```bash
mtq-compose profiles
mtq-compose plan vllm_lmcache
mtq-compose route --task rag --prompt-tokens 65536 --repeated-prefix
mtq-compose simulate-capacity --layers 32 --kv-heads 8 --head-dim 128 \
  --context-tokens 131072 --available-memory-gib 24 --k-bits 4 --v-bits 4
```

Capacity output is deterministic byte accounting, not a speed, latency, or
quality prediction. Benchmark records similarly distinguish measured,
upstream-reported, and simulated evidence and reject incomparable baselines.
LuceBox remains a separate pinned runtime: its reviewed general profile and
documented Qwen 3.6 27B DFlash/DDTree + PFlash + KVFlash route do not transplant
model-specific kernels or headline results into Godzilla.

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

FastDMS, FlashAttention, JetSpec, JetLong, LMCache, MInference, Proxima,
SageAttention, and TriAttention calibration have stricter or mutually
incompatible runtime stacks. Their dependencies remain completely
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
mtq-env plan triattention
mtq-env plan jetspec
mtq-env plan proxima
mtq-env plan jetlong
mtq-env plan rocketkv  # reports its research/license block; changes nothing

# Explicitly create .mtq/environments/fastdms/{pyproject.toml,uv.lock,.venv}
mtq-env create fastdms --yes
mtq-env check fastdms

# Preview and force a local FlashAttention build when a wheel is unsuitable
mtq-env plan fastdms --build-from-source
mtq-env create fastdms --build-from-source --yes

# Build one reviewed add-on package from an existing local checkout
mtq-env plan fastdms --local-source /opt/addons/FastDMS
mtq-env create fastdms --local-source /opt/addons/FastDMS --yes

# Build the official calibrator environment from its checkout with bounded jobs
mtq-env create triattention --local-source /opt/addons/triattention --max-jobs 2 --yes

# Preserve a broken managed .venv, recreate it cleanly, and collect a redacted report
mtq-env create triattention --recreate --yes
mtq-env diagnose triattention --output triattention-diagnostics.json

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

Debian 12 and 13 are explicitly detected and covered by the Linux CI matrix.
Clean repair first preserves the existing managed `.venv`; if synchronization
fails, the incomplete replacement is retained for inspection and the previous
environment is restored. Diagnostics redact token/password-like values and
report distro, lexical and resolved interpreters, prefixes, import failures,
CUDA/toolchain state, and Accelerate environment information.

`--local-source` is a separate option for a checkout you already have. It is
accepted only for the eleven reviewed installable profiles, verifies the
profile-specific source markers, and records an absolute local-path source in
that profile's generated `uv` project. `uv` then builds the selected package
and resolves its declared dependencies inside the isolated environment. It
does not execute scanner-discovered files in the core environment or turn an
arbitrary source folder into an installable add-on.

`--max-jobs 2` controls the `MAX_JOBS` environment value used by source/native
builds; local-checkout builds default to two jobs when it is omitted. The UI
validates existing isolated environments before suggesting another build. Its
manual dependency override suppresses a rebuild recommendation only when the
automatic import check is known to be wrong, and displays a runtime-risk warning.

Native extensions must be compiled with the same CUDA major used by the
profile's PyTorch build. A newer NVIDIA driver may remain installed while a
matching toolkit is selected side by side:

```bash
mtq-env plan fastdms --cuda-toolkit /usr/local/cuda-12.6
mtq-env create fastdms --cuda-toolkit /usr/local/cuda-12.6 --yes
```

The Setup & Add-ons view exposes the same override. CUDA 13 `nvcc` is not used
to compile extensions for the CUDA 12.6 PyTorch profiles.

Reviewed local-source profiles receive managed dependency resolution, bounded
builds, and import validation. Constrained research sources, training systems,
and separate serving runtimes instead receive repository-specific setup
contracts. This includes Lucebox, ChunkLlama, RaBitQCache, ScoPE,
DuoAttention, IceCache, the PFlash/KVFlash llama.cpp fork, and the guarded
Resonance-JetLong composition. The separate `godzilla_composition` catalog
entry documents the exact pinned overlay rather than treating the HawgAuto
fork as an installable add-on. In particular, current Maru checkouts are
recognized through `pyproject.toml`/`setup.py` and the
`maru_resource_manager`/`maru_server` packages—no root `CMakeLists.txt` is
required. Maru remains guided-only because its upstream installer expects a
dedicated Linux host, CXL DAX device setup, and host services.

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
# NVIDIA: TurboQuant/TCQ + Iso/Planar cache types | AMD/Mac: Iso/Planar only

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

The browser UI now has three focused views:

- **Quick Run** keeps hardware detection, cache-method benchmarking, capacity
  planning, presets, and llama.cpp command generation together. It can discover
  models under a configured folder and start or stop a selected GGUF model with
  the generated argument list. Host RAM and GPU VRAM are reported separately;
  Apple unified memory is not double-counted.
- **Setup & Add-ons** stores the model, environment, add-on, and optional
  FlashAttention source folders; automatically scans only those configured
  folders; and reports the roots, depth, and directories inspected. It can
  select a reviewed checkout for an isolated `mtq-env` profile after explicit
  confirmation. A local checkout changes the package source, not its CUDA ABI,
  so the selected toolkit must still match the profile's PyTorch CUDA major.
  The view also recognizes renamed Godzilla trees by their marker script,
  reports KVarN/TriAttention support and existing builds, validates or creates
  the official calibrator environment, and can either calibrate and convert or
  convert existing official statistics after its prerequisites pass. Failed
  managed dependency checks can be repaired from the plan, and optional generic
  starter text can be generated locally. Advanced
  Quick Run controls and infrequent Setup sections are collapsed by default.
  The source picker can inspect the reviewed installable, guided, and blocked
  issue #43 projects alongside the earlier add-ons, domvox, the CUDA
  weight-share source, and separate llama.cpp forks without importing or
  executing source code; guided and blocked profiles are not made installable
  by discovery. Prepared exact-commit composition trees are recognized by their
  manifest and routed through the hash-bounded inspector. The pinned Godzilla + Gigatoken source preparation
  and build remains an explicit `mtq-godzilla-gigatoken` CLI operation.
- **Composition Lab** exposes the same read-only profile planning, deterministic
  workload routing, and analytical capacity simulator as `mtq-compose`. It also
  previews the exact pinned Godzilla build commands and qualified CUDA targets
  (SM86 and SM89) without patching, compiling, or launching from the browser.

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
  hardware.py            GPU and host-memory detection (NVIDIA, AMD, Metal)
  compatibility.py       Method/platform compatibility checks
  tokenizer_backends.py  Bounded Gigatoken interpreter discovery
  optimizations/         Catalog, pairwise matrix, guarded profiles/router, isolated envs
  methods/               5 method families, all with encode/decode
  kernels/triton/        Attention backend, vectorized encode, dispatch
  calibration/           Weight-norm analysis, TriAttention adapters, parity wrapper
  integration/           llama.cpp flags, pinned Godzilla builders, weight sharing, vLLM patch
  benchmark/             Comparisons, provenance contracts, capacity simulation
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

This project reimplements selected algorithms and integrates reviewed external
workflows from published research. Upstream licensing varies; blocked catalog
entries remain blocked when a usable software license or artifact grant is
missing:

| Contribution | Source |
|-------------|--------|
| Walsh-Hadamard KV compression | TheTom/llama-cpp-turboquant |
| Trellis Coded Quantization | spiritbuun/buun-llama-cpp |
| IsoQuant / PlanarQuant / RotorQuant | scrya-com/rotorquant (ParaMind2025) |
| CUDA + Metal kernels | johndpope/llama-cpp-turboquant |
| TriAttention token eviction | WeianMao/triattention |
| domvox TRIA v2 format and calibrator | [domvox/triattention-ggml](https://github.com/domvox/triattention-ggml) |
| Godzilla llama.cpp profile, KVarN alias surface, DFlash flags | [atomicmilkshake/godzilla-llama.cpp](https://github.com/atomicmilkshake/godzilla-llama.cpp) |
| BeeLlama / DFlash lineage | [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp) |
| KVarN research and reference implementation | [huawei-csl/KVarN](https://github.com/huawei-csl/KVarN) |
| Context-extension research notes: Position Interpolation, YaRN, Resonance RoPE, LongRoPE | [llama.cpp](https://github.com/ggml-org/llama.cpp), [sheryc/resonance_rope](https://github.com/sheryc/resonance_rope), published papers |
| Gigatoken Python tokenizer accelerator | [marcelroed/gigatoken](https://github.com/marcelroed/gigatoken) |
| Gigatoken llama.cpp runtime integration lineage | [chynggi/gigatoken-llama.cpp](https://github.com/chynggi/gigatoken-llama.cpp) |
| Issue #43 optimization research and source contracts | [JetSpec](https://github.com/hao-ai-lab/JetSpec), [Lucebox](https://github.com/Luce-Org/lucebox), [Proxima](https://github.com/Tenosra/Proxima), [Jet-Long](https://github.com/jet-ai-projects/jet-long), [ChunkLlama](https://github.com/HKUNLP/ChunkLlama), [RaBitQCache](https://github.com/Sakuraaa0/RaBitQCache), [ScoPE](https://github.com/oncemoe/ScoPE), [DuoAttention](https://github.com/mit-han-lab/duo-attention), [IceCache](https://github.com/yuzhenmao/IceCache), and [PFlash/KVFlash llama.cpp](https://github.com/HawgAuto/llama.cpp-dflash-pflash-kvflash) |
| Exact Godzilla PFlash/KVFlash composition request | [Godzilla llama.cpp](https://github.com/atomicmilkshake/godzilla-llama.cpp), [PFlash/KVFlash fork](https://github.com/HawgAuto/llama.cpp-dflash-pflash-kvflash), and issue [#44](https://github.com/aivrar/multi-turboquant/issues/44) |
| Full guarded composition, routing, simulation, LuceBox review, and SM86/SM89 Godzilla qualification | [Godzilla llama.cpp](https://github.com/atomicmilkshake/godzilla-llama.cpp), [LuceBox](https://github.com/Luce-Org/lucebox), and issue [#46](https://github.com/aivrar/multi-turboquant/issues/46) |

We reimplemented the Python-native algorithms in Python. Godzilla/KVarN support
is a command-generation, source-inspection, and preparation-workflow
integration; context-extension
support is a llama.cpp command-generation and capability-scanning integration
only. This repository does not bundle Godzilla, BeeLlama, KVarN, Resonance
RoPE, LongRoPE, domvox, Gigatoken, CUDA weight sharing, the issue #43 research
projects, or llama.cpp source trees. Installable source profiles use reviewed
revisions in isolated environments; guided-only entries remain read-only source
contracts and do not imply runtime compatibility.

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
| Suggested selecting local add-on source folders, resolving their dependencies, and recognizing Godzilla's KVarN/TriAttention setup needs | [@jawadala](https://github.com/jawadala) | Issue [#25](https://github.com/aivrar/multi-turboquant/issues/25) |
| Identified the official TriAttention calibration script and requested a no-`llama-cli` workflow plus clearer CUDA weight-share guidance | [@jawadala](https://github.com/jawadala) | Issue [#29](https://github.com/aivrar/multi-turboquant/issues/29) |
| Reported remaining dependency-state and calibration workflow friction, prompting bounded local builds, installed-environment validation, and a streamlined official-stats conversion path | [@jawadala](https://github.com/jawadala) | Issue [#31](https://github.com/aivrar/multi-turboquant/issues/31) |
| Requested domvox TRIA v2 calibration support, a 200k-token ceiling, local source selection for blocked add-ons, and a less-cluttered UI; these requests informed the experimental adapter, source inspector, progressive disclosure, and guardrails | [@jawadala](https://github.com/jawadala) | Issue [#32](https://github.com/aivrar/multi-turboquant/issues/32) |
| Reported the incomplete TriAttention environment and current Maru layout, and suggested clearer memory accounting, calibration starter text, repair, and source-specific setup guidance | [@jawadala](https://github.com/jawadala) | Issue [#35](https://github.com/aivrar/multi-turboquant/issues/35) |
| Reported the Linux managed-interpreter symlink regression that caused `uv`'s base Python to be selected instead of the TriAttention virtual environment | [@jawadala](https://github.com/jawadala) | Issue [#37](https://github.com/aivrar/multi-turboquant/issues/37) |
| Suggested evaluating Gigatoken for TriAttention calibration, discovering compatible Python/pyenv environments, and reviewing the separate llama.cpp integration | [@jawadala](https://github.com/jawadala) | Issue [#38](https://github.com/aivrar/multi-turboquant/issues/38) |
| Requested a direct Gigatoken tokenizer path for Godzilla runtime/inference, prompting the pinned combined-source workflow and differential qualification suite | [@jawadala](https://github.com/jawadala) | Issue [#39](https://github.com/aivrar/multi-turboquant/issues/39) |
| Requested Debian 12/13 hardening, deeper diagnostics, the exact Godzilla `09214b160` compatibility profile, domvox/Gigatoken support, and reviewed CUDA weight-share source handling | [@jawadala](https://github.com/jawadala) | Issue [#40](https://github.com/aivrar/multi-turboquant/issues/40) |
| Reported a domvox calibration launch under an interpreter without Torch, prompting compatible-environment discovery, fail-closed final preflight, exact-environment conversion, and detailed redacted failure bundles | [@jawadala](https://github.com/jawadala) | Issue [#42](https://github.com/aivrar/multi-turboquant/issues/42) |
| Proposed the JetSpec, Lucebox, Proxima, Jet-Long, ChunkLlama, RaBitQCache, ScoPE, DuoAttention, IceCache, PFlash/KVFlash, and Resonance-JetLong review, prompting pinned source profiles, read-only discovery contracts, runtime capability scanning, and fail-closed composition metadata | [@jawadala](https://github.com/jawadala) | Issue [#43](https://github.com/aivrar/multi-turboquant/issues/43) |
| Requested a safe PFlash/KVFlash composition path for the exact Godzilla `09214b160` baseline, prompting the pinned overlay, request and runtime guardrails, and build verification | [@jawadala](https://github.com/jawadala) | Issue [#44](https://github.com/aivrar/multi-turboquant/issues/44) |
| Requested full guarded treatment of the reviewed add-on catalog, workload routing, simulation, LuceBox composition research, UI coverage, and SM86/SM89 qualification for the canonical Godzilla baseline | [@jawadala](https://github.com/jawadala) | Issue [#46](https://github.com/aivrar/multi-turboquant/issues/46) |
| Reported model-specific TriAttention calibration failures with Mythos-nano-heretic `IQ4_XS`, prompting source-model context validation, nested RoPE correction, memory preflight, and explicit multi-GPU device selection | [@jawadala](https://github.com/jawadala) | Community testing report (August 2026) |
| Requested a newest-first inference research pass covering memory, context reuse, speed, quality, and safe composition; this prompted the RestoreKV and ARCHead profiles, guarded NOVA-KV and DSpark records, and the evidence-backed research roadmap | [@jawadala](https://github.com/jawadala) | Community research suggestion (August 2026) |
| ForgeAttention — fused MLX kernels for Apple Silicon (`multi_turboquant/kernels/metal/`): packed-3-bit fused QK, tiled SV, flash decode, sparse SV with phase-1/2 early exit, per-head attention budget calibration | [@user-23xyz](https://github.com/user-23xyz) | PR [#1](https://github.com/aivrar/multi-turboquant/pull/1) · sibling project [user-23xyz/forgeattention](https://github.com/user-23xyz/forgeattention) |

Thank you to [@jawadala](https://github.com/jawadala) for the sustained issue
reports and concrete feature suggestions. They have materially shaped the
Godzilla/KVarN support, context-extension tooling, optimization catalog,
isolated dependency system, practical UI workflow, and the official and
domvox TriAttention calibration paths, including the interpreter-path
correction, parity-checked Gigatoken option, and the broader issue #43 research
catalog with explicit safety boundaries, including the exact-commit composition
workflow prompted by issue #44 and the full guarded composition, routing,
simulation, LuceBox, UI, and SM86/SM89 qualification follow-up in issue #46.
Their subsequent Mythos-nano-heretic calibration report also prompted the
model-aware context, RoPE, memory, and CUDA-device safeguards. Their latest
research suggestion prompted the newest-first inference review and its guarded
quality-repair, output-head compression, KV quantization, speculative-decoding,
and context-sharing directions.

The Metal path is community-maintained — the maintainer does not have Apple Silicon hardware, so issues specific to MLX/Metal should tag the contributor for context.

## Support the Project

If Multi-TurboQuant is useful to you, you can support its continued development
through [GitHub Sponsors](https://github.com/sponsors/aivrar).

## License

MIT
