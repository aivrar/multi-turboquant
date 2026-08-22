# Optional Optimization Integrations

Multi-TurboQuant keeps external inference optimizations disabled by default.
The optimization catalog describes what each project actually supports and the
planner rejects missing dependencies, unsupported platforms, unvalidated KV
formats, and known conflicts before launch. Importing the package never imports
or discovers third-party optimization packages.

## Inspect compatibility

```bash
mtq-optimizations --engine vllm --kv-format fp16 --select lmcache
mtq-optimizations --engine vllm --select lmcache --select maru --json
```

The same functionality is available through Python:

```python
from multi_turboquant.optimizations import (
    detect_optimization_context,
    plan_optimizations,
)

context = detect_optimization_context(engine="vllm", kv_format="fp16")
plan = plan_optimizations(["lmcache"], context)
if not plan.ready:
    for issue in plan.issues:
        print(issue.severity, issue.message)
```

Third-party integrations can implement `OptimizationPlugin` and register an
instance explicitly with `OptimizationRegistry.register()`. There is no
automatic entry-point loading because merely installing a plugin must not alter
inference behavior.

## LMCache launch plans

LMCache is the first executable integration because it exposes maintained vLLM
connector contracts and can run as a separate service. Multiprocess mode is the
recommended default:

```python
from multi_turboquant.integration import (
    LMCacheIntegrationConfig,
    build_lmcache_launch_plan,
)

lmcache = build_lmcache_launch_plan(LMCacheIntegrationConfig(
    mp_host="localhost",
    mp_port=5555,
    server_l1_size_gb=20,
))

print(lmcache.server_command)
vllm_command = lmcache.extend_vllm_command([
    "vllm", "serve", "Qwen/Qwen3-8B", "--port", "8000",
])
print(vllm_command)
```

For in-process mode, select `mode="in_process"` and optionally provide an
LMCache YAML file through `config_file`. The launch plan returns the required
`LMCACHE_CONFIG_FILE` environment entry rather than mutating `os.environ`.

For vLLM 0.20.0 or newer, LMCache documents an optional connector implementation
shipped by LMCache itself. Set `use_lmcache_shipped_connector=True` and provide
`vllm_version`; the planner refuses this option for older or unknown versions.

LMCache currently supports standard serving cache layouts in this integration:
FP16, BF16, and FP8. Multi-TurboQuant does not claim that LMCache can serialize
TurboQuant, KVarN, IsoQuant, or PlanarQuant layouts. Those combinations require
a separately implemented and validated LMCache SERDE.

## Reviewed catalog

| Project | Current treatment | Reason |
|---|---|---|
| LMCache | Experimental external vLLM launch-plan integration | Maintained connector and service contracts; local runtime validation still required |
| Maru | Optional LMCache dependency | Requires Python 3.12 and CXL `/dev/dax` hardware |
| MInference | Experimental manifest | vLLM/model/version-specific sparse prefill patches |
| FlashAttention | Experimental Python backend + isolated environment profile | Requires a matched Linux/CUDA/PyTorch native stack |
| SageAttention | Experimental Python manifest | No maintained llama.cpp or vLLM connector |
| Speculative Prefill | Experimental manifest | Old vLLM monkeypatch and draft-model quality tradeoffs |
| FastDMS | Standalone engine only | Requires DMS checkpoints and scheduler/cache changes |
| TriAttention | Experimental Godzilla calibration path | Requires model-specific stats; conflicts with KVarN in both reviewed Godzilla profiles |
| Gigatoken | Optional TriAttention tokenizer | Requires TriAttention and exact Hugging Face token-ID parity for the selected text |
| CUDA LLM Weight Share | Reviewed Linux preload source | Linux/CUDA/x86-64 only; shares one measured model-weight allocation, not KV cache |
| RocketKV | Research only | Research snapshot and non-commercial source license |
| Lexico | Research only | WIP and requires per-model dictionaries |
| AdaDecode | Blocked | No source license and requires trained prediction heads |
| Resonance YaRN | Native backend required | Fine-tuning implementation is not a llama.cpp plugin |
| JetSpec | Installable isolated source profile | CUDA/Triton speculative decoding research; headline results are hardware-specific and its vLLM path uses a separate fork |
| Lucebox | Guided separate runtime | Full CUDA/ROCm serving runtime with DFlash, PFlash, KVFlash, DDTree, and SpecLA; not a Python add-on |
| Proxima | Installable isolated source profile | Source-only vLLM plugin pinned to its reviewed vLLM baseline; reference calibrated checkpoint is not public |
| Jet-Long | Installable reference profile | Exact Python 3.12/Torch 2.9.1/Transformers 5.3 stack; fused path requires Hopper/SM90 and CUDA 13 |
| ChunkLlama | Guided research source | Older DCA reference with separate non-commercial data/weight terms and no reviewed serving connector |
| RaBitQCache | Blocked | No usable repository software license was found |
| ScoPE | Guided training system | TorchTitan training method, not an existing-checkpoint serving add-on |
| DuoAttention | Guided research source | Model-specific head patterns and multiple native CUDA components require separate qualification |
| IceCache | Guided research source | Requires the separate M-DCI artifact, native builds, OpenBLAS/OpenMP, and a high-core-count CPU workflow |
| PFlash/KVFlash llama.cpp | Guided separate runtime | Hawg33 fork is a separate llama.cpp family, not an arbitrary patch for Godzilla |
| Godzilla composition | Experimental pinned native workflow | Exact `09214b160` overlay with request-gated PFlash and complete-idle-slot KVFlash residency |
| Resonance-JetLong | Native backend required | Reviewed only for JetLong `yarn`/`jetlong_freq`; Qwen3 remains a gated research hypothesis |
| RestoreKV | Installable isolated KVPress profile | Learned repair for KVzip-family eviction; requires a released model-matched adapter and remains lossy |
| NOVA-KV | Guarded separate research runtime | Apache-2.0 SGLang 0.5.10 fork with model-specific calibration artifacts and H100-oriented evidence |
| ARCHead | Installable isolated Transformers profile | Compresses a dense LM output head; requires activation calibration and does not patch GGUF |
| DSpark | Guarded serving research | Distinct confidence-scheduled drafter, but the serving path spans moving vLLM code with reported integration failures |

The [August 2026 inference research review](recent-inference-research.md) records
the primary-paper evidence, duplicate check, side-effect mitigations, and the
paper-only OasisKV, CoinRAG, ELDR, and VeriCache directions that are not exposed
as installable software.

## Guarded execution profiles

`mtq-compose` is the executable planning layer above the catalog. Its complete
pairwise matrix covers all reviewed entries; it does not infer compatibility
from two projects merely sharing CUDA, vLLM, or an attention domain. Thirteen
profiles represent combinations with a concrete guarded path. Each declares
its engine, OS/compute support, source or model artifacts, required model
traits, forbidden active features, version constraints, and quality risk.

```bash
mtq-compose profiles
mtq-compose plan godzilla_guarded --os linux --compute cuda \
  --artifact godzilla_09214b160_source --artifact gguf_model
mtq-compose route --task long_context --prompt-tokens 65536 \
  --preferred-engine vllm --artifact minference_pattern
```

Routing is deterministic. It filters candidates through the same validation
contract, then ranks eligible profiles by explicit workload signals and stable
priority/tie-breaking rules. If none is safe, the result is `baseline`; it does
not silently activate the least-incompatible add-on. Profiles create plans
only—installation remains in `mtq-env`, while native Godzilla preparation and
build remain in `mtq-godzilla-compose`.

The two LuceBox profiles preserve the runtime boundary. `lucebox_guarded`
requires a reviewed LuceBox source and a supported model. The narrower
`lucebox_qwen36_composed` route represents LuceBox's documented Qwen 3.6 27B
DFlash/DDTree, PFlash, and KVFlash path and requires both drafter artifacts plus
a supported GPU. PFlash makes this route lossy; neither its quality nor its
GPU-specific upstream measurements are transferred to another model/runtime.

The capacity simulator reports baseline and candidate KV bytes per sequence
and maximum byte-budget concurrency:

```bash
mtq-compose simulate-capacity --layers 32 --kv-heads 8 --head-dim 128 \
  --context-tokens 131072 --available-memory-gib 24 \
  --model-weights-gib 16 --runtime-overhead-gib 2 \
  --k-bits 4 --v-bits 4 --retained-fraction 0.5
```

This is analytical byte accounting. It does not predict throughput, latency,
kernel support, allocator behavior beyond the supplied efficiency, or output
quality. Benchmark contracts separately label evidence as measured,
upstream-reported, or simulated and require comparable model, hardware,
software, workload, and metric definitions before calculating a speedup.

## Composition rules

Every catalog entry declares the runtime domains it changes. The planner
rejects overlapping domains with `unvalidated_composition` unless both entries
explicitly allow the pairing. Coexisting source folders or isolated
environments therefore do not imply that two optimizations may run in the same
model process. The plan JSON also exposes required artifacts, validation gates,
quality-risk labels, and reviewed source commits.

The planner treats token eviction methods as alternatives unless a combination
has a dedicated implementation and evaluation. It also rejects cache-storage
plugins when the selected KV representation has not been validated. Adapters
are not inserted automatically: decompression or layout conversion can consume
more time and memory than the optimization saves.

Startup benchmarking and accuracy qualification should operate only on a
shortlist that has already passed this compatibility plan. Results must be
cached by model, runtime, plugin versions, GPU, driver, and workload shape; a
full Cartesian benchmark at every startup is intentionally out of scope.

For the Godzilla workflow, `triattention` and `gigatoken` are first-class
planner entries. Selecting Gigatoken without TriAttention is an error. An
active `kvarn` feature conflicts with TriAttention because both the v0.3.7 and
`09214b160` source profiles reject that pairing until KVarN-aware pruning is
implemented. `cuda_weight_share` is eligible only for llama.cpp/Godzilla on
Linux CUDA x86-64 with GCC available. These checks describe a single runtime;
independent add-ons for other engines remain separate processes/environments.

The exact `09214b160` composition builder accepts explicit
`--cuda-architectures 86 89` targets. On Windows with a Visual Studio generator
it pins the requested CUDA toolkit through CMake's toolset selection (rather
than allowing Visual Studio to select another installed CUDA compiler);
non-Visual-Studio generators receive `CMAKE_CUDA_COMPILER`. Verification checks
the source/overlay hashes, selected compiler, selected architectures,
executable, and supported flags.

```bash
mtq-optimizations --engine godzilla --select triattention --select gigatoken
mtq-optimizations --engine godzilla --active-feature kvarn --select triattention
mtq-optimizations --engine godzilla --select cuda_weight_share
mtq-optimizations --engine transformers --select jetlong --select resonance_jetlong
```

The only newly allowlisted overlap is `jetlong` with `resonance_jetlong`, and
only in JetLong's YaRN/frequency path. It remains a research composition until
the declared artifact, long-context correctness, perplexity, retrieval,
throughput, memory, and fallback gates pass for the exact model and source
revisions. Plain JetLong, arbitrary Qwen3 checkpoints, and other overlapping
attention/KV methods remain rejected.

## Isolated dependency environments

The compatibility planner above is side-effect-free and examines packages in
the current interpreter. The separate `mtq-env` command handles add-ons whose
dependency stacks should not be installed into that interpreter.

```bash
mtq-env list
mtq-env plan fastdms
mtq-env plan flashattention --json
mtq-env plan lmcache
mtq-env plan triattention
mtq-env plan rocketkv
```

`plan` never writes files, installs packages, or launches a runtime. It checks
the host OS, compute backend, `uv`, and native build tools, then displays the
independent project that would be created. Blocked profiles return one reviewed
reason and no command or generated project. On a compatible Linux CUDA host:

```bash
mtq-env create fastdms --yes
mtq-env check fastdms
mtq-env run fastdms -- python -c "from fastdms import LLM"
```

If the upstream FlashAttention wheel is unavailable or unsuitable, preview and
request a local build explicitly:

```bash
mtq-env plan flashattention --build-from-source
mtq-env create flashattention --build-from-source --yes

mtq-env plan fastdms --build-from-source
mtq-env create fastdms --build-from-source --yes

# Use an already checked-out, reviewed package instead of the default source
mtq-env plan fastdms --local-source /opt/addons/FastDMS
mtq-env create fastdms --local-source /opt/addons/FastDMS --yes

# Official TriAttention checkout and calibrator dependencies
mtq-env plan triattention --local-source /opt/addons/triattention --max-jobs 2
mtq-env create triattention --local-source /opt/addons/triattention --max-jobs 2 --yes

# Preserve and replace an incomplete owned environment, then collect redacted evidence
mtq-env create triattention --recreate --yes
mtq-env diagnose triattention --output triattention-diagnostics.json
```

Source mode is opt-in and is currently reviewed only for the `flashattention`
and `fastdms` profiles. It tells `uv` not to install a binary distribution for
`flash-attn`, sets FlashAttention's own `FLASH_ATTENTION_FORCE_BUILD=TRUE`
switch, and reinstalls that package so an existing environment or cached
artifact cannot bypass the request. Other profiles reject the option rather
than applying an unreviewed source-build procedure.

`--local-source PATH` addresses a different case: installing the profile's
primary package from an existing checkout. It is available for
`flashattention`, `fastdms`, `lmcache`, `minference`, `sageattention`, and
`triattention`, plus the source-only `jetspec`, `restorekv`, `archead`,
`proxima`, and `jetlong` profiles.
Planning resolves the path and verifies a profile-specific marker set (for
example `setup.py`, the import package, and `csrc` where applicable). The
generated project replaces only that package requirement with a
`[tool.uv.sources]` local path and asks `uv` to rebuild it without cache; every
other reviewed pin, Python constraint, CUDA check, validation import, and
isolation boundary stays in force. A missing marker or unsupported profile is
an error before any write.

Local-checkout and forced-source builds default to `MAX_JOBS=2`. Pass
`--max-jobs N` (1-64) to choose another bounded concurrency value. The value is
applied to the child `uv sync` process and never changes the caller's environment.

Local paths are non-editable by default, matching `uv` project-source
semantics. Scanner discovery remains read-only: choosing and creating the
environment is a separate, confirmed action. This is dependency build
orchestration for known Python packages, not arbitrary script execution or a
generic plugin loader.

Each profile lives under `.mtq/environments/<profile>/` by default and owns a
separate `pyproject.toml`, `uv.lock`, and `.venv`. This intentionally avoids a
shared workspace lock: FastDMS, FlashAttention, vLLM-related integrations, and
future research projects can require incompatible Python or PyTorch versions.
The generated project includes an ownership marker, and `mtq-env` refuses to
overwrite a pre-existing project without the matching marker.

Environment creation is opt-in and requires `--yes`. It does not alter the
active virtual environment, install GPU drivers, or change the CUDA toolkit.
Use `--root PATH` to choose another environment root and `--python VERSION` or
`--python /path/to/python` to select an interpreter. A Python installed by
`pyenv` works through the latter form; pyenv itself is not required.

POSIX virtual-environment Python entries are preserved as lexical paths rather
than canonicalized through their final symlink. For example,
`.venv/bin/python` may point at a `uv` or system base executable, but invoking
the symlink is what activates the virtual environment's prefix and packages.
Resolving it first would silently inspect or run the wrong environment.

Debian 12 and Debian 13 are explicitly detected and covered by the Linux CI
matrix (Python 3.11 and 3.13 respectively). `--recreate` moves the current owned
`.venv` to a timestamped backup before synchronizing a clean replacement. A
failed sync preserves the incomplete replacement for inspection and restores
the previous environment. `mtq-env diagnose` emits a redacted JSON report with
the distribution, lexical/resolved interpreter paths, Python prefixes, module
versions and import tracebacks, Torch/CUDA state, uv/Git/NVIDIA/CUDA tools, and
Accelerate environment details.

The Godzilla calibration planner adds a stricter runtime boundary. If no Python
is selected, it probes no more than eight conventional candidates and accepts
the first interpreter that imports the complete calibration stack. It never
combines `site-packages` from multiple environments. Core import failures cannot
be overridden, and the chosen interpreter is probed again immediately before
launch. FlashAttention is required only for the official
`flash_attention_2`/CUDA path; Gigatoken is required only when selected.
domvox's `triattention_common.py`, calibration, and conversion are kept inside
the same interpreter boundary.

A failed UI calibration writes a redacted atomic diagnostic JSON file next to
the requested artifact. In addition to import tracebacks, it records the exact
command and working directory, selected and host interpreter state, relevant
paths, source revision, bounded discovery attempts, OS/CUDA/VRAM/tool versions,
disk capacity, log tails, and recovery guidance.

For profiles that compile CUDA extensions, `nvcc` must use the same CUDA major
as the profile's PyTorch build. NVIDIA driver backward compatibility does not
make a CUDA 13 compiler interchangeable with a CUDA 12.6 PyTorch extension
build. Keep the newer driver and select a side-by-side toolkit root or its
`nvcc` executable:

```bash
mtq-env plan fastdms --cuda-toolkit /usr/local/cuda-12.6
mtq-env create fastdms --cuda-toolkit /usr/local/cuda-12.6 --yes
mtq-env check fastdms --cuda-toolkit /usr/local/cuda-12.6
```

The selected root is exported as `CUDA_HOME` and `CUDA_PATH`, and its `bin`
folder is placed first on `PATH` for creation, validation, and commands run
through `mtq-env`. A major mismatch remains an error because PyTorch's native
extension builder rejects it; minor differences within the same major remain
eligible for the reviewed profile.

### Built-in profiles

| Profile | Status | Locked top-level requirements or block | Current host gate / validation |
|---|---|---|---|
| `flashattention` | Installable | PyTorch 2.7.1 from the CUDA 12.6 index, FlashAttention 2, and build helpers | Linux + CUDA 12.x + `nvcc`; imports Torch and FlashAttention |
| `fastdms` | Installable | FastDMS 0.2.x, PyTorch 2.7.1 CUDA 12.6, and transitive FlashAttention | Linux + CUDA 12.x + `nvcc`; imports Torch, Triton, FlashAttention, and FastDMS |
| `lmcache` | Installable | LMCache 0.5.2, PyTorch 2.11.0 CUDA 13.0, and OpenAI 2.46.0 | Linux CUDA; imports `lmcache.c_ops`, checks Torch CUDA 13, and exposes the standalone CLI |
| `minference` | Installable | Official v0.1.6 source commit `d76b76e`, Transformers 4.x, and PyTorch 2.7.1 CUDA 12.6 | Linux + CUDA 12.x + Git + `nvcc`; compiles and imports Torch, Triton, and MInference |
| `sageattention` | Installable | Audited upstream commit `d1a57a5`, PyTorch 2.7.1 CUDA 12.6, and build helpers | Linux + CUDA 12.x + Git + `nvcc`; compiles and imports SageAttention |
| `triattention` | Installable | Official commit `81552bb`, PyTorch 2.7.1 CUDA 12.6, Transformers 4.57.6, Accelerate 1.14.0, SentencePiece 0.2.2, and Gigatoken 0.10.0 | Linux CUDA + Git; imports the official/domvox calibration stack and supplies the UI's automatic calibration Python |
| `restorekv` | Installable | KVPress commit `161705a`, PyTorch 2.7.1 CUDA 12.6, Transformers 4.57.6, and PEFT 0.20.0 | Linux CUDA + Git; validates imports, while the adapter and quality budget remain separate gates |
| `archead` | Installable | ARCHead commit `2900272`, PyTorch 2.7.1 CUDA 12.6, and Transformers 4.57.6 | Linux CUDA + Git; validates the base implementation without optional Triton/baseline packages |
| `novakv` | Blocked | Complete SGLang 0.5.10 research fork plus model-specific rotations/codebooks | Reproduce upstream independently; do not merge its engine into another profile |
| `dspark` | Blocked | Matching drafter plus exact vLLM proposer/hidden-state path | Wait for a reviewed release that passes parity and multi-concurrency checks |
| `jetspec` | Installable | Reviewed JetSpec commit, Python 3.11, PyTorch 2.7.1 CUDA 12.6, Transformers 4.57.1, and Triton | Linux CUDA + Git; validates the local package and preserves the separate-vLLM-fork boundary |
| `proxima` | Installable | Reviewed Proxima commit and its source-declared vLLM 0.10.1.1 baseline | Linux CUDA + Git; validates the plugin while requiring a separately supplied calibrated model artifact |
| `jetlong` | Installable | Reviewed Jet-Long commit, Python 3.12, PyTorch 2.9.1 CUDA 13.0, Transformers 5.3, and FlashAttention 2.8.3 | Linux CUDA 13 + Git; validates the reference path; fused SM90 execution is a separate guided profile |
| `maru` | Blocked | Upstream installer builds a host C++ resource manager and expects CXL `/dev/dax` | Use upstream installation on a dedicated CXL host |
| `speculative_prefill` | Blocked | Unpackaged monkeypatch pinned to Torch 2.4.0 and vLLM 0.6.3.post1 | Requires a separately qualified legacy source checkout |
| `rocketkv` | Blocked | Unpackaged research snapshot under a non-commercial research license | Not exposed as a supported serving add-on |
| `lexico` | Blocked | WIP source tree requiring a trained dictionary per model/configuration | Dependencies alone cannot produce a usable runtime |
| `adadecode` | Blocked | No repository software license and requires task-specific prediction heads | Automatic installation is not legally or operationally complete |
| `resonance_yarn` | Blocked | Old training environment and Hugging Face LLaMA fork | Needs a native serving-backend implementation, not an environment install |
| `jetlong_fused` | Blocked | Hopper/SM90 fused-kernel path with CUDA 13 and exact upstream stack | Use the reference `jetlong` profile unless the host and model pass fused-path qualification |
| `lucebox` | Blocked | Separate CUDA/ROCm runtime with its own build, container, model, and evaluation contracts | Build and qualify upstream independently; discovery and scanner output are informational |
| `chunkllama` | Blocked | DCA research source with separate code/data/weight terms | Requires model-specific long-context evaluation and a maintained serving target |
| `rabitqcache` | Blocked | Repository has no usable software license | No automated install or integration |
| `scope_pe` | Blocked | TorchTitan training workflow and trained artifacts | Not applicable as a drop-in inference profile |
| `duoattention` | Blocked | Model-specific head patterns and native CUDA stack | Requires exact-model artifact and runtime qualification |
| `icecache` | Blocked | M-DCI artifact, C++/CUDA builds, OpenBLAS/OpenMP, and CPU-heavy indexing | Requires upstream prerequisites and independent evaluation |
| `pflash_llamacpp` | Blocked | Separate Hawg33 llama.cpp runtime family | Inspect/build the fork separately; do not apply it as a Godzilla patch |
| `godzilla_composition` | Guided | Exact Godzilla `09214b160` source overlay | Use `mtq-godzilla-compose`; PFlash is lossy and KVFlash is limited to complete idle slots |
| `resonance_jetlong` | Blocked | Research composition requiring a trained compatible artifact and long-context gates | Planner metadata only; no automatic training or serving claim |

Blocked rows are catalog records, not failed installations. They intentionally
have no create command until the missing hardware contract, licensing,
artifacts, maintenance baseline, or serving integration exists. The UI labels
them as informational and does not offer an automatic Create action.

The Setup source picker can still inspect a local checkout for each blocked
profile. It checks reviewed marker files and reports upstream metadata without
importing or executing the source; selecting a folder never changes the
profile's blocked status or creates an environment. It also returns the
profile-specific host, licensing, runtime, or artifact requirements and the
reviewed next steps. Current Maru source uses Python project metadata and the
`maru_resource_manager`/`maru_server` packages; the inspector no longer requires
a root `CMakeLists.txt`. Automatic Maru installation remains unavailable because
the upstream workflow configures CXL DAX access and host services. The same read-only inspector
recognizes `domvox/triattention-ggml` for the separate experimental TriAttention
adapter. That adapter belongs to the Godzilla calibration workflow, not to the
installable Python add-on profiles.

Gigatoken is an opt-in tokenizer backend for the reviewed official and domvox
TriAttention calibrators, not a separate installable profile. The `triattention` profile pins
the reviewed 0.10.0 release. Before loading model weights, the wrapper requires
exact Hugging Face/Gigatoken token-ID parity for the complete selected text with
matching truncation and maximum length; a mismatch aborts the run. The UI can
inspect up to 64 interpreters across current, `PATH`, active virtual/Conda,
managed `.mtq`, and conventional pyenv locations. Inspection is isolated,
bounded, and does not recursively scan a drive.

The source inspector also recognizes `chynggi/gigatoken-llama.cpp` as a separate
experimental Windows x64/Linux x86_64 runtime fork. Discovery remains
informational and never executes it. The explicit `mtq-godzilla-gigatoken`
workflow performs a deliberate port onto either the pinned v0.3.7 baseline
(`ea1e799`) or the exact `09214b160` compatibility baseline in a new target
tree, verifies all source revisions and hashes, builds with
`LLAMA_GIGATOKEN=ON`, and runs both the differential and legacy tokenizer
suites. It does not patch a selected or arbitrary checkout.

The Hawg33 PFlash/KVFlash catalog record remains blocked as a separate upstream
runtime family. The narrower `godzilla_composition` workflow is distinct: it
creates a new tree from exact Godzilla commit
`09214b160b402011359f0ef9d5fa8f8be1112e85` and applies a reviewed local
overlay. PFlash is disabled unless both `--pflash` and per-request
`"pflash": true` are present, and it bypasses chat, multimodal, embedding,
rerank, and parallel-parent requests. KVFlash is implemented only as an LRU
budget over complete idle slots. Arbitrary KV pages, disk restore, and prefill
skip remain explicitly unsupported.

This narrower residency boundary avoids modifying KVarN's representation or
TriAttention's intra-sequence pruning. DFlash/DDTree can remain enabled, and
PFlash may run before either KVarN or TriAttention. The upstream
TriAttention/KVarN conflict is unchanged, and SpecLA is excluded because it is
a specialized linear-attention runtime. Use `mtq-godzilla-compose plan` before
preparation, then run CPU/CUDA build qualification and workload-specific
PFlash quality plus KVFlash concurrency/soak tests.

### CUDA weight-share source

The source picker recognizes `pontostroy/cuda-llm-weight-share` only when its
origin, marker files, and exact commit
`15bcecaebdbcec479f13df1c4396d5318b5bb85d` match. The dedicated CLI keeps
inspection/planning read-only and requires confirmation before compiling:

```bash
git clone https://github.com/pontostroy/cuda-llm-weight-share.git
git -C cuda-llm-weight-share checkout 15bcecaebdbcec479f13df1c4396d5318b5bb85d
mtq-weight-share inspect cuda-llm-weight-share
mtq-weight-share plan cuda-llm-weight-share --cuda-toolkit /usr/local/cuda
mtq-weight-share build cuda-llm-weight-share --cuda-toolkit /usr/local/cuda --yes
mtq-weight-share validate cuda-llm-weight-share/cuda-llm-weight-share.so
```

Build planning requires Linux x86-64, GCC, CUDA runtime headers, and the
`file`/`nm`/`ldd` validators. Validation checks for an ELF shared object,
exported `cudaMalloc`/`cudaFree` hooks, and no hard `libcudart` dependency.
Production configuration is rejected until reconnaissance has supplied a
nonzero measured allocation and a non-default IPC name. This integration does
not claim KV-cache sharing or cross-model compatibility.

Build isolation is disabled only for the packages whose setup scripts import
the selected Torch or require the active CUDA build context. If no compatible
wheel is available, uv may compile a native extension. The plan reports that
possibility before creation and limits native build parallelism to avoid
exhausting memory on smaller hosts. Forced source mode always performs the
FlashAttention CUDA/C++ compilation and can therefore take substantially
longer than the default wheel-first path. The pinned SageAttention and
MInference Git sources also have explicit static package metadata because their
legacy `setup.py` files import build dependencies before uv can otherwise
resolve the projects.

FastDMS remains a standalone engine and requires a DMS-trained checkpoint. The
environment profile makes its dependencies reproducible; it does not claim to
turn FastDMS into a vLLM or llama.cpp plugin.

### Native validation record

The FastDMS profile was validated end-to-end on July 26, 2026 in Ubuntu 24.04
under WSL2 with an RTX 3090, NVIDIA driver 596.36, and CUDA toolkit 12.0. The
locked environment resolved to:

- Python 3.11.15
- FastDMS 0.2.0
- PyTorch 2.7.1+cu126
- Triton 3.3.1
- FlashAttention 2.8.3.post1

Validation imported all four runtime modules, confirmed CUDA access, executed a
finite FP16 FlashAttention kernel, and generated 16 tokens with
`shisa-ai/Llama-3.2-1B-DMS-8x`. WSL enumerated the RTX 3090 as Torch device 0
even though `nvidia-smi` listed it second, so GPU selection should use Torch's
device order or an explicit `CUDA_VISIBLE_DEVICES` check.

An unconstrained trial selected PyTorch 2.13.0+cu130 and failed correctly
against the CUDA 12.0 compiler. The built-in profile therefore pins PyTorch
2.7.1 to the official CUDA 12.6 wheel index and rejects non-CUDA-12 toolkits
before installation.

The LMCache profile was validated on the same host from a separate Python 3.12
lock. It resolved LMCache 0.5.2, PyTorch 2.11.0+cu130, and Triton 3.6.0; imported
the upstream-recommended `lmcache.c_ops` extension; reported CUDA access; and
successfully rendered `lmcache --help`. That CLI test found that LMCache 0.5.2
imports the OpenAI client from its benchmark command without declaring it in
package metadata. The profile therefore pins OpenAI 2.46.0 explicitly and
validates its import. The prebuilt CUDA 13 runtime worked with driver 596.36 and
did not require changing the host's CUDA 12.0 compiler.

The SageAttention profile was then built natively from pinned commit `d1a57a5`
on the RTX 3090. Its CUDA extensions compiled for SM86 in about six minutes and
resolved to SageAttention 2.2.0, PyTorch 2.7.1+cu126, Triton 3.3.1, and NumPy
2.2.6. A real FP16 attention kernel over shape `(1, 8, 128, 64)` returned finite
output with mean absolute error 0.00136 against PyTorch SDPA. The first attempt
also proved why the profile supplies static package metadata: upstream's legacy
`setup.py` imports build dependencies before dependency resolution. NumPy is
pinned explicitly because the exercised runtime path imports it even though the
upstream package metadata does not declare it.

The MInference profile was built from official v0.1.6 release commit `d76b76e`
in about three minutes with one native-build worker. It resolved to MInference
0.1.6.0, Transformers 4.57.6, PyTorch 2.7.1+cu126, and Triton 3.3.1, and its
compiled `convert_vertical_slash_indexes` CUDA operator ran successfully for a
128-token, two-head input on the RTX 3090. The Git pin is intentional: the
published package imported optional `kivi_gemv` unconditionally, while newer
upstream development also imports optional LeanK dependencies. Transformers is
kept below 5 because MInference relies on a private package-probe API whose
return type changed in Transformers 5. This validates the isolated package and
native operator, not a model-specific sparse-attention configuration.

## Upstream references

- LMCache quickstart: <https://docs.lmcache.ai/getting_started/quickstart.html>
- LMCache configuration: <https://docs.lmcache.ai/api_reference/configurations.html>
- LMCache storage plugins: <https://docs.lmcache.ai/developer_guide/extending_lmcache/storage_plugins.html>
- Maru: <https://github.com/xcena-dev/maru>
- MInference: <https://github.com/microsoft/MInference>
- FlashAttention: <https://github.com/Dao-AILab/flash-attention>
- SageAttention: <https://github.com/thu-ml/SageAttention>
- FastDMS: <https://github.com/shisa-ai/FastDMS>
- RocketKV: <https://github.com/NVlabs/RocketKV>
- Speculative Prefill: <https://github.com/Jingyu6/speculative_prefill>
- Lexico: <https://github.com/krafton-ai/lexico>
- AdaDecode: <https://github.com/weizhepei/AdaDecode>
- Resonance RoPE: <https://github.com/sheryc/resonance_rope>
- domvox TriAttention: <https://github.com/domvox/triattention-ggml>
- Gigatoken: <https://github.com/marcelroed/gigatoken>
- Gigatoken llama.cpp fork: <https://github.com/chynggi/gigatoken-llama.cpp>
- JetSpec: <https://github.com/hao-ai-lab/JetSpec>
- Lucebox hub: <https://github.com/Luce-Org/lucebox-hub>
- Lucebox runtime: <https://github.com/Luce-Org/lucebox>
- Proxima: <https://github.com/Tenosra/Proxima>
- Jet-Long: <https://github.com/jet-ai-projects/jet-long>
- ChunkLlama: <https://github.com/HKUNLP/ChunkLlama>
- RaBitQCache: <https://github.com/Sakuraaa0/RaBitQCache>
- ScoPE: <https://github.com/oncemoe/ScoPE>
- DuoAttention: <https://github.com/mit-han-lab/duo-attention>
- IceCache: <https://github.com/yuzhenmao/IceCache>
- PFlash/KVFlash llama.cpp fork: <https://github.com/HawgAuto/llama.cpp-dflash-pflash-kvflash>
