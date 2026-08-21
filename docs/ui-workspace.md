# Local UI Workspace

`run_ui.py` is a localhost browser interface for Multi-TurboQuant. It uses the
Python standard-library HTTP server and embedded HTML, CSS, and JavaScript, so
there is no npm install or frontend build step.

## Start the UI

```bash
python run_ui.py
```

The default address is `http://localhost:9092`. The server binds to
`127.0.0.1`, even when the page is opened through the `localhost` name.

Useful options:

```bash
python run_ui.py --port 8080
python run_ui.py --no-browser
python run_ui.py --settings-file ./private/ui-settings.json
python run_ui.py --read-only
```

`--read-only` retains planning, benchmark, command-generation, and discovery
features while disabling settings writes, dependency-environment creation, and
Godzilla preparation and model process controls.

## Quick Run

Quick Run contains the frequently used controls:

- detected hardware, cache methods, and presets;
- multi-agent capacity planning and the synthetic cache-method benchmark;
- llama.cpp capability scanning;
- KV-cache quantization, CUDA weight-sharing, and RoPE/YaRN context-extension
  options;
- model selection from the configured model folder;
- safe command preview plus managed `llama-server` start, status, logs, and stop.

The less-frequently changed context-extension, speculative-decoding,
TriAttention, and weight-sharing controls are grouped in collapsed advanced
sections. Expand them when you need to change a non-default launch option;
the generated command still shows every selected flag before a process starts.

The model library recognizes `.gguf`, `.safetensors`, `.bin`, `.pt`, and `.pth`
files, plus Transformers directories containing `config.json` and weights.
Discovery does not load or execute model files. The managed llama.cpp launcher
is intentionally limited to existing `.gguf` files inside the saved model
folder; the other formats are discovery-only.

The launcher uses the generated argument array directly without a shell. Only
one managed `llama-server` process can run at a time, and it is stopped when the
UI server shuts down. A binary path can be supplied in the command controls, or
`llama-server` can be resolved from `PATH`.

For patched Godzilla TriAttention, the stats-path control expects the finished
binary `.triattention` file. Producing that file is a model-specific offline
calibration against the matching Hugging Face checkpoint; a discovered GGUF
alone does not contain the required pre-RoPE query statistics.

The default **Generate stats + convert** mode uses the official TriAttention
checkout's `scripts/calibrate.py`, keeps its `.pt` output, converts that payload
to Godzilla v1, and strictly reads the final artifact back. Selecting a
recognized official checkout also selects its reviewed `mtq-env` profile. Once
that isolated environment is created and validated, the UI finds its Python
automatically. Select a non-empty calibration text and the exact Hugging Face
model. The official script may download model data and uses
`trust_remote_code=True`; only continue with a model source you trust.

Hugging Face is the default tokenizer. **Gigatoken (parity required)** is an
opt-in choice for the reviewed official and domvox Python modes. The wrapper accepts the reviewed
0.10.x API and, before model loading, compares every token ID for the complete
selected calibration text against Hugging Face with the same truncation and
maximum length. It stops on any mismatch.

**Convert existing official .pt** accepts stats that were already produced by
the official script and skips the expensive model forward pass. It still loads
the matching Hugging Face configuration to verify layer/head/RoPE metadata and
then performs the same strict Godzilla artifact validation.

This path does not call `llama-cli`. The UI does not offer a native llama-cli
mode because the current Godzilla binary has no implemented calibration command.
The **Godzilla checkout script** mode retains the older
`scripts/ensure-triattention.ps1` flow for compatible checkouts. KVarN is not
calibrated: it remains a launch-time K/V cache selection.

The **domvox TRIA v2 (experimental)** mode accepts a recognized
`domvox/triattention-ggml` checkout and its `triattention_calibrate.py` script.
It runs the calibrator in the selected Python environment, validates the
resulting TRIA v2 file, and adapts it to Godzilla v1 only after the explicit
lossy-conversion acknowledgement is selected. Layer-budget scales and
attention scale are not representable in Godzilla v1 and are reported as
dropped. The matching Hugging Face model is required for shape and RoPE checks;
a GGUF by itself is not sufficient.
When Gigatoken is selected, domvox is launched through the same fail-closed
parity wrapper as the official calibrator; its script receives only supported
arguments after parity succeeds.

Calibration lengths from 128 through 200,000 tokens are supported. Values above
32,768 require the explicit **Allow long calibration** checkbox and produce a
one-shot memory/runtime warning; the UI does not silently chunk or aggregate
the upstream calibrator's sequence. It permits one calibration job at a time so
two model loads cannot overlap. A successful dependency preflight reports the
selected CUDA device's free and total VRAM, but that snapshot is not a memory
estimate or guarantee for a 200,000-token run.

If the automatically selected managed interpreter is missing `accelerate` or
another declared dependency, the plan offers **Repair TriAttention
dependencies** only after the reviewed environment profile passes its host/tool
preflight. The confirmed action ignores unrelated interpreter and local-source
overrides, re-synchronizes the pinned `triattention` environment with a
conservative two-job limit, validates Torch, Transformers, Accelerate, and
TriAttention, and checks the preparation plan again when the background job
finishes. An incompatible manually selected Python is cleared when managed
repair starts; its environment is not modified.

With no explicit selection, calibration checks at most eight candidates in
order: the owned TriAttention environment, active virtual/Conda, current Python,
other managed environments, `PATH`, and conventional pyenv roots. Every probe
runs in isolation and reports every missing import across Torch, Transformers,
Accelerate, NumPy, Safetensors, Hugging Face Hub, Tokenizers, and SentencePiece,
plus Gigatoken or FlashAttention only when used. Multi-TurboQuant does not mix
`site-packages` between environments. A missing import cannot be bypassed by a
manual dependency override, and the selected interpreter is checked again just
before the background process starts.

**Scan Python and pyenv environments for Gigatoken** checks at most 64
interpreters from the current process, `PATH`, active virtual/Conda environments,
the managed environment root, and conventional pyenv/pyenv-win roots. The scan
does not recurse through drives and uses bounded concurrent probes. Selecting a
compatible result fills both the Python and tokenizer controls. On Linux and
macOS the application deliberately retains `.venv/bin/python` as a lexical path;
following its final symlink to `uv`'s or the system's base executable would lose
the virtual-environment prefix and cause false missing-package reports.

The domvox source inspector also requires the sibling
`triattention_common.py`. Calibration and the subsequent Transformers-backed
conversion both execute under the selected, validated Python. On failure, the
job card exposes a redacted diagnostic report and writes an atomic
`<output>.<job-id>.diagnostics.json` beside the requested artifact. It covers
the exact process and working directory, host and selected interpreter details,
all dependency imports and tracebacks, relevant path/file state, source
revision, discovery attempts, OS and CUDA/toolchain state, VRAM, disk capacity,
log tails, and recovery steps; common token and URL credentials are redacted.

**Generate generic starter text** creates deterministic offline text under
`<model-root>/.mtq/calibration/` and selects it as the calibration input. It
never overwrites unrelated files, uses schema/completion markers, and safely
handles simultaneous requests and filesystems without hard-link support. Exact
tokenization depends on the selected model, and representative domain text
remains preferable for final quality qualification.

The CUDA weight-share controls only prepare the external Linux/CUDA helper's
environment:

- `LD_PRELOAD` loads the helper before llama.cpp so it can intercept CUDA
  allocations.
- `MODEL_SIZE` is the measured model-weight allocation in bytes, not the GGUF
  file size. Use `0` for one discovery run, then reuse the reported value.
- `MODEL_SIZE_TOLERANCE` permits a small allocation-size difference. Start at
  zero; a broad value can match an unrelated allocation.
- `CUDA_VRAM_IPC_NAME` identifies one sharing group. All members use the same
  unique name; unrelated groups use different names.
- `CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC` lets workers wait for master metadata.
- `CUDA_VRAM_IPC_SUPPRESS_MASTER_FREE` is a specialized lifetime override;
  leave it off unless the helper workflow requires it.
- the caller, depth, and normal-allocation trace settings are diagnostics that
  add log volume and overhead.

The helper shares weights, not KV caches or contexts. Participating processes
must use matching model, build, CUDA device, size, and IPC-name settings.
The source picker recognizes the helper only at the reviewed upstream origin
and exact commit. Use `mtq-weight-share inspect`, `plan`, `build --yes`, and
`validate` to compile and verify the Linux x86-64 ELF helper before selecting
it. Validation checks the exported allocation hooks and dynamic dependencies;
`MODEL_SIZE=0` reconnaissance remains mandatory afterward.

## Setup & Add-ons

Setup & Add-ons holds configuration that is changed less often:

- the default model folder;
- the isolated dependency-environment root;
- an optional side-by-side CUDA toolkit root or `nvcc` path for native builds;
- one or more add-on/source roots;
- an optional FlashAttention source checkout;
- environment profile status and explicit creation controls.

Composition Lab is the read-only decision workspace. It lists the guarded
execution profiles, validates one profile against host/artifact/feature inputs,
routes a workload with an explicit baseline fallback, and performs analytical
KV byte-capacity simulation. Its Godzilla card previews a pinned
`09214b160` CUDA build plan for the reviewed SM86/SM89 targets. These actions do
not install packages, alter source trees, compile binaries, or start runtimes.

The scanners are bounded and inspect only configured roots. Add-on scanning
runs at UI startup and shortly after its root list changes, while the manual
button remains available. Results report the resolved roots, scan depth,
directory count, invalid roots, and missing source markers. The add-on source
scanner does not search an entire drive, follow directory symlinks, import
third-party packages, or run source code. Recognized add-ons currently include
FlashAttention, FastDMS, JetSpec, Jet-Long, LMCache, MInference, Proxima,
SageAttention, TriAttention, Godzilla, and llama.cpp checkouts. The source
picker can also inspect local folders for guided or blocked entries including
Maru, Speculative Prefill, RocketKV, Lexico, AdaDecode, Resonance YaRN,
Lucebox, ChunkLlama, RaBitQCache, ScoPE, DuoAttention, IceCache, and the
PFlash/KVFlash llama.cpp fork, plus domvox TriAttention and the reviewed CUDA
weight-share source. A prepared `mtq-godzilla-compose` tree is recognized from
its manifest and checked with the hash-bounded composition inspector rather
than labeled as a generic Godzilla checkout. Resonance-JetLong is instead a planner-only composition
record because it does not have a standalone source checkout. The picker also recognizes the separate
`chynggi/gigatoken-llama.cpp` checkout as an informational experimental
Windows x64/Linux x86_64 runtime fork. Discovery does not make it a Godzilla
build or compile it. Use `mtq-godzilla-gigatoken` separately to create and
qualify either pinned Godzilla v0.3.7 or the exact `09214b160` compatibility
profile. Use `mtq-godzilla-compose` for the separate exact-`09214b160`
request-gated PFlash and complete-idle-slot KVFlash overlay. Both commands
refuse arbitrary or existing target trees. Renamed
Godzilla trees are recognized by `scripts/godzilla-paths.ps1`. FlashAttention inspection checks
the expected source markers and reports version and Git remote metadata when
available.

For the nine reviewed Python add-ons, a recognized checkout has a **Use for
profile** action. It fills the local-source profile and path controls. Refresh
the profile plan before creation: the plan validates the checkout markers,
then `uv` builds that package and resolves its dependencies in the selected
isolated environment. The scanner never imports or executes the checkout, and
unrecognized folders cannot be substituted into a profile.

Selecting a blocked add-on source is informational only: the scanner reports
the reviewed markers, upstream metadata, repository-specific host/artifact
requirements, and the reason automatic installation is not offered. It never
turns a blocked profile into an installable one or executes the selected source.
Current Maru layouts are recognized by `pyproject.toml`/`setup.py` plus
`maru_resource_manager`, `maru_server`, or `maru`; a root `CMakeLists.txt` is
not expected. Maru setup remains guided because upstream requires Linux host
services and a configured CXL DAX device or its documented emulation.

A local checkout changes only the selected package's source. It does not relax
the profile's operating-system, Python, PyTorch, or CUDA ABI requirements. In
particular, native extensions must still use an `nvcc` toolkit whose major
matches the profile's PyTorch CUDA build; the plan now displays this explicitly.

A recognized Godzilla tree has a separate action. Inspection reports its known
source markers, KVarN and TriAttention flags, preparation/resolver scripts,
bundled-calibrator status, and known `llama-server` build locations. A recognized
official TriAttention checkout has a separate action that fills its validated
`scripts/calibrate.py` path and selects its dependency profile.
The domvox action fills `triattention_calibrate.py` and the experimental
calibration mode when its `triattention_common.py` and `TRIA_FORMAT.md` markers
are present.
If the chosen script belongs to the other calibration mode, the planner now
identifies the official/domvox mismatch and names the appropriate mode.
The UI does not configure or compile a Godzilla CMake project automatically.
Use Godzilla's documented process for ordinary checkouts, or the explicit
`mtq-godzilla-gigatoken` workflow for its pinned tokenizer runtime, or
`mtq-godzilla-compose` for the bounded PFlash/KVFlash profile. Composition Lab
can render the latter's exact build plan, including SM86/SM89 selection, but
execution remains CLI-only and does not patch a selected UI source folder.
TriAttention remains experimental and
model-specific. Existing `.triattention` output is reused only after its v1
header, dimensions, sampled indices, numeric arrays, and exact file length pass
validation.

Managed Linux plans report Debian 12 and 13 explicitly. Repair cleanly
preserves the old owned `.venv`, validates a replacement, and restores the old
environment if synchronization fails. For deeper troubleshooting,
`mtq-env diagnose PROFILE` emits a redacted report covering distribution,
lexical/resolved interpreters, Python prefixes, imports, Accelerate, and the
CUDA/toolchain state.

Creating a dependency environment reuses the reviewed `mtq-env` profiles and
runs as a background job. The UI requires explicit confirmation because the
operation creates files, resolves packages, and can build native extensions.
The optional source-build checkbox is accepted only for profiles that declare a
reviewed source-build path. Progress and command output appear in the job list.
Local-checkout builds use `MAX_JOBS=2` by default and expose a bounded job-count
control. Existing environments are import-validated before the UI offers
Create/Repair. If that check is a false negative, the manual override marks the
environment as manually accepted and displays a warning instead of claiming it
was validated.
The CUDA override does not install or replace a toolkit. It selects an existing
toolkit whose major version matches the profile's PyTorch build and exports it
through `CUDA_HOME`, `CUDA_PATH`, and `PATH` for the background job. Blocked
profiles are informational records and intentionally have no Create action.

Godzilla preparation also runs as a background job and verifies that the
expected `.triattention` output exists. Custom outputs are limited to the
selected checkout or model folder. The checkout must remain inside a saved
add-on root, and the GGUF must remain inside the saved model root.

The Hardware card reports host RAM and discrete GPU VRAM independently, plus an
optional combined capacity inventory. RAM does not replace VRAM for CUDA model
loading or KV-cache allocations. Apple unified memory is labeled and is not
added to itself in the combined figure.

## Persistent defaults

By default, the UI saves a versioned JSON document at:

```text
~/.multi-turboquant/ui-settings.json
```

It remembers workspace paths and normal form values, including the active UI
view. The default path is outside the Git checkout, so normal pulls do not
remove it. Writes use an atomic file replacement. Setup & Add-ons also provides JSON
export and import for moving a configuration between machines; imported paths
still need to exist on the destination machine.

Use `--settings-file` to isolate settings for a particular checkout or test
session. Reset restores the built-in defaults and does not delete models,
add-on checkouts, or dependency environments.

## Safety boundary

The UI is designed as a local operator tool, not a remotely exposed service. It
binds only to `127.0.0.1`, does not enable wildcard cross-origin access, caps
JSON request bodies, and sends a restrictive content security policy. Do not
place it behind a public proxy without adding authentication and a production
HTTP boundary.
