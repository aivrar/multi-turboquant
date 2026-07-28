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
alone does not contain the required pre-RoPE query statistics. The UI does not
mislabel Multi-TurboQuant's Python `.pt` stats as Godzilla-compatible data.

The Setup view can prepare this file through a recognized Godzilla checkout's
own `scripts/ensure-triattention.ps1`. Unless the expected output already
exists, the plan requires a GGUF in the saved model root, a Python executable
with the required packages, and a validated compatible
`calibrate-triattention.py`. The current checkout does not bundle that script.
A matching Hugging Face model can be entered explicitly; when Godzilla's
resolver is present, it can instead try the GGUF metadata mapping. The UI warns
that this may download and load the source checkpoint. KVarN is not calibrated:
it remains a launch-time K/V cache selection and is reported separately.

## Setup & Add-ons

Setup & Add-ons holds configuration that is changed less often:

- the default model folder;
- the isolated dependency-environment root;
- an optional side-by-side CUDA toolkit root or `nvcc` path for native builds;
- one or more add-on/source roots;
- an optional FlashAttention source checkout;
- environment profile status and explicit creation controls.

The scanners are bounded and inspect only configured roots. Add-on scanning
runs at UI startup and shortly after its root list changes, while the manual
button remains available. Results report the resolved roots, scan depth,
directory count, invalid roots, and missing source markers. The scanner does
not search an entire drive, follow directory symlinks, import third-party
packages, or run source code. Recognized add-ons currently include
FlashAttention, FastDMS, LMCache, MInference, SageAttention, Godzilla, and
llama.cpp checkouts. Renamed Godzilla trees are recognized by
`scripts/godzilla-paths.ps1`. FlashAttention inspection checks the expected
source markers and reports version and Git remote metadata when available.

For the five reviewed Python add-ons, a recognized checkout has a **Use for
profile** action. It fills the local-source profile and path controls. Refresh
the profile plan before creation: the plan validates the checkout markers,
then `uv` builds that package and resolves its dependencies in the selected
isolated environment. The scanner never imports or executes the checkout, and
unrecognized folders cannot be substituted into a profile.

A local checkout changes only the selected package's source. It does not relax
the profile's operating-system, Python, PyTorch, or CUDA ABI requirements. In
particular, native extensions must still use an `nvcc` toolkit whose major
matches the profile's PyTorch CUDA build; the plan now displays this explicitly.

A recognized Godzilla tree has a separate action. Inspection reports its known
source markers, KVarN and TriAttention flags, preparation/resolver scripts,
bundled-calibrator status, and known `llama-server` build locations.
Multi-TurboQuant does not configure or compile the CMake project automatically;
use Godzilla's documented build process. Current Godzilla policy treats
TriAttention as experimental and manually calibrated. Its checkout does not
currently bundle `calibrate-triattention.py`, so the UI reports that prerequisite
instead of inventing model statistics. If a compatible checkout later bundles
one, it is selected automatically. Existing `.triattention` output is reused
without requiring Python, a calibrator, or PowerShell.

Creating a dependency environment reuses the reviewed `mtq-env` profiles and
runs as a background job. The UI requires explicit confirmation because the
operation creates files, resolves packages, and can build native extensions.
The optional source-build checkbox is accepted only for profiles that declare a
reviewed source-build path. Progress and command output appear in the job list.
The CUDA override does not install or replace a toolkit. It selects an existing
toolkit whose major version matches the profile's PyTorch build and exports it
through `CUDA_HOME`, `CUDA_PATH`, and `PATH` for the background job. Blocked
profiles are informational records and intentionally have no Create action.

Godzilla preparation also runs as a background job and verifies that the
expected `.triattention` output exists. Custom outputs are limited to the
selected checkout or model folder. The checkout must remain inside a saved
add-on root, and the GGUF must remain inside the saved model root.

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
