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
model process controls.

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

## Setup & Add-ons

Setup & Add-ons holds configuration that is changed less often:

- the default model folder;
- the isolated dependency-environment root;
- one or more add-on/source roots;
- an optional FlashAttention source checkout;
- environment profile status and explicit creation controls.

The scanners are bounded and inspect only configured roots. They do not search
an entire drive, follow directory symlinks, import third-party packages, or run
source code. Recognized add-ons currently include FlashAttention, FastDMS,
LMCache, MInference, SageAttention, and llama.cpp checkouts. FlashAttention
inspection checks the expected source markers and reports version and Git
remote metadata when available.

Creating a dependency environment reuses the reviewed `mtq-env` profiles and
runs as a background job. The UI requires explicit confirmation because the
operation creates files, resolves packages, and can build native extensions.
The optional source-build checkbox is accepted only for profiles that declare a
reviewed source-build path. Progress and command output appear in the job list.

## Persistent defaults

By default, the UI saves a versioned JSON document at:

```text
~/.multi-turboquant/ui-settings.json
```

It remembers workspace paths and normal form values, including the active UI
view. Writes use an atomic file replacement. Setup & Add-ons also provides JSON
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
