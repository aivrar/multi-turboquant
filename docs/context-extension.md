# Context Extension Notes

This note covers the context-length work added for issue
[#11](https://github.com/aivrar/multi-turboquant/issues/11). It is separate
from KV-cache compression: context extension changes how positions are encoded,
while Multi-TurboQuant's compression methods change how the KV cache is stored
or pruned.

## What ships

Multi-TurboQuant now exposes llama.cpp startup flags for context extension:

- `-c` / `--ctx-size` through the existing `context_size` argument.
- `--rope-scaling {none,linear,yarn}`
- `--rope-scale`
- `--rope-freq-base`
- `--rope-freq-scale`
- `--yarn-orig-ctx`
- `--yarn-ext-factor`
- `--yarn-attn-factor`
- `--yarn-beta-slow`
- `--yarn-beta-fast`

Use `LlamaCppContextExtensionConfig` with `get_llamacpp_args()` or
`get_llamacpp_command()`:

```python
from multi_turboquant import get_preset
from multi_turboquant.integration import (
    LlamaCppContextExtensionConfig,
    get_llamacpp_command,
)

config = get_preset("balanced")
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
```

The web UI command generator includes the same controls plus a `llama-server`
scanner. The scanner runs `--help` on the selected binary and reports whether it
advertises RoPE/YaRN, KVarN, TriAttention, speculative decoding, DFlash, and
`/props` support.

## Guardrails

The wrapper validates combinations before emitting a command:

- YaRN fields require `rope_scaling="yarn"` or no explicit scaling value. If a
  YaRN field is provided and `rope_scaling` is omitted, the wrapper selects
  YaRN.
- `rope_scale` and `rope_freq_scale` are mutually exclusive.
- `rope_scaling="none"` rejects explicit scale/frequency-scale overrides.
- Numeric scale/base/context values must be positive, except
  `yarn_ext_factor`, which may be zero.
- Extra context-extension args are accepted only as non-empty strings.

These are launch-time flags. Upstream llama.cpp documents `POST /props`, but
the endpoint currently does not expose runtime RoPE/YaRN mutation options, so
Multi-TurboQuant does not claim runtime context extension.

## Research summary

**Position Interpolation** extends context by interpolating position indices
inside the original training range. It is the baseline idea behind many RoPE
extension strategies, but it can reduce positional resolution when pushed too
far.

**YaRN** improves RoPE extrapolation by mixing interpolation with frequency
corrections. llama.cpp exposes YaRN-related startup flags, so this is the best
current integration point for a command generator.

**Resonance RoPE** refines RoPE interpolation to improve trained-short,
tested-long behavior and can improve YaRN-style extension. The public
`sheryc/resonance_rope` implementation is useful research material, but it is
not a drop-in llama.cpp wrapper. Integrating it properly would require backend
changes to the RoPE embedding path or model conversion/training work.

**LongRoPE** uses non-uniform positional interpolation and progressive
fine-tuning to reach very long contexts. It is a model/runtime method, not a
simple `llama-server` flag.

## Implementation decision

For this repo, the robust v1 is a llama.cpp flag wrapper plus scanner:

- It works with current llama.cpp binaries that already expose RoPE/YaRN flags.
- It does not vendor research repos with stale or model-specific code paths.
- It keeps context extension independent from KV-cache compression.
- It gives the UI enough scanner feedback to warn when a selected binary cannot
  support the requested feature set.

Resonance RoPE and LongRoPE remain documented research targets. They should be
implemented only as a backend/model-extension module when there is a maintained
runtime target or a controlled conversion/fine-tuning path.

## KVarN and TriAttention note

KVarN and TriAttention are still intentionally blocked together for Godzilla
llama.cpp. The Godzilla code path rejects the pair until KVarN-aware
TriAttention pruning is implemented (KVX-2). Multi-TurboQuant mirrors that
guardrail in command generation, compatibility checks, and UI warnings.

Godzilla TriAttention calibration is a separate, per-model offline step. Its
runtime `.triattention` file is produced by a compatible Python calibrator from
the matching Hugging Face checkpoint, not from the GGUF. The current Godzilla
checkout does not bundle `calibrate-triattention.py`, and its current lab policy
treats TriAttention as experimental, opt-in, and manually calibrated. The
llama-cpp-turboquant repository cited by older documentation also does not
currently contain the documented Python script or an implemented native
calibration flag, so it is not a safe implementation source. Multi-TurboQuant
reuses a finished stats file and automatically selects a bundled calibrator if
a compatible future checkout ships one; it does not synthesize unverified
model statistics.

## Sources

- llama.cpp server documentation:
  <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- Position Interpolation:
  <https://arxiv.org/abs/2306.15595>
- YaRN:
  <https://arxiv.org/abs/2309.00071>
- Resonance RoPE paper:
  <https://arxiv.org/abs/2403.00071>
- Resonance RoPE reference repository:
  <https://github.com/sheryc/resonance_rope>
- LongRoPE:
  <https://arxiv.org/abs/2402.13753>
- Godzilla llama.cpp:
  <https://github.com/atomicmilkshake/godzilla-llama.cpp>
- KVarN:
  <https://arxiv.org/abs/2606.03458>
- KVarN reference repository:
  <https://github.com/huawei-csl/KVarN>
- TriAttention:
  <https://arxiv.org/abs/2604.04921>
