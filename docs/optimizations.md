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
| SageAttention | Experimental Python manifest | No maintained llama.cpp or vLLM connector |
| Speculative Prefill | Experimental manifest | Old vLLM monkeypatch and draft-model quality tradeoffs |
| FastDMS | Standalone engine only | Requires DMS checkpoints and scheduler/cache changes |
| RocketKV | Research only | Research snapshot and non-commercial source license |
| Lexico | Research only | WIP and requires per-model dictionaries |
| AdaDecode | Blocked | No source license and requires trained prediction heads |
| Resonance YaRN | Native backend required | Fine-tuning implementation is not a llama.cpp plugin |

## Composition rules

The planner treats token eviction methods as alternatives unless a combination
has a dedicated implementation and evaluation. It also rejects cache-storage
plugins when the selected KV representation has not been validated. Adapters
are not inserted automatically: decompression or layout conversion can consume
more time and memory than the optimization saves.

Startup benchmarking and accuracy qualification should operate only on a
shortlist that has already passed this compatibility plan. Results must be
cached by model, runtime, plugin versions, GPU, driver, and workload shape; a
full Cartesian benchmark at every startup is intentionally out of scope.

## Upstream references

- LMCache quickstart: <https://docs.lmcache.ai/getting_started/quickstart.html>
- LMCache configuration: <https://docs.lmcache.ai/api_reference/configurations.html>
- LMCache storage plugins: <https://docs.lmcache.ai/developer_guide/extending_lmcache/storage_plugins.html>
- Maru: <https://github.com/xcena-dev/maru>
- MInference: <https://github.com/microsoft/MInference>
- SageAttention: <https://github.com/thu-ml/SageAttention>
- FastDMS: <https://github.com/shisa-ai/FastDMS>
- RocketKV: <https://github.com/NVlabs/RocketKV>
- Speculative Prefill: <https://github.com/Jingyu6/speculative_prefill>
- Lexico: <https://github.com/krafton-ai/lexico>
- AdaDecode: <https://github.com/weizhepei/AdaDecode>
- Resonance RoPE: <https://github.com/sheryc/resonance_rope>
