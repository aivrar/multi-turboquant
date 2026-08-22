# Recent inference research review — August 2026

This is a dated research snapshot, not a promise that every paper is a supported
runtime. It was reviewed on **2026-08-22** against Multi-TurboQuant's existing
catalog so older ideas already represented by LMCache, MInference, TriAttention,
JetSpec, Proxima, Jet-Long, DuoAttention, IceCache, PFlash/KVFlash, CUDA weight
sharing, and the other current entries were not proposed again under new names.

The review prioritizes primary papers and official source repositories. Published
performance numbers are evidence for deciding what to investigate; they are not
project benchmarks and must not be extrapolated to a different model, GPU, context,
or concurrency level.

## Newest distinct candidates

| Date | Candidate | New lever | Evidence and code status | Multi-TurboQuant action |
|---|---|---|---|---|
| 2026-08-08 | [OasisKV](https://arxiv.org/abs/2608.08097) | Keeps the full decode KV cache below HBM and uses speculative lookahead to prefetch a sparse working set | The paper reports up to 2.1x multi-GPU throughput and materially lower admitted/host KV in its vLLM experiments; no public implementation was identified | Track as a high-value off-HBM design; do not advertise an install or invent a patch without source |
| 2026-08-07 | [CoinRAG](https://arxiv.org/abs/2608.07458) | Reuses fine-grained semantic “nugget” KV caches instead of entire retrieved chunks | The paper reports a better latency/quality frontier for its LongBench RAG setup; no public runtime was identified | Track as the strongest context-sharing direction for repeated RAG, separate from ordinary prefix caching |
| 2026-08-04 | [NOVA-KV](https://arxiv.org/abs/2608.04074) / [code](https://github.com/Amir-zsh/nova-kv) | Attention-product-aware transforms plus vector quantization at two bits per KV element | Apache-2.0 code is available as a research fork of SGLang 0.5.10; the repository ships a gpt-oss-20b bundle and documents H100 evaluation | Added to the catalog and environment planner as a guarded, non-installable separate runtime |
| 2026-08-03 | [ARCHead](https://arxiv.org/abs/2608.02703) / [code](https://github.com/suayptalha/archead) | Compresses the dense LM output head that block weight quantizers can leave behind | MIT code is available; the paper reports 3.7–3.9x output-head storage reduction with a small measured quality delta | Added as a pinned isolated source profile for Transformers experiments |
| 2026-08-02 | [RestoreKV](https://arxiv.org/abs/2608.01247) / [KVPress implementation](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/restorekv_press.py) | Adds learned restore tokens to repair information lost by aggressive query-agnostic KV eviction | Apache-2.0 inference code and model-specific adapters are available; the paper reports large recovery at tight budgets | Added as a pinned isolated KVPress profile with explicit functional-quality gates |
| 2026-07-06 | [DSpark](https://arxiv.org/abs/2607.05147) / [speculator code](https://github.com/vllm-project/speculators) | Semi-autoregressive drafting plus confidence- and load-aware verification length | Training/speculator code and a narrow pretrained path exist, but serving spans a separate moving vLLM implementation and recent integration failures have been reported | Added as research-only; blocked until an exact vLLM release passes parity and concurrency tests |
| 2026-07-01 | [ELDR](https://arxiv.org/abs/2607.00466) | Routes disaggregated MoE decode requests by predicted expert locality rather than load alone | The paper reports unchanged outputs and lower TPOT on deployments up to 40 GPUs; no maintained standalone integration was identified | Track for a future distributed/MoE router, not local single-node inference |
| 2026-05-17 | [VeriCache](https://arxiv.org/abs/2605.17613) | Uses lossy compressed KV for drafting while an off-GPU full KV verifies output | The paper reports identical full-cache output and up to 4x throughput, but no public implementation was identified | Use as the design target for a future verified-cache composition; no install claim |

### Secondary directions reviewed

| Date | Candidate | Decision |
|---|---|---|
| 2026-07-29 revision | [RedKnot](https://arxiv.org/abs/2606.06256) | Its head-aware KV substrate unifies position-independent reuse, compression, hot/cold separation, and distributed placement. No public runtime was identified, so it remains architecture guidance rather than a catalog entry. |
| 2026-06-02 | [Kernel Forge](https://arxiv.org/abs/2607.24762) / [code](https://github.com/TheJoshBrod/KernelForge) | This is relevant to generating hardware-specific CUDA/Triton patches, but it is an external LLM-driven optimization harness, not an inference add-on. Any generated kernel must stay isolated until differential correctness, adversarial-shape, compiler, memory-safety, and target-hardware performance gates pass. |
| 2026-05-10 | [PEEK](https://arxiv.org/abs/2607.02525) | Queue-aware prefix clustering and eviction protection are a strong scheduler direction for repeated-prefix serving. No public implementation was identified, so it is not presented as an available vLLM/SGLang patch. |

## Why these are not duplicates

- **RestoreKV is a quality-repair layer**, not another cache selector. It wraps a
  supported KVzip-family eviction policy with model-specific learned restoration.
- **NOVA-KV changes the representation and distortion objective**. Existing
  TurboQuant/Lexico/Proxima entries do not provide its fixed-width vector-quantized
  SGLang page layout or its query-statistics-derived transforms.
- **ARCHead targets model weights outside the KV cache**. CUDA weight sharing reduces
  duplicate allocations across compatible processes; ARCHead reduces one model's
  persistent output-head representation. Those benefits and failure modes differ.
- **DSpark is not JetSpec renamed**. Both occupy speculative decoding and therefore
  fail closed if combined, but DSpark adds a sequential Markov component and
  confidence/load-aware verification scheduling.
- **OasisKV, CoinRAG, ELDR, and VeriCache occupy system layers not currently
  implemented here**: decode-time tiered prefetch, semantic RAG cache composition,
  MoE locality routing, and full-cache verification respectively.

## Recommended experimental sequence

1. **RestoreKV quality-repair trial.** Start with a released adapter and compare
   full cache, the base KVzip press, and RestoreKV at identical budgets. Measure
   RULER/needle retrieval, executable code, tool-call schema validity, peak memory,
   construction cost, and decode throughput. A perplexity-only pass is insufficient.
2. **ARCHead complement trial.** Apply it only to a supported Transformers model
   whose block weights are already quantized but whose output head remains dense.
   Gate on logits, perplexity, downstream accuracy, serialization/reload, tied
   embeddings, persistent bytes, and end-to-end latency.
3. **NOVA-KV reproduction in its own runtime.** Reproduce BF16 versus NOVA in the
   upstream fork before considering any port. Keep its calibration artifacts,
   SGLang fork, and GPU-specific kernels out of every existing add-on environment.
4. **Verified adaptive-cache prototype.** If VeriCache code becomes available, test
   a full-KV lower tier plus compressed-draft/verification loop. This is the best
   current route to aggressive memory savings with a principled remedy for silent
   code/tool-call degradation.
5. **Context-sharing prototype.** When CoinRAG code is released, compare semantic
   nugget reuse with ordinary prefix reuse using identical retrieved documents and
   include cache-build/storage costs. Do not splice arbitrary KV segments without
   position and attention correctness evidence.
6. **Distributed-only lane.** Evaluate OasisKV and ELDR only on infrastructure that
   actually has multiple memory tiers or prefill/decode workers. They are not honest
   speed claims for a single consumer GPU.

## Composition and side-effect controls

The default policy remains fail closed:

- Two methods that rewrite KV representation, selection, attention kernels, or the
  speculative decoder do not compose merely because both import successfully.
- Lossy cache trials need task-level quality checks, long-generation checks, and a
  full-cache fallback. Perplexity alone can miss functional breakage.
- Report TTFT, time per output token, aggregate throughput, peak VRAM/RAM, accepted
  speculative length, and quality together. Optimizing one metric can regress another.
- Calibration artifacts are model-, tokenizer-, head-layout-, and often runtime-
  specific. Record hashes and refuse mismatches.
- Published H100 or multi-node results are hypotheses for other hardware, not
  expected values.
- Hugging Face tokens remain optional for public artifacts. Authentication can
  improve limits and is required for gated content, but the project does not bypass
  provider access controls or silently redirect downloads to untrusted mirrors.

## Self-suggested build directions

The literature points to three useful designs that are not safe to claim as built yet:

- **Verified tiered context:** retain authoritative full KV in CPU/remote memory,
  draft from a compressed or sparse HBM working set, and verify before committing
  output. This combines the risk remedy of VeriCache with the memory hierarchy of
  OasisKV.
- **Reusable semantic cache with repair:** cache fine-grained RAG units, but attach
  position metadata and selectively recompute boundary tokens after composition.
  This would target CoinRAG-like reuse without assuming arbitrary KV concatenation
  is exact.
- **Metric-aware automatic routing:** select full, compressed, restored, or
  speculative paths from measured memory pressure, prefix reuse, task exactness,
  concurrency, and acceptance history. Automatic routing should choose only among
  profiles that have already passed their own validation gates; it must not generate
  compatibility patches at runtime.
- **Offline patch laboratory:** a Kernel Forge-style search may generate candidate
  kernels in a disposable environment, but promotion must require deterministic
  differential tests, odd/edge shapes, multiple seeds and dtypes, sanitizer/tool
  checks where available, target-device benchmarks, and an ordinary-kernel fallback.
  Generated code is never an automatic compatibility fix merely because it compiles.

These are roadmap proposals. Implementing them requires source availability, an
explicit runtime target, correctness witnesses, and hardware-backed benchmarks.
