# SPDX-License-Identifier: MIT
"""Named preset configurations for common use cases."""

from __future__ import annotations

from .config import CacheConfig, CacheMethod


# --- Named Presets ---

PRESETS: dict[str, CacheConfig] = {

    # === K-only presets (zero speed cost, K compressed, V stays FP16) ===

    "k_only_iso": CacheConfig(
        k_method=CacheMethod.ISO3,
        v_method=CacheMethod.FP16,
        # ISO3 K-only: beats FP16 decode speed on Ampere.
        # No calibration needed. ~5x K compression.
    ),
    "k_only_planar": CacheConfig(
        k_method=CacheMethod.PLANAR3,
        v_method=CacheMethod.FP16,
        # Planar3 K-only: simplest transform, Metal support.
        # ~1.4% slower than FP16 but no calibration needed.
    ),

    # === Symmetric presets (full K+V compression, maximum VRAM savings) ===

    "balanced": CacheConfig(
        k_method=CacheMethod.TURBO3_TCQ,
        v_method=CacheMethod.TURBO3_TCQ,
        # Best quality at 5x compression. TCQ improves quality vs base turbo3.
        # Requires calibration (turboquant_kv.json).
    ),
    "speed": CacheConfig(
        k_method=CacheMethod.TURBO3,
        v_method=CacheMethod.TURBO3,
        # Fastest symmetric decode on Ampere GPUs.
        # ~12.5% slower than FP16 but 5x K+V compression.
        # Requires calibration.
    ),
    "max_compression": CacheConfig(
        k_method=CacheMethod.TURBO2_TCQ,
        v_method=CacheMethod.TURBO2_TCQ,
        # Maximum compression: ~7x K+V savings.
        # Noticeable quality degradation. TCQ helps.
        # Requires calibration.
    ),
    "quality": CacheConfig(
        k_method=CacheMethod.TURBO4,
        v_method=CacheMethod.TURBO4,
        # Near-lossless: ~3.8x compression.
        # Minimal quality impact. Requires calibration.
    ),

    # === Combined presets (TriAttention + quantization) ===

    "extreme": CacheConfig(
        k_method=CacheMethod.TURBO3_TCQ,
        v_method=CacheMethod.TURBO3_TCQ,
        triattention_enabled=True,
        triattention_budget=2048,
        triattention_window=512,
        # Token eviction (16x) * bit compression (5x) = ~80x total.
        # For fitting 70B models on 12GB GPUs at long context.
        # Quality drops noticeably — for throughput-critical workloads.
    ),
    "long_context": CacheConfig(
        k_method=CacheMethod.ISO3,
        v_method=CacheMethod.FP16,
        triattention_enabled=True,
        triattention_budget=4096,
        triattention_window=1024,
        # Token eviction + free K compression.
        # Good quality retention at moderate token budgets.
    ),

    # === Calibration-free presets (no setup required) ===

    "no_calibration_symmetric": CacheConfig(
        k_method=CacheMethod.ISO3,
        v_method=CacheMethod.ISO3,
        # No calibration files needed. ~5x compression.
        # Slower than turbo3 symmetric on Ampere (~21% vs FP16).
    ),
    "no_calibration_quality": CacheConfig(
        k_method=CacheMethod.ISO4,
        v_method=CacheMethod.ISO4,
        # No calibration, ~3.8x compression.
        # Higher quality than iso3.
    ),

    # === Agent-optimized presets (multi-slot, dual GPU) ===

    "agents_8x16k": CacheConfig(
        k_method=CacheMethod.TURBO4,
        v_method=CacheMethod.TURBO4,
        # 8 agents at 16K context on 32B Q4 + dual GPU.
        # turbo4 gives near-lossless 3.8x compression.
        # Needs calibration. ~9 GB headroom on 3090+3060.
    ),
    "agents_4x8k_70b": CacheConfig(
        k_method=CacheMethod.TURBO4,
        v_method=CacheMethod.TURBO4,
        # 4 agents at 8K on 70B Q3 + dual GPU tensor split.
        # Tight VRAM — turbo4 is the least aggressive that fits.
    ),
    "agents_8x8k_70b": CacheConfig(
        k_method=CacheMethod.TURBO2_TCQ,
        v_method=CacheMethod.TURBO2_TCQ,
        # 8 agents at 8K on 70B Q3 + dual GPU.
        # Needs aggressive compression to fit. TCQ improves quality.
    ),
    "agents_16x4k": CacheConfig(
        k_method=CacheMethod.TURBO2_TCQ,
        v_method=CacheMethod.TURBO2_TCQ,
        # 16 agents at 4K context — maximum parallelism.
        # For orchestrator + sub-agent swarms.
    ),

    # === Baselines ===

    "fp16": CacheConfig(
        k_method=CacheMethod.FP16,
        v_method=CacheMethod.FP16,
    ),
    "q8": CacheConfig(
        k_method=CacheMethod.Q8_0,
        v_method=CacheMethod.Q8_0,
    ),
}


def get_preset(name: str) -> CacheConfig:
    """Get a named preset configuration.

    Raises:
        KeyError: If the preset name is not found.
    """
    if name not in PRESETS:
        raise KeyError(
            f"Unknown preset '{name}'. "
            f"Available: {', '.join(sorted(PRESETS.keys()))}"
        )
    return PRESETS[name]


def list_presets() -> dict[str, str]:
    """Return preset names with short descriptions."""
    descriptions = {
        "k_only_iso": "ISO3 K-only — free 5x K compression, beats FP16 speed",
        "k_only_planar": "Planar3 K-only — simplest transform, Metal support",
        "balanced": "Turbo3 TCQ symmetric — best quality at 5x compression",
        "speed": "Turbo3 symmetric — fastest symmetric on Ampere",
        "max_compression": "Turbo2 TCQ symmetric — ~7x compression",
        "quality": "Turbo4 symmetric — near-lossless ~3.8x compression",
        "extreme": "TriAttention + Turbo3 TCQ — ~80x total reduction",
        "long_context": "TriAttention + ISO3 K-only — quality-preserving",
        "no_calibration_symmetric": "ISO3 symmetric — no setup required",
        "no_calibration_quality": "ISO4 symmetric — no setup, higher quality",
        "agents_8x16k": "8 agents at 16K ctx — turbo4 on 32B+dual GPU",
        "agents_4x8k_70b": "4 agents at 8K ctx — turbo4 on 70B+dual GPU",
        "agents_8x8k_70b": "8 agents at 8K ctx — turbo2_tcq on 70B+dual GPU",
        "agents_16x4k": "16 agents at 4K ctx — max parallelism swarm",
        "fp16": "No compression (baseline)",
        "q8": "8-bit quantization (minimal compression)",
    }
    return descriptions


def recommend_preset(
    *,
    vram_gb: float,
    model_size_b: float,
    context_length: int,
    has_calibration: bool = False,
    priority: str = "balanced",
) -> str:
    """Recommend a preset based on hardware and model constraints.

    Args:
        vram_gb: Available GPU VRAM in GB.
        model_size_b: Model size in billions of parameters.
        context_length: Target context length in tokens.
        has_calibration: Whether turboquant_kv.json is available.
        priority: "speed", "quality", or "balanced".

    Returns:
        Preset name string.
    """
    # Rough estimate: model weights + KV cache must fit in VRAM
    # ~2 bytes per param for fp16 weights
    model_bytes_gb = model_size_b * 2.0
    remaining_gb = vram_gb - model_bytes_gb

    if remaining_gb <= 0:
        return "max_compression"

    # KV cache size estimate for FP16 baseline
    # Rough: 2 * num_layers * num_kv_heads * head_dim * context * 2 bytes
    # For a 7B model: ~32 layers, 8 KV heads, 128 dim
    kv_fp16_gb = (
        2 * 32 * 8 * 128 * context_length * 2
    ) / (1024**3) * (model_size_b / 7.0)

    ratio_needed = kv_fp16_gb / max(remaining_gb, 0.1)

    if ratio_needed > 20:
        return "extreme"
    elif ratio_needed > 5:
        if has_calibration:
            return "max_compression"
        return "no_calibration_symmetric"
    elif ratio_needed > 2:
        if not has_calibration:
            return "no_calibration_symmetric"
        if priority == "speed":
            return "speed"
        elif priority == "quality":
            return "quality"
        return "balanced"
    elif ratio_needed > 1:
        if has_calibration:
            return "quality" if priority == "quality" else "k_only_iso"
        return "k_only_iso"
    else:
        return "fp16"
