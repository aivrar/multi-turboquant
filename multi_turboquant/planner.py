# SPDX-License-Identifier: MIT
"""Capacity planner for multi-agent inference workloads.

Given GPUs, a model, and desired agent count/context, calculates the
optimal compression config and shows exactly what fits.

Usage:
    from multi_turboquant.planner import plan_agents

    result = plan_agents(
        gpus=[{"name": "RTX 3090", "vram_gb": 24}, {"name": "RTX 3060", "vram_gb": 12}],
        model_params_b=32,
        model_quant="Q4_K_M",
        desired_agents=8,
        desired_context=8192,
    )
    result.print_report()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .config import CacheConfig, CacheMethod, METHOD_BITS


# ─── Model weight size estimates ────────────────────────────────────────────────

# Approximate bytes per parameter for common quantization levels
QUANT_BPP: dict[str, float] = {
    "FP16": 2.0,
    "Q8_0": 1.1,
    "Q6_K": 0.83,
    "Q5_K_M": 0.73,
    "Q5_K_S": 0.70,
    "Q4_K_M": 0.60,
    "Q4_K_S": 0.57,
    "Q4_0": 0.55,
    "Q3_K_M": 0.48,
    "Q3_K_S": 0.44,
    "Q2_K": 0.35,
    "IQ4_XS": 0.52,
    "IQ3_M": 0.43,
    "IQ2_M": 0.33,
}

# Typical model architecture parameters (layers, kv_heads, head_dim)
MODEL_ARCHS: dict[str, dict[str, int]] = {
    "7B":  {"layers": 32, "kv_heads": 8,  "head_dim": 128, "heads": 32},
    "8B":  {"layers": 32, "kv_heads": 8,  "head_dim": 128, "heads": 32},
    "13B": {"layers": 40, "kv_heads": 40, "head_dim": 128, "heads": 40},
    "14B": {"layers": 40, "kv_heads": 8,  "head_dim": 128, "heads": 40},
    "32B": {"layers": 64, "kv_heads": 8,  "head_dim": 128, "heads": 64},
    "35B": {"layers": 64, "kv_heads": 8,  "head_dim": 128, "heads": 64},
    "70B": {"layers": 80, "kv_heads": 8,  "head_dim": 128, "heads": 64},
    "72B": {"layers": 80, "kv_heads": 8,  "head_dim": 128, "heads": 64},
}


def estimate_model_size_gb(params_b: float, quant: str = "Q4_K_M") -> float:
    """Estimate model weight size in GB."""
    bpp = QUANT_BPP.get(quant, 0.60)
    return params_b * 1e9 * bpp / (1024**3)


def get_arch(params_b: float) -> dict[str, int]:
    """Get architecture params for a model size."""
    # Find closest match
    size_key = f"{int(params_b)}B"
    if size_key in MODEL_ARCHS:
        return MODEL_ARCHS[size_key]
    # Scale from 7B baseline
    base = MODEL_ARCHS["7B"]
    scale = params_b / 7.0
    return {
        "layers": max(32, int(base["layers"] * math.sqrt(scale))),
        "kv_heads": base["kv_heads"],
        "head_dim": base["head_dim"],
        "heads": max(32, int(base["heads"] * math.sqrt(scale))),
    }


def kv_bytes_per_token(arch: dict[str, int], bits_per_element: float = 16.0) -> float:
    """KV cache bytes per token for one layer."""
    # K + V, each is kv_heads * head_dim * bits/8
    return 2 * arch["kv_heads"] * arch["head_dim"] * bits_per_element / 8


def total_kv_bytes(
    arch: dict[str, int],
    context_len: int,
    num_sequences: int = 1,
    k_bits: float = 16.0,
    v_bits: float = 16.0,
    triattention_budget: int | None = None,
) -> int:
    """Total KV cache bytes across all layers and sequences."""
    effective_ctx = context_len
    if triattention_budget is not None:
        effective_ctx = min(context_len, triattention_budget)

    k_per_token_per_layer = arch["kv_heads"] * arch["head_dim"] * k_bits / 8
    v_per_token_per_layer = arch["kv_heads"] * arch["head_dim"] * v_bits / 8
    per_token = (k_per_token_per_layer + v_per_token_per_layer) * arch["layers"]
    return int(per_token * effective_ctx * num_sequences)


# ─── Planning results ──────────────────────────────────────────────────────────

@dataclass
class AgentSlot:
    """One agent's resource allocation."""
    agent_id: int
    context_length: int
    kv_bytes: int
    gpu: str


@dataclass
class PlanResult:
    """Complete capacity plan for a multi-agent deployment."""
    # Input parameters
    model_name: str
    model_params_b: float
    model_quant: str
    desired_agents: int
    desired_context: int

    # GPU allocation
    gpus: list[dict[str, Any]]
    total_vram_gb: float
    model_weight_gb: float
    remaining_vram_gb: float

    # Compression config
    config: CacheConfig
    preset_name: str

    # Capacity results
    max_agents: int
    actual_context: int
    kv_per_agent_mb: float
    total_kv_mb: float
    vram_headroom_mb: float

    # Agent slots
    slots: list[AgentSlot] = field(default_factory=list)

    # Feasibility
    feasible: bool = True
    bottleneck: str = ""

    # Dual-GPU split
    tensor_split: str | None = None
    layers_on_primary: int = 0
    layers_on_secondary: int = 0

    def print_report(self) -> str:
        """Generate a human-readable capacity report."""
        lines = []
        lines.append("=" * 65)
        lines.append("  MULTI-TURBOQUANT CAPACITY PLAN (estimates)")
        lines.append("=" * 65)
        lines.append("")

        # Hardware
        lines.append("HARDWARE:")
        for g in self.gpus:
            lines.append(f"  {g['name']:20s}  {g['vram_gb']:5.1f} GB VRAM")
        lines.append(f"  {'Total':20s}  {self.total_vram_gb:5.1f} GB")
        lines.append("")

        # Model
        lines.append("MODEL:")
        lines.append(f"  {self.model_name} ({self.model_quant})")
        lines.append(f"  Weights:  {self.model_weight_gb:5.1f} GB")
        lines.append(f"  Free:     {self.remaining_vram_gb:5.1f} GB for KV cache")
        if self.tensor_split:
            lines.append(f"  Split:    {self.tensor_split}")
            lines.append(f"            {self.layers_on_primary} layers on primary, "
                         f"{self.layers_on_secondary} on secondary")
        lines.append("")

        # Compression
        lines.append("COMPRESSION:")
        lines.append(f"  Preset:   {self.preset_name}")
        lines.append(f"  K cache:  {self.config.k_method.value} "
                     f"({METHOD_BITS[self.config.k_method]:.2f} bits, "
                     f"{self.config.k_compression:.1f}x)")
        lines.append(f"  V cache:  {self.config.v_method.value} "
                     f"({METHOD_BITS[self.config.v_method]:.2f} bits, "
                     f"{self.config.v_compression:.1f}x)")
        if self.config.triattention_enabled:
            lines.append(f"  TriAttn:  budget={self.config.triattention_budget}, "
                         f"window={self.config.triattention_window}")
        lines.append("")

        # Capacity
        if self.feasible:
            lines.append("CAPACITY:")
            lines.append(f"  Agents:   {self.max_agents} concurrent")
            lines.append(f"  Context:  {self.actual_context:,} tokens each")
            lines.append(f"  KV/agent: {self.kv_per_agent_mb:5.1f} MB")
            lines.append(f"  Total KV: {self.total_kv_mb:5.1f} MB")
            lines.append(f"  Headroom: {self.vram_headroom_mb:5.0f} MB free")
        else:
            lines.append("INFEASIBLE:")
            lines.append(f"  {self.bottleneck}")
        lines.append("")

        # Agent slots
        if self.slots:
            lines.append("AGENT SLOTS:")
            for s in self.slots:
                lines.append(f"  Agent {s.agent_id:2d}: {s.context_length:6,} ctx  "
                             f"{s.kv_bytes / (1024**2):5.1f} MB  [{s.gpu}]")
        lines.append("")
        lines.append("=" * 65)

        report = "\n".join(lines)
        print(report)
        return report

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "model_quant": self.model_quant,
            "model_weight_gb": round(self.model_weight_gb, 2),
            "gpus": self.gpus,
            "total_vram_gb": round(self.total_vram_gb, 1),
            "preset": self.preset_name,
            "config": self.config.to_dict(),
            "max_agents": self.max_agents,
            "context_per_agent": self.actual_context,
            "kv_per_agent_mb": round(self.kv_per_agent_mb, 2),
            "total_kv_mb": round(self.total_kv_mb, 2),
            "vram_headroom_mb": round(self.vram_headroom_mb, 1),
            "feasible": self.feasible,
            "bottleneck": self.bottleneck,
            "tensor_split": self.tensor_split,
        }


# ─── Planner ────────────────────────────────────────────────────────────────────

# VRAM overhead for runtime buffers (activations, scratch, etc.)
RUNTIME_OVERHEAD_GB = 0.5


def plan_agents(
    *,
    gpus: list[dict[str, Any]],
    model_params_b: float,
    model_quant: str = "Q4_K_M",
    desired_agents: int = 4,
    desired_context: int = 8192,
    model_name: str | None = None,
    prefer_speed: bool = False,
    compute: str = "cuda",
) -> PlanResult:
    """Plan a multi-agent deployment on given hardware.

    Finds the best compression config that fits the desired number of
    agents at the desired context length, or reduces until it fits.

    Args:
        gpus: List of GPU specs [{"name": "RTX 3090", "vram_gb": 24}, ...]
        model_params_b: Model size in billions of parameters.
        model_quant: Quantization level for weights.
        desired_agents: Target number of concurrent agents.
        desired_context: Target context length per agent.
        model_name: Human-readable model name.
        prefer_speed: Prefer faster methods over higher compression.

    Returns:
        PlanResult with the optimal configuration.
    """
    if model_name is None:
        model_name = f"{model_params_b:.0f}B model"

    arch = get_arch(model_params_b)
    model_gb = estimate_model_size_gb(model_params_b, model_quant)
    total_vram = sum(g["vram_gb"] for g in gpus)

    # Determine tensor split strategy and VRAM available for KV cache.
    # Key insight: KV cache must reside on the same GPU as the model layers.
    # llama.cpp --tensor-split distributes layers proportionally by VRAM,
    # and each GPU holds KV cache for its own layers.
    tensor_split = None
    layers_primary = arch["layers"]
    layers_secondary = 0
    gpu_layer_counts: list[int] = [arch["layers"]]  # per-GPU layer counts

    if len(gpus) > 1:
        # Always split across all GPUs to maximize KV headroom
        vram_values = [g["vram_gb"] for g in gpus]
        tensor_split = ",".join(f"{v:.0f}" for v in vram_values)

        # Distribute layers proportionally by VRAM
        gpu_layer_counts = []
        assigned = 0
        for i, g in enumerate(gpus):
            ratio = g["vram_gb"] / total_vram
            if i == len(gpus) - 1:
                layers = arch["layers"] - assigned  # last GPU gets remainder
            else:
                layers = int(arch["layers"] * ratio)
            gpu_layer_counts.append(layers)
            assigned += layers

        layers_primary = gpu_layer_counts[0]
        layers_secondary = sum(gpu_layer_counts[1:])

    # Calculate remaining VRAM for KV cache
    # Each GPU holds weights for its layers + KV cache for those layers.
    # The bottleneck GPU determines total usable KV capacity.
    if len(gpus) > 1 and tensor_split:
        # Per-GPU: free VRAM = gpu_vram - (weight share) - overhead
        # KV is split proportionally, so bottleneck GPU limits total
        per_gpu_kv_gb = []
        for i, g in enumerate(gpus):
            ratio = g["vram_gb"] / total_vram
            weight_share = model_gb * ratio
            overhead = RUNTIME_OVERHEAD_GB if i == 0 else 0.2
            free = g["vram_gb"] - weight_share - overhead
            per_gpu_kv_gb.append((free, ratio))

        # Total KV = min(free_i / ratio_i) across all GPUs
        # This finds which GPU is the bottleneck
        remaining_gb = float("inf")
        for free, ratio in per_gpu_kv_gb:
            if ratio > 0:
                remaining_gb = min(remaining_gb, free / ratio)
        remaining_gb = max(remaining_gb, 0)
        if remaining_gb == float("inf"):
            remaining_gb = 0
    else:
        remaining_gb = gpus[0]["vram_gb"] - model_gb - RUNTIME_OVERHEAD_GB
    if remaining_gb <= 0:
        return PlanResult(
            model_name=model_name,
            model_params_b=model_params_b,
            model_quant=model_quant,
            desired_agents=desired_agents,
            desired_context=desired_context,
            gpus=gpus,
            total_vram_gb=total_vram,
            model_weight_gb=model_gb,
            remaining_vram_gb=0,
            config=CacheConfig(),
            preset_name="N/A",
            max_agents=0,
            actual_context=0,
            kv_per_agent_mb=0,
            total_kv_mb=0,
            vram_headroom_mb=0,
            feasible=False,
            bottleneck=f"Model weights ({model_gb:.1f} GB) exceed available VRAM ({total_vram:.1f} GB)",
            tensor_split=tensor_split,
            layers_on_primary=layers_primary,
            layers_on_secondary=layers_secondary,
        )

    remaining_bytes = remaining_gb * (1024**3)

    # Try configs from least to most aggressive, find the best that fits.
    # Filter by compute backend — TurboQuant/TCQ need CUDA, iso/planar work everywhere.
    _all_configs = [
        # (name, k_method, v_method, triattention, tri_budget, needs_cuda)
        ("fp16", CacheMethod.FP16, CacheMethod.FP16, False, 0, False),
        ("k_only_iso", CacheMethod.ISO3, CacheMethod.FP16, False, 0, False),
        ("turbo4", CacheMethod.TURBO4, CacheMethod.TURBO4, False, 0, True),
        ("turbo3", CacheMethod.TURBO3, CacheMethod.TURBO3, False, 0, True),
        ("turbo3_tcq", CacheMethod.TURBO3_TCQ, CacheMethod.TURBO3_TCQ, False, 0, True),
        ("iso3_symmetric", CacheMethod.ISO3, CacheMethod.ISO3, False, 0, False),
        ("iso4_symmetric", CacheMethod.ISO4, CacheMethod.ISO4, False, 0, False),
        ("planar3_symmetric", CacheMethod.PLANAR3, CacheMethod.PLANAR3, False, 0, False),
        ("turbo2_tcq", CacheMethod.TURBO2_TCQ, CacheMethod.TURBO2_TCQ, False, 0, True),
        ("turbo3_tcq+triatt_8k", CacheMethod.TURBO3_TCQ, CacheMethod.TURBO3_TCQ, True, 8192, True),
        ("turbo3_tcq+triatt_4k", CacheMethod.TURBO3_TCQ, CacheMethod.TURBO3_TCQ, True, 4096, True),
        ("turbo3_tcq+triatt_2k", CacheMethod.TURBO3_TCQ, CacheMethod.TURBO3_TCQ, True, 2048, True),
    ]
    # Filter to methods available on this compute backend
    has_cuda = compute == "cuda"
    configs_to_try = [
        (name, k, v, tri, budget)
        for name, k, v, tri, budget, needs_cuda in _all_configs
        if not needs_cuda or has_cuda
    ]

    best_result = None

    for preset_name, k_method, v_method, tri_enabled, tri_budget in configs_to_try:
        k_bits = METHOD_BITS[k_method]
        v_bits = METHOD_BITS[v_method]
        budget = tri_budget if tri_enabled else None

        kv_total = total_kv_bytes(
            arch, desired_context, desired_agents,
            k_bits=k_bits, v_bits=v_bits,
            triattention_budget=budget,
        )

        if kv_total <= remaining_bytes:
            config = CacheConfig(
                k_method=k_method,
                v_method=v_method,
                triattention_enabled=tri_enabled,
                triattention_budget=tri_budget if tri_enabled else 4096,
                head_dim=arch["head_dim"],
                num_kv_heads=arch["kv_heads"],
                num_layers=arch["layers"],
            )

            kv_per_agent = kv_total / desired_agents
            headroom = remaining_bytes - kv_total

            result = PlanResult(
                model_name=model_name,
                model_params_b=model_params_b,
                model_quant=model_quant,
                desired_agents=desired_agents,
                desired_context=desired_context,
                gpus=gpus,
                total_vram_gb=total_vram,
                model_weight_gb=model_gb,
                remaining_vram_gb=remaining_gb,
                config=config,
                preset_name=preset_name,
                max_agents=desired_agents,
                actual_context=desired_context,
                kv_per_agent_mb=kv_per_agent / (1024**2),
                total_kv_mb=kv_total / (1024**2),
                vram_headroom_mb=headroom / (1024**2),
                tensor_split=tensor_split,
                layers_on_primary=layers_primary,
                layers_on_secondary=layers_secondary,
                slots=[
                    AgentSlot(
                        agent_id=i,
                        context_length=desired_context,
                        kv_bytes=kv_per_agent,
                        # Distribute agents across GPUs proportionally
                        gpu=gpus[i % len(gpus)]["name"] if len(gpus) > 1 else gpus[0]["name"],
                    )
                    for i in range(desired_agents)
                ],
            )

            if prefer_speed:
                return result  # Take first (least aggressive) that fits
            best_result = result
            if not tri_enabled:
                # Found a non-TriAttention config that fits — prefer it
                # unless we need more compression
                break

    if best_result is None:
        # Nothing fits — find what's achievable
        # How many agents can we fit with max compression?
        max_config = configs_to_try[-1]
        _, k_m, v_m, tri, tri_b = max_config
        single_agent_kv = total_kv_bytes(
            arch, desired_context, 1,
            k_bits=METHOD_BITS[k_m], v_bits=METHOD_BITS[v_m],
            triattention_budget=tri_b if tri else None,
        )
        if single_agent_kv > 0:
            max_agents_possible = int(remaining_bytes / single_agent_kv)
        else:
            max_agents_possible = 0

        return PlanResult(
            model_name=model_name,
            model_params_b=model_params_b,
            model_quant=model_quant,
            desired_agents=desired_agents,
            desired_context=desired_context,
            gpus=gpus,
            total_vram_gb=total_vram,
            model_weight_gb=model_gb,
            remaining_vram_gb=remaining_gb,
            config=CacheConfig(k_method=k_m, v_method=v_m,
                               triattention_enabled=tri,
                               triattention_budget=tri_b),
            preset_name="max_compression",
            max_agents=max_agents_possible,
            actual_context=desired_context,
            kv_per_agent_mb=single_agent_kv / (1024**2),
            total_kv_mb=single_agent_kv * max_agents_possible / (1024**2),
            vram_headroom_mb=(remaining_bytes - single_agent_kv * max_agents_possible) / (1024**2),
            feasible=max_agents_possible > 0,
            bottleneck=f"Max {max_agents_possible} agents at {desired_context} ctx "
                       f"(requested {desired_agents})",
            tensor_split=tensor_split,
            layers_on_primary=layers_primary,
            layers_on_secondary=layers_secondary,
        )

    return best_result


def plan_scenarios(
    *,
    gpus: list[dict[str, Any]],
    model_params_b: float,
    model_quant: str = "Q4_K_M",
    model_name: str | None = None,
) -> list[PlanResult]:
    """Plan multiple scenarios to show the full range of possibilities.

    Returns plans for 1, 2, 4, 8 agents at various context lengths.
    """
    scenarios = [
        (1, 32768),  # 1 agent, max context
        (1, 65536),  # 1 agent, very long context
        (2, 16384),  # 2 agents
        (4, 8192),   # 4 agents
        (4, 16384),  # 4 agents, bigger context
        (8, 4096),   # 8 agents
        (8, 8192),   # 8 agents, bigger context
        (16, 4096),  # 16 agents
    ]

    results = []
    for agents, context in scenarios:
        result = plan_agents(
            gpus=gpus,
            model_params_b=model_params_b,
            model_quant=model_quant,
            desired_agents=agents,
            desired_context=context,
            model_name=model_name,
        )
        results.append(result)

    return results
