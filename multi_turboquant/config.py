# SPDX-License-Identifier: MIT
"""Unified configuration for all KV cache compression methods."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CacheMethod(str, Enum):
    """All supported KV cache compression methods."""

    # --- TurboQuant (Walsh-Hadamard Transform) ---
    TURBO2 = "turbo2"          # 2.25-bit, ~7x compression
    TURBO3 = "turbo3"          # 3.25-bit, ~5x compression
    TURBO4 = "turbo4"          # 4.25-bit, ~3.8x compression

    # --- TCQ (Trellis Coded Quantization on WHT) ---
    TURBO2_TCQ = "turbo2_tcq"  # 2.25-bit + trellis, better quality
    TURBO3_TCQ = "turbo3_tcq"  # 3.25-bit + trellis, better quality

    # --- IsoQuant (Quaternion 4D rotation) ---
    ISO3 = "iso3"              # 3.25-bit, 512 FMAs, no calibration
    ISO4 = "iso4"              # 4.25-bit, 512 FMAs, no calibration

    # --- PlanarQuant (Givens 2D rotation) ---
    PLANAR3 = "planar3"        # 3.25-bit, 256 FMAs, no calibration
    PLANAR4 = "planar4"        # 4.25-bit, 256 FMAs, no calibration

    # --- RotorQuant (Cl(3,0) sandwich, SO(3) rotation on 3D groups) ---
    ROTOR3 = "rotor3"          # 3.25-bit, no calibration
    ROTOR4 = "rotor4"          # 4.25-bit, no calibration, experimental

    # --- TriAttention (token eviction) ---
    TRIATTENTION = "triattention"  # full precision, fewer tokens

    # --- Baselines ---
    FP16 = "f16"               # no compression
    Q8_0 = "q8_0"              # 8-bit, minimal compression


class MethodFamily(str, Enum):
    """Method families grouping related cache types."""
    TURBOQUANT = "turboquant"
    TCQ = "tcq"
    ISOQUANT = "isoquant"
    PLANARQUANT = "planarquant"
    ROTORQUANT = "rotorquant"
    TRIATTENTION = "triattention"
    BASELINE = "baseline"


# Method -> family mapping
METHOD_FAMILIES: dict[CacheMethod, MethodFamily] = {
    CacheMethod.TURBO2: MethodFamily.TURBOQUANT,
    CacheMethod.TURBO3: MethodFamily.TURBOQUANT,
    CacheMethod.TURBO4: MethodFamily.TURBOQUANT,
    CacheMethod.TURBO2_TCQ: MethodFamily.TCQ,
    CacheMethod.TURBO3_TCQ: MethodFamily.TCQ,
    CacheMethod.ISO3: MethodFamily.ISOQUANT,
    CacheMethod.ISO4: MethodFamily.ISOQUANT,
    CacheMethod.PLANAR3: MethodFamily.PLANARQUANT,
    CacheMethod.PLANAR4: MethodFamily.PLANARQUANT,
    CacheMethod.ROTOR3: MethodFamily.ROTORQUANT,
    CacheMethod.ROTOR4: MethodFamily.ROTORQUANT,
    CacheMethod.TRIATTENTION: MethodFamily.TRIATTENTION,
    CacheMethod.FP16: MethodFamily.BASELINE,
    CacheMethod.Q8_0: MethodFamily.BASELINE,
}


# Methods that require per-model calibration files
CALIBRATION_REQUIRED: set[CacheMethod] = {
    CacheMethod.TURBO2,
    CacheMethod.TURBO3,
    CacheMethod.TURBO4,
    CacheMethod.TURBO2_TCQ,
    CacheMethod.TURBO3_TCQ,
    CacheMethod.TRIATTENTION,
}

# Methods that work without calibration
CALIBRATION_FREE: set[CacheMethod] = {
    CacheMethod.ISO3,
    CacheMethod.ISO4,
    CacheMethod.PLANAR3,
    CacheMethod.PLANAR4,
    CacheMethod.ROTOR3,
    CacheMethod.ROTOR4,
    CacheMethod.FP16,
    CacheMethod.Q8_0,
}

# Bits per element for each method
METHOD_BITS: dict[CacheMethod, float] = {
    CacheMethod.TURBO2: 2.25,
    CacheMethod.TURBO3: 3.25,
    CacheMethod.TURBO4: 4.25,
    CacheMethod.TURBO2_TCQ: 2.25,
    CacheMethod.TURBO3_TCQ: 3.25,
    CacheMethod.ISO3: 3.25,
    CacheMethod.ISO4: 4.25,
    CacheMethod.PLANAR3: 3.25,
    CacheMethod.PLANAR4: 4.25,
    CacheMethod.ROTOR3: 3.25,
    CacheMethod.ROTOR4: 4.25,
    CacheMethod.TRIATTENTION: 16.0,  # full precision, token eviction
    CacheMethod.FP16: 16.0,
    CacheMethod.Q8_0: 8.0,
}

# Approximate compression ratios vs FP16
METHOD_COMPRESSION: dict[CacheMethod, float] = {
    method: 16.0 / bits for method, bits in METHOD_BITS.items()
}

# Supported head dimensions per method
SUPPORTED_HEAD_DIMS: dict[MethodFamily, tuple[int, ...]] = {
    MethodFamily.TURBOQUANT: (64, 128, 192, 256),
    MethodFamily.TCQ: (64, 128, 192, 256),
    MethodFamily.ISOQUANT: (64, 128),     # quaternion needs head_dim % 4 == 0
    MethodFamily.PLANARQUANT: (64, 128),  # givens needs head_dim % 2 == 0
    MethodFamily.ROTORQUANT: (64, 128),   # Cl(3,0) groups-of-3; padded to multiple of 3
    MethodFamily.TRIATTENTION: (64, 128, 192, 256),
    MethodFamily.BASELINE: (64, 128, 192, 256),
}


@dataclass(frozen=True)
class CacheConfig:
    """Unified configuration for KV cache compression.

    Supports asymmetric configs (different methods for K and V caches),
    combined modes (TriAttention + quantization), and preset-based configs.

    Examples:
        # Symmetric TurboQuant
        CacheConfig(k_method=CacheMethod.TURBO3, v_method=CacheMethod.TURBO3)

        # K-only IsoQuant (V stays FP16 — beats FP16 decode speed)
        CacheConfig(k_method=CacheMethod.ISO3, v_method=CacheMethod.FP16)

        # Combined: TriAttention eviction + TurboQuant compression
        CacheConfig(
            k_method=CacheMethod.TURBO3_TCQ,
            v_method=CacheMethod.TURBO3_TCQ,
            triattention_budget=2048,
        )
    """

    k_method: CacheMethod = CacheMethod.FP16
    v_method: CacheMethod = CacheMethod.FP16

    # TriAttention token eviction settings (composable with any method)
    triattention_enabled: bool = False
    triattention_budget: int = 4096
    triattention_window: int = 512  # recent-token window that's never evicted
    use_custom_triattention_llamacpp: bool = False
    triattention_log: bool = False

    # Calibration paths
    turboquant_metadata_path: str | None = None
    triattention_stats_path: str | None = None

    # Model info (populated during initialization)
    head_dim: int = 128
    num_kv_heads: int = 8
    num_layers: int = 32

    @property
    def is_symmetric(self) -> bool:
        return self.k_method == self.v_method

    @property
    def is_k_only(self) -> bool:
        return self.k_method != CacheMethod.FP16 and self.v_method == CacheMethod.FP16

    @property
    def needs_calibration(self) -> bool:
        return (
            self.k_method in CALIBRATION_REQUIRED
            or self.v_method in CALIBRATION_REQUIRED
            or self.triattention_enabled
        )

    @property
    def k_compression(self) -> float:
        return METHOD_COMPRESSION[self.k_method]

    @property
    def v_compression(self) -> float:
        return METHOD_COMPRESSION[self.v_method]

    @property
    def total_compression(self) -> float:
        """Total KV memory reduction factor including token eviction."""
        kv_compression = (self.k_compression + self.v_compression) / 2.0
        if self.triattention_enabled:
            # Assume full context vs budget ratio
            # Actual ratio depends on runtime context length
            return kv_compression  # Token eviction multiplies at runtime
        return kv_compression

    @property
    def k_family(self) -> MethodFamily:
        return METHOD_FAMILIES[self.k_method]

    @property
    def v_family(self) -> MethodFamily:
        return METHOD_FAMILIES[self.v_method]

    def validate(self) -> list[str]:
        """Validate this config and return any warnings."""
        warnings = []
        for label, method in [("K", self.k_method), ("V", self.v_method)]:
            family = METHOD_FAMILIES[method]
            supported = SUPPORTED_HEAD_DIMS.get(family, ())
            if supported and self.head_dim not in supported:
                warnings.append(
                    f"{label} method {method.value} may not support "
                    f"head_dim={self.head_dim}. "
                    f"Tested dimensions: {supported}"
                )
        if self.triattention_enabled and self.triattention_budget <= 0:
            warnings.append("TriAttention budget must be > 0")
        if self.triattention_enabled and self.triattention_window < 0:
            warnings.append("TriAttention window must be >= 0")
        if (
            self.triattention_enabled
            and self.use_custom_triattention_llamacpp
            and not self.triattention_stats_path
        ):
            warnings.append(
                "Patched llama.cpp TriAttention requires triattention_stats_path"
            )
        return warnings

    def estimate_kv_bytes(self, context_length: int) -> int:
        """Estimate total KV cache bytes for a given context length."""
        k_bits = METHOD_BITS[self.k_method]
        v_bits = METHOD_BITS[self.v_method]
        tokens = context_length
        if self.triattention_enabled:
            tokens = min(tokens, self.triattention_budget + self.triattention_window)
        bytes_per_token = (
            self.num_kv_heads * self.head_dim * (k_bits + v_bits) / 8.0
        )
        return int(math.ceil(tokens * bytes_per_token * self.num_layers))

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_method": self.k_method.value,
            "v_method": self.v_method.value,
            "triattention_enabled": self.triattention_enabled,
            "triattention_budget": self.triattention_budget,
            "triattention_window": self.triattention_window,
            "use_custom_triattention_llamacpp": self.use_custom_triattention_llamacpp,
            "triattention_log": self.triattention_log,
            "turboquant_metadata_path": self.turboquant_metadata_path,
            "triattention_stats_path": self.triattention_stats_path,
            "head_dim": self.head_dim,
            "num_kv_heads": self.num_kv_heads,
            "num_layers": self.num_layers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheConfig:
        return cls(
            k_method=CacheMethod(data["k_method"]),
            v_method=CacheMethod(data["v_method"]),
            triattention_enabled=data.get("triattention_enabled", False),
            triattention_budget=data.get("triattention_budget", 4096),
            triattention_window=data.get("triattention_window", 512),
            use_custom_triattention_llamacpp=data.get(
                "use_custom_triattention_llamacpp", False
            ),
            triattention_log=data.get("triattention_log", False),
            turboquant_metadata_path=data.get("turboquant_metadata_path"),
            triattention_stats_path=data.get("triattention_stats_path"),
            head_dim=data.get("head_dim", 128),
            num_kv_heads=data.get("num_kv_heads", 8),
            num_layers=data.get("num_layers", 32),
        )
