# SPDX-License-Identifier: MIT
"""Method compatibility per GPU vendor and platform.

Not every method works on every GPU. This module maps what's available
where, generates the right build flags, and warns about unsupported
combinations before the user hits a runtime error.

Usage:
    from multi_turboquant.compatibility import check_config, get_cmake_flags
    from multi_turboquant.hardware import detect_platform

    platform = detect_platform()
    issues = check_config(config, platform)
    cmake = get_cmake_flags(platform)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import CacheMethod, MethodFamily, METHOD_FAMILIES

if TYPE_CHECKING:
    from .config import CacheConfig
    from .hardware import PlatformInfo


# ─── Method support matrix ──────────────────────────────────────────────────────

# Which compute backends support which methods for INFERENCE (llama.cpp)
LLAMACPP_SUPPORT: dict[str, set[CacheMethod]] = {
    "cuda": {
        CacheMethod.FP16, CacheMethod.Q8_0,
        CacheMethod.TURBO2, CacheMethod.TURBO3, CacheMethod.TURBO4,
        CacheMethod.TURBO2_TCQ, CacheMethod.TURBO3_TCQ,
        CacheMethod.ISO3, CacheMethod.ISO4,
        CacheMethod.PLANAR3, CacheMethod.PLANAR4,
    },
    "rocm": {
        CacheMethod.FP16, CacheMethod.Q8_0,
        # TurboQuant needs CUDA flash attention kernels — not available on ROCm
        # IsoQuant/PlanarQuant C implementations are vendor-neutral
        CacheMethod.ISO3, CacheMethod.ISO4,
        CacheMethod.PLANAR3, CacheMethod.PLANAR4,
    },
    "metal": {
        CacheMethod.FP16, CacheMethod.Q8_0,
        # PlanarQuant and IsoQuant have Metal kernels (johndpope fork)
        CacheMethod.ISO3, CacheMethod.ISO4,
        CacheMethod.PLANAR3, CacheMethod.PLANAR4,
    },
    "cpu": {
        CacheMethod.FP16, CacheMethod.Q8_0,
    },
}

# vLLM support (Linux + CUDA only)
VLLM_SUPPORT: set[CacheMethod] = {
    CacheMethod.FP16, CacheMethod.Q8_0,
    CacheMethod.TURBO2, CacheMethod.TURBO3, CacheMethod.TURBO4,
    CacheMethod.TURBO2_TCQ, CacheMethod.TURBO3_TCQ,
    CacheMethod.ISO3, CacheMethod.ISO4,
    CacheMethod.PLANAR3, CacheMethod.PLANAR4,
    CacheMethod.TRIATTENTION,
}


# ─── Build flags ────────────────────────────────────────────────────────────────

def get_cmake_flags(plat: "PlatformInfo") -> list[str]:
    """Generate cmake flags for building llama.cpp on this platform."""
    flags = []

    if plat.cuda_available:
        flags.extend([
            "-DGGML_CUDA=ON",
            "-DGGML_CUDA_FA=ON",
            "-DGGML_CUDA_FA_ALL_QUANTS=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=native",
        ])
    elif plat.rocm_available:
        flags.extend([
            "-DGGML_HIP=ON",
            # ROCm flash attention support varies by version
            # FA_ALL_QUANTS may not be available
        ])
    elif plat.metal_available:
        flags.extend([
            "-DGGML_METAL=ON",
        ])
    else:
        # CPU-only
        flags.append("-DGGML_BLAS=ON")

    # Use Ninja if available (faster builds)
    import shutil
    if shutil.which("ninja"):
        flags.extend(["-G", "Ninja"])

    return flags


def get_build_command(plat: "PlatformInfo", clone_dir: str = "/opt/llama.cpp") -> str:
    """Generate the full clone + build command."""
    cmake_flags = " ".join(get_cmake_flags(plat))

    # Detect core count
    if plat.os == "darwin":
        jobs = "$(sysctl -n hw.ncpu)"
    else:
        jobs = "$(nproc)"

    return (
        f"git clone https://github.com/ggml-org/llama.cpp {clone_dir}\n"
        f"cd {clone_dir}\n"
        f"cmake -B build {cmake_flags}\n"
        f"cmake --build build --config Release -j{jobs}"
    )


# ─── Compatibility checks ──────────────────────────────────────────────────────

@dataclass
class CompatIssue:
    """A compatibility issue with a configuration."""
    severity: str       # "error", "warning", "info"
    method: str         # which method has the issue
    message: str        # human-readable description
    suggestion: str     # what to do about it


def check_config(
    config: "CacheConfig",
    plat: "PlatformInfo",
    engine: str = "llamacpp",
) -> list[CompatIssue]:
    """Check a CacheConfig against the detected platform.

    Returns a list of issues (empty = fully compatible).
    """
    issues: list[CompatIssue] = []
    compute = plat.primary_compute

    for label, method in [("K", config.k_method), ("V", config.v_method)]:
        if method == CacheMethod.FP16 or method == CacheMethod.Q8_0:
            continue

        if engine == "llamacpp":
            supported = LLAMACPP_SUPPORT.get(compute, set())
        else:
            supported = VLLM_SUPPORT

        if method not in supported:
            family = METHOD_FAMILIES[method]
            # Find what IS available on this platform
            available = [
                m.value for m in supported
                if m not in (CacheMethod.FP16, CacheMethod.Q8_0)
            ]

            if compute == "rocm" and family in (MethodFamily.TURBOQUANT, MethodFamily.TCQ):
                issues.append(CompatIssue(
                    severity="error",
                    method=f"{label}={method.value}",
                    message=f"TurboQuant/TCQ requires CUDA flash attention kernels — "
                            f"not available on AMD/ROCm.",
                    suggestion=f"Use IsoQuant or PlanarQuant instead: "
                               f"iso3, iso4, planar3, planar4",
                ))
            elif compute == "metal" and family in (MethodFamily.TURBOQUANT, MethodFamily.TCQ):
                issues.append(CompatIssue(
                    severity="error",
                    method=f"{label}={method.value}",
                    message=f"TurboQuant/TCQ requires CUDA — "
                            f"not available on Apple Metal.",
                    suggestion=f"Use IsoQuant or PlanarQuant which have Metal kernels: "
                               f"iso3, iso4, planar3, planar4",
                ))
            elif compute == "cpu":
                issues.append(CompatIssue(
                    severity="error",
                    method=f"{label}={method.value}",
                    message=f"No GPU detected — KV cache compression requires a GPU.",
                    suggestion=f"Use FP16 or Q8_0 for CPU-only inference.",
                ))
            else:
                issues.append(CompatIssue(
                    severity="error",
                    method=f"{label}={method.value}",
                    message=f"{method.value} not supported on {compute}.",
                    suggestion=f"Available methods: {', '.join(available)}",
                ))

    # TriAttention checks
    if config.triattention_enabled:
        if engine == "llamacpp":
            issues.append(CompatIssue(
                severity="warning",
                method="triattention",
                message="TriAttention is vLLM-only — ignored in llama.cpp mode.",
                suggestion="Use vLLM or disable TriAttention.",
            ))
        if not plat.can_run_vllm:
            issues.append(CompatIssue(
                severity="error",
                method="triattention",
                message="TriAttention requires vLLM (Linux + CUDA).",
                suggestion="Run on Linux with NVIDIA GPU for TriAttention support.",
            ))

    # vLLM platform check
    if engine == "vllm" and not plat.can_run_vllm:
        issues.append(CompatIssue(
            severity="error",
            method="engine",
            message="vLLM requires Linux + CUDA.",
            suggestion="Use llama.cpp instead — supports CUDA, ROCm, and Metal.",
        ))

    return issues


def get_available_methods(
    plat: "PlatformInfo",
    engine: str = "llamacpp",
) -> list[CacheMethod]:
    """Return all methods available on this platform."""
    compute = plat.primary_compute
    if engine == "llamacpp":
        return sorted(LLAMACPP_SUPPORT.get(compute, set()), key=lambda m: m.value)
    elif engine == "vllm":
        if plat.can_run_vllm:
            return sorted(VLLM_SUPPORT, key=lambda m: m.value)
        return []
    return [CacheMethod.FP16, CacheMethod.Q8_0]


def get_recommended_config(
    plat: "PlatformInfo",
    engine: str = "llamacpp",
) -> "CacheConfig":
    """Return the best default config for this platform.

    NVIDIA: turbo3_tcq symmetric (best quality + speed)
    AMD/Metal: iso3 K-only (best available, no calibration)
    CPU: FP16 (no compression available)
    """
    from .config import CacheConfig

    compute = plat.primary_compute

    if compute == "cuda":
        return CacheConfig(
            k_method=CacheMethod.TURBO3_TCQ,
            v_method=CacheMethod.TURBO3_TCQ,
        )
    elif compute in ("rocm", "metal"):
        return CacheConfig(
            k_method=CacheMethod.ISO3,
            v_method=CacheMethod.FP16,
        )
    else:
        return CacheConfig(
            k_method=CacheMethod.FP16,
            v_method=CacheMethod.FP16,
        )
