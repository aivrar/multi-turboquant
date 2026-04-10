# SPDX-License-Identifier: MIT
"""Generate llama.cpp CLI flags for KV cache compression.

Maps CacheConfig to the correct --cache-type-k / --cache-type-v flags
and any other required arguments.

Usage:
    from multi_turboquant.integration import get_llamacpp_args
    from multi_turboquant import CacheConfig, CacheMethod

    config = CacheConfig(k_method=CacheMethod.ISO3, v_method=CacheMethod.FP16)
    args = get_llamacpp_args(config)
    # ['--cache-type-k', 'iso3', '--cache-type-v', 'f16', '-fa', 'on']
"""

from __future__ import annotations

from ..config import CacheConfig, CacheMethod, MethodFamily, METHOD_FAMILIES


# Map CacheMethod -> llama.cpp cache type string
LLAMACPP_CACHE_TYPES: dict[CacheMethod, str] = {
    CacheMethod.TURBO2: "turbo2",
    CacheMethod.TURBO3: "turbo3",
    CacheMethod.TURBO4: "turbo4",
    CacheMethod.TURBO2_TCQ: "turbo2_tcq",
    CacheMethod.TURBO3_TCQ: "turbo3_tcq",
    CacheMethod.ISO3: "iso3",
    CacheMethod.ISO4: "iso4",
    CacheMethod.PLANAR3: "planar3",
    CacheMethod.PLANAR4: "planar4",
    CacheMethod.FP16: "f16",
    CacheMethod.Q8_0: "q8_0",
    # TriAttention is vLLM-only, not supported in llama.cpp
}


def get_llamacpp_args(
    config: CacheConfig,
    *,
    flash_attention: bool = True,
    model_path: str | None = None,
    context_size: int | None = None,
    gpu_layers: int | None = None,
    tensor_split: str | None = None,
    parallel_slots: int | None = None,
) -> list[str]:
    """Generate llama.cpp CLI arguments for a CacheConfig.

    Args:
        config: Cache configuration.
        flash_attention: Enable flash attention (required for turbo/iso/planar).
        model_path: Path to GGUF model file.
        context_size: Context window size.
        gpu_layers: Number of layers to offload to GPU.

    Returns:
        List of CLI argument strings.
    """
    args: list[str] = []

    if model_path:
        args.extend(["--model", model_path])

    # Cache type flags
    k_type = LLAMACPP_CACHE_TYPES.get(config.k_method)
    v_type = LLAMACPP_CACHE_TYPES.get(config.v_method)

    if k_type is None:
        raise ValueError(
            f"Method {config.k_method.value} is not supported in llama.cpp"
        )
    if v_type is None:
        raise ValueError(
            f"Method {config.v_method.value} is not supported in llama.cpp"
        )

    args.extend(["--cache-type-k", k_type])
    args.extend(["--cache-type-v", v_type])

    # Flash attention is required for all non-baseline methods
    non_baseline = {
        config.k_method, config.v_method,
    } - {CacheMethod.FP16, CacheMethod.Q8_0}
    if non_baseline or flash_attention:
        args.extend(["-fa", "on"])

    if context_size:
        args.extend(["-c", str(context_size)])

    if gpu_layers is not None:
        args.extend(["-ngl", str(gpu_layers)])

    # Tensor split for multi-GPU
    if tensor_split:
        args.extend(["--tensor-split", tensor_split])

    # Parallel slots for multi-agent
    if parallel_slots and parallel_slots > 1:
        args.extend(["--parallel", str(parallel_slots)])

    # TriAttention warning
    if config.triattention_enabled:
        import warnings
        warnings.warn(
            "TriAttention is vLLM-only and will be ignored in llama.cpp mode.",
            RuntimeWarning,
            stacklevel=2,
        )

    return args


def get_llamacpp_command(
    config: CacheConfig,
    *,
    binary: str = "llama-server",
    model_path: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    context_size: int = 4096,
    gpu_layers: int = 99,
    tensor_split: str | None = None,
    parallel_slots: int | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Generate a complete llama-server command line.

    Args:
        config: Cache configuration.
        binary: Path to llama-server binary.
        model_path: Path to GGUF model file.
        host: Listen address.
        port: Listen port.
        context_size: Context window size.
        gpu_layers: GPU layer count.
        extra_args: Additional CLI arguments.

    Returns:
        Complete command as list of strings.
    """
    cmd = [binary]
    cmd.extend(get_llamacpp_args(
        config,
        model_path=model_path,
        context_size=context_size,
        gpu_layers=gpu_layers,
        tensor_split=tensor_split,
        parallel_slots=parallel_slots,
    ))
    cmd.extend(["--host", host, "--port", str(port)])

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def get_cmake_flags(config: CacheConfig) -> list[str]:
    """Generate CMake build flags needed for the configured methods.

    Returns the flags needed when building llama.cpp with support
    for the configured cache types.
    """
    flags = [
        "-DGGML_CUDA=ON",
        "-DGGML_CUDA_FA=ON",
        "-DGGML_CUDA_FA_ALL_QUANTS=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=native",
    ]

    # Metal support for PlanarQuant/IsoQuant on macOS
    families = {METHOD_FAMILIES[config.k_method], METHOD_FAMILIES[config.v_method]}
    if MethodFamily.PLANARQUANT in families or MethodFamily.ISOQUANT in families:
        import sys
        if sys.platform == "darwin":
            flags.append("-DGGML_METAL=ON")

    return flags
