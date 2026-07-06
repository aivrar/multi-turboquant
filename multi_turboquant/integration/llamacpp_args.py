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

from dataclasses import dataclass
from enum import Enum

from ..config import CacheConfig, CacheMethod, MethodFamily, METHOD_FAMILIES


class LlamaCppProfile(str, Enum):
    """Supported llama.cpp-compatible integration profiles."""

    UPSTREAM = "upstream"
    PATCHED_TRIATTENTION = "patched_triattention"
    GODZILLA = "godzilla"


@dataclass(frozen=True)
class LlamaCppSpeculativeConfig:
    """Godzilla llama.cpp speculative-decoding arguments."""

    spec_type: str = "dflash"
    draft_model: str | None = None
    draft_hf: str | None = None
    draft_context_size: int | None = None
    draft_gpu_layers: int | str | None = None
    draft_device: str | None = None
    draft_cache_type_k: CacheMethod | str | None = None
    draft_cache_type_v: CacheMethod | str | None = None
    draft_n_max: int | None = None
    draft_n_min: int | None = None
    branch_budget: int | None = None
    draft_top_k: int | None = None
    draft_p_split: float | None = None
    draft_p_min: float | None = None
    draft_temp: float | str | None = None
    dflash_cross_ctx: int | None = None
    dflash_max_slots: int | None = None
    adaptive: bool | None = None
    dm_controller: str | None = None
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class LlamaCppContextExtensionConfig:
    """llama.cpp RoPE and YaRN context-extension arguments.

    These settings are launch-time llama.cpp options. Upstream llama.cpp does
    not currently support changing RoPE/YaRN scaling through POST /props.
    """

    rope_scaling: str | None = None
    rope_scale: float | None = None
    rope_freq_base: float | None = None
    rope_freq_scale: float | None = None
    yarn_orig_ctx: int | None = None
    yarn_ext_factor: float | None = None
    yarn_attn_factor: float | None = None
    yarn_beta_slow: float | None = None
    yarn_beta_fast: float | None = None
    extra_args: tuple[str, ...] = ()


LLAMACPP_ROPE_SCALING_TYPES = {"none", "linear", "yarn"}
LLAMACPP_YARN_FIELDS = (
    "yarn_orig_ctx",
    "yarn_ext_factor",
    "yarn_attn_factor",
    "yarn_beta_slow",
    "yarn_beta_fast",
)


def normalize_llamacpp_profile(
    profile: LlamaCppProfile | str | None,
) -> LlamaCppProfile:
    """Normalize a llama.cpp profile value and reject typos early."""
    if profile is None:
        return LlamaCppProfile.UPSTREAM
    if isinstance(profile, LlamaCppProfile):
        return profile
    try:
        return LlamaCppProfile(profile)
    except ValueError as exc:
        allowed = ", ".join(p.value for p in LlamaCppProfile)
        raise ValueError(
            f"Unknown llama.cpp profile {profile!r}; expected one of: {allowed}"
        ) from exc


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


GODZILLA_KVARN_CACHE_TYPES: dict[CacheMethod, str] = {
    CacheMethod.KVARN2: "kvarn2",
    CacheMethod.KVARN3: "kvarn3",
    CacheMethod.KVARN4: "kvarn4",
    CacheMethod.KVARN5: "kvarn5",
    CacheMethod.KVARN6: "kvarn6",
    CacheMethod.KVARN8: "kvarn8",
}

GODZILLA_SPEC_TYPES = {
    "none",
    "ngram-cache",
    "ngram-simple",
    "ngram-map-k",
    "ngram-map-k4v",
    "ngram-mod",
    "suffix",
    "copyspec",
    "recycle",
    "dflash",
    "draft-mtp",
}


def _is_kvarn_method(method: CacheMethod) -> bool:
    return METHOD_FAMILIES[method] == MethodFamily.KVARN


def _validate_godzilla_kvarn_config(
    config: CacheConfig,
    profile: LlamaCppProfile,
) -> None:
    k_is_kvarn = _is_kvarn_method(config.k_method)
    v_is_kvarn = _is_kvarn_method(config.v_method)
    if not (k_is_kvarn or v_is_kvarn):
        return

    if profile != LlamaCppProfile.GODZILLA:
        raise ValueError(
            "KVarN cache types require fork_profile='godzilla' because they "
            "are Godzilla llama.cpp pseudo cache aliases, not upstream "
            "llama.cpp cache types."
        )
    if k_is_kvarn != v_is_kvarn:
        raise ValueError(
            "Godzilla KVarN must be configured for both K and V. Use matching "
            "kvarn* methods on both sides instead of mixing KVarN with a "
            "normal cache type."
        )
    if config.triattention_enabled:
        raise ValueError(
            "Godzilla currently rejects KVarN with TriAttention until "
            "KVarN-aware prune is implemented (KVX-2). Disable "
            "triattention_enabled or choose non-KVarN cache types."
        )


def _positive_int(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be > 0")


def _non_negative_int(name: str, value: int | None) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be >= 0")


def _probability(name: str, value: float | None) -> None:
    if value is not None and not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _positive_float(name: str, value: float | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be > 0")


def _non_negative_float(name: str, value: float | None) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be >= 0")


def _normalize_rope_scaling(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in LLAMACPP_ROPE_SCALING_TYPES:
        allowed = ", ".join(sorted(LLAMACPP_ROPE_SCALING_TYPES))
        raise ValueError(
            f"Unknown rope_scaling {value!r}; expected one of: {allowed}"
        )
    return normalized


def _uses_yarn_options(context_extension: LlamaCppContextExtensionConfig) -> bool:
    return any(
        getattr(context_extension, field) is not None
        for field in LLAMACPP_YARN_FIELDS
    )


def _validate_context_extension(
    context_extension: LlamaCppContextExtensionConfig,
) -> str | None:
    """Validate context-extension settings and return normalized scaling."""
    rope_scaling = _normalize_rope_scaling(context_extension.rope_scaling)

    _positive_float("rope_scale", context_extension.rope_scale)
    _positive_float("rope_freq_base", context_extension.rope_freq_base)
    _positive_float("rope_freq_scale", context_extension.rope_freq_scale)
    _positive_int("yarn_orig_ctx", context_extension.yarn_orig_ctx)
    _non_negative_float("yarn_ext_factor", context_extension.yarn_ext_factor)
    _positive_float("yarn_attn_factor", context_extension.yarn_attn_factor)
    _positive_float("yarn_beta_slow", context_extension.yarn_beta_slow)
    _positive_float("yarn_beta_fast", context_extension.yarn_beta_fast)

    if (
        context_extension.rope_scale is not None
        and context_extension.rope_freq_scale is not None
    ):
        raise ValueError("Use rope_scale or rope_freq_scale, not both")

    if _uses_yarn_options(context_extension):
        if rope_scaling in {"none", "linear"}:
            raise ValueError(
                "YaRN options require rope_scaling='yarn' or omitted "
                "rope_scaling so the wrapper can select YaRN."
            )
        if rope_scaling is None:
            rope_scaling = "yarn"

    if rope_scaling == "none" and (
        context_extension.rope_scale is not None
        or context_extension.rope_freq_scale is not None
    ):
        raise ValueError(
            "rope_scaling='none' cannot be combined with rope_scale or "
            "rope_freq_scale"
        )

    for arg in context_extension.extra_args:
        if not str(arg).strip():
            raise ValueError("context extension extra_args cannot contain blanks")

    return rope_scaling


def _get_context_extension_args(
    context_extension: LlamaCppContextExtensionConfig,
) -> list[str]:
    rope_scaling = _validate_context_extension(context_extension)
    args: list[str] = []

    if rope_scaling is not None:
        args.extend(["--rope-scaling", rope_scaling])
    if context_extension.rope_scale is not None:
        args.extend(["--rope-scale", str(context_extension.rope_scale)])
    if context_extension.rope_freq_base is not None:
        args.extend(["--rope-freq-base", str(context_extension.rope_freq_base)])
    if context_extension.rope_freq_scale is not None:
        args.extend(["--rope-freq-scale", str(context_extension.rope_freq_scale)])
    if context_extension.yarn_orig_ctx is not None:
        args.extend(["--yarn-orig-ctx", str(context_extension.yarn_orig_ctx)])
    if context_extension.yarn_ext_factor is not None:
        args.extend(["--yarn-ext-factor", str(context_extension.yarn_ext_factor)])
    if context_extension.yarn_attn_factor is not None:
        args.extend(["--yarn-attn-factor", str(context_extension.yarn_attn_factor)])
    if context_extension.yarn_beta_slow is not None:
        args.extend(["--yarn-beta-slow", str(context_extension.yarn_beta_slow)])
    if context_extension.yarn_beta_fast is not None:
        args.extend(["--yarn-beta-fast", str(context_extension.yarn_beta_fast)])
    args.extend(str(arg) for arg in context_extension.extra_args)
    return args


def _draft_cache_type(label: str, cache_type: CacheMethod | str) -> str:
    if isinstance(cache_type, CacheMethod):
        if _is_kvarn_method(cache_type):
            raise ValueError(
                f"{label} draft cache type cannot use KVarN; Godzilla only "
                "accepts KVarN aliases for target cache types."
            )
        mapped = LLAMACPP_CACHE_TYPES.get(cache_type)
        if mapped is None:
            raise ValueError(
                f"{label} draft cache type {cache_type.value} is not supported "
                "by llama.cpp draft cache flags."
            )
        return mapped

    value = str(cache_type).strip()
    if not value:
        raise ValueError(f"{label} draft cache type cannot be empty")
    if value.lower().startswith("kvarn"):
        raise ValueError(
            f"{label} draft cache type cannot use KVarN; Godzilla only accepts "
            "KVarN aliases for target cache types."
        )
    return value


def _get_godzilla_speculative_args(
    speculative: LlamaCppSpeculativeConfig,
    profile: LlamaCppProfile,
) -> list[str]:
    if profile != LlamaCppProfile.GODZILLA:
        raise ValueError(
            "Godzilla speculative decoding flags require fork_profile='godzilla'."
        )

    spec_type = str(speculative.spec_type).strip()
    if spec_type not in GODZILLA_SPEC_TYPES:
        allowed = ", ".join(sorted(GODZILLA_SPEC_TYPES))
        raise ValueError(
            f"Unknown Godzilla speculative type {spec_type!r}; expected one of: "
            f"{allowed}"
        )
    if speculative.draft_model and speculative.draft_hf:
        raise ValueError("Use draft_model or draft_hf, not both")
    if spec_type == "dflash" and not (
        speculative.draft_model or speculative.draft_hf
    ):
        raise ValueError("DFlash requires draft_model or draft_hf")

    _positive_int("draft_context_size", speculative.draft_context_size)
    _positive_int("draft_n_max", speculative.draft_n_max)
    _non_negative_int("draft_n_min", speculative.draft_n_min)
    _non_negative_int("branch_budget", speculative.branch_budget)
    _positive_int("draft_top_k", speculative.draft_top_k)
    _positive_int("dflash_cross_ctx", speculative.dflash_cross_ctx)
    _positive_int("dflash_max_slots", speculative.dflash_max_slots)
    _probability("draft_p_split", speculative.draft_p_split)
    _probability("draft_p_min", speculative.draft_p_min)
    if (
        isinstance(speculative.draft_temp, (int, float))
        and speculative.draft_temp < 0
    ):
        raise ValueError("draft_temp must be >= 0 or 'auto'")
    if isinstance(speculative.draft_temp, str) and speculative.draft_temp != "auto":
        try:
            if float(speculative.draft_temp) < 0:
                raise ValueError
        except ValueError as exc:
            raise ValueError("draft_temp must be >= 0 or 'auto'") from exc
    if speculative.dm_controller is not None and speculative.dm_controller not in {
        "profit",
        "fringe",
    }:
        raise ValueError("dm_controller must be 'profit' or 'fringe'")

    args = ["--spec-type", spec_type]
    if speculative.draft_model:
        args.extend(["--spec-draft-model", speculative.draft_model])
    if speculative.draft_hf:
        args.extend(["--spec-draft-hf", speculative.draft_hf])
    if speculative.draft_context_size is not None:
        args.extend(["--spec-draft-ctx-size", str(speculative.draft_context_size)])
    if speculative.draft_gpu_layers is not None:
        args.extend(["--spec-draft-ngl", str(speculative.draft_gpu_layers)])
    if speculative.draft_device:
        args.extend(["--spec-draft-device", speculative.draft_device])
    if speculative.draft_cache_type_k is not None:
        args.extend([
            "--spec-draft-type-k",
            _draft_cache_type("K", speculative.draft_cache_type_k),
        ])
    if speculative.draft_cache_type_v is not None:
        args.extend([
            "--spec-draft-type-v",
            _draft_cache_type("V", speculative.draft_cache_type_v),
        ])
    if speculative.draft_n_max is not None:
        args.extend(["--spec-draft-n-max", str(speculative.draft_n_max)])
    if speculative.draft_n_min is not None:
        args.extend(["--spec-draft-n-min", str(speculative.draft_n_min)])
    if speculative.branch_budget is not None:
        args.extend(["--spec-branch-budget", str(speculative.branch_budget)])
    if speculative.draft_top_k is not None:
        args.extend(["--spec-draft-top-k", str(speculative.draft_top_k)])
    if speculative.draft_p_split is not None:
        args.extend(["--spec-draft-p-split", str(speculative.draft_p_split)])
    if speculative.draft_p_min is not None:
        args.extend(["--spec-draft-p-min", str(speculative.draft_p_min)])
    if speculative.draft_temp is not None:
        args.extend(["--spec-draft-temp", str(speculative.draft_temp)])
    if speculative.dflash_cross_ctx is not None:
        args.extend(["--spec-dflash-cross-ctx", str(speculative.dflash_cross_ctx)])
    if speculative.dflash_max_slots is not None:
        args.extend(["--spec-dflash-max-slots", str(speculative.dflash_max_slots)])
    if speculative.adaptive is True:
        args.append("--spec-dm-adaptive")
    elif speculative.adaptive is False:
        args.append("--no-spec-dm-adaptive")
    if speculative.dm_controller is not None:
        args.extend(["--spec-dm-controller", speculative.dm_controller])
    args.extend(str(arg) for arg in speculative.extra_args)
    return args


def get_llamacpp_args(
    config: CacheConfig,
    *,
    flash_attention: bool = True,
    model_path: str | None = None,
    context_size: int | None = None,
    gpu_layers: int | None = None,
    tensor_split: str | None = None,
    parallel_slots: int | None = None,
    fork_profile: LlamaCppProfile | str | None = LlamaCppProfile.UPSTREAM,
    context_extension: LlamaCppContextExtensionConfig | None = None,
    speculative: LlamaCppSpeculativeConfig | None = None,
) -> list[str]:
    """Generate llama.cpp CLI arguments for a CacheConfig.

    Args:
        config: Cache configuration.
        flash_attention: Enable flash attention (required for turbo/iso/planar).
        model_path: Path to GGUF model file.
        context_size: Context window size.
        gpu_layers: Number of layers to offload to GPU.
        fork_profile: llama.cpp-compatible profile to target.
        context_extension: RoPE/YaRN context-extension options.
        speculative: Godzilla speculative-decoding options.

    Returns:
        List of CLI argument strings.
    """
    profile = normalize_llamacpp_profile(fork_profile)
    _validate_godzilla_kvarn_config(config, profile)
    args: list[str] = []

    if model_path:
        args.extend(["--model", model_path])

    # Cache type flags
    k_type = LLAMACPP_CACHE_TYPES.get(
        config.k_method,
        GODZILLA_KVARN_CACHE_TYPES.get(config.k_method),
    )
    v_type = LLAMACPP_CACHE_TYPES.get(
        config.v_method,
        GODZILLA_KVARN_CACHE_TYPES.get(config.v_method),
    )

    def _unsupported(label: str, method: CacheMethod) -> ValueError:
        family = METHOD_FAMILIES[method]
        if family == MethodFamily.TRIATTENTION:
            return ValueError(
                f"{label} method {method.value} is token eviction, not a "
                f"llama.cpp cache type. Use a normal K/V cache type and set "
                f"triattention_enabled=True. For patched llama.cpp forks, also "
                f"set use_custom_triattention_llamacpp=True."
            )
        if family == MethodFamily.ROTORQUANT:
            return ValueError(
                f"{label} method {method.value} is not supported by llama.cpp upstream yet — "
                f"cache-type registration pending. Use the Python API directly "
                f"(multi_turboquant.compress / decompress), or fall back to "
                f"iso3/iso4/planar3/planar4 for llama.cpp inference."
            )
        return ValueError(
            f"{label} method {method.value} is not supported in llama.cpp"
        )

    if k_type is None:
        raise _unsupported("K", config.k_method)
    if v_type is None:
        raise _unsupported("V", config.v_method)

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

    if context_extension is not None:
        args.extend(_get_context_extension_args(context_extension))

    if gpu_layers is not None:
        args.extend(["-ngl", str(gpu_layers)])

    # Tensor split for multi-GPU
    if tensor_split:
        args.extend(["--tensor-split", tensor_split])

    # Parallel slots for multi-agent
    if parallel_slots and parallel_slots > 1:
        args.extend(["--parallel", str(parallel_slots)])

    if speculative is not None:
        args.extend(_get_godzilla_speculative_args(speculative, profile))

    # TriAttention token eviction is separate from K/V cache dtype flags.
    if config.triattention_enabled:
        if config.use_custom_triattention_llamacpp:
            args.extend(_get_patched_triattention_args(config))
        else:
            import warnings
            warnings.warn(
                "TriAttention is vLLM-only unless using a patched llama.cpp fork; "
                "it will be ignored in upstream llama.cpp mode.",
                RuntimeWarning,
                stacklevel=2,
            )

    return args


def _get_patched_triattention_args(config: CacheConfig) -> list[str]:
    """Generate TriAttention flags for patched llama.cpp forks."""
    if not config.triattention_stats_path:
        raise ValueError(
            "triattention_stats_path is required when "
            "use_custom_triattention_llamacpp=True"
        )
    if config.triattention_budget <= 0:
        raise ValueError("triattention_budget must be > 0")
    if config.triattention_window < 0:
        raise ValueError("triattention_window must be >= 0")

    args = [
        "--triattention-stats", config.triattention_stats_path,
        "--triattention-budget", str(config.triattention_budget),
        "--triattention-window", str(config.triattention_window),
    ]
    if config.triattention_log:
        args.append("--triattention-log")
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
    cuda_weight_share: object | None = None,
    fork_profile: LlamaCppProfile | str | None = LlamaCppProfile.UPSTREAM,
    context_extension: LlamaCppContextExtensionConfig | None = None,
    speculative: LlamaCppSpeculativeConfig | None = None,
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
        fork_profile: llama.cpp-compatible profile to target.
        context_extension: RoPE/YaRN context-extension options.
        speculative: Godzilla speculative-decoding options.
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
        fork_profile=fork_profile,
        context_extension=context_extension,
        speculative=speculative,
    ))
    cmd.extend(["--host", host, "--port", str(port)])

    if extra_args:
        cmd.extend(extra_args)

    if cuda_weight_share is not None:
        from .weight_share import wrap_cuda_weight_share_command
        cmd = wrap_cuda_weight_share_command(cmd, cuda_weight_share)

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
