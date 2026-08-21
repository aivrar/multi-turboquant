# SPDX-License-Identifier: MIT
"""Experimental low-memory TriAttention statistics from a local GGUF model.

Transformers currently dequantizes GGUF tensors before execution.  This module
therefore avoids a source-weight download and bounds retained query state, but
it is not yet a native packed-GGUF/GGML calibration backend.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Mapping

from .godzilla_triattention import (
    convert_official_triattention_stats,
    normalize_calibration_device,
    validate_model_calibration_length,
)

PROTOTYPE_MAX_TOKENS = 32_768
DEFAULT_PROJECTION_CHUNK_TOKENS = 2_048
_GGUF_QUANTIZATION = re.compile(
    r"(?:^|[-_.])((?:I?Q|TQ|F|BF)\d+(?:_[A-Z0-9]+)*)(?=[-_.]|$)", re.IGNORECASE
)


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash one local artifact without reading it all into memory."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"GGUF model not found: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: object, field: str) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _config_metadata(config: object) -> dict[str, object]:
    heads = _positive_int(getattr(config, "num_attention_heads", None), "num_attention_heads")
    layers = _positive_int(getattr(config, "num_hidden_layers", None), "num_hidden_layers")
    kv_heads = _positive_int(
        getattr(config, "num_key_value_heads", heads), "num_key_value_heads"
    )
    hidden_size = _positive_int(getattr(config, "hidden_size", None), "hidden_size")
    head_dim = _positive_int(
        getattr(config, "head_dim", hidden_size // heads), "head_dim"
    )
    context = getattr(config, "max_position_embeddings", None)
    if context is not None:
        context = _positive_int(context, "max_position_embeddings")
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is None:
        rope_parameters = getattr(config, "rope_scaling", None)
    nested_theta = (
        rope_parameters.get("rope_theta")
        if isinstance(rope_parameters, Mapping)
        else None
    )
    legacy_theta = getattr(config, "rope_theta", None)
    theta = nested_theta if nested_theta is not None else legacy_theta
    if theta is None:
        raise ValueError("GGUF metadata does not declare rope_theta")
    return {
        "model_type": getattr(config, "model_type", None),
        "num_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "max_position_embeddings": context,
        "rope_theta": float(theta),
        "rope_type": (
            rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
            if isinstance(rope_parameters, Mapping)
            else "default"
        ),
    }


def _local_gguf_arguments(path: str | Path) -> tuple[str, str]:
    """Return a local directory and relative filename safe for Transformers on Windows."""
    source = Path(path).expanduser().resolve()
    return str(source.parent), source.name


def _gguf_quantization(path: str | Path) -> str:
    """Infer a common GGUF quantization label from its filename."""
    match = _GGUF_QUANTIZATION.search(Path(path).stem)
    return match.group(1).upper() if match else "unknown"


def load_local_gguf_metadata(path: str | Path) -> dict[str, object]:
    """Read model metadata directly from one local GGUF without remote code."""
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".gguf":
        raise ValueError(f"GGUF model not found: {source}")
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("transformers is required for GGUF metadata inspection") from exc
    model_directory, gguf_filename = _local_gguf_arguments(source)
    try:
        config = AutoConfig.from_pretrained(
            model_directory,
            gguf_file=gguf_filename,
            local_files_only=True,
            trust_remote_code=False,
        )
    except ImportError as exc:
        raise RuntimeError("GGUF inspection requires the optional 'gguf' package") from exc
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        **_config_metadata(config),
    }


class StreamingQueryStats:
    """Accumulate official-compatible base-Q moments without retaining Q traces."""

    def __init__(self, *, num_heads: int, head_dim: int, chunk_tokens: int = 2_048):
        import torch

        self.num_heads = _positive_int(num_heads, "num_heads")
        self.head_dim = _positive_int(head_dim, "head_dim")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even")
        self.chunk_tokens = _positive_int(chunk_tokens, "chunk_tokens")
        self._real: dict[int, torch.Tensor] = {}
        self._imag: dict[int, torch.Tensor] = {}
        self._absolute: dict[int, torch.Tensor] = {}
        self._count: dict[int, int] = {}

    def hook(self, layer_index: int):
        """Return a bounded forward pre-hook for one attention module."""
        import torch

        if layer_index < 0:
            raise ValueError("layer_index must be non-negative")

        def collect(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                raise RuntimeError(f"Layer {layer_index} did not provide hidden_states")
            batch, tokens, _ = hidden_states.shape
            frequency_count = self.head_dim // 2
            real_sum = torch.zeros(
                self.num_heads, frequency_count, dtype=torch.float64, device="cpu"
            )
            imag_sum = torch.zeros_like(real_sum)
            absolute_sum = torch.zeros_like(real_sum)
            with torch.no_grad():
                for start in range(0, tokens, self.chunk_tokens):
                    projected = module.q_proj(
                        hidden_states[:, start : start + self.chunk_tokens]
                    )
                    projected = projected.reshape(
                        batch, -1, self.num_heads, self.head_dim
                    ).transpose(1, 2)
                    projected = projected.float()
                    real = projected[..., :frequency_count]
                    imag = projected[..., frequency_count:]
                    real_sum += real.sum(dim=(0, 2), dtype=torch.float64).cpu()
                    imag_sum += imag.sum(dim=(0, 2), dtype=torch.float64).cpu()
                    absolute_sum += torch.hypot(real, imag).sum(
                        dim=(0, 2), dtype=torch.float64
                    ).cpu()
            sample_count = batch * tokens
            if layer_index in self._count:
                self._real[layer_index] += real_sum
                self._imag[layer_index] += imag_sum
                self._absolute[layer_index] += absolute_sum
                self._count[layer_index] += sample_count
            else:
                self._real[layer_index] = real_sum
                self._imag[layer_index] = imag_sum
                self._absolute[layer_index] = absolute_sum
                self._count[layer_index] = sample_count

        return collect

    def payload(self, *, metadata: Mapping[str, object]) -> dict[str, object]:
        """Build the strict official payload consumed by the Godzilla converter."""
        stats: dict[str, dict[str, object]] = {}
        sampled: list[list[int]] = []
        for layer_index in sorted(self._count):
            count = self._count[layer_index]
            if count <= 0:
                raise RuntimeError(f"Layer {layer_index} has no query samples")
            for head_index in range(self.num_heads):
                key = f"layer{layer_index:02d}_head{head_index:02d}"
                stats[key] = {
                    "q_mean_real": (self._real[layer_index][head_index] / count).float(),
                    "q_mean_imag": (self._imag[layer_index][head_index] / count).float(),
                    "q_abs_mean": (self._absolute[layer_index][head_index] / count).float(),
                }
                sampled.append([layer_index, head_index])
        if not stats:
            raise RuntimeError("No query statistics were captured")
        return {
            "metadata": {
                **dict(metadata),
                "head_dim": self.head_dim,
                "rope_style": "half",
                "sampled_heads": sampled,
            },
            "stats": stats,
        }


def _attention_layers(model: object) -> list[object]:
    backbone = getattr(model, "model", model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise RuntimeError("Cannot locate model.model.layers for GGUF calibration")
    attention_layers = []
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None or not hasattr(attention, "q_proj"):
            raise RuntimeError(f"Layer {index} does not expose self_attn.q_proj")
        attention_layers.append(attention)
    return attention_layers


def _atomic_torch_save(payload: Mapping[str, object], output: Path) -> None:
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_name = handle.name
        torch.save(dict(payload), temporary_name)
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def calibrate_local_gguf_streaming(
    *,
    gguf: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    godzilla_output_path: str | Path | None = None,
    max_length: int = 2_048,
    device: str = "cpu",
    projection_chunk_tokens: int = DEFAULT_PROJECTION_CHUNK_TOKENS,
    attention_implementation: str = "sdpa",
    confirm_fp32_dequantization: bool = False,
) -> dict[str, object]:
    """Run the bounded first-stage GGUF prototype and write official-compatible stats."""
    if not confirm_fp32_dequantization:
        raise ValueError(
            "This prototype uses Transformers, which dequantizes GGUF weights for execution; "
            "set confirm_fp32_dequantization=True to acknowledge the RAM cost"
        )
    length = _positive_int(max_length, "max_length")
    if length > PROTOTYPE_MAX_TOKENS:
        raise ValueError(
            f"The GGUF streaming prototype is currently capped at {PROTOTYPE_MAX_TOKENS} tokens"
        )
    chunk_tokens = _positive_int(projection_chunk_tokens, "projection_chunk_tokens")
    normalized_device = normalize_calibration_device(device)
    source = Path(gguf).expanduser().resolve()
    calibration_input = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    godzilla_output = (
        Path(godzilla_output_path).expanduser().resolve()
        if godzilla_output_path is not None
        else None
    )
    if not source.is_file() or source.suffix.lower() != ".gguf":
        raise ValueError(f"GGUF model not found: {source}")
    if not calibration_input.is_file() or calibration_input.stat().st_size == 0:
        raise ValueError(f"Calibration input must be non-empty: {calibration_input}")
    if output.suffix.lower() != ".pt":
        raise ValueError("Streaming GGUF statistics output must end in .pt")
    if godzilla_output is not None:
        if godzilla_output.suffix.lower() != ".triattention":
            raise ValueError("Godzilla statistics output must end in .triattention")
        if godzilla_output == output:
            raise ValueError("Intermediate .pt and Godzilla outputs must differ")

    source_stat = source.stat()
    source_identity = (source_stat.st_size, source_stat.st_mtime_ns)
    source_sha256 = sha256_file(source)

    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "GGUF streaming calibration requires Torch, Transformers, Accelerate, and gguf"
        ) from exc

    model_directory, gguf_filename = _local_gguf_arguments(source)
    config = AutoConfig.from_pretrained(
        model_directory,
        gguf_file=gguf_filename,
        local_files_only=True,
        trust_remote_code=False,
    )
    model_metadata = _config_metadata(config)
    validate_model_calibration_length(length, model_metadata)
    tokenizer = AutoTokenizer.from_pretrained(
        model_directory,
        config=config,
        gguf_file=gguf_filename,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_directory,
        gguf_file=gguf_filename,
        config=config,
        device_map=normalized_device,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation=attention_implementation,
    )
    model.eval()
    text = calibration_input.read_text(encoding="utf-8")
    input_ids = tokenizer.encode(
        text, return_tensors="pt", truncation=True, max_length=length
    )
    input_device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(input_device)
    collector = StreamingQueryStats(
        num_heads=int(model_metadata["num_attention_heads"]),
        head_dim=int(model_metadata["head_dim"]),
        chunk_tokens=chunk_tokens,
    )
    attention_layers = _attention_layers(model)
    handles = [
        attention.register_forward_pre_hook(collector.hook(index), with_kwargs=True)
        for index, attention in enumerate(attention_layers)
    ]
    try:
        with torch.no_grad():
            model(input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    first_parameter = next(model.parameters())
    final_stat = source.stat()
    if (final_stat.st_size, final_stat.st_mtime_ns) != source_identity:
        raise RuntimeError("GGUF model changed while calibration was running; no output was written")
    payload = collector.payload(
        metadata={
            "num_traces": 1,
            "dtype": str(first_parameter.dtype).replace("torch.", ""),
            "use_chat_template": False,
            "system_prompt": "",
            "attn_implementation": attention_implementation,
            "rope_type": model_metadata["rope_type"],
            "source_format": "gguf-transformers-dequantized",
            "source_quantization": _gguf_quantization(source),
            "source_sha256": source_sha256,
            "source_size_bytes": source_identity[0],
            "runtime_parameter_dtype": str(first_parameter.dtype).replace("torch.", ""),
            "tokenized_length": int(input_ids.shape[1]),
            "projection_chunk_tokens": chunk_tokens,
        }
    )
    _atomic_torch_save(payload, output)
    report: dict[str, object] = {
        "gguf": str(source),
        "output": str(output),
        "tokenized_length": int(input_ids.shape[1]),
        "source_sha256": payload["metadata"]["source_sha256"],  # type: ignore[index]
        "runtime_parameter_dtype": payload["metadata"]["runtime_parameter_dtype"],  # type: ignore[index]
        "retained_query_bytes": 0,
        "projection_chunk_tokens": chunk_tokens,
        "warning": "Transformers dequantizes GGUF weights; this is not native packed-IQ4 execution.",
    }
    if godzilla_output is not None:
        converted = convert_official_triattention_stats(
            output,
            godzilla_output,
            model_name=source.stem,
            num_layers=int(model_metadata["num_layers"]),
            num_attention_heads=int(model_metadata["num_attention_heads"]),
            num_key_value_heads=int(model_metadata["num_key_value_heads"]),
            rope_theta=float(model_metadata["rope_theta"]),
            expected_head_dim=int(model_metadata["head_dim"]),
        )
        report["godzilla_output"] = str(godzilla_output)
        report["godzilla_sampled_heads"] = converted["sampled_heads"]
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experimental bounded-memory TriAttention stats from a local GGUF"
    )
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--godzilla-output",
        help="Optional final Godzilla v1 .triattention output, validated after writing",
    )
    parser.add_argument("--max-length", type=int, default=2_048)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument(
        "--projection-chunk-tokens", type=int, default=DEFAULT_PROJECTION_CHUNK_TOKENS
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--confirm-fp32-dequantization", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = calibrate_local_gguf_streaming(
        gguf=args.gguf,
        input_path=args.input,
        output_path=args.output,
        godzilla_output_path=args.godzilla_output,
        max_length=args.max_length,
        device=args.device,
        projection_chunk_tokens=args.projection_chunk_tokens,
        attention_implementation=args.attn_implementation,
        confirm_fp32_dequantization=args.confirm_fp32_dequantization,
    )
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
