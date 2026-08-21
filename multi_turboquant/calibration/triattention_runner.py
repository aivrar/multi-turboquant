# SPDX-License-Identifier: MIT
"""Run reviewed TriAttention calibrators with model/device compatibility patches."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Callable, Mapping


def _value(source: object, name: str) -> object | None:
    value = getattr(source, name, None)
    if value is not None:
        return value
    return source.get(name) if isinstance(source, Mapping) else None


def _authoritative_rope_theta(config: object) -> float | None:
    text_config = getattr(config, "text_config", config)
    parameters = _value(text_config, "rope_parameters")
    if parameters is None:
        parameters = _value(text_config, "rope_scaling")
    nested = _value(parameters, "rope_theta") if parameters is not None else None
    return float(nested) if nested is not None else None


def patch_auto_config(*, stderr: object = sys.stderr) -> Callable[..., object]:
    """Promote nested RoPE metadata for calibrators written against older Transformers."""
    from transformers import AutoConfig

    original = AutoConfig.from_pretrained

    def compatible_from_pretrained(*args: object, **kwargs: object) -> object:
        config = original(*args, **kwargs)
        text_config = getattr(config, "text_config", config)
        authoritative = _authoritative_rope_theta(config)
        legacy = getattr(text_config, "rope_theta", None)
        if authoritative is not None and (
            legacy is None or float(legacy) != authoritative
        ):
            setattr(text_config, "rope_theta", authoritative)
            print(
                "TriAttention compatibility: using nested rope_parameters.rope_theta="
                f"{authoritative:g} instead of legacy value {legacy!r}.",
                file=stderr,
            )
        return config

    AutoConfig.from_pretrained = staticmethod(compatible_from_pretrained)
    return original


def patch_auto_model_device(device: str) -> Callable[..., object]:
    """Force reviewed calibrators' ``device_map='auto'`` onto the selected CUDA device."""
    from transformers import AutoModelForCausalLM

    original = AutoModelForCausalLM.from_pretrained

    def selected_from_pretrained(*args: object, **kwargs: object) -> object:
        if device.startswith("cuda") and kwargs.get("device_map") == "auto":
            kwargs["device_map"] = device
        return original(*args, **kwargs)

    AutoModelForCausalLM.from_pretrained = staticmethod(selected_from_pretrained)
    return original


def _select_cuda_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    index = int(device.split(":", 1)[1]) if ":" in device else 0
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {index} was requested, but only {torch.cuda.device_count()} device(s) "
            "are visible"
        )
    torch.cuda.set_device(index)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a reviewed TriAttention calibrator with compatibility guards"
    )
    parser.add_argument("--kind", choices=("official", "domvox"), required=True)
    parser.add_argument(
        "--tokenizer-backend", choices=("transformers", "gigatoken"), required=True
    )
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--attn-implementation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    calibrator = Path(args.calibrator).expanduser().resolve()
    calibration_input = Path(args.input).expanduser().resolve()
    if not calibrator.is_file():
        raise ValueError(f"{args.kind} TriAttention calibrator not found: {calibrator}")
    if not calibration_input.is_file() or calibration_input.stat().st_size == 0:
        raise ValueError(f"Calibration input must be non-empty: {calibration_input}")
    if args.kind == "official" and not args.attn_implementation:
        raise ValueError("Official calibration requires --attn-implementation")
    if args.kind == "domvox" and args.attn_implementation:
        raise ValueError("domvox calibration does not accept --attn-implementation")

    _select_cuda_device(args.device)
    patch_auto_config()
    patch_auto_model_device(args.device)
    if args.tokenizer_backend == "gigatoken":
        from gigatoken_runner import _patch_auto_tokenizer

        text = calibration_input.read_text(encoding="utf-8")
        _patch_auto_tokenizer(validation_text=text, max_length=args.max_length)

    sys.argv = [
        str(calibrator),
        "--model",
        args.model,
        "--input",
        str(calibration_input),
        "--output",
        args.output,
        "--max-length",
        str(args.max_length),
        "--device",
        args.device,
    ]
    if args.kind == "official":
        sys.argv.extend(("--attn-implementation", args.attn_implementation))
    runpy.run_path(str(calibrator), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
