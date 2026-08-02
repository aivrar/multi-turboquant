# SPDX-License-Identifier: MIT
"""Fail-closed Gigatoken wrapper for the official TriAttention calibrator."""

from __future__ import annotations

import argparse
import importlib.metadata
import runpy
import sys
from pathlib import Path
from typing import Callable


def _ids(tokenizer: object, text: str, *, max_length: int) -> list[int]:
    encoded = tokenizer.encode(  # type: ignore[attr-defined]
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, list) or any(not isinstance(item, int) for item in encoded):
        raise RuntimeError("Tokenizer parity check returned non-integer token IDs")
    return encoded


def validate_tokenizer_parity(
    reference: object,
    accelerated: object,
    text: str,
    *,
    max_length: int,
) -> int:
    """Require exact token-ID parity for the actual selected calibration text."""
    expected = _ids(reference, text, max_length=max_length)
    observed = _ids(accelerated, text, max_length=max_length)
    if expected != observed:
        mismatch = next(
            (index for index, pair in enumerate(zip(expected, observed)) if pair[0] != pair[1]),
            min(len(expected), len(observed)),
        )
        expected_value = expected[mismatch] if mismatch < len(expected) else "<end>"
        observed_value = observed[mismatch] if mismatch < len(observed) else "<end>"
        raise RuntimeError(
            "Gigatoken token-ID parity failed at index "
            f"{mismatch}: Hugging Face={expected_value}, Gigatoken={observed_value}; "
            f"lengths {len(expected)} and {len(observed)}"
        )
    return len(expected)


def _patch_auto_tokenizer(
    *,
    validation_text: str,
    max_length: int,
    stderr: object = sys.stderr,
) -> Callable[..., object]:
    import gigatoken
    from transformers import AutoTokenizer

    version = importlib.metadata.version("gigatoken")
    parts = version.split(".")
    if len(parts) < 2 or parts[:2] != ["0", "10"]:
        raise RuntimeError(
            f"Gigatoken {version} is not in the reviewed 0.10.x compatibility series"
        )
    original = AutoTokenizer.from_pretrained

    def accelerated_from_pretrained(*args: object, **kwargs: object) -> object:
        reference = original(*args, **kwargs)
        accelerated = gigatoken.Tokenizer(reference).as_hf()
        token_count = validate_tokenizer_parity(
            reference,
            accelerated,
            validation_text,
            max_length=max_length,
        )
        print(
            f"Gigatoken {version} parity validation passed for {token_count} token IDs.",
            file=stderr,
        )
        return accelerated

    AutoTokenizer.from_pretrained = staticmethod(accelerated_from_pretrained)
    return original


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the official TriAttention calibrator with validated Gigatoken"
    )
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--attn-implementation", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    calibrator = Path(args.calibrator).expanduser().resolve()
    calibration_input = Path(args.input).expanduser().resolve()
    if not calibrator.is_file():
        raise ValueError(f"Official TriAttention calibrator not found: {calibrator}")
    if not calibration_input.is_file() or calibration_input.stat().st_size == 0:
        raise ValueError(f"Calibration input must be non-empty: {calibration_input}")
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
        "--attn-implementation",
        args.attn_implementation,
    ]
    runpy.run_path(str(calibrator), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
