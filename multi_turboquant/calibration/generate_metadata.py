# SPDX-License-Identifier: MIT
# Derived from Llama_TQ generate_tq_metadata.py
"""Generate turboquant_kv.json calibration metadata for TurboQuant/TCQ.

Uses weight-norm analysis of K/V projection matrices to identify
high-variance dimensions per attention head. These dimensions get
higher-precision encoding in TurboQuant's mixed-bitwidth KV cache.

Usage as CLI:
    python -m multi_turboquant.calibration.generate_metadata /path/to/model

Usage as library:
    from multi_turboquant.calibration import generate_turboquant_metadata
    metadata = generate_turboquant_metadata("/path/to/model", recipe="turbo3")
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

from ..methods.turboquant import get_outlier_count, GROUP_ALIGNMENT


def _load_tensor(safetensors_files: list[str], tensor_name: str) -> torch.Tensor:
    """Load a named tensor from the correct safetensors shard."""
    from safetensors import safe_open
    for sf_path in safetensors_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            if tensor_name in f.keys():
                return f.get_tensor(tensor_name)
    raise KeyError(f"Tensor {tensor_name!r} not found in any shard")


def _detect_weight_pattern(
    sf_files: list[str], projection: str = "k_proj",
) -> str | None:
    """Auto-detect the weight naming convention for K/V projections."""
    from safetensors import safe_open

    patterns = [
        f"model.layers.{{}}.self_attn.{projection}.weight",
        f"transformer.h.{{}}.attn.{projection}.weight",
        f"model.layers.{{}}.attention.{projection}.weight",
    ]

    for pattern in patterns:
        test_name = pattern.format(0)
        try:
            _load_tensor(sf_files, test_name)
            return pattern
        except (KeyError, Exception):
            continue

    # Fallback: scan tensor names
    for sf_path in sf_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if projection in name and ".0." in name:
                    return name.replace(".0.", ".{}.")
    return None


def generate_turboquant_metadata(
    model_path: str,
    recipe: str = "turbo3",
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate TurboQuant calibration metadata from model weights.

    Args:
        model_path: Path to model directory containing config.json
            and safetensors files.
        recipe: "turbo2", "turbo3", or "turbo4".
        verbose: Print progress to stdout.

    Returns:
        Metadata dict ready for JSON serialization.
    """
    # Map recipe names
    recipe_map = {
        "turbo2": "turboquant25",
        "turbo3": "turboquant35",
        "turbo4": "turboquant35",  # turbo4 uses same outlier ratio as turbo3
        "turboquant25": "turboquant25",
        "turboquant35": "turboquant35",
        "turbo2_tcq": "turboquant25",
        "turbo3_tcq": "turboquant35",
    }
    internal_recipe = recipe_map.get(recipe, "turboquant35")
    outlier_ratios = {"turboquant25": 0.25, "turboquant35": 0.50}

    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    num_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    hidden_size = config["hidden_size"]
    head_dim = config.get("head_dim", hidden_size // num_heads)
    model_type = config.get("model_type", "unknown")

    outlier_count = get_outlier_count(head_dim, recipe.replace("_tcq", ""))

    if verbose:
        print(f"Model: {os.path.basename(model_path)} ({model_type})")
        print(f"  layers={num_layers}  heads={num_heads}  "
              f"kv_heads={num_kv_heads}  head_dim={head_dim}")
        print(f"  recipe={recipe}  outlier_count={outlier_count}/{head_dim}")

    sf_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not sf_files:
        raise FileNotFoundError(f"No .safetensors files in {model_path}")

    if verbose:
        print(f"Found {len(sf_files)} safetensors shard(s)")

    k_pattern = _detect_weight_pattern(sf_files, "k_proj")
    v_pattern = _detect_weight_pattern(sf_files, "v_proj")

    if k_pattern is None:
        raise ValueError("Could not detect K projection weight naming convention")
    if v_pattern is None:
        v_pattern = k_pattern.replace("k_proj", "v_proj")

    if verbose:
        print(f"  K pattern: {k_pattern}")
        print(f"  V pattern: {v_pattern}")

    layers = {}
    for layer_idx in range(num_layers):
        k_name = k_pattern.format(layer_idx)
        v_name = v_pattern.format(layer_idx)

        k_weight = _load_tensor(sf_files, k_name).float()
        v_weight = _load_tensor(sf_files, v_name).float()

        k_weight = k_weight.view(num_kv_heads, head_dim, -1)
        v_weight = v_weight.view(num_kv_heads, head_dim, -1)

        k_norms = k_weight.norm(dim=-1)  # [num_kv_heads, head_dim]
        v_norms = v_weight.norm(dim=-1)

        k_indices = []
        v_indices = []
        for head in range(num_kv_heads):
            _, k_idx = k_norms[head].topk(outlier_count)
            _, v_idx = v_norms[head].topk(outlier_count)
            k_indices.append(sorted(k_idx.tolist()))
            v_indices.append(sorted(v_idx.tolist()))

        layer_name = f"model.layers.{layer_idx}.self_attn"
        layers[layer_name] = {
            "key_high_precision_indices": k_indices,
            "value_high_precision_indices": v_indices,
        }

        if verbose:
            sys.stdout.write(f"\r  Calibrating layer {layer_idx + 1}/{num_layers}...")
            sys.stdout.flush()

    if verbose:
        print(" done")

    return {
        "version": 1,
        "recipe": internal_recipe,
        "head_size": head_dim,
        "model_name": os.path.basename(model_path),
        "transform_version": "structured_hadamard_v1",
        "codebook_version": "lloyd_beta_v1",
        "layers": layers,
        "calibration": {
            "method": "weight_norm",
            "objective": "l2_norm",
            "num_prompts": 0,
            "max_seq_len": 0,
            "batch_size": 0,
            "num_observed_tokens": 0,
            "dtype": str(config.get("torch_dtype", "float16")),
            "device": "cpu",
            "prompts_sha256": "",
        },
    }


def save_metadata(metadata: dict, output_path: str | None = None, model_path: str | None = None) -> str:
    """Save metadata to JSON file."""
    if output_path is None:
        if model_path is None:
            raise ValueError("Either output_path or model_path must be provided")
        output_path = os.path.join(model_path, "turboquant_kv.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate TurboQuant KV cache metadata"
    )
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument(
        "--recipe", default="turbo3",
        choices=["turbo2", "turbo3", "turbo4", "turbo2_tcq", "turbo3_tcq"],
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        print(f"Error: {args.model_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    metadata = generate_turboquant_metadata(args.model_path, args.recipe)
    out_path = save_metadata(metadata, args.output, args.model_path)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nSaved {out_path} ({size_kb:.1f} KB)")
    print(f"  {len(metadata['layers'])} layers calibrated for {args.recipe}")


if __name__ == "__main__":
    main()
