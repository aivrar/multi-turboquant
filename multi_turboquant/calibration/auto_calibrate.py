# SPDX-License-Identifier: MIT
"""Unified auto-calibration: detects required methods and generates everything."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..config import CacheConfig, CacheMethod, CALIBRATION_REQUIRED, MethodFamily, METHOD_FAMILIES


def auto_calibrate(
    config: CacheConfig,
    model_path: str,
    *,
    output_dir: str | None = None,
    verbose: bool = True,
) -> dict[str, str]:
    """Auto-calibrate all methods that need it for the given config.

    Detects which methods in the config require calibration, runs the
    appropriate generators, and returns paths to the generated files.

    Args:
        config: CacheConfig specifying the methods to use.
        model_path: Path to the model directory.
        output_dir: Where to save calibration files (default: model_path).
        verbose: Print progress.

    Returns:
        Dict of {method_name: calibration_file_path}.
    """
    if output_dir is None:
        output_dir = model_path

    results: dict[str, str] = {}
    methods_needing_calibration = set()

    for method in [config.k_method, config.v_method]:
        if method in CALIBRATION_REQUIRED:
            methods_needing_calibration.add(method)

    if config.triattention_enabled:
        methods_needing_calibration.add(CacheMethod.TRIATTENTION)

    # ─── TurboQuant / TCQ calibration ───────────────────────────────────
    tq_methods = {
        m for m in methods_needing_calibration
        if METHOD_FAMILIES.get(m) in (MethodFamily.TURBOQUANT, MethodFamily.TCQ)
    }
    if tq_methods:
        from .generate_metadata import generate_turboquant_metadata, save_metadata

        # Pick the recipe from the first TQ method found
        recipe_map = {
            CacheMethod.TURBO2: "turbo2",
            CacheMethod.TURBO3: "turbo3",
            CacheMethod.TURBO4: "turbo4",
            CacheMethod.TURBO2_TCQ: "turbo2_tcq",
            CacheMethod.TURBO3_TCQ: "turbo3_tcq",
        }
        recipe = recipe_map.get(next(iter(tq_methods)), "turbo3")

        # Check if metadata already exists
        existing = os.path.join(output_dir, "turboquant_kv.json")
        if os.path.isfile(existing):
            if verbose:
                print(f"TurboQuant metadata already exists: {existing}")
            for m in tq_methods:
                results[m.value] = existing
        else:
            if verbose:
                print(f"Generating TurboQuant metadata for {recipe}...")
            metadata = generate_turboquant_metadata(
                model_path, recipe, verbose=verbose,
            )
            out_path = save_metadata(metadata, model_path=output_dir)
            if verbose:
                print(f"  Saved: {out_path}")
            for m in tq_methods:
                results[m.value] = out_path

    # ─── TriAttention calibration ───────────────────────────────────────
    if CacheMethod.TRIATTENTION in methods_needing_calibration:
        stats_path = os.path.join(output_dir, "triattention_stats.pt")
        if os.path.isfile(stats_path):
            if verbose:
                print(f"TriAttention stats already exist: {stats_path}")
            results["triattention"] = stats_path
        else:
            if verbose:
                print("Generating TriAttention stats...")
                print("  (requires model loaded in memory — may take a while)")
            try:
                from .generate_stats import generate_triattention_stats, save_stats
                stats = generate_triattention_stats(
                    model_path=model_path, verbose=verbose,
                )
                out_path = save_stats(stats, stats_path)
                results["triattention"] = out_path
                if verbose:
                    print(f"  Saved: {out_path}")
            except Exception as e:
                if verbose:
                    print(f"  Warning: TriAttention calibration failed: {e}")
                    print("  You can generate stats manually with:")
                    print(f"    python -m multi_turboquant.calibration.generate_stats {model_path}")

    if not methods_needing_calibration:
        if verbose:
            print("No calibration needed — all selected methods are calibration-free.")

    return results


def check_calibration(
    config: CacheConfig,
    model_path: str,
) -> dict[str, bool]:
    """Check if all required calibration files exist.

    Returns:
        Dict of {method_name: exists_bool}.
    """
    checks: dict[str, bool] = {}

    for method in [config.k_method, config.v_method]:
        if method not in CALIBRATION_REQUIRED:
            continue
        family = METHOD_FAMILIES[method]
        if family in (MethodFamily.TURBOQUANT, MethodFamily.TCQ):
            path = os.path.join(model_path, "turboquant_kv.json")
            checks[method.value] = os.path.isfile(path)
        elif family == MethodFamily.TRIATTENTION:
            path = os.path.join(model_path, "triattention_stats.pt")
            checks[method.value] = os.path.isfile(path)

    if config.triattention_enabled:
        path = os.path.join(model_path, "triattention_stats.pt")
        checks["triattention"] = os.path.isfile(path)

    return checks
