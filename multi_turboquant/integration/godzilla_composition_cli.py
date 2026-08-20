# SPDX-License-Identifier: MIT
"""CLI for the pinned Godzilla composition overlay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .godzilla_composition import (
    build_godzilla_composition,
    plan_godzilla_composition,
    prepare_godzilla_composition,
    verify_godzilla_composition,
)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("target", type=Path, help="New or prepared composition source directory")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-jobs", type=int, default=2)
    parser.add_argument("--generator", help="Optional CMake generator")
    parser.add_argument("--cuda-toolkit", type=Path, help="CUDA toolkit root or nvcc path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and qualify the pinned Godzilla PFlash/KVFlash composition overlay"
    )
    children = parser.add_subparsers(dest="action", required=True)
    for action, help_text in (
        ("plan", "Inspect the workflow without changing files"),
        ("prepare", "Create a new pinned overlay source tree"),
        ("build", "Build the prepared llama-server"),
        ("all", "Prepare, build, and verify consecutively"),
        ("verify", "Verify an existing prepared build"),
    ):
        child = children.add_parser(action, help=help_text)
        _add_common(child)
        child.add_argument("--json", action="store_true")
        if action == "plan":
            child.add_argument(
                "--for-action", choices=("prepare", "build", "verify"), default="prepare"
            )
        if action in {"prepare", "build", "all"}:
            child.add_argument("--yes", action="store_true", help="Confirm downloads/builds")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    action = args.for_action if args.action == "plan" else args.action
    plan = plan_godzilla_composition(
        args.target,
        action=action,
        backend=args.backend,
        max_jobs=args.max_jobs,
        generator=args.generator,
        cuda_toolkit=args.cuda_toolkit,
    )
    if args.action == "plan":
        print(json.dumps(plan.to_dict(), indent=2))
        print("No files, downloads, or builds were changed.")
        return 0 if plan.ready else 2
    if not plan.ready:
        print(json.dumps(plan.to_dict(), indent=2))
        return 2
    try:
        if args.action == "prepare":
            result = prepare_godzilla_composition(plan, confirmed=args.yes)
        elif args.action == "build":
            result = build_godzilla_composition(plan, confirmed=args.yes)
        elif args.action == "verify":
            result = verify_godzilla_composition(plan)
        else:
            prepare_godzilla_composition(plan, confirmed=args.yes)
            build_plan = plan_godzilla_composition(
                args.target,
                action="build",
                backend=args.backend,
                max_jobs=args.max_jobs,
                generator=args.generator,
                cuda_toolkit=args.cuda_toolkit,
            )
            result = build_godzilla_composition(build_plan, confirmed=args.yes)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result.get("valid", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
