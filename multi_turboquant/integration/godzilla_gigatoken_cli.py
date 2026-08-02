# SPDX-License-Identifier: MIT
"""CLI for the pinned Godzilla + Gigatoken runtime workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .godzilla_gigatoken import (
    DEFAULT_GODZILLA_PROFILE,
    GODZILLA_SOURCE_PROFILES,
    build_godzilla_gigatoken,
    plan_godzilla_gigatoken,
    prepare_godzilla_gigatoken,
    verify_godzilla_gigatoken,
)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("target", type=Path, help="New or prepared combined source directory")
    parser.add_argument(
        "--godzilla-profile",
        choices=tuple(profile.id for profile in GODZILLA_SOURCE_PROFILES),
        default=DEFAULT_GODZILLA_PROFILE,
        help="Exact reviewed Godzilla source baseline",
    )
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-jobs", type=int, default=2)
    parser.add_argument("--with-curl", action="store_true", help="Build server URL-download support")
    parser.add_argument("--generator", help="Optional CMake generator")
    parser.add_argument("--cuda-toolkit", type=Path, help="CUDA toolkit root or nvcc path")
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="Optional directory with DeepSeek V3, GPT-OSS, or Kimi K2.7 vocab GGUF fixtures",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and qualify a reviewed Godzilla + Gigatoken runtime"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action, help_text in (
        ("plan", "Inspect preparation without changing anything"),
        ("prepare", "Create the pinned combined source tree"),
        ("build", "Build and run tokenizer differential tests"),
        ("all", "Prepare, build, and test consecutively"),
        ("verify", "Rerun tests for an existing build"),
    ):
        child = subparsers.add_parser(action, help=help_text)
        _add_common(child)
        child.add_argument("--json", action="store_true")
        if action == "plan":
            child.add_argument(
                "--for-action",
                choices=("prepare", "build", "verify"),
                default="prepare",
                help="Operation to inspect without changing anything",
            )
        if action in {"prepare", "build", "all"}:
            child.add_argument("--yes", action="store_true", help="Confirm downloads and builds")
    return parser


def _print(value: dict[str, object], as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    print(json.dumps(value, indent=2, sort_keys=False))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    planned_action = args.for_action if args.action == "plan" else args.action
    plan = plan_godzilla_gigatoken(
        args.target,
        godzilla_profile=args.godzilla_profile,
        action=planned_action,
        backend=args.backend,
        max_jobs=args.max_jobs,
        with_curl=args.with_curl,
        generator=args.generator,
        cuda_toolkit=args.cuda_toolkit,
        fixture_dir=args.fixture_dir,
    )
    if args.action == "plan":
        _print(plan.to_dict(), args.json)
        print("No files, downloads, or builds were changed.")
        return 0 if plan.ready else 2
    if not plan.ready:
        _print(plan.to_dict(), args.json)
        return 2
    try:
        if args.action == "prepare":
            result = prepare_godzilla_gigatoken(plan, confirmed=args.yes)
        elif args.action == "build":
            result = build_godzilla_gigatoken(plan, confirmed=args.yes)
        elif args.action == "verify":
            result = verify_godzilla_gigatoken(plan)
        else:
            prepare_godzilla_gigatoken(plan, confirmed=args.yes)
            build_plan = plan_godzilla_gigatoken(
                args.target,
                godzilla_profile=args.godzilla_profile,
                action="build",
                backend=args.backend,
                max_jobs=args.max_jobs,
                with_curl=args.with_curl,
                generator=args.generator,
                cuda_toolkit=args.cuda_toolkit,
                fixture_dir=args.fixture_dir,
            )
            result = build_godzilla_gigatoken(build_plan, confirmed=args.yes)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _print(result, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
