# SPDX-License-Identifier: MIT
"""CLI for reviewed CUDA LLM weight-share source builds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .weight_share import (
    build_cuda_weight_share,
    inspect_cuda_weight_share_source,
    plan_cuda_weight_share_build,
    validate_cuda_weight_share_library,
)


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", type=Path, help="Pinned cuda-llm-weight-share checkout")
    parser.add_argument("--output", type=Path, help="Output .so path; defaults inside the checkout")
    parser.add_argument("--cuda-toolkit", type=Path, help="CUDA toolkit root or nvcc path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect, build, and validate the pinned CUDA LLM weight-share helper"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="Inspect source without executing it")
    inspect_parser.add_argument("source", type=Path)
    inspect_parser.add_argument("--json", action="store_true")
    plan_parser = subparsers.add_parser("plan", help="Plan a source build without changing files")
    _add_build_arguments(plan_parser)
    plan_parser.add_argument("--json", action="store_true")
    build = subparsers.add_parser("build", help="Compile and validate the shared object")
    _add_build_arguments(build)
    build.add_argument("--yes", action="store_true", help="Confirm compilation")
    build.add_argument("--json", action="store_true")
    validate = subparsers.add_parser("validate", help="Validate an existing shared object")
    validate.add_argument("library", type=Path)
    validate.add_argument("--json", action="store_true")
    return parser


def _print(value: dict[str, object], *, as_json: bool) -> None:
    print(json.dumps(value, indent=2, sort_keys=as_json))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "inspect":
            result = inspect_cuda_weight_share_source(args.source)
        elif args.action == "validate":
            result = validate_cuda_weight_share_library(args.library)
        else:
            plan = plan_cuda_weight_share_build(
                args.source,
                output=args.output,
                cuda_toolkit=args.cuda_toolkit,
            )
            if args.action == "plan":
                result = plan.to_dict()
                _print(result, as_json=args.json)
                print("No files or compilation outputs were changed.")
                return 0 if plan.ready else 2
            result = build_cuda_weight_share(plan, confirmed=args.yes)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _print(result, as_json=args.json)
    return 0 if bool(result.get("valid", True)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
