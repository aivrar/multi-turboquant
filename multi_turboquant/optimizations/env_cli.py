# SPDX-License-Identifier: MIT
"""Command-line interface for opt-in optimization environments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .environments import (
    BUILTIN_ENVIRONMENT_PROFILES,
    DEFAULT_ENVIRONMENT_ROOT,
    check_environment,
    plan_environment,
    run_in_environment,
    synchronize_environment,
)


def _add_common_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("profile", choices=[item.id for item in BUILTIN_ENVIRONMENT_PROFILES])
    parser.add_argument("--root", type=Path, default=DEFAULT_ENVIRONMENT_ROOT)
    parser.add_argument(
        "--python",
        help="Python version or interpreter path; pyenv interpreter paths are accepted",
    )


def _print_plan(plan, *, as_json: bool = False, read_only: bool = True) -> None:
    if as_json:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return
    print(f"Profile: {plan.profile.name} ({plan.profile.id})")
    print(f"Status:  {'installable' if plan.profile.installable else 'blocked'}")
    print(f"Target:  {plan.target}")
    print(f"Python:  {plan.python_request} ({plan.profile.python_spec})")
    if plan.profile.packages:
        print("Packages:")
        for package in plan.profile.packages:
            print(f"  - {package}")
    for issue in plan.issues:
        print(f"{issue.severity.upper()}: {issue.message}")
    if read_only:
        print("No files or packages were changed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan and create isolated environments for optional optimizations"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    list_parser = subparsers.add_parser("list", help="List available dependency profiles")
    list_parser.add_argument("--json", action="store_true")

    plan_parser = subparsers.add_parser("plan", help="Inspect an environment without changing it")
    _add_common_profile_arguments(plan_parser)
    plan_parser.add_argument("--json", action="store_true")

    create_parser = subparsers.add_parser("create", help="Create and lock an isolated environment")
    _add_common_profile_arguments(create_parser)
    create_parser.add_argument("--yes", action="store_true", help="Confirm downloads/builds")
    create_parser.add_argument("--upgrade", action="store_true", help="Refresh locked versions")
    create_parser.add_argument("--no-check", action="store_true", help="Skip post-install imports")

    check_parser = subparsers.add_parser("check", help="Validate an existing environment")
    _add_common_profile_arguments(check_parser)
    check_parser.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run a command in an existing environment")
    _add_common_profile_arguments(run_parser)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "list":
        profiles = [profile.to_dict() for profile in BUILTIN_ENVIRONMENT_PROFILES]
        if args.json:
            print(json.dumps(profiles, indent=2, sort_keys=True))
        else:
            for profile in BUILTIN_ENVIRONMENT_PROFILES:
                status = "installable" if profile.installable else "blocked"
                print(f"{profile.id:22} {status:11} {profile.name}")
        return 0

    plan = plan_environment(args.profile, root=args.root, python=args.python)
    if args.action == "plan":
        _print_plan(plan, as_json=args.json)
        return 0 if plan.ready else 2
    if args.action == "create":
        _print_plan(plan, read_only=False)
        if not plan.ready:
            return 2
        if not args.yes:
            print("Creation not confirmed; rerun with --yes after reviewing the plan.")
            return 2
        try:
            synchronize_environment(plan, upgrade=args.upgrade)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"Created locked environment at {plan.target}")
        if not args.no_check:
            report = check_environment(plan)
            print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.action == "check":
        try:
            report = check_environment(plan)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            for name, version in sorted(report.items()):
                print(f"{name:18} {version}")
        return 0
    if args.action == "run":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            print("ERROR: provide a command after --", file=sys.stderr)
            return 2
        try:
            return run_in_environment(plan, command)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
