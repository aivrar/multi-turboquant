# SPDX-License-Identifier: MIT
"""CLI for composition profiles and guarded workload routing."""

from __future__ import annotations

import argparse
import json
import math
import sys

from ..hardware import detect_platform
from ..benchmark.capacity import CachePolicy, CapacityScenario, KVModelShape, simulate_capacity
from .profiles import (
    BUILTIN_EXECUTABLE_PROFILES,
    ProfileHost,
    plan_execution_profile,
)
from .routing import SUPPORTED_TASKS, WorkloadRequest, route_workload


def _host(args: argparse.Namespace) -> ProfileHost:
    detected = detect_platform()
    return ProfileHost(
        args.os or detected.os,
        args.compute or detected.primary_compute,
        args.architecture or detected.arch,
        args.gpu_memory_gb if args.gpu_memory_gb is not None else detected.available_vram_gb,
    )


def _host_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--os")
    parser.add_argument("--compute")
    parser.add_argument("--architecture")
    parser.add_argument("--gpu-memory-gb", type=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect guarded optimization profiles and route workloads"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    profiles = subparsers.add_parser("profiles", help="List reviewed execution profiles")
    profiles.add_argument("--json", action="store_true")

    plan = subparsers.add_parser("plan", help="Validate one profile without launching it")
    plan.add_argument("profile", choices=[item.id for item in BUILTIN_EXECUTABLE_PROFILES])
    _host_args(plan)
    plan.add_argument("--artifact", action="append", default=[])
    plan.add_argument("--active-feature", action="append", default=[])
    plan.add_argument("--exact-output", action="store_true")
    plan.add_argument("--json", action="store_true")

    route = subparsers.add_parser("route", help="Select a profile or return baseline fallback")
    _host_args(route)
    route.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    route.add_argument("--prompt-tokens", type=int, required=True)
    route.add_argument("--expected-output-tokens", type=int, default=0)
    route.add_argument("--repeated-prefix", action="store_true")
    route.add_argument("--exact-output", action="store_true")
    route.add_argument("--preferred-engine")
    route.add_argument("--artifact", action="append", default=[])
    route.add_argument("--trait", action="append", default=[])
    route.add_argument("--active-feature", action="append", default=[])
    route.add_argument(
        "--profile",
        action="append",
        default=[],
        choices=[item.id for item in BUILTIN_EXECUTABLE_PROFILES],
    )
    route.add_argument("--json", action="store_true")

    simulate = subparsers.add_parser(
        "simulate-capacity",
        help="Estimate KV memory and concurrency without making performance claims",
    )
    simulate.add_argument("--layers", type=int, required=True)
    simulate.add_argument("--kv-heads", type=int, required=True)
    simulate.add_argument("--head-dim", type=int, required=True)
    simulate.add_argument("--context-tokens", type=int, required=True)
    simulate.add_argument("--available-memory-gib", type=float, required=True)
    simulate.add_argument("--model-weights-gib", type=float, default=0)
    simulate.add_argument("--runtime-overhead-gib", type=float, default=0)
    simulate.add_argument("--k-bits", type=float, default=16)
    simulate.add_argument("--v-bits", type=float, default=16)
    simulate.add_argument("--retained-fraction", type=float, default=1)
    simulate.add_argument("--allocator-efficiency", type=float, default=1)
    simulate.add_argument("--metadata-bytes-per-token", type=int, default=0)
    simulate.add_argument("--fixed-overhead-bytes", type=int, default=0)
    simulate.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "profiles":
        if args.json:
            print(json.dumps([profile.to_dict() for profile in BUILTIN_EXECUTABLE_PROFILES], indent=2, sort_keys=True))
        else:
            for profile in BUILTIN_EXECUTABLE_PROFILES:
                print(f"{profile.id:24} {profile.engine:12} {profile.summary}")
        return 0

    if args.action == "simulate-capacity":
        try:
            gib = 1024 ** 3
            result = simulate_capacity(
                KVModelShape(args.layers, args.kv_heads, args.head_dim),
                CapacityScenario(
                    context_tokens=args.context_tokens,
                    available_memory_bytes=_gib_to_bytes(args.available_memory_gib, gib),
                    model_weights_bytes=_gib_to_bytes(args.model_weights_gib, gib),
                    runtime_overhead_bytes=_gib_to_bytes(args.runtime_overhead_gib, gib),
                ),
                CachePolicy(
                    k_bits=args.k_bits,
                    v_bits=args.v_bits,
                    retained_fraction=args.retained_fraction,
                    allocator_efficiency=args.allocator_efficiency,
                    per_token_metadata_bytes=args.metadata_bytes_per_token,
                    fixed_overhead_bytes=args.fixed_overhead_bytes,
                ),
            )
        except (OverflowError, TypeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"Provenance:           {result.provenance.value}")
            print(f"Baseline bytes/seq:   {result.baseline_bytes_per_sequence}")
            print(f"Candidate bytes/seq:  {result.candidate_bytes_per_sequence}")
            print(f"Baseline concurrency: {result.baseline_max_concurrency}")
            print(f"Candidate concurrency:{result.candidate_max_concurrency}")
            print("This is byte-accounting, not a measured performance result.")
        return 0

    host = _host(args)
    if args.action == "plan":
        plan = plan_execution_profile(
            args.profile,
            host,
            available_artifacts=frozenset(args.artifact),
            active_features=frozenset(args.active_feature),
            exact_output_required=args.exact_output,
        )
        if args.json:
            print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"Profile: {plan.profile.name} ({plan.profile.id})")
            print(f"Ready:   {'yes' if plan.ready else 'no'}")
            for issue in plan.issues:
                print(f"{issue.severity.upper()}: {issue.message}")
            print("No process was launched.")
        return 0 if plan.ready else 2

    try:
        request = WorkloadRequest(
            task=args.task,
            prompt_tokens=args.prompt_tokens,
            expected_output_tokens=args.expected_output_tokens,
            repeated_prefix=args.repeated_prefix,
            exact_output_required=args.exact_output,
            preferred_engine=args.preferred_engine,
            artifacts=frozenset(args.artifact),
            model_traits=frozenset(args.trait),
            active_features=frozenset(args.active_feature),
        )
        decision = route_workload(
            request,
            host,
            candidate_profile_ids=tuple(args.profile) or None,
        )
    except (KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"Route:   {decision.selected_profile or 'baseline'}")
        print(f"Reason:  {decision.reason}")
        print("No process was launched.")
    return 0


def _gib_to_bytes(value: float, gib: int) -> int:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Memory values must be finite and non-negative")
    return int(value * gib)


if __name__ == "__main__":
    raise SystemExit(main())
