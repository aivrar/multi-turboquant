# SPDX-License-Identifier: MIT
"""CLI for inspecting optional optimization eligibility."""

from __future__ import annotations

import argparse
import json

from .catalog import create_builtin_registry
from .planner import detect_optimization_context, plan_optimizations


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect optional inference optimizations")
    parser.add_argument("--engine", default="vllm")
    parser.add_argument("--kv-format", default="fp16")
    parser.add_argument("--select", action="append", default=[])
    parser.add_argument("--active-feature", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    context = detect_optimization_context(
        engine=args.engine,
        kv_format=args.kv_format,
        active_features=tuple(args.active_feature),
    )
    registry = create_builtin_registry()
    selected = args.select or [plugin.descriptor.id for plugin in registry.list()]
    plan = plan_optimizations(selected, context, registry=registry)

    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return
    for probe in plan.probes:
        state = "ready" if probe.eligible else "unavailable"
        print(f"{probe.descriptor.id:22} {state:11} {probe.descriptor.summary}")
        for issue in probe.issues:
            print(f"  {issue.severity.upper()}: {issue.message}")
    for issue in plan.issues:
        if issue not in tuple(item for probe in plan.probes for item in probe.issues):
            print(f"{issue.severity.upper()}: {issue.message}")


if __name__ == "__main__":
    main()
