#!/usr/bin/env python3
"""Plan a multi-agent deployment and generate the launch command.

Usage:
    python scripts/plan_and_launch.py --model 32 --agents 8 --context 16384
    python scripts/plan_and_launch.py --model 70 --quant Q3_K_M --agents 4
    python scripts/plan_and_launch.py --scenarios  # show all possibilities
"""

import argparse
import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_turboquant.planner import plan_agents, plan_scenarios
from multi_turboquant.integration.llamacpp_args import get_llamacpp_command


def main():
    parser = argparse.ArgumentParser(
        description="Multi-TurboQuant deployment planner"
    )
    parser.add_argument("--model", type=float, default=32,
                        help="Model size in billions (e.g., 7, 13, 32, 70)")
    parser.add_argument("--quant", default="Q4_K_M",
                        help="Weight quantization (Q4_K_M, Q3_K_M, etc.)")
    parser.add_argument("--agents", type=int, default=4,
                        help="Desired number of concurrent agents")
    parser.add_argument("--context", type=int, default=8192,
                        help="Context length per agent")
    parser.add_argument("--model-path", default=None,
                        help="Path to GGUF model file")
    parser.add_argument("--gpus", type=float, nargs="+", default=[24, 12],
                        help="GPU VRAM in GB, one per GPU (e.g., --gpus 24 12 or --gpus 24)")
    parser.add_argument("--port", type=int, default=8180,
                        help="Server port")
    parser.add_argument("--scenarios", action="store_true",
                        help="Show all scenarios for this model")
    parser.add_argument("--launch", action="store_true",
                        help="Print the launch command")
    args = parser.parse_args()

    gpus = [{"name": f"GPU {i}", "vram_gb": v} for i, v in enumerate(args.gpus)]

    model_name = f"{args.model:.0f}B {args.quant}"

    if args.scenarios:
        gpu_desc = " + ".join(f"{g['vram_gb']:.0f}GB" for g in gpus)
        print(f"\n>>> ALL SCENARIOS: {model_name} on {gpu_desc} <<<\n")
        results = plan_scenarios(
            gpus=gpus,
            model_params_b=args.model,
            model_quant=args.quant,
            model_name=model_name,
        )
        print(f"{'Agents':>6} {'Context':>8} {'Preset':>28} {'KV/Agent':>10} {'Total KV':>10} {'Free':>10} {'OK':>4}")
        print("-" * 80)
        for r in results:
            ok = "Y" if r.feasible else "N"
            note = f"  {r.bottleneck}" if not r.feasible and r.bottleneck else ""
            print(f"{r.max_agents:>6} {r.actual_context:>8,} {r.preset_name:>28} "
                  f"{r.kv_per_agent_mb:>9.1f}M {r.total_kv_mb:>9.1f}M "
                  f"{r.vram_headroom_mb:>9.0f}M  {ok:>3}{note}")
        return

    # Plan specific deployment
    plan = plan_agents(
        gpus=gpus,
        model_params_b=args.model,
        model_quant=args.quant,
        desired_agents=args.agents,
        desired_context=args.context,
        model_name=model_name,
    )
    plan.print_report()

    if args.launch or args.model_path:
        if not args.model_path:
            print("Add --model-path /path/to/model.gguf to get the launch command\n")
            return

        cmd = get_llamacpp_command(
            plan.config,
            model_path=args.model_path,
            port=args.port,
            context_size=plan.actual_context * plan.max_agents,  # total context
            tensor_split=plan.tensor_split,
            parallel_slots=plan.max_agents,
        )
        print("LAUNCH COMMAND:")
        print("  " + " ".join(cmd))
        print()


if __name__ == "__main__":
    main()
