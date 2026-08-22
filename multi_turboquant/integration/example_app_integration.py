"""Example: How to integrate multi_turboquant into an existing app.

This shows the exact pattern for using multi_turboquant inside
the Llama_TQ app (or any app that manages llama.cpp / vLLM services).

Drop-in replacement for hardcoded cache type flags with config-driven
method selection, capacity planning, and auto-calibration.
"""


# ─── BEFORE (hardcoded in Llama_TQ) ────────────────────────────────────────────
#
# cmd = (
#     "/opt/llama.cpp/build/bin/llama-server "
#     f"--model {model_path} --host 0.0.0.0 --port 8080 "
#     "-ngl auto -fa on --cache-type-k turbo3_tcq --cache-type-v turbo3_tcq"
# )


# ─── AFTER (with multi_turboquant) ─────────────────────────────────────────────

def build_service_command(
    model_path: str,
    port: int = 8080,
    *,
    # User picks a preset from the UI dropdown, or the planner recommends one
    preset: str = "balanced",
    # Or explicit method selection
    k_method: str | None = None,
    v_method: str | None = None,
    # Auto-detected or from UI
    gpu_vram: list[float] | None = None,
    model_params_b: float | None = None,
    desired_agents: int = 1,
    desired_context: int = 4096,
) -> str:
    """Build the llama-server launch command using multi_turboquant.

    This replaces the hardcoded command construction in bridge_services.py.
    """
    from multi_turboquant import CacheConfig, CacheMethod, get_preset
    from multi_turboquant.integration import get_llamacpp_command
    from multi_turboquant.hardware import detect_gpus

    # Method 1: Use a named preset
    if k_method is None:
        config = get_preset(preset)
    else:
        # Method 2: Explicit method selection from UI
        config = CacheConfig(
            k_method=CacheMethod(k_method),
            v_method=CacheMethod(v_method or k_method),
        )

    # Auto-detect GPUs if not provided
    if gpu_vram is None:
        gpus = detect_gpus()
        gpu_vram = [g.vram_total_mb / 1024 for g in gpus]

    # Calculate tensor split and parallel slots if planning
    tensor_split = None
    parallel_slots = desired_agents if desired_agents > 1 else None

    if len(gpu_vram) > 1:
        tensor_split = ",".join(f"{v:.0f}" for v in gpu_vram)

    # Generate the command
    cmd = get_llamacpp_command(
        config,
        model_path=model_path,
        port=port,
        context_size=desired_context * desired_agents,
        tensor_split=tensor_split,
        parallel_slots=parallel_slots,
    )

    return " ".join(cmd)


def get_ui_cache_options() -> list[dict]:
    """Get cache type options for a UI dropdown.

    Returns a list suitable for populating a <select> element.
    """
    from multi_turboquant import list_methods
    from multi_turboquant.config import METHOD_BITS, CALIBRATION_FREE

    options = [{"value": "f16", "label": "FP16 (no compression)", "group": "Baseline"}]

    for info in list_methods():
        needs_cal = "Yes" if info.requires_calibration else "No"
        label = (
            f"{info.method.value} — {info.bits:.1f}-bit, "
            f"{16.0/info.bits:.1f}x compression"
        )
        if not info.requires_calibration:
            label += " (no calibration)"

        options.append({
            "value": info.method.value,
            "label": label,
            "group": info.family.value.title(),
            "bits": info.bits,
            "compression": round(16.0 / info.bits, 1),
            "needs_calibration": info.requires_calibration,
        })

    return options


def plan_for_ui(
    model_params_b: float,
    model_quant: str,
    desired_agents: int,
    desired_context: int,
) -> dict:
    """Run the capacity planner and return results for the UI.

    Call this when the user changes model/agents/context in the config panel.
    """
    from multi_turboquant import plan_agents
    from multi_turboquant.hardware import detect_gpus

    gpus = detect_gpus()
    if not gpus:
        return {"error": "No GPUs detected", "feasible": False}

    result = plan_agents(
        gpus=[g.to_planner_dict() for g in gpus],
        model_params_b=model_params_b,
        model_quant=model_quant,
        desired_agents=desired_agents,
        desired_context=desired_context,
    )

    return result.to_dict()


def auto_calibrate_if_needed(model_path: str, preset: str = "balanced") -> dict:
    """Check if calibration is needed and generate it.

    Call this before starting a service with a TurboQuant preset.
    """
    from multi_turboquant import get_preset
    from multi_turboquant.calibration import auto_calibrate

    config = get_preset(preset)
    if not config.needs_calibration:
        return {"status": "not_needed", "message": "Selected method is calibration-free"}

    results = auto_calibrate(config, model_path, verbose=True)
    return {"status": "ok", "files": results}


# ─── Usage in bridge_services.py ───────────────────────────────────────────────

def example_bridge_integration():
    """Shows how bridge_services.py would use this."""

    # 1. User opens config panel → populate cache type dropdown
    options = get_ui_cache_options()
    # Send to UI: handler._json(options)

    # 2. User adjusts agents/context slider → update capacity display
    plan = plan_for_ui(
        model_params_b=32,
        model_quant="Q4_K_M",
        desired_agents=8,
        desired_context=16384,
    )
    # Send to UI: handler._json(plan)
    # UI shows: "8 agents at 16K each — turbo4, 8.5 GB KV, 9 GB free"

    # 3. User clicks Start → build command and launch
    cmd = build_service_command(
        model_path="/opt/models/qwen2.5-32b-q4_k_m.gguf",
        port=8180,
        preset="agents_8x16k",
        desired_agents=8,
        desired_context=16384,
    )
    print(f"Launch: {cmd}")
    # bridge_services would then run: wsl_exec(distro, cmd)


if __name__ == "__main__":
    example_bridge_integration()
