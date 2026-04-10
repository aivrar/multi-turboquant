# SPDX-License-Identifier: MIT
"""GPU integration tests — real inference on real hardware.

These tests start llama-server inside WSL, run actual inference,
and verify the results. They're skipped if no GPU is available
or if the WSL distro isn't set up.

Run with: pytest tests/test_gpu.py -v
Skip with: pytest tests/ --ignore=tests/test_gpu.py
"""

import json
import os
import subprocess
import time

import pytest

# ─── GPU availability detection ────────────────────────────────────────────────

# Try Multi_TQ distro first, fall back to Llama_TQ (for dev environments
# where the Multi_TQ distro hasn't been set up yet)
_WSL_DISTRO = None

def _find_wsl_distro():
    global _WSL_DISTRO
    if _WSL_DISTRO is not None:
        return _WSL_DISTRO
    for distro in ("linbox-Multi_TQ", "linbox-Llama_TQ"):
        try:
            r = subprocess.run(
                ["wsl.exe", "-d", distro, "--", "echo", "ok"],
                capture_output=True, text=True, timeout=10,
                env={**os.environ, "MSYS_NO_PATHCONV": "1"},
            )
            if r.returncode == 0 and "ok" in r.stdout:
                _WSL_DISTRO = distro
                return distro
        except Exception:
            continue
    _WSL_DISTRO = ""
    return ""


def _has_nvidia_gpu():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0 and r.stdout.strip()
    except Exception:
        return False


def _has_wsl_distro():
    return bool(_find_wsl_distro())


def _has_llama_server():
    distro = _find_wsl_distro()
    if not distro:
        return False
    try:
        r = subprocess.run(
            ["wsl.exe", "-d", distro, "--",
             "test", "-f", "/opt/llama.cpp/build/bin/llama-server"],
            capture_output=True, timeout=10,
            env={**os.environ, "MSYS_NO_PATHCONV": "1"},
        )
        return r.returncode == 0
    except Exception:
        return False


def _has_test_model():
    distro = _find_wsl_distro()
    if not distro:
        return False
    try:
        r = subprocess.run(
            ["wsl.exe", "-d", distro, "--",
             "test", "-f", "/opt/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"],
            capture_output=True, timeout=10,
            env={**os.environ, "MSYS_NO_PATHCONV": "1"},
        )
        return r.returncode == 0
    except Exception:
        return False


HAS_GPU = _has_nvidia_gpu()
HAS_WSL = _has_wsl_distro()
HAS_SERVER = _has_llama_server() if HAS_WSL else False
HAS_MODEL = _has_test_model() if HAS_WSL else False
CAN_RUN_GPU_TESTS = HAS_GPU and HAS_WSL and HAS_SERVER and HAS_MODEL

skip_no_gpu = pytest.mark.skipif(
    not CAN_RUN_GPU_TESTS,
    reason=f"GPU tests require: GPU={HAS_GPU} WSL={HAS_WSL} server={HAS_SERVER} model={HAS_MODEL}",
)

PORT = 8191  # Use a unique port to avoid conflicts
MODEL = "/opt/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
ENV = {**os.environ, "MSYS_NO_PATHCONV": "1"}


# ─── Helpers ────────────────────────────────────────────────────────────────────

def wsl_run(cmd, timeout=10):
    """Run a command in the WSL distro."""
    distro = _find_wsl_distro()
    r = subprocess.run(
        ["wsl.exe", "-d", distro, "--", "bash", "-c", cmd],
        capture_output=True, text=True, timeout=timeout, env=ENV,
    )
    return r.stdout.strip(), r.stderr.strip(), r.returncode


def start_server(k_type="f16", v_type="f16", ctx=2048, extra=""):
    """Start llama-server and wait for it to be ready."""
    distro = _find_wsl_distro()
    # Kill anything on the port
    wsl_run(f"fuser -k {PORT}/tcp 2>/dev/null", timeout=5)
    time.sleep(1)

    cmd = (
        f"/opt/llama.cpp/build/bin/llama-server "
        f"--model {MODEL} --host 127.0.0.1 --port {PORT} "
        f"-c {ctx} -ngl 99 -fa on "
        f"--cache-type-k {k_type} --cache-type-v {v_type} {extra}"
    )

    # Start in background
    proc = subprocess.Popen(
        ["wsl.exe", "-d", distro, "--", "bash", "-c",
         f"{cmd} > /tmp/pytest_server.log 2>&1"],
        env=ENV,
    )

    # Wait for health
    import urllib.request
    for i in range(120):
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            if resp.status == 200:
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"Server crashed during startup (exit={proc.returncode})")
        time.sleep(1)

    proc.kill()
    raise RuntimeError("Server did not become ready in 120s")


def stop_server(proc):
    """Stop the server and clean up."""
    proc.kill()
    try:
        proc.wait(timeout=5)
    except Exception:
        pass
    wsl_run(f"fuser -k {PORT}/tcp 2>/dev/null", timeout=5)
    time.sleep(1)


def run_inference(prompt="Explain KV cache compression.", max_tokens=32):
    """Send an inference request and return parsed response."""
    import urllib.request
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read())


# ─── GPU Tests ──────────────────────────────────────────────────────────────────

@skip_no_gpu
class TestGPUInference:
    """Real GPU inference tests with llama-server."""

    def test_fp16_baseline(self):
        """FP16 baseline: server starts, generates tokens, reports timing."""
        proc = start_server(k_type="f16", v_type="f16")
        try:
            result = run_inference()

            assert "choices" in result
            assert len(result["choices"]) > 0
            text = result["choices"][0].get("text", "")
            assert len(text) > 0, "Should generate non-empty text"

            tokens = result.get("usage", {}).get("completion_tokens", 0)
            assert tokens > 0, "Should report completion tokens"

            timings = result.get("timings", {})
            decode_speed = timings.get("predicted_per_second", 0)
            assert decode_speed > 10, f"Decode should be >10 tok/s, got {decode_speed}"
        finally:
            stop_server(proc)

    def test_turbo4_symmetric(self):
        """turbo4 symmetric: compressed KV cache generates valid output."""
        proc = start_server(k_type="turbo4", v_type="turbo4")
        try:
            result = run_inference()

            tokens = result.get("usage", {}).get("completion_tokens", 0)
            assert tokens > 0, "turbo4 should generate tokens"

            decode_speed = result.get("timings", {}).get("predicted_per_second", 0)
            assert decode_speed > 10, f"turbo4 decode should be >10 tok/s, got {decode_speed}"
        finally:
            stop_server(proc)

    def test_turbo3_k_only(self):
        """turbo3 K-only: K compressed, V stays FP16."""
        proc = start_server(k_type="turbo3", v_type="f16")
        try:
            result = run_inference()

            tokens = result.get("usage", {}).get("completion_tokens", 0)
            assert tokens > 0, "turbo3 K-only should generate tokens"

            decode_speed = result.get("timings", {}).get("predicted_per_second", 0)
            assert decode_speed > 10, f"turbo3 K-only should be >10 tok/s, got {decode_speed}"
        finally:
            stop_server(proc)

    def test_turbo4_speed_within_range(self):
        """turbo4 decode speed should be within 50% of FP16 baseline."""
        # FP16 baseline
        proc = start_server(k_type="f16", v_type="f16")
        try:
            fp16_result = run_inference(max_tokens=64)
            fp16_speed = fp16_result.get("timings", {}).get("predicted_per_second", 0)
        finally:
            stop_server(proc)

        # turbo4
        proc = start_server(k_type="turbo4", v_type="turbo4")
        try:
            turbo4_result = run_inference(max_tokens=64)
            turbo4_speed = turbo4_result.get("timings", {}).get("predicted_per_second", 0)
        finally:
            stop_server(proc)

        assert fp16_speed > 0, "FP16 baseline should have measurable speed"
        assert turbo4_speed > 0, "turbo4 should have measurable speed"

        ratio = turbo4_speed / fp16_speed
        assert ratio > 0.5, (
            f"turbo4 ({turbo4_speed:.1f} tok/s) should be within 50% of "
            f"FP16 ({fp16_speed:.1f} tok/s), got {ratio:.1%}"
        )

    def test_multi_slot(self):
        """Multi-slot: server accepts --parallel flag and responds."""
        proc = start_server(k_type="f16", v_type="f16", extra="--parallel 2")
        try:
            result = run_inference()
            tokens = result.get("usage", {}).get("completion_tokens", 0)
            assert tokens > 0, "Multi-slot server should generate tokens"
        finally:
            stop_server(proc)


@skip_no_gpu
class TestGPUCalibration:
    """Test calibration generation on real model weights."""

    def test_generate_metadata(self):
        """Generate turboquant_kv.json from safetensors model."""
        safetensors_model = "/opt/models/Qwen2.5-7B-Instruct"

        # Check if safetensors model exists
        _, _, rc = wsl_run(f"test -d {safetensors_model}")
        if rc != 0:
            pytest.skip("Qwen2.5-7B-Instruct safetensors not available")

        # Run calibration
        script = "/mnt/e/Multi_TurboQuant/multi_turboquant/calibration/generate_metadata.py"
        stdout, stderr, rc = wsl_run(
            f"cd /mnt/e/Multi_TurboQuant && python3 -m multi_turboquant.calibration.generate_metadata "
            f"{safetensors_model} --recipe turbo3",
            timeout=120,
        )

        assert rc == 0 or "Saved" in stdout, f"Calibration failed: {stderr}"

        # Verify file was created
        stdout2, _, _ = wsl_run(f"test -f {safetensors_model}/turboquant_kv.json && echo OK")
        assert "OK" in stdout2, "turboquant_kv.json should be created"


@skip_no_gpu
class TestGPUHardwareDetection:
    """Test hardware detection against actual GPUs."""

    def test_detect_real_gpus(self):
        """Hardware detection should find the actual GPUs."""
        from multi_turboquant.hardware import detect_gpus

        gpus = detect_gpus()
        assert len(gpus) > 0, "Should detect at least one GPU"

        # Should find NVIDIA GPUs
        names = [g.name for g in gpus]
        has_nvidia = any("NVIDIA" in n or "GeForce" in n or "RTX" in n for n in names)
        assert has_nvidia, f"Should find NVIDIA GPU, got: {names}"

        # VRAM should be reasonable
        for g in gpus:
            assert g.vram_total_mb > 1000, f"GPU {g.name} VRAM too low: {g.vram_total_mb}MB"

    def test_platform_detection(self):
        """Platform detection should identify the current system."""
        from multi_turboquant.hardware import detect_platform

        plat = detect_platform()
        assert plat.os in ("windows", "linux", "darwin")
        assert plat.gpu_count > 0
        assert plat.cuda_available
        assert plat.can_run_llamacpp

    def test_planner_with_real_gpus(self):
        """Capacity planner should work with detected hardware."""
        from multi_turboquant.hardware import detect_gpus
        from multi_turboquant.planner import plan_agents

        gpus = detect_gpus()
        gpu_list = [g.to_planner_dict() for g in gpus]

        result = plan_agents(
            gpus=gpu_list,
            model_params_b=7,
            model_quant="Q4_K_M",
            desired_agents=4,
            desired_context=4096,
        )

        assert result.feasible, "7B Q4 with 4 agents should be feasible"
        assert result.max_agents >= 4
        assert result.total_kv_mb > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
