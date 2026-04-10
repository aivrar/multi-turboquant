# SPDX-License-Identifier: MIT
"""Hardware detection — GPUs, platform, capabilities.

Auto-detects NVIDIA, AMD (ROCm), and Apple Metal GPUs.
Works on Windows, Linux, and macOS without any dependencies beyond stdlib.

Usage:
    from multi_turboquant.hardware import detect_gpus, detect_platform

    gpus = detect_gpus()
    platform = detect_platform()
    print(platform.summary())
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class GPU:
    """Detected GPU device."""
    index: int
    name: str
    vram_total_mb: int
    vram_used_mb: int = 0
    temp_c: int = 0
    vendor: str = "unknown"      # "nvidia", "amd", "apple"
    compute: str = ""            # "cuda", "rocm", "metal"
    driver: str = ""

    @property
    def vram_gb(self) -> float:
        return self.vram_total_mb / 1024

    @property
    def vram_free_mb(self) -> int:
        return self.vram_total_mb - self.vram_used_mb

    def to_planner_dict(self) -> dict:
        """Format for the capacity planner."""
        return {"name": self.name, "vram_gb": self.vram_gb}


@dataclass
class PlatformInfo:
    """Detected platform and capabilities."""
    os: str                          # "windows", "linux", "darwin"
    arch: str                        # "x86_64", "arm64"
    in_wsl: bool = False             # running inside WSL2
    has_wsl: bool = False            # WSL2 available (Windows host)
    gpus: list[GPU] = field(default_factory=list)
    cuda_available: bool = False
    rocm_available: bool = False
    metal_available: bool = False
    python_version: str = ""

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def total_vram_gb(self) -> float:
        return sum(g.vram_gb for g in self.gpus)

    @property
    def primary_compute(self) -> str:
        """The primary compute backend available."""
        if self.cuda_available:
            return "cuda"
        if self.rocm_available:
            return "rocm"
        if self.metal_available:
            return "metal"
        return "cpu"

    @property
    def can_run_inference(self) -> bool:
        return len(self.gpus) > 0

    @property
    def can_run_triton(self) -> bool:
        return self.cuda_available and self.os == "linux"

    @property
    def can_run_vllm(self) -> bool:
        return self.cuda_available and self.os == "linux"

    @property
    def can_run_llamacpp(self) -> bool:
        return len(self.gpus) > 0  # llama.cpp supports CUDA, ROCm, Metal

    def gpu_planner_list(self) -> list[dict]:
        """Format all GPUs for the capacity planner."""
        return [g.to_planner_dict() for g in self.gpus]

    @property
    def cuda_device_order_warning(self) -> str | None:
        """Warn if nvidia-smi and CUDA have different GPU ordering.

        This happens when the PCI bus order doesn't match CUDA's
        enumeration. The fix is to set CUDA_VISIBLE_DEVICES.
        """
        nvidia_gpus = [g for g in self.gpus if g.vendor == "nvidia"]
        if len(nvidia_gpus) <= 1:
            return None
        # If the first GPU (by nvidia-smi index) isn't the largest,
        # CUDA might put a different GPU as device 0
        largest = max(nvidia_gpus, key=lambda g: g.vram_total_mb)
        if nvidia_gpus[0].vram_total_mb != largest.vram_total_mb:
            fix = get_cuda_visible_devices_for_primary(nvidia_gpus)
            return (
                f"GPU ordering mismatch: nvidia-smi shows {nvidia_gpus[0].name} as index 0, "
                f"but {largest.name} has more VRAM. "
                f"Set CUDA_VISIBLE_DEVICES={fix} to force largest GPU as primary."
            )
        return None

    def summary(self) -> str:
        lines = [
            f"Platform: {self.os} {self.arch}",
            f"  WSL: {'in WSL' if self.in_wsl else 'available' if self.has_wsl else 'no'}",
            f"  GPUs: {self.gpu_count} ({self.total_vram_gb:.1f} GB total)",
        ]
        for g in self.gpus:
            lines.append(
                f"    [{g.index}] {g.name} — {g.vram_total_mb} MB "
                f"({g.vendor}/{g.compute})"
            )
        warning = self.cuda_device_order_warning
        if warning:
            lines.append(f"  WARNING: {warning}")
        lines.append(f"  Compute: {self.primary_compute}")
        lines.append(f"  Triton: {'yes' if self.can_run_triton else 'no'}")
        lines.append(f"  vLLM: {'yes' if self.can_run_vllm else 'no'}")
        lines.append(f"  llama.cpp: {'yes' if self.can_run_llamacpp else 'no (no GPU)'}")
        return "\n".join(lines)


# ─── Detection functions ────────────────────────────────────────────────────────

def get_cuda_visible_devices_for_primary(gpus: list["GPU"]) -> str:
    """Generate CUDA_VISIBLE_DEVICES to ensure the largest GPU is primary.

    nvidia-smi and CUDA often enumerate GPUs in different orders.
    nvidia-smi uses PCI bus order; CUDA uses its own order based on
    compute capability and PCI topology. This can cause the wrong GPU
    to be used as the default device.

    This function returns a CUDA_VISIBLE_DEVICES string that forces
    the largest-VRAM GPU to be CUDA device 0.

    Usage:
        os.environ["CUDA_VISIBLE_DEVICES"] = get_cuda_visible_devices_for_primary(gpus)
    """
    if not gpus or len(gpus) <= 1:
        return ""
    # Sort by VRAM descending, return all indices with biggest first
    sorted_gpus = sorted(gpus, key=lambda g: g.vram_total_mb, reverse=True)
    return ",".join(str(g.index) for g in sorted_gpus)


def _detect_nvidia() -> list[GPU]:
    """Detect NVIDIA GPUs via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,temperature.gpu,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(GPU(
                    index=int(parts[0]),
                    name=parts[1],
                    vram_total_mb=int(parts[2]),
                    vram_used_mb=int(parts[3]),
                    temp_c=int(parts[4]) if parts[4].isdigit() else 0,
                    vendor="nvidia",
                    compute="cuda",
                    driver=parts[5] if len(parts) > 5 else "",
                ))
        return gpus
    except Exception:
        return []


def _detect_amd() -> list[GPU]:
    """Detect AMD GPUs via rocm-smi."""
    if not shutil.which("rocm-smi"):
        return []
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid", "--showmeminfo", "vram", "--csv"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        # Parse rocm-smi CSV output
        gpus = []
        lines = result.stdout.strip().split("\n")
        for i, line in enumerate(lines):
            if line.startswith("device"):
                continue  # header
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    gpus.append(GPU(
                        index=i,
                        name=f"AMD GPU {parts[0]}",
                        vram_total_mb=int(parts[1]) // (1024 * 1024) if parts[1].isdigit() else 0,
                        vram_used_mb=int(parts[2]) // (1024 * 1024) if parts[2].isdigit() else 0,
                        vendor="amd",
                        compute="rocm",
                    ))
                except (ValueError, IndexError):
                    pass
        return gpus
    except Exception:
        return []


def _detect_apple_metal() -> list[GPU]:
    """Detect Apple Silicon GPUs via system_profiler."""
    if platform.system() != "Darwin":
        return []
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        import json
        data = json.loads(result.stdout)
        displays = data.get("SPDisplaysDataType", [])
        gpus = []
        for i, display in enumerate(displays):
            name = display.get("sppci_model", "Apple GPU")
            # Apple reports VRAM in various formats
            vram_str = display.get("spdisplays_vram", "0")
            vram_mb = 0
            if isinstance(vram_str, str):
                # Parse "16 GB" or "16384 MB"
                parts = vram_str.split()
                if len(parts) >= 2:
                    try:
                        val = int(parts[0])
                        if "GB" in parts[1].upper():
                            vram_mb = val * 1024
                        else:
                            vram_mb = val
                    except ValueError:
                        pass
            # Apple Silicon unified memory — estimate GPU share
            if vram_mb == 0:
                try:
                    total_mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                    # GPU typically gets ~75% of unified memory
                    vram_mb = int(total_mem * 0.75 / (1024 * 1024))
                except (ValueError, AttributeError):
                    vram_mb = 8192  # conservative default

            gpus.append(GPU(
                index=i,
                name=name,
                vram_total_mb=vram_mb,
                vendor="apple",
                compute="metal",
            ))
        return gpus
    except Exception:
        return []


def _check_wsl() -> tuple[bool, bool]:
    """Check WSL status.

    Returns:
        (in_wsl, has_wsl) — whether we're inside WSL, whether WSL is available.
    """
    in_wsl = False
    has_wsl = False

    # Check if running inside WSL
    try:
        with open("/proc/version") as f:
            if "microsoft" in f.read().lower():
                in_wsl = True
    except (OSError, IOError):
        pass

    # Check if WSL is available on Windows host
    if platform.system() == "Windows" and shutil.which("wsl"):
        try:
            result = subprocess.run(
                ["wsl", "--status"], capture_output=True, text=True, timeout=5,
            )
            has_wsl = result.returncode == 0
        except Exception:
            has_wsl = shutil.which("wsl") is not None

    return in_wsl, has_wsl


def detect_gpus() -> list[GPU]:
    """Detect all available GPUs across all vendors.

    Tries NVIDIA first, then AMD, then Apple Metal.
    Returns a unified list sorted by VRAM (largest first).
    """
    gpus = []
    gpus.extend(_detect_nvidia())
    gpus.extend(_detect_amd())
    gpus.extend(_detect_apple_metal())

    # Re-index if multiple vendors
    for i, g in enumerate(gpus):
        g.index = i

    return sorted(gpus, key=lambda g: g.vram_total_mb, reverse=True)


def detect_platform() -> PlatformInfo:
    """Detect the full platform: OS, GPUs, capabilities."""
    os_name = platform.system().lower()
    if os_name == "windows":
        os_name = "windows"
    elif os_name == "darwin":
        os_name = "darwin"
    else:
        os_name = "linux"

    arch = platform.machine().lower()
    if arch in ("amd64", "x86_64"):
        arch = "x86_64"
    elif arch in ("arm64", "aarch64"):
        arch = "arm64"

    in_wsl, has_wsl = _check_wsl()
    gpus = detect_gpus()

    cuda = any(g.compute == "cuda" for g in gpus)
    rocm = any(g.compute == "rocm" for g in gpus)
    metal = any(g.compute == "metal" for g in gpus)

    import sys
    python_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    return PlatformInfo(
        os=os_name,
        arch=arch,
        in_wsl=in_wsl,
        has_wsl=has_wsl,
        gpus=gpus,
        cuda_available=cuda,
        rocm_available=rocm,
        metal_available=metal,
        python_version=python_ver,
    )


def get_recommended_engine(plat: PlatformInfo) -> str:
    """Recommend the best inference engine for this platform.

    Returns:
        "llamacpp", "vllm", or "none".
    """
    if plat.can_run_vllm:
        return "vllm"  # Best throughput on Linux + CUDA
    if plat.can_run_llamacpp:
        return "llamacpp"  # Works everywhere with a GPU
    return "none"


def get_build_instructions(plat: PlatformInfo) -> str:
    """Generate platform-specific build instructions for llama.cpp."""
    if plat.cuda_available:
        if plat.os == "linux" or plat.in_wsl:
            return (
                "cd /opt && git clone https://github.com/ggml-org/llama.cpp llama.cpp\n"
                "cd llama.cpp && cmake -B build -G Ninja "
                "-DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON "
                "-DCMAKE_CUDA_ARCHITECTURES=native\n"
                "cmake --build build --config Release -j$(nproc)"
            )
        elif plat.os == "windows":
            return (
                "# Run inside WSL2:\n"
                "wsl -d linbox-Multi_TQ\n"
                "# Then same as Linux build above"
            )
    if plat.rocm_available:
        return (
            "cd /opt && git clone https://github.com/ggml-org/llama.cpp llama.cpp\n"
            "cd llama.cpp && cmake -B build -G Ninja "
            "-DGGML_HIP=ON -DGGML_CUDA_FA=ON\n"
            "cmake --build build --config Release -j$(nproc)"
        )
    if plat.metal_available:
        return (
            "cd /opt && git clone https://github.com/ggml-org/llama.cpp llama.cpp\n"
            "cd llama.cpp && cmake -B build "
            "-DGGML_METAL=ON -DGGML_CUDA_FA_ALL_QUANTS=ON\n"
            "cmake --build build --config Release -j$(sysctl -n hw.ncpu)"
        )
    return "# No GPU detected — CPU-only build:\n" \
           "cd /opt && git clone https://github.com/ggml-org/llama.cpp llama.cpp\n" \
           "cd llama.cpp && cmake -B build && cmake --build build -j$(nproc)"
