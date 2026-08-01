# SPDX-License-Identifier: MIT

from multi_turboquant.hardware import (
    GPU,
    PlatformInfo,
    _memory_from_proc_text,
    detect_system_memory,
)


def test_proc_memory_parser_reports_total_and_available_mib():
    memory = _memory_from_proc_text(
        "MemTotal:       33554432 kB\nMemFree: 1048576 kB\nMemAvailable: 8388608 kB\n"
    )

    assert memory is not None
    assert memory.total_mb == 32768
    assert memory.available_mb == 8192


def test_platform_keeps_system_ram_and_discrete_vram_separate():
    detected = PlatformInfo(
        os="linux",
        arch="x86_64",
        gpus=[GPU(0, "GPU", 24 * 1024, vram_used_mb=4 * 1024, vendor="nvidia")],
        system_memory_total_mb=64 * 1024,
        system_memory_available_mb=40 * 1024,
    )

    assert detected.system_memory_gb == 64
    assert detected.total_vram_gb == 24
    assert detected.available_vram_gb == 20
    assert detected.combined_memory_gb == 88


def test_platform_does_not_double_count_apple_unified_memory():
    detected = PlatformInfo(
        os="darwin",
        arch="arm64",
        gpus=[GPU(0, "Apple GPU", 48 * 1024, vendor="apple", compute="metal")],
        metal_available=True,
        system_memory_total_mb=64 * 1024,
    )

    assert detected.unified_memory is True
    assert detected.combined_memory_gb == 64


def test_platform_adds_discrete_vram_to_apple_unified_capacity():
    detected = PlatformInfo(
        os="darwin",
        arch="arm64",
        gpus=[
            GPU(0, "Apple GPU", 48 * 1024, vendor="apple", compute="metal"),
            GPU(1, "External GPU", 16 * 1024, vendor="amd", compute="rocm"),
        ],
        metal_available=True,
        system_memory_total_mb=64 * 1024,
    )

    assert detected.unified_memory is True
    assert detected.combined_memory_gb == 80


def test_system_memory_detection_is_bounded():
    memory = detect_system_memory()

    assert memory.total_mb >= 0
    assert 0 <= memory.available_mb <= memory.total_mb
