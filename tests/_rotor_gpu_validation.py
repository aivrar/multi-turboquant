"""Standalone GPU validation for RotorQuant.

Runs the key correctness checks on a real CUDA device without requiring
pytest, so it can execute inside any torch-enabled Python interpreter.

Usage:
    python tests/_rotor_gpu_validation.py
"""
from __future__ import annotations

import math
import sys
import warnings

import torch

from multi_turboquant import CacheConfig, CacheMethod, compress, decompress, get_method
from multi_turboquant.methods import rotorquant as _rq
from multi_turboquant.methods.rotorquant import (
    _apply_rotor_sandwich,
    _rotation_matrix_3d,
)


def _banner(s: str) -> None:
    print(f"\n=== {s} ===")


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        print("FATAL: torch.cuda.is_available() is False — this script requires a real GPU.")
        sys.exit(2)
    dev = torch.device("cuda:0")
    print(f"device: {dev}  |  name: {torch.cuda.get_device_name(0)}  "
          f"|  torch: {torch.__version__}")
    return dev


def check_rotation_properties(dev: torch.device) -> None:
    _banner("3x3 rotation matrix properties (GPU)")
    R = _rotation_matrix_3d(dev.type, dev.index)
    assert R.device.type == "cuda", f"rotation matrix landed on {R.device}, not CUDA"
    I = torch.eye(3, device=dev)
    ortho = torch.allclose(R.T @ R, I, atol=1e-5)
    det = torch.linalg.det(R).item()
    print(f"  orthogonal (R^T R = I): {ortho}")
    print(f"  det(R) = {det:.6f} (should be +1 for SO(3))")
    assert ortho, "rotation matrix is not orthogonal on GPU"
    assert abs(det - 1.0) < 1e-5, f"det(R)={det} not ~+1 — not a proper rotation"


def check_sandwich_invertibility(dev: torch.device) -> None:
    _banner("Rotor-sandwich invertibility (GPU)")
    x = torch.randn(2, 3, 129, device=dev, dtype=torch.float32)
    r = _apply_rotor_sandwich(x)
    x_hat = _apply_rotor_sandwich(r, inverse=True)
    max_err = (x - x_hat).abs().max().item()
    print(f"  max |x - R^T R x| = {max_err:.2e}")
    assert max_err < 1e-4, f"sandwich inverse not recovering input: {max_err}"


def check_roundtrip(method: CacheMethod, dev: torch.device, threshold: float) -> float:
    _banner(f"{method.value} round-trip quality (GPU, head_dim=128)")
    torch.manual_seed(1234)
    x = torch.randn(64, 8, 128, device=dev, dtype=torch.float16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = get_method(method)
    compressed = m.encode(x)
    assert compressed.data.device.type == "cuda", \
        f"encode output on {compressed.data.device}, expected cuda"
    decoded = m.decode(compressed, dtype=torch.float16)
    assert decoded.device.type == "cuda"
    assert decoded.shape == x.shape
    cos = torch.nn.functional.cosine_similarity(
        x.float().reshape(-1, 128), decoded.float().reshape(-1, 128), dim=-1
    ).mean().item()
    packed_bytes = compressed.data.shape[-1]
    compression = (128 * 2) / packed_bytes  # fp16 raw bytes / packed bytes
    print(f"  cosine similarity = {cos:.4f}   (threshold {threshold})")
    print(f"  packed bytes/head = {packed_bytes}   compression = {compression:.2f}x")
    assert cos > threshold, f"{method.value} cos similarity {cos} below {threshold}"
    return cos


def check_padding_head_dim_64(dev: torch.device) -> None:
    _banner("head_dim=64 padding round-trip (GPU)")
    torch.manual_seed(99)
    x = torch.randn(16, 4, 64, device=dev, dtype=torch.float16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = get_method(CacheMethod.ROTOR3)
    packed = m.encode(x)
    decoded = m.decode(packed, dtype=torch.float16)
    assert decoded.shape == x.shape
    cos = torch.nn.functional.cosine_similarity(
        x.float().reshape(-1, 64), decoded.float().reshape(-1, 64), dim=-1
    ).mean().item()
    print(f"  cosine similarity = {cos:.4f}")
    assert cos > 0.95, f"head_dim=64 cos similarity {cos} too low"


def check_cpu_gpu_agreement() -> None:
    """CPU and GPU must produce identical results for a fixed rotation."""
    _banner("CPU vs GPU numerical agreement")
    torch.manual_seed(7)
    x_cpu = torch.randn(4, 2, 128, dtype=torch.float32)
    x_gpu = x_cpu.to("cuda:0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = get_method(CacheMethod.ROTOR3)
    dec_cpu = m.decode(m.encode(x_cpu), dtype=torch.float32)
    dec_gpu = m.decode(m.encode(x_gpu), dtype=torch.float32).cpu()
    diff = (dec_cpu - dec_gpu).abs().max().item()
    print(f"  max |cpu - gpu| = {diff:.2e}")
    assert diff < 1e-3, f"CPU/GPU divergence too large: {diff}"


def check_end_to_end_dispatch(dev: torch.device) -> None:
    _banner("End-to-end compress()/decompress() dispatch on GPU")
    x = torch.randn(4, 8, 128, device=dev, dtype=torch.float16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cfg = CacheConfig(k_method=CacheMethod.ROTOR3, v_method=CacheMethod.FP16)
        c = compress(x, cfg, which="k")
    print(f"  method={c.method.value}  device={c.data.device}  shape={tuple(c.data.shape)}")
    assert c.method == CacheMethod.ROTOR3
    assert c.data.device.type == "cuda"


def main() -> None:
    dev = require_cuda()
    check_rotation_properties(dev)
    check_sandwich_invertibility(dev)
    rotor3_cos = check_roundtrip(CacheMethod.ROTOR3, dev, threshold=0.97)
    rotor4_cos = check_roundtrip(CacheMethod.ROTOR4, dev, threshold=0.99)
    check_padding_head_dim_64(dev)
    check_cpu_gpu_agreement()
    check_end_to_end_dispatch(dev)

    print("\n=== SUMMARY ===")
    print(f"  rotor3 GPU cosine similarity: {rotor3_cos:.4f}")
    print(f"  rotor4 GPU cosine similarity: {rotor4_cos:.4f}")
    print("  ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
