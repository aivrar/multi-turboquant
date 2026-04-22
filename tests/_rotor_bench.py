"""Side-by-side GPU benchmark: iso vs planar vs rotor.

Prints a table of cosine similarity, packed bytes/head, compression ratio,
and encode/decode latency for each method at multiple shapes on CUDA.

Usage:
    PYTHONPATH=. python tests/_rotor_bench.py
"""
from __future__ import annotations

import time
import warnings

import torch

from multi_turboquant import CacheMethod, get_method


def bench(method: CacheMethod, x: torch.Tensor, iters: int = 20):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = get_method(method)
    for _ in range(3):
        c = m.encode(x)
        _ = m.decode(c, dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = m.encode(x)
    torch.cuda.synchronize()
    enc_ms = (time.perf_counter() - t0) * 1000 / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        d = m.decode(c, dtype=torch.float16)
    torch.cuda.synchronize()
    dec_ms = (time.perf_counter() - t0) * 1000 / iters
    cos = torch.nn.functional.cosine_similarity(
        x.float().reshape(-1, x.shape[-1]),
        d.float().reshape(-1, x.shape[-1]),
        dim=-1,
    ).mean().item()
    bytes_per_head = c.data.shape[-1]
    fp16_bytes = x.shape[-1] * 2
    ratio = fp16_bytes / bytes_per_head
    return cos, bytes_per_head, ratio, enc_ms, dec_ms


def main() -> None:
    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda:0")
    print(f"# GPU: {torch.cuda.get_device_name(0)}  torch: {torch.__version__}")
    print()

    shapes = [(64, 8, 128), (256, 8, 128), (64, 8, 64)]
    methods = [
        CacheMethod.ISO3, CacheMethod.PLANAR3, CacheMethod.ROTOR3,
        CacheMethod.ISO4, CacheMethod.PLANAR4, CacheMethod.ROTOR4,
    ]

    torch.manual_seed(42)
    for shape in shapes:
        x = torch.randn(*shape, device=dev, dtype=torch.float16)
        print(f"## shape {shape} (fp16)")
        header = f"{'method':<10}{'cos':>8}{'bytes':>8}{'ratio':>8}{'enc_ms':>10}{'dec_ms':>10}"
        print(header)
        print("-" * len(header))
        for mt in methods:
            cos, bph, r, em, dm = bench(mt, x)
            print(f"{mt.value:<10}{cos:>8.4f}{bph:>8}{r:>8.2f}{em:>10.2f}{dm:>10.2f}")
        print()


if __name__ == "__main__":
    main()
