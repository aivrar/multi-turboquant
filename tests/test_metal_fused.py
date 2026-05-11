"""Tests for Metal fused attention kernels. Skip on non-Apple."""
import pytest
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")

def test_fused_qk():
    from multi_turboquant.kernels.metal.fused_attention import (
        planar_fused_qk_scores, _compress, _decompress, _planar_rotate, _planar_unrotate, _CODEBOOKS)
    import math
    mx.random.seed(42); B,H,T,D,bits = 1,2,50,64,3
    k = mx.random.normal((B*H*T, D)).astype(mx.float32)
    kp, kn = _compress(k, bits, _planar_rotate)
    kd = _decompress(kp, kn, D, bits, _planar_unrotate, mx.float32).reshape(B,H,T,D)
    q = mx.random.normal((B,H,1,D)).astype(mx.float16)
    scale = 1.0/math.sqrt(D)
    ref = (q.astype(mx.float32) @ kd.swapaxes(-1,-2)) * scale
    fused = planar_fused_qk_scores(q, kp.reshape(B,H,T,-1), kn.reshape(B,H,T),
                                    mx.array(_CODEBOOKS[bits], dtype=mx.float32), scale, D, bits)
    mx.eval(ref, fused)
    assert mx.max(mx.abs(fused - ref)).item() < 0.001

def test_sparse_sv():
    from multi_turboquant.kernels.metal.fused_attention import (
        planar_sparse_sv_values, _compress, _planar_rotate, _CODEBOOKS)
    mx.random.seed(0); B,H,T,D,bits = 1,2,100,64,3
    v = mx.random.normal((B*H*T, D)).astype(mx.float32)
    vp, vn = _compress(v, bits, _planar_rotate)
    probs = mx.softmax(mx.random.normal((B,H,1,T)), axis=-1)
    out = planar_sparse_sv_values(probs, vp.reshape(B,H,T,-1), vn.reshape(B,H,T),
                                   mx.array(_CODEBOOKS[bits], dtype=mx.float32), D, bits)
    mx.eval(out)
    assert out.shape == (B,H,1,D) and not mx.any(mx.isnan(out)).item()
