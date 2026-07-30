"""
test_compiler.py
===============
Core pytest suite for metatron_compiler.py.

Covers the contract fixes (IMPLEMENTATION_CONTRACT.md §1):
  * morton_reorder(n) is a bijection (permutation of 0..n-1) AND matches its
    docstring example (seq_len=8 -> [0,1,4,5,2,3,6,7]).
  * compile_to_csr returns indices sorted within each row (CSR invariant).
  * int8 quantization of a zero tensor is finite (zero-scale guard) and
    round-trips exactly.
  * bandwidth_estimate uses 4 transfers (Q, K, V, O), not 3, and accepts d_k.
"""

from __future__ import annotations

import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from config import CONFIG
from metatron_compiler import (
    TokenReorder,
    compile_to_csr,
    compile_to_coo,
    SparseKernelCompiler,
    Quantizer,
)
from metatron_sparse_attention import MetatronSparseMask


# ── morton_reorder is a bijection ────────────────────────────────────────────

def test_morton_reorder_is_bijection():
    """morton_reorder(n) must be a permutation of 0..n-1 for various n."""
    for n in (1, 2, 7, 8, 16, 31, 64, 127, 128):
        order = TokenReorder.morton_reorder(n)
        assert order.dtype == torch.long
        assert order.numel() == n
        assert sorted(order.tolist()) == list(range(n)), (
            f"n={n}: morton_reorder is not a bijection: {order.tolist()}"
        )


def test_morton_reorder_matches_docstring_example():
    """The docstring example: seq_len=8 -> [0,1,4,5,2,3,6,7]."""
    order = TokenReorder.morton_reorder(8)
    assert order.tolist() == [0, 1, 4, 5, 2, 3, 6, 7], (
        f"docstring example mismatch: {order.tolist()}"
    )


def test_morton_reorder_reverse_inverse():
    """reverse_reorder is the inverse of morton_reorder."""
    n = 64
    fwd = TokenReorder.morton_reorder(n)
    inv = TokenReorder.reverse_reorder(fwd)
    # inv[fwd[i]] == i, so fwd[inv] should be identity.
    identity = fwd[inv]
    assert identity.tolist() == list(range(n))


# ── polyhedral_ring_order is a bijection (real BFS) ─────────────────────────

def test_polyhedral_ring_order_is_bijection():
    """polyhedral_ring_order must be a bijection over 0..n-1 for every
    supported solid."""
    from metatron_sparse_attention import POLYHEDRA
    for solid in CONFIG['supported_solids']:
        n = 32
        order = TokenReorder.polyhedral_ring_order(n, solid)
        assert order.numel() == n
        assert sorted(order.tolist()) == list(range(n)), (
            f"{solid}: ring order not a bijection"
        )
        # Sanity: the solid's edges dict is populated (BFS uses it).
        assert POLYHEDRA[solid]['edges'], f"{solid} has empty edges dict"


# ── compile_to_csr: sorted within each row ──────────────────────────────────

def test_compile_to_csr_sorted_within_rows():
    """compile_to_csr must return indices sorted within each row."""
    seq_len = 32
    for solid in CONFIG['supported_solids']:
        mask = MetatronSparseMask.build(seq_len, solid)
        indptr, indices = compile_to_csr(mask)
        assert indptr.dtype == torch.long
        assert indices.dtype == torch.long
        assert indptr.numel() == seq_len + 1
        for r in range(seq_len):
            row_cols = indices[indptr[r]:indptr[r + 1]].tolist()
            assert row_cols == sorted(row_cols), (
                f"{solid} row {r} not sorted: {row_cols}"
            )
            for c in row_cols:
                assert 0 <= c < seq_len


def test_compile_to_csr_matches_mask_csr():
    """compile_to_csr must produce the same (indptr, indices) as
    MetatronSparseMask.csr."""
    mask = MetatronSparseMask.build(32, 'icosahedron')
    indptr_c, indices_c = compile_to_csr(mask)
    indptr_m, indices_m = mask.csr
    assert torch.equal(indptr_c, indptr_m)
    assert torch.equal(indices_c, indices_m)


def test_compile_to_csr_device_arg():
    """compile_to_csr(device=...) honours the device argument."""
    mask = MetatronSparseMask.build(16, 'icosahedron')
    indptr, indices = compile_to_csr(mask, device=torch.device('cpu'))
    assert indptr.device == torch.device('cpu')
    assert indices.device == torch.device('cpu')


def test_compile_to_csr_usable_with_sparse_csr_tensor():
    """compile_to_csr output must be accepted by torch.sparse_csr_tensor."""
    mask = MetatronSparseMask.build(16, 'icosahedron')
    indptr, indices = compile_to_csr(mask)
    values = torch.ones(indices.numel(), dtype=torch.float32)
    sp = torch.sparse_csr_tensor(indptr, indices, values, (16, 16))
    assert sp.shape == (16, 16)


# ── compile_to_coo ───────────────────────────────────────────────────────────

def test_compile_to_coo_shapes():
    """compile_to_coo returns (row, col) tensors matching mask.edges length."""
    mask = MetatronSparseMask.build(32, 'icosahedron')
    r, c = compile_to_coo(mask)
    assert r.numel() == len(mask.edges[0])
    assert c.numel() == len(mask.edges[1])
    assert r.dtype == torch.long and c.dtype == torch.long


# ── int8 quantization of a zero tensor ───────────────────────────────────────

def test_int8_quantize_zero_tensor_finite():
    """Quantizing an all-zero tensor with int8 must not divide by zero and
    must produce a finite int8 tensor that round-trips exactly."""
    q = Quantizer('int8')
    zeros = torch.zeros(64, 64)
    out, info = q.quantize(zeros)
    assert out.dtype == torch.int8
    assert torch.isfinite(out.float()).all()
    # Round-trip: dequant must reproduce zeros exactly.
    deq = q.dequantize(out, info)
    assert torch.allclose(deq, zeros)
    # The scale used must be finite (1.0 guard, not inf/nan).
    assert torch.isfinite(info['scale'])
    assert info['scale'].item() == 1.0


def test_int8_quantize_nonzero_roundtrip():
    """int8 quantization of a non-zero tensor round-trips approximately."""
    torch.manual_seed(0)
    q = Quantizer('int8')
    x = torch.randn(128, 256)
    out, info = q.quantize(x)
    assert out.dtype == torch.int8
    deq = q.dequantize(out, info)
    # int8 has limited precision; allow a small relative tolerance.
    max_abs = x.abs().max().item()
    assert (deq - x).abs().max().item() <= max_abs * (1.0 / 127.0 + 1e-6)


def test_int8_quantize_clamps_to_range():
    """int8 output must be within [-127, 127]."""
    q = Quantizer('int8')
    x = torch.randn(100) * 1000  # large magnitude
    out, _ = q.quantize(x)
    assert out.to(torch.int).min().item() >= -127
    assert out.to(torch.int).max().item() <= 127


def test_quantizer_rejects_unsupported_dtype():
    """Unsupported dtype must raise ValueError."""
    with pytest.raises(ValueError):
        Quantizer('int4')


def test_fp16_quantize_dtype():
    """fp16 quantization casts to torch.float16."""
    q = Quantizer('fp16')
    x = torch.randn(32, 32, dtype=torch.float32)
    out, info = q.quantize(x)
    assert out.dtype == torch.float16
    assert info == {}


# ── bandwidth_estimate uses 4 transfers (Q, K, V, O) ───────────────────────

def test_bandwidth_estimate_uses_4_transfers():
    """bandwidth_estimate must be n_edges * d_k * 4 (float32) * 4 (Q,K,V,O)."""
    mask = MetatronSparseMask.build(64, 'icosahedron')
    compiler = SparseKernelCompiler(mask, 'icosahedron')
    n_edges = len(mask.edges[0])
    d_k = 64
    bw = compiler.bandwidth_estimate(d_k=d_k)
    # 4 transfers (Q,K,V,O), each d_k float32 (4 bytes).
    assert bw == n_edges * d_k * 4 * 4, (
        f"bandwidth={bw} expected {n_edges * d_k * 4 * 4} (4 transfers, not 3)"
    )
    # The old buggy *3 value must NOT match.
    assert bw != n_edges * d_k * 4 * 3


def test_bandwidth_estimate_accepts_dk_param():
    """bandwidth_estimate must accept d_k (default 64) and scale linearly."""
    mask = MetatronSparseMask.build(64, 'icosahedron')
    compiler = SparseKernelCompiler(mask, 'icosahedron')
    n_edges = len(mask.edges[0])
    bw_default = compiler.bandwidth_estimate()  # d_k=64
    bw_32 = compiler.bandwidth_estimate(d_k=32)
    assert bw_default == n_edges * 64 * 4 * 4
    assert bw_32 == n_edges * 32 * 4 * 4
    assert bw_32 == bw_default // 2


# ── flop_estimate / flops_per_token / summary accept d_k ───────────────────

def test_flop_estimate_accepts_dk():
    """flop_estimate must accept d_k (default 64): 2 * n_edges * d_k."""
    mask = MetatronSparseMask.build(64, 'icosahedron')
    compiler = SparseKernelCompiler(mask, 'icosahedron')
    n_edges = len(mask.edges[0])
    assert compiler.flop_estimate() == 2 * n_edges * 64
    assert compiler.flop_estimate(d_k=32) == 2 * n_edges * 32


def test_flops_per_token_accepts_dk():
    """flops_per_token(d_k) = flop_estimate(d_k) / seq_len."""
    mask = MetatronSparseMask.build(64, 'icosahedron')
    compiler = SparseKernelCompiler(mask, 'icosahedron')
    n_edges = len(mask.edges[0])
    assert compiler.flops_per_token(d_k=32) == (2 * n_edges * 32) / 64


def test_summary_accepts_dk():
    """summary(d_k) must include d_k-dependent fields."""
    mask = MetatronSparseMask.build(64, 'icosahedron')
    compiler = SparseKernelCompiler(mask, 'icosahedron')
    s = compiler.summary(d_k=32)
    assert s['solid'] == 'icosahedron'
    assert s['seq_len'] == 64
    assert s['flops_estimate'] == compiler.flop_estimate(d_k=32)
    assert s['bandwidth_bytes'] == compiler.bandwidth_estimate(d_k=32)


# ── __main__ smoke runs clean ───────────────────────────────────────────────

def test_main_smoke_runs_clean():
    """`python metatron_compiler.py` must exit 0 with no traceback."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, "metatron_compiler.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"metatron_compiler.py exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Traceback" not in result.stderr
    assert "smoke OK" in result.stdout


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pytest.main([__file__, "-q"])