"""
test_sparse_attention.py
========================
Core pytest suite for metatron_sparse_attention.py.

Covers the contract fixes (IMPLEMENTATION_CONTRACT.md §1):
  * MetatronSparseAttention.forward takes (B, L, d_model) and returns
    (B, L, d_model) — NOT pre-split heads.
  * mask override at call time works (forward(x, mask=other)).
  * Octahedron has exactly 8 triangular faces; dodecahedron has exactly 12
    pentagonal faces; every face has the correct vertex count.
  * CSR indices are sorted within each row (CSR invariant for
    torch.sparse_csr_tensor).
  * measure_sparsity exposes `mask_sparsity_pct` (not the old
    `ops_reduction_pct`).
"""

from __future__ import annotations

import math
import sys
import os

# Ensure repo root is importable when pytest runs from the tests/ dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from config import CONFIG
from metatron_sparse_attention import (
    POLYHEDRA,
    MetatronSparseMask,
    MetatronSparseAttention,
    measure_sparsity,
)


# ── Polyhedra correctness ───────────────────────────────────────────────────

def test_octahedron_has_8_triangular_faces():
    """Octahedron must have exactly 8 faces, each a triangle (3 vertices)."""
    octa = POLYHEDRA['octahedron']
    assert len(octa['faces']) == 8, f"octahedron must have 8 faces, got {len(octa['faces'])}"
    for f in octa['faces']:
        assert len(f) == 3, f"octahedron face must be a triangle, got {f}"
    # apexes 0 (top) and 5 (bottom); equator 1-2-3-4
    verts = {v for f in octa['faces'] for v in f}
    assert verts == set(range(6)), f"octahedron vertices must be 0..5, got {verts}"


def test_dodecahedron_has_12_pentagonal_faces():
    """Dodecahedron must have exactly 12 faces, each pentagonal (5 vertices)."""
    dod = POLYHEDRA['dodecahedron']
    assert len(dod['faces']) == 12, (
        f"dodecahedron must have 12 faces, got {len(dod['faces'])}"
    )
    for f in dod['faces']:
        assert len(f) == 5, f"dodecahedron face must be pentagonal, got {f}"
        # No repeated vertex inside one face.
        assert len(set(f)) == 5, f"dodecahedron face has repeated vertex: {f}"
    # Vertex set must be exactly 0..19.
    verts = {v for f in dod['faces'] for v in f}
    assert verts == set(range(20)), (
        f"dodecahedron vertices must be 0..19, got {sorted(verts)}"
    )


def test_all_solids_face_vertex_counts():
    """Sanity: every supported solid's faces have a consistent vertex count."""
    for name in CONFIG['supported_solids']:
        poly = POLYHEDRA[name]
        face_lens = {len(f) for f in poly['faces']}
        assert len(face_lens) == 1, (
            f"{name} has mixed face sizes: {face_lens}"
        )
        # Every face lists distinct vertices.
        for f in poly['faces']:
            assert len(set(f)) == len(f), (
                f"{name} face has repeated vertex: {f}"
            )


# ── CSR invariant: indices sorted within each row ───────────────────────────

def test_csr_indices_sorted_within_rows():
    """CSR `indices` must be sorted within every row."""
    seq_len = 32
    for solid in CONFIG['supported_solids']:
        mask = MetatronSparseMask.build(seq_len, solid)
        indptr, indices = mask.csr
        assert indptr.dtype == torch.long
        assert indices.dtype == torch.long
        assert indptr.numel() == seq_len + 1
        for r in range(seq_len):
            row_cols = indices[indptr[r]:indptr[r + 1]].tolist()
            assert row_cols == sorted(row_cols), (
                f"{solid}: row {r} cols not sorted: {row_cols}"
            )
            # All column indices in range.
            for c in row_cols:
                assert 0 <= c < seq_len


def test_csr_total_matches_edges():
    """CSR indices length must equal the total number of edges."""
    mask = MetatronSparseMask.build(32, 'icosahedron')
    _, indices = mask.csr
    assert indices.numel() == len(mask.edges[0])


def test_csr_is_usable_with_sparse_csr_tensor():
    """The CSR (indptr, indices) must be accepted by torch.sparse_csr_tensor."""
    mask = MetatronSparseMask.build(16, 'icosahedron')
    indptr, indices = mask.csr
    values = torch.ones(indices.numel(), dtype=torch.float32)
    sp = torch.sparse_csr_tensor(indptr, indices, values, (16, 16))
    assert sp.shape == (16, 16)


# ── measure_sparsity exposes the renamed key ────────────────────────────────

def test_measure_sparsity_has_mask_sparsity_pct():
    """measure_sparsity must expose `mask_sparsity_pct` (renamed from
    ops_reduction_pct)."""
    info = measure_sparsity(seq_len=32, solid='icosahedron')
    assert 'mask_sparsity_pct' in info
    assert info['mask_sparsity_pct'] >= 0.0
    assert info['mask_sparsity_pct'] <= 100.0
    # No ops_reduction_pct key should linger.
    assert 'ops_reduction_pct' not in info


# ── Forward shape contract: (B, L, d_model) -> (B, L, d_model) ───────────────

def test_forward_shape_contract():
    """Forward takes (B, L, d_model) — NOT pre-split heads — and returns
    (B, L, d_model)."""
    d_model = CONFIG['d_model']
    n_heads = CONFIG['n_heads']
    B, L = 2, 32
    mask = MetatronSparseMask.build(L, CONFIG['default_solid'])
    attn = MetatronSparseAttention(d_model=d_model, n_heads=n_heads, mask=mask)
    x = torch.randn(B, L, d_model)        # NOT pre-split
    out = attn(x)
    assert out.shape == (B, L, d_model), f"expected {(B, L, d_model)}, got {tuple(out.shape)}"


def test_forward_without_mask_runs():
    """Forward with no mask at all must still produce the right shape."""
    d_model = CONFIG['d_model']
    n_heads = CONFIG['n_heads']
    B, L = 2, 16
    attn = MetatronSparseAttention(d_model=d_model, n_heads=n_heads, mask=None)
    x = torch.randn(B, L, d_model)
    out = attn(x)
    assert out.shape == (B, L, d_model)


# ── Mask override at call time ───────────────────────────────────────────────

def test_mask_override_at_call_time():
    """forward(x, mask=other) must use the override, not self.mask."""
    d_model = CONFIG['d_model']
    n_heads = CONFIG['n_heads']
    B, L = 2, 32

    mask_a = MetatronSparseMask.build(L, 'icosahedron')
    mask_b = MetatronSparseMask.build(L, 'tetrahedron')

    attn = MetatronSparseAttention(d_model=d_model, n_heads=n_heads, mask=mask_a)
    x = torch.randn(B, L, d_model)

    # Run with the override (mask_b) — must not raise and must return the
    # correct shape. We assert the override is actually consulted by checking
    # the forward accepts a different mask and produces finite output.
    out_override = attn(x, mask=mask_b)
    assert out_override.shape == (B, L, d_model)
    assert torch.isfinite(out_override).all()

    # And the default (mask_a) path still works.
    out_default = attn(x)
    assert out_default.shape == (B, L, d_model)


def test_mask_override_changes_output():
    """Overriding the mask must change the output (different mask → different
    attention pattern)."""
    torch.manual_seed(CONFIG.get('seed', 42))
    d_model = CONFIG['d_model']
    n_heads = CONFIG['n_heads']
    B, L = 1, 32
    mask_a = MetatronSparseMask.build(L, 'icosahedron')
    mask_b = MetatronSparseMask.build(L, 'dodecahedron')
    # Different solids produce different edge sets.
    assert mask_a.edges[0] != mask_b.edges[0] or mask_a.edges[1] != mask_b.edges[1]

    attn = MetatronSparseAttention(d_model=d_model, n_heads=n_heads, mask=mask_a)
    attn.eval()  # disable dropout for determinism
    x = torch.randn(B, L, d_model)
    with torch.no_grad():
        out_a = attn(x)
        out_b = attn(x, mask=mask_b)
    assert not torch.allclose(out_a, out_b), (
        "mask override should change the output"
    )


# ── Device derivation ────────────────────────────────────────────────────────

def test_forward_derives_device_from_input():
    """The mask edges are built on x.device inside forward (no stored buffers
    that force CPU)."""
    d_model = CONFIG['d_model']
    n_heads = CONFIG['n_heads']
    B, L = 1, 16
    mask = MetatronSparseMask.build(L, 'icosahedron')
    attn = MetatronSparseAttention(d_model=d_model, n_heads=n_heads, mask=mask)
    x = torch.randn(B, L, d_model)
    out = attn(x)
    assert out.device == x.device


# ── Backward / gradient flow ─────────────────────────────────────────────────

def test_backward_runs():
    """A backward pass through the sparse forward must populate gradients."""
    d_model = CONFIG['d_model']
    n_heads = CONFIG['n_heads']
    B, L = 1, 16
    mask = MetatronSparseMask.build(L, 'icosahedron')
    attn = MetatronSparseAttention(d_model=d_model, n_heads=n_heads, mask=mask)
    x = torch.randn(B, L, d_model, requires_grad=True)
    out = attn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ── Smoke: build mask for all 5 solids ───────────────────────────────────────

def test_build_all_solids_smoke():
    """MetatronSparseMask.build must work for every supported solid and yield
    a non-empty edge set."""
    L = 32
    for solid in CONFIG['supported_solids']:
        m = MetatronSparseMask.build(L, solid)
        assert m.seq_len == L
        assert m.solid == solid
        assert len(m.edges[0]) == len(m.edges[1])
        assert len(m.edges[0]) > 0, f"{solid} produced empty edge set"
        assert 0.0 <= m.sparsity() <= 1.0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pytest.main([__file__, "-q"])