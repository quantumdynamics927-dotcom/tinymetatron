"""
metatron_sparse_attention.py
=============================
MetatronSparseAttention — polyhedral sparse mask + CSR/COO forward pass.

Matches the disclosure in TMT Patent Draft §7.1
- Hierarchical local windows (radial rings)
- Geometric chords (φ=0.618 golden ratio conjugate, √2=1.414)
- Long-range routes (radial→chord→ring schedule)
- Self-attention forward contract: x:(B,L,d_model) -> (B,L,d_model)

Usage:
    mask = MetatronSparseMask.build(seq_len, solid='icosahedron')
    attn = MetatronSparseAttention(d_model=256, n_heads=4, mask=mask)
    out = attn(x)              # x:(B,L,d_model) -> out:(B,L,d_model)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple, Optional

from config import CONFIG

# ── Polyhedral adjacency tables ──────────────────────────────────────────────
# Each solid: vertex count, faces, edge adjacency (0-indexed vertices)
#
# Octahedron (V=6): apex 0 (top), apex 5 (bottom), equator 1-2-3-4.
# 8 triangular faces, 12 edges, every vertex degree 4.
#   Correct face list per IMPLEMENTATION_CONTRACT.md §1:
#     [[0,1,2],[0,2,3],[0,3,4],[0,4,1],
#      [5,2,1],[5,3,2],[5,4,3],[5,1,4]]
#
# Dodecahedron (V=20): 12 pentagonal faces, 30 edges, every vertex degree 3.
#   Standard labeling (combinatorially correct: every edge shared by exactly
#   two faces, every face a 5-cycle).
POLYHEDRA = {
    'tetrahedron': {
        'V': 4,
        'faces': [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        'edges': {0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2]},
    },
    'hexahedron': {   # Cube — 8 vertices, 12 edges, 6 faces
        'V': 8,
        'faces': [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                  [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]],
        'edges': {
            0: [1, 3, 4], 1: [0, 2, 5], 2: [1, 3, 6], 3: [0, 2, 7],
            4: [0, 5, 7], 5: [1, 4, 6], 6: [2, 5, 7], 7: [3, 4, 6],
        },
    },
    'octahedron': {
        # V=6, apexes 0 (top) / 5 (bottom), equator 1-2-3-4. 8 triangular faces.
        'V': 6,
        'faces': [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
                  [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]],
        'edges': {},   # filled below via face expansion
    },
    'dodecahedron': {
        # V=20, 12 pentagonal faces (standard labeling), 30 edges, degree 3.
        'V': 20,
        'faces': [
            [0, 11, 10, 1, 15],
            [0, 19, 2, 14, 15],
            [0, 19, 17, 4, 11],
            [1, 10, 5, 16, 18],
            [1, 18, 3, 14, 15],
            [2, 9, 6, 17, 19],
            [2, 9, 8, 3, 14],
            [3, 8, 7, 16, 18],
            [4, 11, 10, 5, 13],
            [4, 17, 6, 12, 13],
            [5, 16, 7, 12, 13],
            [6, 9, 8, 7, 12],
        ],
        'edges': {},   # filled below via face expansion
    },
    'icosahedron': {
        'V': 12,
        'faces': [
            [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1],
            [1, 5, 7], [5, 4, 9], [4, 3, 11], [3, 2, 10], [2, 1, 6],
            [6, 1, 7], [7, 5, 9], [9, 4, 11], [11, 3, 10], [10, 2, 6],
            [6, 7, 8], [7, 9, 8], [9, 11, 8], [11, 10, 8], [10, 6, 8],
        ],
        'edges': {},   # filled below via face expansion
    },
}

# ── Build edge dicts for solids that only defined faces ──────────────────────
def _build_edges(poly: dict) -> dict:
    if poly.get('edges'):
        return poly['edges']
    edges = {}
    for face in poly['faces']:
        for i, a in enumerate(face):
            b = face[(i + 1) % len(face)]
            edges.setdefault(a, set()).add(b)
            edges.setdefault(b, set()).add(a)
    poly['edges'] = {k: list(v) for k, v in edges.items()}
    return poly['edges']

for poly in POLYHEDRA.values():
    _build_edges(poly)


# ── Constants (fallbacks; CONFIG is source of truth) ─────────────────────────
PHI = CONFIG.get('phi', 0.618)         # Golden ratio conjugate  (φ)
SQRT2 = CONFIG.get('sqrt2', 1.414)      # √2                     (chord ratio)


def radial_level(idx: int, seq_len: int, n_levels: int = None) -> int:
    """
    Assign token position `idx` to a radial level (ring).
    Uses binary expansion: level = floor(log2(idx | 1))
    This mirrors the patent's 2**level(idx) window sizing.
    """
    if n_levels is None:
        n_levels = max(1, int(math.log2(seq_len)) + 1)
    if idx == 0:
        return 0
    return min(int(math.log2(idx | 1)), n_levels - 1)


def local_window(idx: int, seq_len: int, level: int = None) -> set:
    """
    L(i) — local window of radius w = 2**level.
    Patent §7.1 step 1: "ventanas locales"
    """
    if level is None:
        level = radial_level(idx, seq_len)
    w = 2 ** level
    start = max(0, idx - w)
    end = min(seq_len, idx + w + 1)
    return set(range(start, end))


def geometric_chords(idx: int, seq_len: int, poly_name: str = 'icosahedron',
                     phi: float = PHI, sqrt2: float = SQRT2) -> set:
    """
    C(i) — geometric chord jumps using φ and √2 proportions.
    Patent §7.1 step 2: "cordones (chords)"
    Maps idx → geometric progression positions modulated by polyhedral topology.
    The cross-chord (φ + √2 − 1) is added to the set before the self-index is
    discarded, so all three chord families (φ, √2, cross) contribute.
    """
    poly = POLYHEDRA[poly_name]
    V = poly['V']
    chords = set()

    # φ-chord: golden-ratio hop (mod vertex count)
    phi_idx = int(idx * phi) % V
    target_phi = (idx + phi_idx) % seq_len
    chords.add(target_phi)

    # √2-chord: diagonal hop
    sqrt2_idx = int(idx * sqrt2) % V
    target_s2 = (idx + sqrt2_idx) % seq_len
    chords.add(target_s2)

    # Cross-chord: combination of the two ratios (adds diversity)
    cross = (idx * (phi + sqrt2 - 1)) % seq_len
    chords.add(cross)

    chords.discard(idx)     # never self-connect
    return chords


def long_range_routes(idx: int, seq_len: int,
                      poly_name: str = 'icosahedron',
                      schedule: list = None) -> set:
    """
    R(i) — long-range routes following radial→chord→ring phases.
    Patent §7.1 step 3: "rutas a largo alcance"
    """
    if schedule is None:
        schedule = CONFIG.get('comm_schedule', ['radial', 'chord', 'ring'])

    routes = set()
    poly = POLYHEDRA[poly_name]
    V = poly['V']

    level = radial_level(idx, seq_len)
    max_level = int(math.log2(seq_len)) if seq_len > 1 else 1

    # Radial phase: jump to equivalent position in next level
    if 'radial' in schedule and level < max_level:
        stride = 2 ** (level + 1)
        radial_target = (idx + stride) % seq_len
        routes.add(radial_target)

    # Chord phase: use polyhedral vertex mapping
    if 'chord' in schedule:
        vertex = idx % V
        # Walk to the farthest reachable vertex via edges
        edges = poly['edges'].get(vertex, [])
        if edges:
            far = edges[len(edges) // 2]      # roughly opposite vertex
            far_target = (idx // V) * V + far
            if far_target < seq_len:
                routes.add(far_target)

    # Ring phase: global ring sweep (every V-th token)
    if 'ring' in schedule:
        for hop in range(0, seq_len, V):
            target = (idx + hop) % seq_len
            if target != idx:
                routes.add(target)

    routes.discard(idx)
    return routes


class MetatronSparseMask:
    """
    Polyhedral sparse attention mask M(i) = L(i) ∪ C(i) ∪ R(i)
    Compiles to CSR / COO sparse index format.

    Attributes:
        seq_len:   sequence length
        solid:     polyhedron name
        levels:    number of radial levels
        phi / s2:  chord ratios
        edges:     (row, col) index pairs for non-zero attention entries
        csr:       (indptr, indices) CSR representation (indices sorted within
                   each row — the invariant required by
                   torch.sparse_csr_tensor)
    """

    def __init__(self, seq_len: int,
                 solid: Literal['tetrahedron', 'hexahedron', 'octahedron',
                                 'dodecahedron', 'icosahedron'] = 'icosahedron',
                 phi: float = PHI, sqrt2: float = SQRT2,
                 schedule: list = None):
        self.seq_len = seq_len
        self.solid = solid
        self.phi = phi
        self.sqrt2 = sqrt2
        self.schedule = schedule or CONFIG.get('comm_schedule',
                                               ['radial', 'chord', 'ring'])

        poly = POLYHEDRA[solid]
        # Guard against non-positive seq_len: math.log2(0) is a domain error
        # and a negative length has no meaningful mask. Produce an empty edge
        # set so downstream code sees a well-formed (if trivial) mask instead
        # of crashing. A zero-length sequence legitimately has no attention.
        if seq_len <= 0:
            self.n_levels = 1
            self.edges = ([], [])
            self._csr = None
            self._coo = None
            return

        self.n_levels = max(1, int(math.log2(seq_len)) + 1)

        # Build edge list: (head_token, attend_to_token)
        rows, cols = [], []
        for idx in range(seq_len):
            L = local_window(idx, seq_len, radial_level(idx, seq_len, self.n_levels))
            C = geometric_chords(idx, seq_len, solid, phi, sqrt2)
            R = long_range_routes(idx, seq_len, solid, self.schedule)

            # Union mask for this head token, sorted so the resulting CSR has
            # monotonically non-decreasing indices within each row.
            mask_ids = sorted(L | C | R)
            for target in mask_ids:
                rows.append(idx)
                cols.append(target)

        self.edges = (rows, cols)

        # CSR / COO caches
        self._csr = None
        self._coo = None

    @classmethod
    def build(cls, seq_len: int, solid: str = 'icosahedron',
              phi: float = PHI, sqrt2: float = SQRT2) -> 'MetatronSparseMask':
        """Factory — same as __init__, named per patent §7.1."""
        return cls(seq_len, solid, phi, sqrt2)

    @property
    def coo(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (row_indices, col_indices) COO format as int64 tensors."""
        if self._coo is None:
            r = torch.tensor(self.edges[0], dtype=torch.long)
            c = torch.tensor(self.edges[1], dtype=torch.long)
            self._coo = (r, c)
        return self._coo

    @property
    def csr(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (indptr, indices) CSR format with indices SORTED within each
        row (the invariant required by torch.sparse_csr_tensor).
        """
        if self._csr is None:
            rows, cols = self.edges
            n = self.seq_len
            # Count entries per row.
            counts = [0] * n
            for r in rows:
                counts[r] += 1
            # Prefix sum → indptr.
            indptr_arr = [0] * (n + 1)
            for i in range(n):
                indptr_arr[i + 1] = indptr_arr[i] + counts[i]
            # Bucket the column indices into per-row lists and sort them.
            row_buckets: list[list[int]] = [[] for _ in range(n)]
            for r, c in zip(rows, cols):
                row_buckets[r].append(c)
            indices_arr: list[int] = []
            for r in range(n):
                br = sorted(row_buckets[r])
                indices_arr.extend(br)
            self._csr = (
                torch.tensor(indptr_arr, dtype=torch.long),
                torch.tensor(indices_arr, dtype=torch.long),
            )
        return self._csr

    def sparsity(self) -> float:
        """Returns fraction of pruned attention entries (0–1)."""
        total_possible = self.seq_len * self.seq_len
        kept = len(self.edges[0])
        return 1.0 - (kept / total_possible)

    def __repr__(self) -> str:
        sp = self.sparsity()
        return (f"MetatronSparseMask(seq_len={self.seq_len}, "
                f"solid={self.solid}, sparsity={sp:.1%}, "
                f"edges={len(self.edges[0])})")


class MetatronSparseAttention(nn.Module):
    """
    Multi-head sparse self-attention with a polyhedral mask.

    forward(x, mask=None):
        x: (B, L, d_model) — self-attention input (q = k = v = x internally)
        Returns: (B, L, d_model)

    The forward projects x with four standard Linear(d_model, d_model) layers
    (Q, K, V, O), reshapes to (B, H, L, d_k), scales by 1/sqrt(d_k), applies the
    sparse mask (−inf outside the polyhedral edges), softmaxes, and produces
    the sparse-aware output via a 4-D scatter_add over the mask edges.

    Patent §7.1: "forward(q, k, v, mask=M)" — realized here as self-attention.

    Device is derived from the input tensor `x`; the mask edges are converted
    to tensors on `x.device` inside forward (no pre-registered buffers).
    """

    def __init__(self, d_model: int, n_heads: int,
                 mask: Optional[MetatronSparseMask] = None,
                 dropout: float = 0.1,
                 bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.mask = mask
        self.dropout = nn.Dropout(dropout)

        # Standard MHA projections — one Linear(d_model, d_model) each.
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor,
                mask: Optional[MetatronSparseMask] = None) -> torch.Tensor:
        """
        Self-attention over x.

        Args:
            x: (B, L, d_model) — input hidden states (q = k = v = x).
            mask: optional MetatronSparseMask override (defaults to self.mask).
        Returns:
            (B, L, d_model)
        """
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_k

        # Project and reshape to (B, H, L, d_k).
        q = self.W_q(x).view(B, L, H, D).transpose(1, 2)   # (B, H, L, d_k)
        k = self.W_k(x).view(B, L, H, D).transpose(1, 2)
        v = self.W_v(x).view(B, L, H, D).transpose(1, 2)

        # Scaled dot-product: scale = 1 / sqrt(d_k)
        scale = 1.0 / math.sqrt(D)
        qk = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, H, L, L)

        active_mask = mask if mask is not None else self.mask
        if active_mask is not None:
            # The mask was built for a fixed seq_len; using it on a differently
            # sized tensor would index out of bounds (or silently misalign
            # rows/cols). Refuse rather than produce a wrong, masked output.
            if active_mask.seq_len != L:
                raise ValueError(
                    f"mask seq_len ({active_mask.seq_len}) != input length "
                    f"({L}); build a mask for length {L} or pass a matching "
                    f"override."
                )
            # Empty mask (seq_len==0 at build time) → identity passthrough.
            if len(active_mask.edges[0]) == 0:
                return self.W_o(x)
            # Derive r, c from active_mask.edges on x.device (no stored buffers).
            rows, cols = active_mask.edges
            r = torch.tensor(rows, dtype=torch.long, device=x.device)
            c = torch.tensor(cols, dtype=torch.long, device=x.device)

            # Apply sparse mask: set masked positions to -inf before softmax.
            mask_matrix = torch.full((L, L), float('-inf'), device=x.device,
                                     dtype=qk.dtype)
            mask_matrix[r, c] = 0.0
            qk = qk + mask_matrix.unsqueeze(0)              # broadcast over B,H

            attn_weights = F.softmax(qk, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Sparse-aware output via 4-D scatter_add over the mask edges.
            n_edges = r.numel()
            valid = attn_weights[:, :, r, c]                # (B, H, n_edges)
            v_gathered = v.index_select(2, c)               # (B, H, n_edges, d_k)
            weighted = v_gathered * valid.unsqueeze(-1)     # (B, H, n_edges, d_k)
            idx = r.view(1, 1, n_edges, 1).expand_as(weighted)  # (B, H, n_edges, d_k)
            out = torch.zeros(B, H, L, D, device=v.device, dtype=v.dtype)
            out.scatter_add_(2, idx, weighted)              # (B, H, L, d_k)
        else:
            attn_weights = F.softmax(qk, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, v)             # (B, H, L, d_k)

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)  # (B, L, d_model)
        return self.W_o(out)

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, n_heads={self.n_heads}, "
                f"mask={self.mask}")


# ── Benchmark helper ─────────────────────────────────────────────────────────
def measure_sparsity(seq_len: int = 512, solid: str = 'icosahedron') -> dict:
    """
    Returns sparsity metrics for a given configuration.

    NOTE: `mask_sparsity_pct` reports the fraction of attention entries pruned
    by the polyhedral mask. It is NOT a realized-FLOP-reduction claim — the
    forward pass is a masked-dense reference implementation.
    """
    mask = MetatronSparseMask.build(seq_len, solid)
    sp = mask.sparsity()
    n_edges = len(mask.edges[0])
    total = seq_len * seq_len
    return {
        'solid': solid,
        'seq_len': seq_len,
        'sparsity_pct': sp * 100,
        'edges_kept': n_edges,
        'edges_total': total,
        'mask_sparsity_pct': sp * 100,
    }


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    torch.manual_seed(CONFIG.get('seed', 42))

    # Build mask for all 5 supported solids; print sparsity + edge count.
    solids = CONFIG['supported_solids']
    seq_len = CONFIG['seq_len'] * 4   # 128 — exercise a non-trivial length
    for solid in solids:
        m = MetatronSparseMask.build(seq_len=seq_len, solid=solid)
        print(f"{solid:12s}  sparsity={m.sparsity():.1%}  "
              f"edges={len(m.edges[0]):6d}")

    # Full self-attention smoke test.
    d_model = CONFIG['d_model']          # 256
    n_heads = CONFIG['n_heads']           # 4
    B = CONFIG['batch_size'] // 8         # 2
    L = seq_len                           # 128
    mask = MetatronSparseMask.build(L, CONFIG['default_solid'])
    attn = MetatronSparseAttention(d_model=d_model, n_heads=n_heads, mask=mask)

    # x is (B, L, d_model) — NOT pre-split heads.
    x = torch.randn(B, L, d_model)
    out = attn(x)
    assert out.shape == (B, L, d_model), f"Shape mismatch: {out.shape}"
    print(f"\nForward pass OK -> output shape {tuple(out.shape)}")
    print(f"Max abs output: {out.abs().max().item():.4f}")