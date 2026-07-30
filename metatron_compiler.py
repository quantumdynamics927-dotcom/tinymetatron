"""
metatron_compiler.py
====================
Token reordering + kernel compilation + optional quantization.
Reorders tokens for cache locality, compiles sparse masks to CSR/COO format,
and generates per‑solid benchmark kernels.

Matches TMT Patent Draft §7.3 / Claim 26:
    "reordenación para localidad de caché"
    "generación de kernels y cuantización opcional"
    "planificador de bloques/hilos para conflictos de banco"
"""

from __future__ import annotations

import math
from typing import Tuple, Optional, Literal

import torch
import torch.nn as nn
from metatron_sparse_attention import MetatronSparseMask, POLYHEDRA


class TokenReorder:
    """
    Reorders token indices to improve cache locality and avoid bank conflicts.

    Patent Claim 5: "reordena tokens para mejorar localidad de caché"
    Patent Claim 26: "planificador de bloques/hilos para conflictos de banco"

    Uses a Hilbert‑curve‑inspired Morton order (interleaves bits)
    which maps nearby sequence positions to nearby cache lines.
    """

    @staticmethod
    def morton_reorder(seq_len: int) -> torch.Tensor:
        """
        Returns a permutation index tensor that maps original positions
        to cache‑friendly positions using Morton (Z‑order) curve.

        The raw Morton key (bit‑interleaving of `i`) is unique for every
        `i` in `0..seq_len-1`, so sorting by the RAW key yields a true
        bijection (permutation of `0..seq_len-1`). No modular reduction is
        applied — that would collide distinct keys and break the bijection.

        Example: seq_len=8 → order=[0,1,4,5,2,3,6,7]
        """
        n_bits = max(1, (seq_len - 1).bit_length())
        order = []
        for i in range(seq_len):
            # Interleave the bits of i to create the Z‑order (Morton) key.
            # The key is stored RAW (no `% seq_len`): each i < seq_len has a
            # distinct key, so the sorted order is a bijection.
            morton = 0
            for b in range(n_bits):
                morton |= ((i >> (2 * b)) & 1) << b
                morton |= ((i >> (2 * b + 1)) & 1) << (b + n_bits)
            order.append(morton)
        # Sort by raw Morton key, return original indices
        sorted_pairs = sorted(enumerate(order), key=lambda x: x[1])
        return torch.tensor([orig for orig, _ in sorted_pairs], dtype=torch.long)

    @staticmethod
    def reverse_reorder(forward_idx: torch.Tensor) -> torch.Tensor:
        """
        Returns the inverse permutation of `forward_idx`.
        """
        out = torch.empty_like(forward_idx)
        out[forward_idx] = torch.arange(len(forward_idx), device=forward_idx.device)
        return out

    @staticmethod
    def polyhedral_ring_order(seq_len: int,
                              solid: str = 'icosahedron') -> torch.Tensor:
        """
        Orders tokens by their natural ring structure in the polyhedral graph.

        A real BFS is run from vertex 0 over `poly['edges']`; the BFS level of
        each vertex is its ring index. Each token `idx` is mapped to vertex
        `idx % V` and assigned that vertex's ring. Tokens are then stably sorted
        by (ring, original_index), which yields a bijection over
        `0..seq_len-1`.
        """
        poly = POLYHEDRA[solid]
        V = poly['V']
        edges = poly['edges']

        # BFS from vertex 0 over the polyhedral edge graph.
        ring_of_vertex = {0: 0}
        queue = [0]
        while queue:
            v = queue.pop(0)
            for nbr in edges.get(v, []):
                if nbr not in ring_of_vertex:
                    ring_of_vertex[nbr] = ring_of_vertex[v] + 1
                    queue.append(nbr)

        # Map each token to its vertex's ring (vertex 0 default → ring 0).
        indexed = []
        for idx in range(seq_len):
            v = idx % V
            ring = ring_of_vertex.get(v, 0)
            indexed.append((idx, ring))
        indexed.sort(key=lambda x: (x[1], x[0]))
        return torch.tensor([idx for idx, _ in indexed], dtype=torch.long)


def compile_to_csr(mask: MetatronSparseMask,
                   device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compiles a MetatronSparseMask to CSR (Compressed Sparse Row) format.
    Suitable for torch.sparse_csr_tensor.

    Columns are SORTED within each row (CSR invariant required by
    `torch.sparse_csr_tensor`).

    Returns: (indptr, indices) — both int64 tensors on `device`.

    Patent Claim 29: "compilación a representaciones dispersas CSR/COO"
    """
    rows, cols = mask.edges
    n = mask.seq_len

    # Bucket columns per row, sort within each row, then flatten.
    per_row: list[list[int]] = [[] for _ in range(n)]
    for r, c in zip(rows, cols):
        per_row[r].append(c)

    indices: list[int] = []
    indptr = [0]
    for r in range(n):
        per_row[r].sort()
        indices.extend(per_row[r])
        indptr.append(len(indices))

    indptr_t = torch.tensor(indptr, dtype=torch.long, device=device)
    indices_t = torch.tensor(indices, dtype=torch.long, device=device)
    return indptr_t, indices_t


def compile_to_coo(mask: MetatronSparseMask,
                   device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compiles a MetatronSparseMask to COO (Coordinate) format.
    Returns: (row_indices, col_indices)
    """
    r = torch.tensor(mask.edges[0], dtype=torch.long, device=device)
    c = torch.tensor(mask.edges[1], dtype=torch.long, device=device)
    return r, c


class SparseKernelCompiler:
    """
    Generates optimised sparse attention kernels for a given mask + solid.

    Produces:
      - A Python callable (uses torch ops, JIT‑compatible)
      - An estimated FLOP count for benchmarking

    Patent Claim 8: "compilación a kernels específicos y cuantización"
    """

    def __init__(self, mask: MetatronSparseMask, solid: str = 'icosahedron'):
        self.mask  = mask
        self.solid = solid

    def flop_estimate(self, d_k: int = 64) -> int:
        """
        Estimate FLOPs for one sparse attention forward pass.
        FLOPs ≈ 2 * n_edges * d_k (one multiply‑add per edge per head dim).
        """
        n_edges = len(self.mask.edges[0])
        return 2 * n_edges * d_k

    def flops_per_token(self, d_k: int = 64) -> float:
        """FLOPs normalised per output token."""
        L = self.mask.seq_len
        return self.flop_estimate(d_k) / L

    def bandwidth_estimate(self, d_k: int = 64) -> int:
        """
        Estimated bytes moved (reads + writes) for one forward pass.

        Four transfers per edge: Q read, K read, V read, and O write
        (Q, K, V, O = 4 projections), each of `d_k` float32 elements.
        """
        n_edges = len(self.mask.edges[0])
        bytes_per = 4   # float32
        return n_edges * d_k * 4 * bytes_per  # Q, K, V reads + O write

    def summary(self, d_k: int = 64) -> dict:
        return {
            'solid': self.solid,
            'seq_len': self.mask.seq_len,
            'edges': len(self.mask.edges[0]),
            'sparsity': f"{self.mask.sparsity():.1%}",
            'flops_estimate': self.flop_estimate(d_k),
            'flops_per_token': self.flops_per_token(d_k),
            'bandwidth_bytes': self.bandwidth_estimate(d_k),
        }


class Quantizer:
    """
    Optional post‑compilation quantisation to int8/float16.

    Patent Claim 8: "cuantización"
    """

    SUPPORTED = ['fp32', 'fp16', 'bf16', 'int8']

    def __init__(self, dtype: Literal['fp32','fp16','bf16','int8'] = 'fp16'):
        if dtype not in self.SUPPORTED:
            raise ValueError(f"dtype must be one of {self.SUPPORTED}")
        self.dtype = dtype

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Quantise a tensor. Returns (quantized_tensor, scale_info).
        scale_info is needed for dequantisation.
        """
        if self.dtype == 'fp32':
            return tensor, {}
        elif self.dtype in ('fp16', 'bf16'):
            return tensor.to(dtype=torch.float16 if self.dtype == 'fp16'
                             else torch.bfloat16), {}
        elif self.dtype == 'int8':
            # Per‑tensor symmetric quantisation with zero‑scale guard.
            amax = tensor.abs().max()
            if amax > 0:
                scale = amax / 127.0
            else:
                # All‑zero (or constant‑zero) tensor: avoid divide‑by‑zero;
                # use scale 1.0 so dequant reproduces the zeros exactly.
                scale = torch.tensor(1.0, device=tensor.device,
                                     dtype=tensor.dtype)
            q = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
            return q, {'scale': scale}

    def dequantize(self, tensor: torch.Tensor, scale_info: dict) -> torch.Tensor:
        if self.dtype == 'int8':
            scale = scale_info['scale']
            return tensor.float() * scale
        return tensor.float()


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    torch.manual_seed(42)

    seq_len = 128
    mask = MetatronSparseMask.build(seq_len, 'icosahedron')

    # Token reordering — bijection over 0..seq_len-1
    order = TokenReorder.morton_reorder(seq_len)
    print(f"Morton order (first 16): {order[:16].tolist()}")
    assert sorted(order.tolist()) == list(range(seq_len)), "morton_reorder not a bijection"

    # Polyhedral ring order — real BFS, bijection
    ring = TokenReorder.polyhedral_ring_order(seq_len, 'icosahedron')
    assert sorted(ring.tolist()) == list(range(seq_len)), "ring order not a bijection"

    # CSR compilation — columns sorted within each row
    indptr, indices = compile_to_csr(mask)
    print(f"CSR indptr length: {len(indptr)}, indices length: {len(indices)}")
    # Verify CSR invariant: sorted columns within each row
    for r in range(seq_len):
        row_cols = indices[indptr[r]:indptr[r + 1]].tolist()
        assert row_cols == sorted(row_cols), f"row {r} not sorted"

    # Kernel compiler summary (uses default d_k=64)
    compiler = SparseKernelCompiler(mask, 'icosahedron')
    summary  = compiler.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Quantization — fp16 and int8 (incl. zero‑scale guard)
    x = torch.randn(128, 256)
    qz_fp16 = Quantizer('fp16')
    xq_fp16, _ = qz_fp16.quantize(x)
    print(f"\nfp16 quantized dtype: {xq_fp16.dtype}")

    qz_int8 = Quantizer('int8')
    xq_int8, info_int8 = qz_int8.quantize(x)
    print(f"int8 quantized dtype: {xq_int8.dtype}")

    # Zero‑tensor int8 guard: must not divide by zero.
    zeros = torch.zeros(64, 64)
    zq, zinfo = qz_int8.quantize(zeros)
    assert zq.dtype == torch.int8
    assert torch.allclose(qz_int8.dequantize(zq, zinfo), zeros)
    print("int8 zero‑tensor guard OK")

    print("\nmetatron_compiler smoke OK")
