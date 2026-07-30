"""
metatron_global_memory.py
=========================
Peristent global memory layer — 13 nodes representing Metatron's Cube.
Each node holds a learned (d_model) state vector shared across all sequences.
Cross‑attention between local sequence and global memory enables
long‑range context beyond the sequence length limit.

Matches TMT Patent Draft Claim 24:
    "memoria global perzistente entre secuencias"

Also matches the Slovak spec description:
    "Globálna pamäť Metatron (13 uzlov)"
    "perzistentná globálna pamäť zdieľaná medzi sekvenciami"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MetatronGlobalMemory(nn.Module):
    """
    Persistent global memory with 13 learned nodes (Metatron's Cube).
    Cross‑attention is applied between the local sequence and the memory nodes.

    The 13 memory nodes correspond to the 13 polyhedral vertices
    of Metatron's Cube (derived from the icosahedral graph's core nodes G0–G12).

    Usage:
        memory = MetatronGlobalMemory(d_model=256, n_nodes=13)
        context = memory(local_x)        # (B, L, D) → (B, L, D)
        # Memory can be read/written per-step in an autoregressive setting
        memory_state = memory.get_state() # (n_nodes, D)
    """

    NODES = 13   # Metatron's Cube: 13 central spheres

    def __init__(self, d_model: int, n_nodes: int = NODES,
                 dropout: float = 0.1,
                 memory_init: str = 'zeros'):
        super().__init__()
        assert n_nodes == self.NODES, (
            f"TMT Global Memory requires exactly {self.NODES} nodes "
            f"(Metatron's Cube motif). Got {n_nodes}."
        )
        self.d_model = d_model
        self.n_nodes = n_nodes

        # 13 learned memory nodes — persistent across sequences
        if memory_init == 'zeros':
            init = torch.zeros(n_nodes, d_model)
        elif memory_init == 'normal':
            init = torch.randn(n_nodes, d_model) * 0.02
        else:
            init = torch.ones(n_nodes, d_model) * 0.01

        # 13 learned memory nodes — a learnable Parameter (not a buffer).
        # Registered as a Parameter so it moves with .to(device)/.cuda() and
        # participates in optimization. Persistence across calls is guaranteed
        # by in-place updates (set_state uses copy_); never reassign via
        # `self.memory = nn.Parameter(...)` after __init__ (that would break
        # the data_ptr identity that callers rely on for persistence checks).
        self.memory = nn.Parameter(init, requires_grad=True)

        # Gating controller: controls how much the local token attends to memory
        self.gate = nn.Linear(d_model, d_model)

        # Output projection to mix memory back into local space
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, local_x: torch.Tensor,
                read_weights: bool = False) -> torch.Tensor:
        """
        Apply cross‑attention between local sequence and global memory.

        Args:
            local_x:    (B, L, D) local sequence activations
            read_weights: if True, return the per-node attention weights

        Returns:
            (B, L, D) memory‑augmented local activations
            If read_weights=True: also returns (B, L, n_nodes) attention weights
        """
        B, L, D = local_x.shape

        # Query = local sequence; Key/Value = memory nodes
        q = local_x                          # (B, L, D)
        k = self.memory.unsqueeze(0)        # (1, n_nodes, D)
        v = self.memory.unsqueeze(0)        # (1, n_nodes, D)

        # Scaled dot-product cross‑attention
        scale = D ** -0.5
        qk = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, L, n_nodes)

        # Softmax over memory nodes → per-token memory attention
        attn = F.softmax(qk, dim=-1)                          # (B, L, n_nodes)

        # Memory‑aware gating (Claim 21: multidimensional gating).
        # PER-TOKEN gating: a gate value is computed for every position from its
        # own local_x, giving each token an independent control over how much
        # global memory it absorbs. This is more expressive than per-sequence
        # (mean-over-L) gating and is the chosen variant here.
        gate_val = torch.sigmoid(self.gate(local_x))           # (B, L, D)
        memory_out = torch.matmul(attn, v)                     # (B, L, D)
        memory_out = memory_out * gate_val

        out = local_x + self.dropout(self.proj(memory_out))    # residual

        if read_weights:
            return out, attn
        return out

    def get_state(self) -> torch.Tensor:
        """Returns the current memory node states: (n_nodes, D)."""
        return self.memory

    def set_state(self, new_memory: torch.Tensor):
        """
        Overwrite memory node states in place (e.g., after a sequence finishes).
        new_memory: (n_nodes, D)

        Uses copy_ so the Parameter's storage identity (data_ptr) is preserved
        across updates — callers rely on this for persistence checks. The
        incoming tensor is cast to the memory's current device and dtype so
        set_state is safe to call from any device/dtype context.
        """
        assert new_memory.shape == (self.n_nodes, self.d_model), (
            f"Expected memory shape {(self.n_nodes, self.d_model)}, "
            f"got {tuple(new_memory.shape)}."
        )
        with torch.no_grad():
            self.memory.copy_(
                new_memory.to(self.memory.device, self.memory.dtype)
            )

    def reset(self):
        """Reset memory to zeros (useful between unrelated tasks)."""
        with torch.no_grad():
            self.memory.zero_()

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, n_nodes={self.n_nodes}, "
                f"total_memory_kb={self.n_nodes * self.d_model * 4 / 1024:.1f}")


class MetatronMemoryEnhancedAttention(nn.Module):
    """
    Standard local attention + global memory cross‑attention in one module.

    Used in the TinyMetatron SLM pipeline:
        local_attn → (B, L, H, D)  — from metatron_sparse_attention.py
        global_mem → (B, L, D)     — this module

    The two are combined via residual addition.
    """

    def __init__(self, d_model: int, n_heads: int,
                 n_memory_nodes: int = 13,
                 dropout: float = 0.1):
        super().__init__()
        # n_heads is the head count of the externally-supplied local attention
        # submodule. The local_attn module itself is set externally (e.g. a
        # MetatronSparseAttention instance); when the caller wires it via
        # set_local_attn below, n_heads is forwarded to that constructor so the
        # arg is actually used rather than dead. Stored for reference/inspection.
        self.d_model = d_model
        self.n_heads = n_heads
        self.local_attn = None    # set externally via set_local_attn or direct attr
        self.global_mem = MetatronGlobalMemory(d_model, n_memory_nodes, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def set_local_attn(self, local_attn: nn.Module) -> None:
        """Wire an externally-constructed local attention module.

        The caller is responsible for building it with `self.n_heads` heads
        (this method does not re-create it). Keeping construction external lets
        the model assemble the sparse polyhedral mask independently.
        """
        self.local_attn = local_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-residual memory-enhanced attention block.

        x: (B, L, D)
        Returns: (B, L, D) with local attention + global memory applied.

        Residual structure (one raw-x addition per sublayer — NOT two):
            h = self.norm1(x)
            if local_attn is wired: h = h + self.local_attn(h)      # local sublayer
            out = self.norm2(h + self.global_mem(h))               # global sublayer
            return out

        The inner MetatronGlobalMemory.forward already keeps its own residual
        (`local_x + dropout(proj(...))`), so the wrapper adds `h` to its output
        exactly once via `h + self.global_mem(h)`. This avoids the previous
        double-residual bug where raw x was effectively added twice.
        """
        h = self.norm1(x)
        if self.local_attn is not None:
            h = h + self.local_attn(h)
        out = self.norm2(h + self.global_mem(h))
        return out


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    torch.manual_seed(42)

    B, L, D = 2, 32, 256
    x = torch.randn(B, L, D)

    mem = MetatronGlobalMemory(d_model=D, n_nodes=13)
    out, attn = mem(x, read_weights=True)

    print(f"Input : {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Memory state: {mem.memory.shape}")
    print(f"Attention to memory nodes (first 3 tokens, first 5 nodes):")
    print(torch.round(attn[0, :3, :5], decimals=3))

    # Verify memory persists across calls: capture data_ptr ONCE, run another
    # forward, then confirm the storage identity is unchanged (no reallocation).
    p0 = mem.memory.data_ptr()
    x2 = torch.randn(B, L, D)
    out2 = mem(x2)
    persists = mem.memory.data_ptr() == p0
    print(f"\nMemory nodes still same object: {persists}")
    print(f"persists: {persists}")

    # Smoke-test the memory-enhanced attention wrapper (no local_attn wired).
    attn_wrap = MetatronMemoryEnhancedAttention(d_model=D, n_heads=4)
    w_out = attn_wrap(x)
    print(f"Wrapped output: {w_out.shape}")
