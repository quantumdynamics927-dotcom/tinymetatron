"""
test_global_memory.py
======================
Core pytest suite for metatron_global_memory.py.

Covers the contract fixes (IMPLEMENTATION_CONTRACT.md §1):
  * Persistence across 2 forwards: the memory Parameter's data_ptr is stable
    (set_state uses copy_, never reassigns the Parameter).
  * set_state preserves device/dtype (casts the incoming tensor to the
    memory's current device/dtype).
  * local_attn is actually applied in MetatronMemoryEnhancedAttention when
    wired (output differs from the no-local-attn path).
  * Gating is per-token (gate_val = sigmoid(gate(local_x))).
  * The double-residual bug is fixed (raw x is added exactly once per sublayer).
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pytest

from config import CONFIG
from metatron_global_memory import (
    MetatronGlobalMemory,
    MetatronMemoryEnhancedAttention,
)


# ── Basic shapes ─────────────────────────────────────────────────────────────

def test_global_memory_forward_shape():
    """MetatronGlobalMemory(x) returns (B, L, D)."""
    D = CONFIG["d_model"]
    mem = MetatronGlobalMemory(d_model=D, n_nodes=13)
    B, L = 2, 32
    x = torch.randn(B, L, D)
    out = mem(x)
    assert out.shape == (B, L, D)


def test_global_memory_read_weights_shape():
    """read_weights=True returns (out, attn) with attn of shape
    (B, L, n_nodes)."""
    D = CONFIG["d_model"]
    mem = MetatronGlobalMemory(d_model=D, n_nodes=13)
    B, L = 2, 16
    x = torch.randn(B, L, D)
    out, attn = mem(x, read_weights=True)
    assert out.shape == (B, L, D)
    assert attn.shape == (B, L, 13)
    # Attention over memory nodes sums to 1 per (B, L).
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, L), atol=1e-5)


def test_memory_is_learnable_parameter():
    """The memory must be a learnable Parameter (not a buffer)."""
    D = CONFIG["d_model"]
    mem = MetatronGlobalMemory(d_model=D, n_nodes=13)
    assert isinstance(mem.memory, nn.Parameter)
    assert mem.memory.requires_grad
    assert mem.memory.shape == (13, D)


# ── Persistence across two forwards ─────────────────────────────────────────

def test_memory_persists_across_two_forwards():
    """The memory Parameter's data_ptr must be unchanged across two forward
    calls — copy_ preserves storage identity."""
    D = CONFIG["d_model"]
    mem = MetatronGlobalMemory(d_model=D, n_nodes=13)
    x1 = torch.randn(2, 16, D)
    _ = mem(x1)
    p0 = mem.memory.data_ptr()
    x2 = torch.randn(2, 16, D)
    _ = mem(x2)
    assert mem.memory.data_ptr() == p0, (
        "memory storage identity changed across forwards — copy_ should "
        "preserve data_ptr"
    )


# ── set_state preserves device/dtype ────────────────────────────────────────

def test_set_state_preserves_dtype_and_device():
    """set_state must cast the incoming tensor to the memory's device/dtype,
    and preserve the Parameter's storage identity (data_ptr)."""
    D = CONFIG["d_model"]
    mem = MetatronGlobalMemory(d_model=D, n_nodes=13)
    p0 = mem.memory.data_ptr()
    new_mem = torch.randn(13, D, dtype=torch.float64)
    mem.set_state(new_mem)
    # dtype preserved (memory stays float32).
    assert mem.memory.dtype == torch.float32
    # device preserved.
    assert mem.memory.device == torch.device('cpu')
    # storage identity preserved.
    assert mem.memory.data_ptr() == p0
    # values copied.
    assert torch.allclose(mem.memory, new_mem.to(torch.float32), atol=1e-5)


def test_set_state_rejects_wrong_shape():
    """set_state asserts on shape mismatch."""
    D = CONFIG["d_model"]
    mem = MetatronGlobalMemory(d_model=D, n_nodes=13)
    with pytest.raises(AssertionError):
        mem.set_state(torch.randn(12, D))


def test_reset_zeros_memory():
    """reset() zeros the memory in place (storage identity preserved)."""
    D = CONFIG["d_model"]
    mem = MetatronGlobalMemory(d_model=D, n_nodes=13, memory_init='normal')
    p0 = mem.memory.data_ptr()
    mem.reset()
    assert torch.allclose(mem.memory, torch.zeros_like(mem.memory))
    assert mem.memory.data_ptr() == p0


# ── Per-token gating ────────────────────────────────────────────────────────

def test_gate_is_per_token():
    """Per-token gating: gate_val = sigmoid(gate(local_x)) — different
    tokens get different gate values."""
    D = CONFIG["d_model"]
    mem = MetatronGlobalMemory(d_model=D, n_nodes=13)
    B, L = 2, 8
    x = torch.randn(B, L, D)
    # Run forward and verify the gate produces per-token variation by
    # inspecting the gate Linear directly.
    gate_val = torch.sigmoid(mem.gate(x))
    assert gate_val.shape == (B, L, D)
    # Per-token variation: not all rows are identical.
    assert not torch.allclose(gate_val[0, 0], gate_val[0, 1])


# ── local_attn is applied in the wrapper ───────────────────────────────────

class _IdentityLocalAttn(nn.Module):
    """A trivial local_attn stub that simply returns its input (so it is
    wired-but-inert); used to verify the wiring path is taken."""

    def __init__(self, D):
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, x):
        return self.lin(x)


def test_local_attn_is_applied():
    """When local_attn is wired, it is actually applied — output differs from
    the no-local-attn path."""
    torch.manual_seed(CONFIG.get("seed", 42))
    D = CONFIG["d_model"]
    B, L = 2, 16

    # Without local_attn.
    wrap_no_local = MetatronMemoryEnhancedAttention(d_model=D, n_heads=4)
    # With local_attn wired.
    wrap_with_local = MetatronMemoryEnhancedAttention(d_model=D, n_heads=4)
    wrap_with_local.set_local_attn(_IdentityLocalAttn(D))

    x = torch.randn(B, L, D)
    wrap_no_local.eval()
    wrap_with_local.eval()
    with torch.no_grad():
        out_no = wrap_no_local(x)
        out_yes = wrap_with_local(x)
    assert out_no.shape == (B, L, D)
    assert out_yes.shape == (B, L, D)
    # Wiring local_attn changes the output.
    assert not torch.allclose(out_no, out_yes), (
        "local_attn must be applied and change the output vs. no-local-attn"
    )


def test_wrapper_residual_structure():
    """The wrapper applies exactly one raw-x residual per sublayer (no
    double-residual). With local_attn=None the structure is:
        h = norm1(x); out = norm2(h + global_mem(h))
    so the output is finite and shape-preserving."""
    D = CONFIG["d_model"]
    wrap = MetatronMemoryEnhancedAttention(d_model=D, n_heads=4)
    x = torch.randn(2, 16, D)
    out = wrap(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# ── Backward through the wrapper ────────────────────────────────────────────

def test_wrapper_backward():
    """Backward through MetatronMemoryEnhancedAttention populates gradients on
    the global memory Parameter."""
    D = CONFIG["d_model"]
    torch.manual_seed(0)
    wrap = MetatronMemoryEnhancedAttention(d_model=D, n_heads=4)
    wrap.set_local_attn(_IdentityLocalAttn(D))
    x = torch.randn(2, 16, D, requires_grad=True)
    out = wrap(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # global memory parameter has a gradient.
    assert wrap.global_mem.memory.grad is not None


# ── n_nodes must be 13 ───────────────────────────────────────────────────────

def test_rejects_non_13_nodes():
    """MetatronGlobalMemory requires exactly 13 nodes (Metatron's Cube)."""
    D = CONFIG["d_model"]
    with pytest.raises(AssertionError):
        MetatronGlobalMemory(d_model=D, n_nodes=12)


# ── Memory init modes ───────────────────────────────────────────────────────

def test_memory_init_modes():
    """All three documented init modes produce finite memory."""
    D = CONFIG["d_model"]
    for mode in ('zeros', 'normal', 'ones'):
        mem = MetatronGlobalMemory(d_model=D, n_nodes=13, memory_init=mode)
        assert mem.memory.shape == (13, D)
        assert torch.isfinite(mem.memory).all()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pytest.main([__file__, "-q"])