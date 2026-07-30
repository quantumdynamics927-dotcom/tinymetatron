"""
test_moe.py
===========
Core pytest suite for metatron_moe.py.

Covers the contract fixes (IMPLEMENTATION_CONTRACT.md §1):
  * last_load.sum() == top_k * B * L (aggregate over ALL k slots, not the
    loop-leftover expert_ids).
  * return_load returns a (out, counts) tuple.
  * return_aux returns a (out, counts, aux_loss) tuple with aux_loss > 0.
  * reset_load zeros the cumulative _expert_counts buffer.
  * Device: _expert_counts follows the module via .to(device); expert_utilisation
    returns a tensor on the module device (never CPU zeros).
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from config import CONFIG
from metatron_moe import MetatronMoE, MetatronGatingScore, MetatronExpert


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_moe(**kw) -> MetatronMoE:
    return MetatronMoE(
        d_model=CONFIG["d_model"],
        n_experts=CONFIG["moe_experts"],
        top_k=CONFIG["moe_top_k"],
        d_ff=CONFIG["d_ff"],
        capacity_factor=CONFIG["moe_capacity_factor"],
        **kw,
    )


def make_x(B=2, L=None, D=None) -> torch.Tensor:
    L = L or CONFIG["seq_len"]
    D = D or CONFIG["d_model"]
    torch.manual_seed(CONFIG.get("seed", 42))
    return torch.randn(B, L, D)


# ── Default forward shape ───────────────────────────────────────────────────

def test_default_forward_shape():
    """moe(x) returns (B, L, D)."""
    moe = make_moe()
    B, L = 2, CONFIG["seq_len"]
    x = make_x(B, L)
    out = moe(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, L, CONFIG["d_model"])


# ── return_load: load.sum() == top_k * B * L ─────────────────────────────────

def test_return_load_is_tuple():
    """return_load=True must return a (out, counts) tuple."""
    moe = make_moe()
    x = make_x()
    res = moe(x, return_load=True)
    assert isinstance(res, tuple) and len(res) == 2
    out, counts = res
    assert out.shape == x.shape
    assert counts.shape == (CONFIG["moe_experts"],)


def test_load_sum_equals_top_k_B_L():
    """Per-call counts over ALL k slots must sum to top_k * B * L."""
    moe = make_moe()
    B, L = 3, CONFIG["seq_len"]
    x = make_x(B, L)
    _, counts = moe(x, return_load=True)
    expected = CONFIG["moe_top_k"] * B * L
    assert counts.sum().item() == expected, (
        f"load.sum()={counts.sum().item()} expected {expected}"
    )
    # last_load must equal the returned counts and also satisfy the invariant.
    assert moe.last_load is not None
    assert moe.last_load.sum().item() == expected


def test_load_counts_are_nonneg_integers():
    """Counts are non-negative and sum to top_k*B*L."""
    moe = make_moe()
    B, L = 2, CONFIG["seq_len"]
    x = make_x(B, L)
    _, counts = moe(x, return_load=True)
    assert (counts >= 0).all()
    assert counts.sum().item() == CONFIG["moe_top_k"] * B * L


# ── return_aux: Switch-Transformer balancing loss ──────────────────────────

def test_return_aux_tuple_and_positive():
    """return_load + return_aux returns (out, counts, aux_loss) with aux > 0."""
    moe = make_moe()
    x = make_x()
    res = moe(x, return_load=True, return_aux=True)
    assert isinstance(res, tuple) and len(res) == 3
    out, counts, aux = res
    assert out.shape == x.shape
    assert counts.sum().item() == CONFIG["moe_top_k"] * x.shape[0] * x.shape[1]
    assert aux.item() > 0, f"aux_loss must be > 0, got {aux.item()}"
    assert torch.isfinite(aux)


def test_aux_loss_is_switch_transformer_form():
    """aux_loss must equal n_e * (fractions^2).sum() with fractions=counts/sum."""
    moe = make_moe()
    x = make_x()
    _, counts, aux = moe(x, return_load=True, return_aux=True)
    n_e = CONFIG["moe_experts"]
    fractions = counts / (counts.sum() + 1e-8)
    expected = n_e * fractions.pow(2).sum()
    assert torch.allclose(aux, expected, atol=1e-4), (
        f"aux={aux.item()} expected={expected.item()}"
    )


# ── reset_load zeros the cumulative buffer ──────────────────────────────────

def test_reset_load_zeros_buffer():
    """reset_load must zero _expert_counts; expert_utilisation then sums to 0."""
    moe = make_moe()
    _ = moe(make_x())  # populate cumulative buffer
    util_before = moe.expert_utilisation()
    assert util_before.sum().item() > 0  # populated
    moe.reset_load()
    util_after = moe.expert_utilisation()
    assert torch.allclose(util_after, torch.zeros_like(util_after))
    assert moe._expert_counts.abs().sum().item() == 0


def test_expert_utilisation_normalised():
    """expert_utilisation returns a normalised (sum≈1) vector on the module
    device after at least one forward."""
    moe = make_moe()
    _ = moe(make_x())
    util = moe.expert_utilisation()
    assert util.shape == (CONFIG["moe_experts"],)
    assert (util >= 0).all()
    assert util.sum().item() == pytest.approx(1.0, abs=1e-5)


# ── Device handling ─────────────────────────────────────────────────────────

def test_expert_utilisation_on_module_device():
    """expert_utilisation must live on the module's device, never CPU zeros
    when the module is on a non-CPU device (only meaningful when CUDA is
    available; on CPU this still verifies the device matches)."""
    moe = make_moe()
    _ = moe(make_x())
    util = moe.expert_utilisation()
    assert util.device == moe._expert_counts.device


def test_buffer_follows_to_device():
    """_expert_counts is a registered buffer; .to(device) must move it."""
    moe = make_moe()
    cpu_counts = moe._expert_counts
    # On a CPU-only environment, .to('cpu') is a no-op but still must return
    # a module whose buffer is on cpu.
    moe_cpu = moe.to('cpu')
    assert moe_cpu._expert_counts.device.type == 'cpu'
    # The buffer dtype is preserved.
    assert moe_cpu._expert_counts.dtype == torch.float


# ── Capacity factor reserved (no dropping) ──────────────────────────────────

def test_capacity_factor_stored_not_dropping():
    """capacity_factor is stored for API compatibility but does NOT cause
    token dropping — every token still contributes top_k assignments."""
    moe = make_moe()
    # Default CONFIG capacity_factor is stored untouched.
    assert moe.capacity_factor == CONFIG["moe_capacity_factor"]
    B, L = 2, CONFIG["seq_len"]
    x = make_x(B, L)
    _, counts = moe(x, return_load=True)
    # No dropping: sum still equals top_k * B * L.
    assert counts.sum().item() == CONFIG["moe_top_k"] * B * L


# ── Backward / gradient flow ────────────────────────────────────────────────

def test_backward_runs():
    """A backward through the MoE forward populates parameter gradients."""
    moe = make_moe()
    x = make_x()
    out, _, aux = moe(x, return_load=True, return_aux=True)
    loss = out.sum() + CONFIG["moe_aux_loss_weight"] * aux
    loss.backward()
    # At least one expert parameter must have a gradient.
    grads = [p.grad for p in moe.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


# ── Gating scorer ───────────────────────────────────────────────────────────

def test_gating_score_shape():
    """MetatronGatingScore returns (..., n_experts) logits."""
    g = MetatronGatingScore(CONFIG["d_model"], CONFIG["moe_experts"])
    x = make_x(B=2, L=8)
    logits = g(x)
    assert logits.shape == (2, 8, CONFIG["moe_experts"])


def test_expert_forward_shape():
    """A single expert is a 2-layer FFN mapping D → D."""
    e = MetatronExpert(CONFIG["d_model"], CONFIG["d_ff"])
    x = torch.randn(4, CONFIG["d_model"])
    out = e(x)
    assert out.shape == x.shape


# ── Determinism / repeated calls ─────────────────────────────────────────────

def test_repeated_forward_accumulates_counts():
    """Cumulative _expert_counts grows across calls; per-call last_load
    resets each call to the per-call counts."""
    moe = make_moe()
    x = make_x()
    _ = moe(x)
    c1 = moe._expert_counts.sum().item()
    _ = moe(x)
    c2 = moe._expert_counts.sum().item()
    assert c2 > c1, "cumulative counts should grow across calls"
    # last_load is per-call: sum equals exactly top_k*B*L.
    assert moe.last_load.sum().item() == CONFIG["moe_top_k"] * x.shape[0] * x.shape[1]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pytest.main([__file__, "-q"])