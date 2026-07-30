"""
tests/test_model.py
==================
ECOSYSTEM pytest suite for the TinyMetatron SLM (contract section 2).

Covers (contract section 5 / test descriptions):
    * forward(input_ids[B,32]) -> logits shape (B, 32, 291)
    * total_params in [5e6, 7e6] via param_count()
    * generate(...) returns ids in [0, 291)
    * one forward + backward produces a finite loss
"""

from __future__ import annotations

import os
import sys
import math

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch
import torch.nn.functional as F

from config import CONFIG
from tinymetatron_model import TinyMetatron


@pytest.fixture(scope="module")
def model() -> TinyMetatron:
    torch.manual_seed(CONFIG["seed"])
    m = TinyMetatron(CONFIG)
    return m


# ── forward shape ─────────────────────────────────────────────────────────────
def test_forward_shape(model: TinyMetatron) -> None:
    B, L = 2, CONFIG["seq_len"]  # seq_len == 32
    input_ids = torch.randint(0, CONFIG["vocab_size"], (B, L))
    logits, aux = model(input_ids)
    assert logits.shape == (B, L, CONFIG["vocab_size"]), (
        f"logits shape {tuple(logits.shape)} != {(B, L, CONFIG['vocab_size'])}")
    assert aux.shape == (), f"aux_loss must be scalar, got {tuple(aux.shape)}"


# ── parameter budget ──────────────────────────────────────────────────────────
def test_param_count_in_budget(model: TinyMetatron) -> None:
    pc = model.param_count()
    total = pc["total"]
    assert isinstance(total, int)
    assert 5_000_000 <= total <= 7_000_000, (
        f"total params {total} outside [5e6, 7e6]")
    assert pc["in_5_7M_budget"] is True
    # active MoE per token must be a small fraction of the total.
    assert pc["active_moe_per_token"] < total


# ── generate returns ids in [0, vocab_size) ────────────────────────────────────
def test_generate_ids_in_vocab(model: TinyMetatron) -> None:
    B = 2
    prompt_len = 4
    prompt = torch.randint(0, CONFIG["vocab_size"], (B, prompt_len))
    out = model.generate(prompt, max_length=5, temperature=0.7)
    assert out.shape[0] == B
    # at least the prompt, at most prompt + max_length
    assert prompt_len <= out.shape[1] <= prompt_len + 5
    assert int(out.min().item()) >= 0
    assert int(out.max().item()) < CONFIG["vocab_size"]


# ── forward + backward produces finite loss ───────────────────────────────────
def test_forward_backward_finite_loss(model: TinyMetatron) -> None:
    B, L = 2, CONFIG["seq_len"]
    input_ids = torch.randint(0, CONFIG["vocab_size"], (B, L))
    target = torch.randint(0, CONFIG["vocab_size"], (B, L))

    logits, aux = model(input_ids)
    ce = F.cross_entropy(logits.view(-1, CONFIG["vocab_size"]),
                         target.view(-1))
    total = ce + CONFIG["moe_aux_loss_weight"] * aux
    assert math.isfinite(float(total.item())), (
        f"total loss not finite: {total.item()}")

    total.backward()
    # At least one parameter must have a non-None gradient after backward.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "no gradients produced after backward"
    # All present grads must be finite.
    for g in grads:
        assert torch.isfinite(g).all().item(), "non-finite gradient encountered"


# ── from_config builds an equivalent model ────────────────────────────────────
def test_from_config(model: TinyMetatron) -> None:
    m2 = TinyMetatron.from_config()
    pc2 = m2.param_count()
    assert pc2["total"] == model.param_count()["total"]
    assert 5_000_000 <= pc2["total"] <= 7_000_000