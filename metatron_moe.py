"""
metatron_moe.py
===============
Mixture-of-Experts layer with 13 specialized experts and top-K (K=2) routing.
Top-K routing uses a multidimensional gating scorer (tokens, latency, graphity).

Matches the disclosure in TMT Patent Draft §7.1 / Claim 25:
    "MoE de 13 expertos y top‑k≤2 activos"
    "gating scorer multidimensional (tokens, latencia, graphity, consenso)"

13 experts correspond to the 13 polyhedral nodes (Metatron's cube motif).

Load tracking
-------------
Two parallel counters are maintained:

* ``last_load``  — per-call expert assignment counts over ALL k slots, i.e.
  ``counts = bincount(topk_idx.reshape(-1), minlength=n_e)``. By construction
  ``last_load.sum() == top_k * B * L`` (every token contributes k assignments).
  This is the source of truth for the Switch-Transformer auxiliary load-balancing
  loss.

* ``_expert_counts`` — cumulative across calls (a registered buffer that follows
  the module's device via ``.to(device)``). Exposed through
  :meth:`expert_utilisation` and resettable via :meth:`reset_load`.

``capacity_factor`` is stored as a constructor argument for API compatibility but
is RESERVED — no token dropping is implemented. Expert capacity is effectively
unbounded; the value is documented here so future work can wire it in without
changing the signature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple

from config import CONFIG


class MetatronGatingScore(nn.Module):
    """
    Multidimensional gating scorer.
    Patent Claim 21: "gating scorer multidimensional (tokens, latencia, graphity, consenso)"

    Each dimension is a learned linear projection; scores are combined
    before the top-k selection.
    """

    def __init__(self, d_model: int, n_experts: int = 13,
                 bias: bool = True):
        super().__init__()
        self.n_experts = n_experts
        self.d_model   = d_model

        # One linear projection per gating dimension
        self.gate_tokens = nn.Linear(d_model, n_experts, bias=bias)
        self.gate_latency = nn.Linear(d_model, n_experts, bias=False)
        self.gate_graphity = nn.Linear(d_model, n_experts, bias=False)

        # Learnable dimension weights (consensus)
        self.dim_weight = nn.Parameter(torch.ones(3) / 3)  # softmax→normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D) or (B, D)
        Returns: (..., n_experts) raw gating logits
        """
        s_tokens    = self.gate_tokens(x)
        s_latency   = self.gate_latency(x)
        s_graphity  = self.gate_graphity(x)

        # Normalize dimension weights to sum=1
        w = F.softmax(self.dim_weight, dim=0)
        combined = w[0]*s_tokens + w[1]*s_latency + w[2]*s_graphity
        return combined


class MetatronExpert(nn.Module):
    """
    One expert in the MoE stack.
    Each expert is a 2-layer FFN with GeLU activation — standard MoE sublayer.
    """

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or (d_model * 4)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetatronMoE(nn.Module):
    """
    Mixture-of-Experts with 13 specialized experts, top-K=2 routing.

    Patent Claim 25: "MoE de 13 expertos y top‑k≤2 activos"
    Patent Claim 18:  "sparsity adaptativa controlada por carga"

    The load-balancing term (auxiliary loss) encourages equal expert utilisation,
    which in turn maximises the effective sparsity of the routing.

    Usage:
        moe = MetatronMoE(d_model=256, n_experts=13, top_k=2)
        out = moe(x)                              # (B, L, D)
        out, load = moe(x, return_load=True)      # load.sum() == top_k*B*L
        out, load, aux = moe(x, return_load=True, return_aux=True)
    """

    def __init__(self, d_model: int, n_experts: int = 13,
                 top_k: int = 2, d_ff: int = None,
                 dropout: float = 0.1,
                 bias: bool = False,
                 capacity_factor: float = 1.25):
        super().__init__()
        assert top_k <= n_experts
        self.n_experts      = n_experts
        self.top_k          = top_k
        # RESERVED: expert capacity multiplier. Stored for API compatibility but
        # NOT used — no token dropping is implemented. Expert capacity is
        # effectively unbounded; see module docstring.
        self.capacity_factor = capacity_factor

        self.gating = MetatronGatingScore(d_model, n_experts, bias=bias)
        self.experts = nn.ModuleList([
            MetatronExpert(d_model, d_ff, dropout)
            for _ in range(n_experts)
        ])

        # Learnable bias for load-aware sparsity (Claim 18)
        self.load_bias = nn.Parameter(torch.zeros(n_experts))

        # Registered buffer: cumulative expert assignment counts across calls.
        # Follows the module device via .to(device). Resettable via reset_load().
        self.register_buffer('_expert_counts',
                             torch.zeros(n_experts, dtype=torch.float))
        # Per-call counts (source of truth for aux loss). Set on every forward.
        self.last_load: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor,
                return_load: bool = False,
                return_aux: bool = False
                ) -> Union[torch.Tensor,
                           Tuple[torch.Tensor, torch.Tensor],
                           Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        x: (B, L, D)
        Returns:
            * ``out``            (B, L, D)                       by default
            * ``(out, counts)``                                  if return_load
            * ``(out, counts, aux_loss)``                        if both flags

        ``counts`` are per-call assignment counts over ALL k slots, so
        ``counts.sum() == top_k * B * L``. ``aux_loss`` is the Switch-Transformer
        balancing loss ``n_e * (fractions).pow(2).sum()`` with
        ``fractions = counts / counts.sum()``.
        """
        B, L, D   = x.shape
        n_e       = self.n_experts
        k         = self.top_k
        flat_x    = x.view(-1, D)                     # (B*L, D)

        # ── Gating ──────────────────────────────────────────────────
        gate_logits = self.gating(flat_x)             # (B*L, n_experts)
        gate_logits = gate_logits + self.load_bias     # load-aware adjustment

        # ── Top-K selection ────────────────────────────────────────
        topk_vals, topk_idx = torch.topk(gate_logits, k, dim=-1)  # (B*L, k)
        topk_vals = F.softmax(topk_vals, dim=-1)       # normalize per-token

        # ── Dispatch to experts ────────────────────────────────────
        out = torch.zeros_like(flat_x)                 # (B*L, D)

        for ki in range(k):
            expert_ids = topk_idx[:, ki]               # (B*L,)
            weights    = topk_vals[:, ki]              # (B*L,)
            # Accumulate expert contributions
            for e_id in range(n_e):
                mask_e = (expert_ids == e_id)          # (B*L,)
                count  = mask_e.sum().item()
                if count == 0:
                    continue
                # Gather tokens for this expert
                tok_i   = mask_e.nonzero(as_tuple=True)[0]
                w       = weights[tok_i].unsqueeze(-1) # (n_tok, 1)
                inp_e   = flat_x[tok_i]               # (n_tok, D)
                out_e   = self.experts[e_id](inp_e)   # (n_tok, D)
                out[tok_i] += w * out_e

        # ── Load tracking over ALL k slots ─────────────────────────
        # Aggregate across every selected slot so that
        #   counts.sum() == top_k * B * L
        # (do NOT reference the loop-leftover `expert_ids` which only covers
        # the last ki iteration).
        all_idx = topk_idx.reshape(-1)                 # (B*L * k,)
        counts = torch.bincount(all_idx, minlength=n_e).float()

        # Per-call counts (source of truth for aux loss)
        self.last_load = counts
        # Cumulative counts on the buffer's device (follows module via .to)
        self._expert_counts = self._expert_counts + counts.to(
            self._expert_counts.device)

        # ── Switch-Transformer auxiliary balancing loss ────────────
        # aux_loss = n_e * sum(fractions^2),  fractions = counts / counts.sum()
        aux_loss: Optional[torch.Tensor] = None
        if return_aux:
            fractions = counts / (counts.sum() + 1e-8)
            aux_loss = n_e * fractions.pow(2).sum()

        # ── Return shape ────────────────────────────────────────────
        out = out.view(B, L, D)
        if return_load and return_aux:
            return out, counts, aux_loss
        if return_load:
            return out, counts
        return out

    def reset_load(self) -> None:
        """Zero the cumulative ``_expert_counts`` buffer in place."""
        with torch.no_grad():
            self._expert_counts.zero_()

    def expert_utilisation(self) -> torch.Tensor:
        """
        Returns normalised cumulative expert utilisation (0–1) on the module
        device (never CPU zeros). Computed from the cumulative buffer so the
        value is meaningful even before any forward has set ``last_load``.
        """
        total = self._expert_counts.sum() + 1e-8
        return self._expert_counts / total


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    torch.manual_seed(CONFIG["seed"])

    B, L, D = 2, CONFIG["seq_len"], CONFIG["d_model"]
    top_k   = CONFIG["moe_top_k"]
    n_e     = CONFIG["moe_experts"]
    x = torch.randn(B, L, D)

    moe = MetatronMoE(d_model=D, n_experts=n_e, top_k=top_k,
                      d_ff=CONFIG["d_ff"],
                      capacity_factor=CONFIG["moe_capacity_factor"])

    # Default forward
    out = moe(x)
    assert out.shape == (B, L, D), f"expected {(B,L,D)}, got {tuple(out.shape)}"

    # return_load: per-call counts must sum to top_k*B*L
    out2, load = moe(x, return_load=True)
    assert out2.shape == (B, L, D)
    expected = top_k * B * L
    got = load.sum().item()
    assert got == expected, f"load.sum()={got} expected {expected}"

    # return_aux: Switch-Transformer balancing loss, strictly positive
    out3, load3, aux = moe(x, return_load=True, return_aux=True)
    assert out3.shape == (B, L, D)
    assert load3.sum().item() == expected
    assert aux.item() > 0, f"aux_loss must be > 0, got {aux.item()}"

    # reset_load zeros the cumulative buffer
    moe.reset_load()
    util = moe.expert_utilisation()
    assert util.shape == (n_e,)
    assert torch.allclose(util, torch.zeros_like(util))

    # Re-run to populate utilisation on the module device
    _ = moe(x)
    util = moe.expert_utilisation()
    assert util.device == moe._expert_counts.device

    print(f"Input  : {tuple(x.shape)}")
    print(f"Output : {tuple(out.shape)}")
    print(f"load.sum()       == top_k*B*L : {got} == {expected}")
    print(f"aux_loss         : {aux.item():.6f}  (>0)")
    print(f"Expert utilisation (cumulative, last call):")
    print(util.detach().cpu().tolist())
    print(f"Active experts   : {(util > 0).sum().item()} / {n_e}")
    print("OK")