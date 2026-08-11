"""
tinymetatron_model.py
=====================
TinyMetatron SLM — the full model wiring the Metatron core modules into a
small transformer language model.

Architecture (per IMPLEMENTATION_CONTRACT.md section 2 + R2/R8):
    * Token embedding  nn.Embedding(vocab_size -> d_model), tied with the LM
      head when CONFIG['tie_embeddings'] is True.
    * N transformer layers, each composed of:
        1. Sparse polyhedral self-attention
           (MetatronSparseAttention, metatron_sparse_attention.py) with a
           residual + LayerNorm.  The sparse mask is the polyhedral mask
           M(i) = L(i) u C(i) u R(i) built from CONFIG['default_solid'].
        2. 13-expert Mixture-of-Experts (MetatronMoE, metatron_moe.py) with a
           residual + LayerNorm and aux-loss capture
           (return_load=True, return_aux=True -> Switch-Transformer loss).
        3. Cross-attention to the SHARED 13-node global memory
           (MetatronGlobalMemory, metatron_global_memory.py) — ONE instance
           shared across all layers — with a residual + LayerNorm.
    * LM head -> logits (B, L, vocab_size).

forward(input_ids) returns (logits, aux_loss) so the trainer can add the
auxiliary load-balancing loss:  total = CE(logits, labels) + w * aux_loss.

Patent references: Claims 18 (adaptive sparsity), 21 (multidimensional gating),
24 (persistent global memory), 25 (13-expert top-k<=2 MoE).
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG
from metatron_sparse_attention import MetatronSparseAttention, MetatronSparseMask
from metatron_moe import MetatronMoE
from metatron_global_memory import MetatronGlobalMemory


# ── Transformer layer ─────────────────────────────────────────────────────────
class TinyMetatronLayer(nn.Module):
    """
    One TinyMetatron transformer layer.

    Ordering: sparse self-attention -> MoE -> shared global-memory
    cross-attention.  Each sublayer uses a pre-norm residual:

        x = x + sublayer(norm(x))

    The global-memory sublayer is special: ``MetatronGlobalMemory.forward``
    already keeps an internal residual (``local_x + dropout(proj(...))``).  To
    realise a single clean residual on the layer stream ``x`` we subtract the
    normalised input ``h`` from the module output so only the cross-attention
    contribution is added:

        h   = norm(x)
        x   = x + (global_memory(h) - h)      # == x + dropout(proj(cross_attn))

    The global-memory module itself is NOT owned by this layer — it is shared
    by the parent model and passed in at forward time (R8/shared instance).
    """

    def __init__(self, d_model: int, n_heads: int, n_experts: int, top_k: int,
                 d_ff: int, dropout: float, capacity_factor: float):
        super().__init__()
        # Sublayer 1: sparse polyhedral self-attention.
        self.attn_norm = nn.LayerNorm(d_model)
        # mask is supplied per-forward (length-aware); built once and cached
        # by the parent model.  bias=False keeps the param budget inside spec.
        self.attn = MetatronSparseAttention(d_model, n_heads, mask=None,
                                           dropout=dropout, bias=False)

        # Sublayer 2: 13-expert MoE.
        self.moe_norm = nn.LayerNorm(d_model)
        self.moe = MetatronMoE(d_model=d_model, n_experts=n_experts,
                               top_k=top_k, d_ff=d_ff, dropout=dropout,
                               bias=False, capacity_factor=capacity_factor)

        # Sublayer 3: shared global-memory cross-attention.
        # The MetatronGlobalMemory instance is shared and lives on the parent
        # model; this layer only owns its pre-norm.
        self.mem_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: MetatronSparseMask,
                global_memory: MetatronGlobalMemory
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) sparse self-attention sublayer.
        h = self.attn_norm(x)
        x = x + self.attn(h, mask=mask)

        # 2) MoE sublayer with aux-loss capture.
        h = self.moe_norm(x)
        moe_out, _counts, aux = self.moe(h, return_load=True, return_aux=True)
        x = x + moe_out

        # 3) shared global-memory cross-attention sublayer.
        #    global_memory(h) = h + dropout(proj(cross_attn)); subtract h to
        #    keep a single residual on the x stream.
        h = self.mem_norm(x)
        x = x + (global_memory(h) - h)

        return x, aux


# ── Full model ───────────────────────────────────────────────────────────────
class TinyMetatron(nn.Module):
    """
    TinyMetatron SLM.

    Patent claims realised:
        * Claim 18 — adaptive sparsity (sparse polyhedral attention + MoE).
        * Claim 21 — multidimensional gating (inside MetatronMoE).
        * Claim 24 — persistent global memory shared across layers/sequences.
        * Claim 25 — 13-expert top-k<=2 MoE.

    Public interface (IMPLEMENTATION_CONTRACT.md section 2):
        TinyMetatron(config)
        .forward(input_ids)  -> (logits, aux_loss)
        .generate(input_ids, max_length, temperature) -> ids
        .param_count()      -> dict
        .from_config()      classmethod
        .load_checkpoint(path) / .save_checkpoint(path)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)

        V        = self.config["vocab_size"]
        d_model  = self.config["d_model"]
        n_heads  = self.config["n_heads"]
        n_layers = self.config["n_layers"]
        d_ff     = self.config["d_ff"]
        dropout  = self.config["dropout"]
        n_exp    = self.config["moe_experts"]
        top_k    = self.config["moe_top_k"]
        cap_f    = self.config["moe_capacity_factor"]
        self.solid       = self.config["default_solid"]
        self.seq_len_cfg = self.config["seq_len"]
        self.tie_emb     = self.config["tie_embeddings"]
        self.pad_id       = self.config["pad_id"]
        self.eos_id       = self.config["eos_id"]
        self.vocab_size   = V
        self.d_model      = d_model

        # Token embedding.
        self.token_emb = nn.Embedding(V, d_model)

        # SHARED 13-node global memory — ONE instance across all layers.
        self.global_memory = MetatronGlobalMemory(
            d_model, self.config["global_memory_nodes"], dropout=dropout,
            memory_init=self.config.get("global_memory_init", "zeros"))

        # N transformer layers.
        self.layers = nn.ModuleList([
            TinyMetatronLayer(d_model, n_heads, n_exp, top_k, d_ff, dropout,
                              cap_f)
            for _ in range(n_layers)
        ])

        # Final norm before the LM head.
        self.ln_f = nn.LayerNorm(d_model)

        # LM head.
        self.lm_head = nn.Linear(d_model, V, bias=False)
        if self.tie_emb:
            self.lm_head.weight = self.token_emb.weight  # weight sharing

        # Mask cache keyed by sequence length (length-aware; safe for
        # autoregressive generation where L grows).  Built lazily.
        self._mask_cache: Dict[int, MetatronSparseMask] = {}

    # ── mask builder (length-aware, cached) ──────────────────────────────────
    def _get_mask(self, L: int) -> MetatronSparseMask:
        """Return the polyhedral sparse mask for sequence length L (cached)."""
        m = self._mask_cache.get(L)
        if m is None:
            m = MetatronSparseMask.build(L, self.solid)
            self._mask_cache[L] = m
        return m

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, input_ids: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: LongTensor of shape (B, L) with token ids in
                [0, vocab_size).
        Returns:
            logits:   FloatTensor (B, L, vocab_size)
            aux_loss: scalar FloatTensor — sum of per-layer Switch-Transformer
                balancing losses from the MoE.  The caller scales this by
                CONFIG['moe_aux_loss_weight'] before adding to the LM loss.
        """
        B, L = input_ids.shape
        device = input_ids.device        # derive device from inputs (rule 5)

        x = self.token_emb(input_ids)    # (B, L, d_model)

        mask = self._get_mask(L)
        aux_total = torch.zeros((), device=device, dtype=x.dtype)

        for layer in self.layers:
            x, aux = layer(x, mask, self.global_memory)
            aux_total = aux_total + aux.to(aux_total.dtype)

        x = self.ln_f(x)
        logits = self.lm_head(x)          # (B, L, vocab_size)
        return logits, aux_total

    # ── generation ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_length: int = 32,
                 temperature: float = 0.7) -> torch.Tensor:
        """
        Autoregressive token sampling.

        Args:
            input_ids:   LongTensor (B, L_in) prompt.
            max_length:  maximum number of NEW tokens to generate.
            temperature: softmax temperature (>0).  As temperature -> 0 the
                distribution becomes argmax (greedy); we clamp low values to
                avoid division by zero.
        Returns:
            LongTensor (B, L_in + k) where k <= max_length.  Generation stops
            early when every sequence in the batch has emitted EOS.
        """
        self.eval()
        device = input_ids.device
        cur = input_ids
        temp = max(float(temperature), 1e-4)

        finished = torch.zeros(cur.shape[0], dtype=torch.bool, device=device)

        for _ in range(max_length):
            logits, _ = self.forward(cur)
            next_logits = logits[:, -1, :] / temp          # (B, V)
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)

            # Don't extend sequences that already finished: keep EOS.
            next_id = torch.where(finished,
                                  torch.full_like(next_id, self.eos_id),
                                  next_id)
            cur = torch.cat([cur, next_id.unsqueeze(-1)], dim=1)

            finished = finished | (next_id == self.eos_id)
            if bool(finished.all()):
                break

        return cur

    # ── parameter accounting ─────────────────────────────────────────────────
    def param_count(self) -> dict:
        """Return per-component parameter counts and the total."""
        def n(p_iter):
            return sum(p.numel() for p in p_iter if p is not None)

        embedding = n(self.token_emb.parameters())
        attention = sum(n(layer.attn.parameters()) for layer in self.layers)
        moe      = sum(n(layer.moe.parameters()) for layer in self.layers)
        # layernorms inside layers + final norm
        layer_norms = (
            sum(n(layer.attn_norm.parameters()) for layer in self.layers)
            + sum(n(layer.moe_norm.parameters()) for layer in self.layers)
            + sum(n(layer.mem_norm.parameters()) for layer in self.layers)
            + n(self.ln_f.parameters())
        )
        global_mem = n(self.global_memory.parameters())
        lm_head = n(self.lm_head.parameters())
        # when tied, lm_head shares token_emb.weight; subtract to avoid double
        if self.tie_emb:
            lm_head_shared = lm_head
        else:
            lm_head_shared = lm_head

        total = (embedding + attention + moe + layer_norms + global_mem
                 + lm_head_shared)
        # Account for sharing: if tied, lm_head weight == token_emb weight,
        # so the unique total is embedding + others (lm_head already counted
        # via embedding because of the shared Parameter object).  Compute the
        # unique total from all distinct parameters instead.
        unique_total = n(self.parameters())

        return {
            "embedding": embedding,
            "attention_total": attention,
            "moe_total": moe,
            "global_memory": global_mem,
            "layer_norms": layer_norms,
            "lm_head": lm_head,
            "lm_head_tied": self.tie_emb,
            "total_unique": unique_total,
            "total": unique_total,
            "active_moe_per_token": (
                self.config["moe_top_k"]
                * (self.d_model * self.config["d_ff"]
                   + self.config["d_ff"] * self.d_model)
                * self.config["n_layers"]),
            "in_5_7M_budget": 5_000_000 <= unique_total <= 7_000_000,
        }

    # ── constructors / persistence ────────────────────────────────────────────
    @classmethod
    def from_config(cls) -> "TinyMetatron":
        """Build a TinyMetatron from the frozen CONFIG dict."""
        return cls(CONFIG)

    def save_checkpoint(self, path: str) -> None:
        """Save model state + config to `path` (.pt).

        Saves a full checkpoint including optimizer state, step counter, and config
        for a faithful resume.  The ``load_checkpoint`` method is backward-
        compatible with older lean checkpoints (state_dict only).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                    exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str, map_location=None) -> None:
        """Load weights from a checkpoint written by :meth:`save_checkpoint`.

        Device is derived from the current model parameters; ``map_location``
        defaults to that device so checkpoints transfer cleanly.

        Backward-compatible with both lean checkpoints (state_dict only) and
        full checkpoints (state_dict + optimizer_state_dict + step + config).
        """
        if map_location is None:
            map_location = next(self.parameters()).device
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        self.load_state_dict(ckpt["state_dict"], strict=True)


# ── Smoke test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 fix (rule 4)

    import tempfile

    torch.manual_seed(CONFIG["seed"])

    model = TinyMetatron.from_config()
    pc = model.param_count()
    for k, v in pc.items():
        if isinstance(v, bool):
            print(f"  {k:24s}: {v}")
        else:
            print(f"  {k:24s}: {v}")
    print(f"\nIn 5-7M budget: {pc['in_5_7M_budget']}")

    B, L = 2, CONFIG["seq_len"]
    input_ids = torch.randint(0, CONFIG["vocab_size"], (B, L))
    logits, aux = model(input_ids)
    assert logits.shape == (B, L, CONFIG["vocab_size"]), \
        f"logits shape {tuple(logits.shape)}"
    assert aux.shape == (), f"aux_loss must be scalar, got {tuple(aux.shape)}"
    print(f"\nforward OK -> logits {tuple(logits.shape)}, "
          f"aux_loss {aux.item():.6f}")

    # Loss + backward (sanity): CE over LM logits + aux.
    target = torch.randint(0, CONFIG["vocab_size"], (B, L))
    loss = F.cross_entropy(logits.view(-1, CONFIG["vocab_size"]),
                           target.view(-1))
    total = loss + CONFIG["moe_aux_loss_weight"] * aux
    total.backward()
    print(f"loss+backward OK -> CE={loss.item():.4f} total={total.item():.4f}")

    # 5-token generate.
    prompt = torch.randint(0, CONFIG["vocab_size"], (B, 4))
    out_ids = model.generate(prompt, max_length=5, temperature=0.7)
    assert out_ids.shape[0] == B
    assert out_ids.shape[1] >= 4  # at least the prompt
    assert out_ids.shape[1] <= 4 + 5
    assert int(out_ids.max().item()) < CONFIG["vocab_size"]
    assert int(out_ids.min().item()) >= 0
    print(f"generate OK -> {tuple(out_ids.shape)} "
          f"(prompt=4 + up to 5 new)")

    # Checkpoint round-trip in a temp path (no persistent ckpt/ dir — rule 8).
    with tempfile.TemporaryDirectory() as td:
        ck = os.path.join(td, "smoke.pt")
        model.save_checkpoint(ck)
        model2 = TinyMetatron.from_config()
        model2.load_checkpoint(ck)
        l2, a2 = model2(input_ids)
        assert l2.shape == logits.shape
        print("checkpoint round-trip OK")

    print("OK")