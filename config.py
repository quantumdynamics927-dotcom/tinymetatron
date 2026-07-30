"""
config.py
=========
Single source of truth for the TinyMetatron SLM configuration.

EVERY other module imports CONFIG from here. Do NOT redefine these constants
elsewhere. config.py is FROZEN except that `d_ff` may be adjusted by the model
implementer to land the total parameter count inside [5_000_000, 7_000_000]
(TMT spec: 5-7M total params, ~1M active per token via top-2/13 MoE routing).
"""

from __future__ import annotations

# ── Model architecture ──────────────────────────────────────────────────────
CONFIG = {
    # Tokenizer / vocab
    "vocab_size": 291,
    "pad_id": 0,
    "bos_id": 1,
    "eos_id": 2,
    "unk_id": 3,
    "sep_id": 4,

    # Transformer core
    "d_model": 256,
    "n_heads": 4,
    "d_k": 64,            # d_model // n_heads
    "d_ff": 112,           # TUNING KNOB: adjust so total params in [5e6, 7e6]
    "n_layers": 6,
    "seq_len": 32,
    "dropout": 0.1,
    "tie_embeddings": True,     # share input embedding & LM-head weight

    # Sparse polyhedral attention
    "default_solid": "icosahedron",
    "supported_solids": ["tetrahedron", "hexahedron", "octahedron",
                         "dodecahedron", "icosahedron"],
    "phi": 0.618,
    "sqrt2": 1.414,
    "comm_schedule": ["radial", "chord", "ring"],

    # Mixture of Experts (Metatron's Cube: 13 spheres)
    "moe_experts": 13,
    "moe_top_k": 2,
    "moe_capacity_factor": 1.25,
    "moe_aux_loss": True,
    "moe_aux_loss_weight": 0.01,
    "moe_gating_dims": ["tokens", "latency", "graphity"],
    "moe_layers": "all",     # "all" => MoE in every layer; or a list of layer idx

    # Global memory (Metatron's Cube: 13 nodes), SHARED across all layers
    "global_memory_nodes": 13,
    "global_memory_init": "zeros",
    "global_memory_shared": True,

    # Training
    "batch_size": 16,
    "learning_rate": 1e-3,
    "seed": 42,
    "log_every": 10,

    # Paths / runtime
    "db_path": "metatron.db",
    "checkpoint_dir": "ckpt",
    "data_dir": "data",
    "vocab_path": "vocab.json",
    "host": "0.0.0.0",
    "port": 8010,
    "device": "cpu",        # "cpu" | "cuda"

    # Generation
    "default_temperature": 0.7,
    "default_max_length": 100,

    # Distributed sharding (patent component; NOT used by single-CPU forward path)
    "shard_topology": "dodecahedron",
    "shard_n_devices": 1,
    "shard_bandwidth_bps": 1e9,
}


def get_config() -> dict:
    """Return the frozen CONFIG dict (copy so callers cannot mutate the original)."""
    return dict(CONFIG)


# ── Parameter estimator (used to size d_ff so total lands in 5-7M) ──────────
def estimate_total_params(d_ff: int = None, n_layers: int = None) -> dict:
    """
    Rough parameter estimate for the TinyMetatron architecture described in
    IMPLEMENTATION_CONTRACT.md (standard MHA projections + shared 13-node
    global memory + per-layer 13-expert MoE + tied embeddings).

    Returns a dict with per-component and total counts.
    """
    c = get_config()
    d_model = c["d_model"]
    n_heads = c["n_heads"]
    d_k = c["d_k"]
    V = c["vocab_size"]
    experts = c["moe_experts"]
    nL = n_layers if n_layers is not None else c["n_layers"]
    dff = d_ff if d_ff is not None else c["d_ff"]

    emb = V * d_model if not c["tie_embeddings"] else V * d_model  # tied: counted once
    # attention: 4 standard projections Linear(d_model,d_model)
    attn_per_layer = 4 * d_model * d_model
    # global memory: 13*d_model memory + gate Linear(d,d) + proj Linear(d,d)
    # shared across layers => counted once
    memory = c["global_memory_nodes"] * d_model + 2 * d_model * d_model
    # MoE per layer: experts * (Linear(d,d_ff) + Linear(d_ff,d))
    moe_per_layer = experts * (d_model * dff + dff * d_model)
    # layernorms: ~2 per layer
    norms_per_layer = 2 * 2 * d_model

    attn_total = attn_per_layer * nL
    moe_total = moe_per_layer * nL
    norms_total = norms_per_layer * nL

    total = emb + attn_total + memory + moe_total + norms_total
    return {
        "embedding": emb,
        "attention_total": attn_total,
        "global_memory": memory,
        "moe_total": moe_total,
        "norms_total": norms_total,
        "total": total,
        "active_moe_per_token": c["moe_top_k"] * (d_model * dff + dff * d_model) * nL,
        "in_5_7M_budget": 5_000_000 <= total <= 7_000_000,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    est = estimate_total_params()
    for k, v in est.items():
        print(f"  {k:24s}: {v}")
    print(f"\nIn 5-7M budget: {est['in_5_7M_budget']}")
    if not est["in_5_7M_budget"]:
        # suggest a d_ff that lands mid-budget (~6.25M)
        target = 6_250_000
        c = get_config()
        fixed = (est["embedding"] + est["attention_total"] + est["global_memory"]
                 + est["norms_total"])
        # moe_total = nL * experts * 2 * d_model * d_ff
        denom = c["n_layers"] * c["moe_experts"] * 2 * c["d_model"]
        sug = max(1, (target - fixed) // denom)
        print(f"Suggested d_ff to hit ~6.25M: {sug}")