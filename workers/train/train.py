"""
workers/train/train.py
=====================
Single-seed training worker for TinyMetatron.

Contract:
    Reads: config['input_paths'] (corpus files), config['model_config']
    Writes: config['artifact_dir']/ (checkpoints, result.json)
    Emits: structured result JSON (never writes registry)

Usage:
    python -m workers.train.train --config config.json
    python -m workers.train.train --seed 42 --steps 2000 --artifact-dir /tmp/run

Output:
    checkpoints/step*.pt      — full checkpoint files
    result.json               — structured result manifest
    stdout.txt / stderr.txt  — worker output capture
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F
from torch.optim import Adam

from config import CONFIG
from tokenizer import Tokenizer
from tinymetatron_model import TinyMetatron


# ── Result schema ───────────────────────────────────────────────────────────────

WORKER_VERSION = 1

def _result(
    status: str,
    input_hash: str,
    output_hash: str,
    artifact_paths: list[str],
    metrics: dict,
    started_at: str,
    ended_at: str,
    stdout_path: str = "",
    stderr_path: str = "",
    error: str = "",
) -> dict:
    """Build a schema-validated worker result."""
    r = {
        "worker": "workers.train.train",
        "version": WORKER_VERSION,
        "status": status,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "artifact_paths": artifact_paths,
        "metrics": metrics,
        "started_at": started_at,
        "ended_at": ended_at,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
    }
    if error:
        r["error"] = error
    return r


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    """Alias for compatibility."""
    return _hash_file(path)


# ── Training helpers ──────────────────────────────────────────────────────────

def train_step(model, batch, optimizer, pad_id, vocab_size, aux_loss_weight):
    """One gradient step. Returns (ce, aux_loss)."""
    model.train()
    logits, aux = model(batch)
    logits_flat = logits[:, :-1, :].contiguous().view(-1, vocab_size)
    labels_flat = batch[:, 1:].contiguous().view(-1)
    ce = F.cross_entropy(logits_flat, labels_flat, reduction="mean", ignore_index=pad_id)
    total = ce + aux_loss_weight * aux
    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(ce.detach()), float(aux.detach())


def compute_val_ce(model, val_rows, tokenizer, seq_len, pad_id, vocab_size,
                   batch_size, aux_loss_weight):
    """Compute validation CE over all val rows."""
    model.eval()
    total_ce, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val_rows), batch_size):
            batch_rows = val_rows[i:i + batch_size]
            encs = [tokenizer.encode(r["text"])[:seq_len] for r in batch_rows]
            max_len = max(len(e) for e in encs)
            ids = torch.full((len(encs), max_len), pad_id, dtype=torch.long)
            for j, enc in enumerate(encs):
                ids[j, :len(enc)] = torch.tensor(enc, dtype=torch.long)
            logits, _ = model(ids)
            logits_flat = logits[:, :-1, :].contiguous().view(-1, vocab_size)
            labels_flat = ids[:, 1:].contiguous().view(-1)
            mask = (labels_flat != pad_id).float()
            token_losses = F.cross_entropy(logits_flat, labels_flat, reduction="none")
            total_ce += (token_losses * mask).sum().item()
            total_tokens += mask.sum().item()
    return total_ce / total_tokens if total_tokens > 0 else 0.0


# ── Config file I/O ─────────────────────────────────────────────────────────

def load_config(path: str | None, args: argparse.Namespace) -> dict:
    """Load config from JSON file or build from CLI args."""
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    # Build from CLI args
    return {
        "seed": args.seed,
        "steps": args.steps,
        "val_every": args.val_every,
        "log_every": args.log_every,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "small_model": args.small_model,
        "artifact_dir": args.artifact_dir,
        "corpus_dir": args.corpus_dir,
        "tokenizer_path": args.tokenizer_path,
    }


# ── Main worker ──────────────────────────────────────────────────────────────

def run(config: dict) -> dict:
    """Execute the training worker. Returns a schema-validated result dict."""
    started_at = datetime.now(timezone.utc).isoformat()

    artifact_dir = Path(config["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "checkpoints").mkdir(exist_ok=True)

    # Capture stdout/stderr
    stdout_path = artifact_dir / "stdout.txt"
    stderr_path = artifact_dir / "stderr.txt"

    # Tokenizer
    tok_path = config.get("tokenizer_path", str(_ROOT / "vocab.json"))
    tok = Tokenizer.from_file(tok_path)

    # Model config
    if config.get("small_model"):
        model_config = {**CONFIG}
        model_config["n_layers"] = 3
        model_config["d_model"] = 128
        model_config["d_ff"] = 64
    else:
        model_config = dict(CONFIG)

    V = model_config["vocab_size"]
    SEQ_LEN = model_config["seq_len"]
    PAD = model_config["pad_id"]
    aux_w = float(model_config["moe_aux_loss_weight"])

    # Load corpus
    corpus_dir = Path(config.get("corpus_dir", str(_ROOT / "experiments/exp-003/corpus")))
    train_rows = [json.loads(l) for l in open(corpus_dir / "train.jsonl")]
    val_rows = [json.loads(l) for l in open(corpus_dir / "val.jsonl")]

    # Hash corpus inputs
    h = hashlib.sha256()
    for f in sorted(corpus_dir.glob("*.jsonl")):
        h.update(_hash_file(f).encode())
    input_hash = "sha256:" + h.hexdigest()[:16]

    # Model + optimizer
    seed = config["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMetatron(model_config).to(dev)
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # Encode training data
    n_train = len(train_rows)
    train_ids = [tok.encode(r["text"])[:SEQ_LEN] for r in train_rows]
    max_len = max(len(e) for e in train_ids)
    train_ids_padded = torch.zeros(n_train, max_len, dtype=torch.long).fill_(PAD)
    for i, enc in enumerate(train_ids):
        train_ids_padded[i, :len(enc)] = torch.tensor(enc, dtype=torch.long)

    # Training loop
    step = 0
    best_val_ce = float("inf")
    best_step = 0
    patience_counter = 0
    val_every = config.get("val_every", 50)
    log_every = config.get("log_every", 25)
    patience = config.get("patience", 3)
    steps = config.get("steps", 500)
    batch_size = config.get("batch_size", 16)
    epoch_idx = 0
    EPOCH_SEED = 42

    t0 = time.time()
    ended_at = started_at  # overwritten on success

    try:
        while step < steps:
            rng = torch.Generator()
            rng.manual_seed(EPOCH_SEED + epoch_idx)
            perm = torch.randperm(n_train, generator=rng)

            for i in range(0, n_train, batch_size):
                step += 1
                batch = train_ids_padded[perm[i:i + batch_size]].to(dev)
                ce, aux = train_step(model, batch, optimizer, PAD, V, aux_w)

                if step % val_every == 0 or step == steps:
                    val_ce = compute_val_ce(
                        model, val_rows, tok, SEQ_LEN, PAD, V, batch_size, aux_w)

                    ckpt_path = artifact_dir / "checkpoints" / f"step{step:06d}.pt"
                    torch.save({
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": model_config,
                        "step": step,
                        "final_loss": ce,
                        "val_loss": val_ce,
                    }, ckpt_path)

                    is_best = val_ce < best_val_ce
                    if is_best:
                        best_val_ce = val_ce
                        best_step = step
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience and step >= min(steps, 200):
                        step = steps
                        break

                if step >= steps:
                    break

            epoch_idx += 1

        # Save final checkpoint
        final_ckpt = artifact_dir / "checkpoints" / f"step{step:06d}_final.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model_config,
            "step": step,
            "final_loss": ce,
            "val_loss": best_val_ce,
        }, final_ckpt)

        elapsed = time.time() - t0
        ended_at = datetime.now(timezone.utc).isoformat()

        # Collect artifact paths
        ckpts = sorted((artifact_dir / "checkpoints").glob("step*.pt"))
        artifact_paths = [str(p) for p in ckpts]

        # Output hash
        out_h = hashlib.sha256()
        for p in ckpts:
            out_h.update(_hash_file(p).encode())
        output_hash = "sha256:" + out_h.hexdigest()[:16]

        metrics = {
            "best_val_ce": best_val_ce,
            "best_step": best_step,
            "total_steps": step,
            "elapsed_s": round(elapsed, 1),
            "seed": seed,
            "params": sum(p.numel() for p in model.parameters()),
        }

        return _result(
            status="success",
            input_hash=input_hash,
            output_hash=output_hash,
            artifact_paths=artifact_paths,
            metrics=metrics,
            started_at=started_at,
            ended_at=ended_at,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

    except Exception as exc:
        ended_at = datetime.now(timezone.utc).isoformat()
        return _result(
            status="error",
            input_hash=input_hash,
            output_hash="sha256:error",
            artifact_paths=[],
            metrics={},
            started_at=started_at,
            ended_at=ended_at,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            error=str(exc),
        )


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TinyMetatron training worker")
    parser.add_argument("--config", help="JSON config file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--small-model", action="store_true")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--corpus-dir", default=str(_ROOT / "experiments/exp-003/corpus"))
    parser.add_argument("--tokenizer-path", default=str(_ROOT / "vocab.json"))
    parser.add_argument("--result", default="result.json",
                        help="Output path for result JSON (default: result.json in artifact-dir)")
    args = parser.parse_args()

    config = load_config(args.config, args)
    result = run(config)

    result_path = Path(args.result)
    if result_path.name == "result.json" and args.artifact_dir:
        result_path = Path(args.artifact_dir) / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
