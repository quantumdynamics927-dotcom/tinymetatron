"""
workers/evaluate/per_row_loss.py
================================
Compute per-row CE breakdown to identify memorized vs novel rows.

Contract:
    Reads: config['corpus_paths'], config['checkpoint_path']
    Writes: config['artifact_dir']/ (result.json)
    Emits: structured result JSON with row-level CE data

Usage:
    python -m workers.evaluate.per_row_loss --run-id exp-004-train-seed42 --step 500 --eval-set hard_dev
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F

from config import CONFIG
from tokenizer import Tokenizer
from tinymetatron_model import TinyMetatron

WORKER_VERSION = 1


def _result(status, input_hash, output_hash, artifact_paths, metrics,
            started_at, ended_at, error=""):
    r = {
        "worker": "workers.evaluate.per_row_loss",
        "version": WORKER_VERSION,
        "status": status,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "artifact_paths": artifact_paths,
        "metrics": metrics,
        "started_at": started_at,
        "ended_at": ended_at,
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


EVAL_SETS = {
    "val": "val.jsonl",
    "hard_dev": "test_final.jsonl",
    "novel_eval": "novel_eval.jsonl",
    "test_final": "test_final.jsonl",
}


def load_eval_rows(eval_set: str, corpus_dir: Path) -> list[dict]:
    filename = EVAL_SETS.get(eval_set, eval_set + ".jsonl")
    path = corpus_dir / filename
    return [json.loads(l) for l in open(path)]


def per_row_ce(
    checkpoint_path: str,
    eval_rows: list[dict],
    tokenizer_path: str,
    seq_len: int,
    pad_id: int,
    vocab_size: int,
) -> list[dict]:
    """Compute CE per row."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TinyMetatron.from_config()
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    tok = Tokenizer.from_file(tokenizer_path)
    results = []

    with torch.no_grad():
        for row in eval_rows:
            enc = tok.encode(row["text"])[:seq_len]
            ids = torch.tensor([enc], dtype=torch.long)
            logits, _ = model(ids)
            logits_flat = logits[0, :-1, :]  # [L-1, V]
            labels_flat = ids[0, 1:]            # [L-1]
            mask = (labels_flat != pad_id).float()
            token_losses = F.cross_entropy(logits_flat, labels_flat, reduction="none")
            ce = (token_losses * mask).sum().item() / max(mask.sum().item(), 1)
            results.append({
                "subdomain": row.get("subdomain", ""),
                "family": row.get("family", ""),
                "ce": round(ce, 6),
                "ppl": round(math.exp(ce), 4) if ce > 0 else 1.0,
                "tokens": mask.sum().item(),
                "text": row["text"][:80],
            })

    return results


def run(config: dict) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()

    eval_set = config["eval_set"]
    corpus_dir = Path(config["corpus_dir"])
    checkpoint_path = config["checkpoint_path"]
    tokenizer_path = config.get("tokenizer_path", str(_ROOT / "vocab.json"))
    seq_len = config.get("seq_len", CONFIG["seq_len"])
    pad_id = config.get("pad_id", CONFIG["pad_id"])
    vocab_size = config.get("vocab_size", CONFIG["vocab_size"])

    artifact_dir = Path(config.get("artifact_dir", "."))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    ckpt_hash = _hash_file(Path(checkpoint_path))
    tok_hash = _hash_file(Path(tokenizer_path))
    input_hash = f"sha256:{ckpt_hash[:8]}_{tok_hash[:8]}"

    try:
        eval_rows = load_eval_rows(eval_set, corpus_dir)
        rows = per_row_ce(checkpoint_path, eval_rows, tokenizer_path,
                          seq_len, pad_id, vocab_size)

        avg_ce = sum(r["ce"] for r in rows) / len(rows)
        metrics = {
            "avg_ce": round(avg_ce, 6),
            "avg_ppl": round(math.exp(avg_ce), 4),
            "num_rows": len(rows),
            "by_subdomain": _aggregate_by(rows, "subdomain"),
            "by_family": _aggregate_by(rows, "family"),
        }

        result_path = artifact_dir / "result.json"
        result = _result(
            status="success",
            input_hash=input_hash,
            output_hash="sha256:" + ckpt_hash[:16],
            artifact_paths=[str(result_path)],
            metrics=metrics,
            started_at=started_at,
            ended_at=datetime.now(timezone.utc).isoformat(),
        )

        # Include per-row data in the result for the loop to store
        result["per_row"] = rows

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as exc:
        return _result(
            status="error",
            input_hash=input_hash,
            output_hash="sha256:error",
            artifact_paths=[],
            metrics={},
            started_at=started_at,
            ended_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


def _aggregate_by(rows: list[dict], key: str) -> dict:
    groups = {}
    for r in rows:
        k = r.get(key, "(unknown)")
        if k not in groups:
            groups[k] = []
        groups[k].append(r["ce"])
    return {k: round(sum(v) / len(v), 4) for k, v in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--eval-set", required=True,
                        choices=["val", "hard_dev", "novel_eval", "test_final"])
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--corpus-dir",
                        default=str(_ROOT / "experiments/exp-003/corpus"))
    parser.add_argument("--tokenizer-path", default=str(_ROOT / "vocab.json"))
    parser.add_argument("--artifact-dir", default=".")
    parser.add_argument("--result", default=None)
    args = parser.parse_args()

    config = vars(args)
    result = run(config)

    out_path = Path(args.result) if args.result else \
        Path(args.artifact_dir) / "result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"avg_ce={result['metrics']['avg_ce']:.4f}  "
          f"avg_ppl={result['metrics']['avg_ppl']:.2f}  "
          f"rows={result['metrics']['num_rows']}")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
