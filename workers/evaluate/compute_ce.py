"""
workers/evaluate/compute_ce.py
=============================
Compute CE/PPL over a named evaluation set.

Contract:
    Reads: config['corpus_paths'], config['checkpoint_path']
    Writes: config['artifact_dir']/ (result.json)
    Emits: structured result JSON with ce, ppl, total_tokens

Usage:
    python -m workers.evaluate.compute_ce --run-id exp-004-train-seed42 --step 500 --eval-set val
    python -m workers.evaluate.compute_ce --run-id exp-004-train-seed42 --step 500 --eval-set hard_dev
    python -m workers.evaluate.compute_ce --run-id exp-004-train-seed42 --step 500 --eval-set novel_eval
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
            started_at, ended_at, stdout_path="", stderr_path="", error=""):
    r = {
        "worker": "workers.evaluate.compute_ce",
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


EVAL_SETS = {
    "val": "val.jsonl",
    "hard_dev": "test_final.jsonl",
    "novel_eval": "novel_eval.jsonl",
    "test_final": "test_final.jsonl",
}


def load_eval_rows(eval_set: str, corpus_dir: Path) -> list[dict]:
    """Load rows for a named evaluation set."""
    filename = EVAL_SETS.get(eval_set, eval_set + ".jsonl")
    path = corpus_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found: {path}")
    return [json.loads(l) for l in open(path)]


def compute_ce(
    checkpoint_path: str,
    eval_rows: list[dict],
    tokenizer_path: str,
    seq_len: int,
    pad_id: int,
    vocab_size: int,
    batch_size: int = 32,
) -> tuple[float, float, int]:
    """
    Compute per-token CE over eval_rows.

    Returns (ce, ppl, total_tokens).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TinyMetatron.from_config()
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    tok = Tokenizer.from_file(tokenizer_path)
    total_ce, total_tokens = 0.0, 0

    with torch.no_grad():
        for i in range(0, len(eval_rows), batch_size):
            batch = eval_rows[i:i + batch_size]
            encs = [tok.encode(r["text"])[:seq_len] for r in batch]
            max_len = max(len(e) for e in encs)
            ids = torch.full((len(encs), max_len), pad_id, dtype=torch.long)
            for j, enc in enumerate(encs):
                ids[j, :len(enc)] = torch.tensor(enc, dtype=torch.long)

            logits, _ = model(ids)
            logits_flat = logits[:, :-1, :].reshape(-1, vocab_size)
            labels_flat = ids[:, 1:].reshape(-1)
            mask = (labels_flat != pad_id).float()
            token_losses = F.cross_entropy(logits_flat, labels_flat, reduction="none")
            total_ce += (token_losses * mask).sum().item()
            total_tokens += mask.sum().item()

    ce = total_ce / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(ce)
    return ce, ppl, total_tokens


def run(config: dict) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()

    eval_set = config["eval_set"]
    corpus_dir = Path(config["corpus_dir"])
    checkpoint_path = config["checkpoint_path"]
    tokenizer_path = config.get("tokenizer_path", str(_ROOT / "vocab.json"))
    seq_len = config.get("seq_len", CONFIG["seq_len"])
    pad_id = config.get("pad_id", CONFIG["pad_id"])
    vocab_size = config.get("vocab_size", CONFIG["vocab_size"])
    batch_size = config.get("batch_size", 32)

    artifact_dir = Path(config.get("artifact_dir", "."))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Hash inputs
    ckpt_hash = _hash_file(Path(checkpoint_path))
    tok_hash = _hash_file(Path(tokenizer_path))
    input_hash = f"sha256:{ckpt_hash[:8]}_{tok_hash[:8]}"

    try:
        eval_rows = load_eval_rows(eval_set, corpus_dir)
        ce, ppl, total_tokens = compute_ce(
            checkpoint_path, eval_rows, tokenizer_path,
            seq_len, pad_id, vocab_size, batch_size)

        result_json = artifact_dir / "result.json"
        metrics = {"ce": round(ce, 6), "ppl": round(ppl, 4),
                  "total_tokens": total_tokens, "num_rows": len(eval_rows)}

        result = _result(
            status="success",
            input_hash=input_hash,
            output_hash="sha256:" + ckpt_hash[:16],
            artifact_paths=[str(result_json)],
            metrics=metrics,
            started_at=started_at,
            ended_at=datetime.now(timezone.utc).isoformat(),
        )

        with open(result_json, "w") as f:
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
    parser.add_argument("--result", default=None,
                        help="Output path for result JSON")
    args = parser.parse_args()

    config = vars(args)
    result = run(config)

    out_path = Path(args.result) if args.result else \
        Path(args.artifact_dir) / "result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result["metrics"], indent=2))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
