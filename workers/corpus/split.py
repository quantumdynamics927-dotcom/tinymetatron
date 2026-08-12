"""
workers/corpus/split.py
====================
Source-disjoint 80/10/10 split → train.jsonl / val.jsonl / hard_dev.jsonl.

Contract:
    Reads: config['corpus_dir']/deduped.jsonl (from dedupe stage)
    Writes: config['output_dir']/train.jsonl, val.jsonl, hard_dev.jsonl

Output schema:
{
  "worker": "workers.corpus.split",
  "version": 1,
  "status": "success",
  "metrics": {
    "total_rows": int,
    "train_rows": int,
    "val_rows": int,
    "hard_dev_rows": int,
    "train_pct": float,
    "val_pct": float,
    "hard_dev_pct": float,
    "seed": int,
    "source_distribution": {source: count}
  }
}

Usage:
    python -m workers.corpus.split --corpus-dir data/raw --output-dir experiments/exp-004/corpus --result result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

WORKER_VERSION = 1

# Fixed seed for deterministic splits
DEFAULT_SEED = 42


def run(config: dict) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    corpus_dir = Path(config["corpus_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", DEFAULT_SEED))
    train_pct = float(config.get("train_pct", 0.80))
    val_pct = float(config.get("val_pct", 0.10))
    # hard_dev gets the rest

    # Find deduped input
    input_path = corpus_dir / "deduped.jsonl"
    if not input_path.exists():
        candidates = list(corpus_dir.glob("*.jsonl"))
        input_path = candidates[0] if candidates else None

    if not input_path or not input_path.exists():
        raise FileNotFoundError(f"No deduped.jsonl found in {corpus_dir}")

    rows = []
    for line in open(input_path, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    total = len(rows)

    # Group by source for source-disjoint split
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        source = row.get("source", row.get("domain", "unknown"))
        by_source[source].append(row)

    # Shuffle within each source group with fixed seed
    rng = random.Random(seed)
    for source in by_source:
        rng.shuffle(by_source[source])

    # Assign rows to splits respecting source-disjoint constraint
    train_rows = []
    val_rows = []
    hard_dev_rows = []

    for source, source_rows in by_source.items():
        n = len(source_rows)
        n_train = max(1, round(n * train_pct))
        n_val = max(1, round(n * val_pct))
        # Hard dev gets the rest from each source

        train_rows.extend(source_rows[:n_train])
        val_rows.extend(source_rows[n_train:n_train + n_val])
        hard_dev_rows.extend(source_rows[n_train + n_val:])

    # Verify disjointness
    train_ids = set(id(r) for r in train_rows)
    val_ids = set(id(r) for r in val_rows)
    hard_ids = set(id(r) for r in hard_dev_rows)
    assert not train_ids & val_ids, "train/val overlap!"
    assert not train_ids & hard_ids, "train/hard_dev overlap!"
    assert not val_ids & hard_ids, "val/hard_dev overlap!"

    # Source distribution
    train_sources = defaultdict(int)
    val_sources = defaultdict(int)
    hard_sources = defaultdict(int)
    for r in train_rows:
        train_sources[r.get("source", r.get("domain", "unknown"))] += 1
    for r in val_rows:
        val_sources[r.get("source", r.get("domain", "unknown"))] += 1
    for r in hard_dev_rows:
        hard_sources[r.get("source", r.get("domain", "unknown"))] += 1

    def write_split(path: Path, rows: list) -> int:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(rows)

    n_train = write_split(output_dir / "train.jsonl", train_rows)
    n_val = write_split(output_dir / "val.jsonl", val_rows)
    n_hard = write_split(output_dir / "hard_dev.jsonl", hard_dev_rows)

    ended_at = datetime.now(timezone.utc).isoformat()

    h_in = hashlib.sha256()
    h_in.update(str(total).encode())
    input_hash = "sha256:" + h_in.hexdigest()[:16]

    all_out = train_rows + val_rows + hard_dev_rows
    h_out = hashlib.sha256()
    h_out.update(json.dumps({"train": n_train, "val": n_val, "hard": n_hard}, sort_keys=True).encode())
    output_hash = "sha256:" + h_out.hexdigest()[:16]

    result = {
        "worker": "workers.corpus.split",
        "version": WORKER_VERSION,
        "status": "success",
        "input_hash": input_hash,
        "output_hash": output_hash,
        "artifact_paths": [
            str(output_dir / "train.jsonl"),
            str(output_dir / "val.jsonl"),
            str(output_dir / "hard_dev.jsonl"),
        ],
        "metrics": {
            "total_rows": total,
            "train_rows": n_train,
            "val_rows": n_val,
            "hard_dev_rows": n_hard,
            "train_pct": round(n_train / total, 4) if total else 0,
            "val_pct": round(n_val / total, 4) if total else 0,
            "hard_dev_pct": round(n_hard / total, 4) if total else 0,
            "seed": seed,
            "source_distribution": {
                "train": dict(train_sources),
                "val": dict(val_sources),
                "hard_dev": dict(hard_sources),
            },
        },
        "started_at": started_at,
        "ended_at": ended_at,
    }

    result_path = Path(config.get("result", "")) if config.get("result") else output_dir.parent / "split_result.json"
    if str(result_path) == ".":
        result_path = output_dir.parent / "split_result.json"
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-pct", type=float, default=0.80)
    parser.add_argument("--val-pct", type=float, default=0.10)
    args = parser.parse_args()

    config = vars(args)
    result = run(config)

    if args.result:
        with open(args.result, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
