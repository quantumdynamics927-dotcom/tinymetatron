"""
workers/corpus/split.py
====================
Source-disjoint 80/10/10 split → train.jsonl / val.jsonl / hard_dev.jsonl.

Every source group is assigned wholly to exactly one partition; no source is
sliced across train/val/hard_dev. This is what makes the split source-disjoint
(as opposed to merely row/text-disjoint within a source).

Contract:
    Reads: config['corpus_dir']/deduped.jsonl (from dedupe stage)
    Writes: config['output_dir']/train.jsonl, val.jsonl, hard_dev.jsonl,
            config['output_dir']/split_meta.json

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
    "split_policy": "source_disjoint_v1",
    "n_source_groups": int,
    "source_counts": {"train": int, "val": int, "hard_dev": int},
    "source_overlap": {"train_val": int, "train_hard_dev": int, "val_hard_dev": int},
    "text_overlap": {"train_val": int, "train_hard_dev": int, "val_hard_dev": int},
    "source_distribution": {"train": {source: count}, ...}
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

# Split policy identifier recorded in split_meta.json and MANIFEST.json.
# v1 = whole source groups assigned to exactly one partition.
SPLIT_POLICY = "source_disjoint_v1"

# A meaningful 3-way source-disjoint split needs at least this many groups.
MIN_SOURCE_GROUPS = 3


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

    # Group rows by source. Each whole group is assigned to exactly one split.
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        source = row.get("source", row.get("domain", "unknown"))
        by_source[source].append(row)

    n_groups = len(by_source)

    # A meaningful source-disjoint 3-way split requires enough independent
    # source groups. With fewer than 3 we refuse rather than slice a tiny
    # source across all three partitions to make ratios look correct.
    if n_groups < MIN_SOURCE_GROUPS:
        ended_at = datetime.now(timezone.utc).isoformat()
        err = {
            "worker": "workers.corpus.split",
            "version": WORKER_VERSION,
            "status": "error",
            "error": (
                f"source-disjoint split requires >={MIN_SOURCE_GROUPS} independent "
                f"source groups, got {n_groups}. Refusing to slice sources across "
                f"partitions — real corpus freezing must fail when groups are too "
                f"few for a meaningful split."
            ),
            "metrics": {"total_rows": total, "n_source_groups": n_groups},
            "started_at": started_at,
            "ended_at": ended_at,
        }
        result_path = Path(config.get("result", "")) if config.get("result") else output_dir.parent / "split_result.json"
        if str(result_path) == ".":
            result_path = output_dir.parent / "split_result.json"
        result_path = Path(result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(err, f, indent=2)
        return err

    # Deterministic ordering: shuffle the group keys, then rows within each
    # group, both driven by the fixed seed.
    rng = random.Random(seed)
    group_keys = list(by_source.keys())
    rng.shuffle(group_keys)
    for k in group_keys:
        rng.shuffle(by_source[k])

    # Whole-group allocation. The ratio target applies to groups/sources, not
    # necessarily individual rows, so resulting partitions may have uneven sample
    # counts (permitted). We walk the shuffled groups filling train, then val,
    # then hard_dev, always reserving enough remaining groups to keep every
    # partition non-empty.
    target_train = train_pct * total
    target_val = val_pct * total

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    hard_dev_rows: list[dict] = []

    i = 0
    acc_train = 0
    while i < n_groups and acc_train < target_train and (n_groups - i) > 2:
        g = group_keys[i]
        train_rows.extend(by_source[g])
        acc_train += len(by_source[g])
        i += 1

    acc_val = 0
    while i < n_groups and acc_val < target_val and (n_groups - i) > 1:
        g = group_keys[i]
        val_rows.extend(by_source[g])
        acc_val += len(by_source[g])
        i += 1

    for j in range(i, n_groups):
        g = group_keys[j]
        hard_dev_rows.extend(by_source[g])

    # Verify the source-disjoint + text-disjoint invariants.
    def _src(r: dict) -> str:
        return r.get("source", r.get("domain", "unknown"))

    train_sources = {_src(r) for r in train_rows}
    val_sources = {_src(r) for r in val_rows}
    hard_sources = {_src(r) for r in hard_dev_rows}
    train_texts = {r.get("text", "") for r in train_rows}
    val_texts = {r.get("text", "") for r in val_rows}
    hard_texts = {r.get("text", "") for r in hard_dev_rows}

    assert train_sources.isdisjoint(val_sources), "train/val source overlap!"
    assert train_sources.isdisjoint(hard_sources), "train/hard_dev source overlap!"
    assert val_sources.isdisjoint(hard_sources), "val/hard_dev source overlap!"
    assert train_texts.isdisjoint(val_texts), "train/val text overlap!"
    assert train_texts.isdisjoint(hard_texts), "train/hard_dev text overlap!"
    assert val_texts.isdisjoint(hard_texts), "val/hard_dev text overlap!"

    source_counts = {
        "train": len(train_sources),
        "val": len(val_sources),
        "hard_dev": len(hard_sources),
    }
    source_overlap = {
        "train_val": len(train_sources & val_sources),
        "train_hard_dev": len(train_sources & hard_sources),
        "val_hard_dev": len(val_sources & hard_sources),
    }
    text_overlap = {
        "train_val": len(train_texts & val_texts),
        "train_hard_dev": len(train_texts & hard_texts),
        "val_hard_dev": len(val_texts & hard_texts),
    }

    # Per-split source → row counts (informational).
    train_src_counts: dict[str, int] = defaultdict(int)
    val_src_counts: dict[str, int] = defaultdict(int)
    hard_src_counts: dict[str, int] = defaultdict(int)
    for r in train_rows:
        train_src_counts[_src(r)] += 1
    for r in val_rows:
        val_src_counts[_src(r)] += 1
    for r in hard_dev_rows:
        hard_src_counts[_src(r)] += 1

    def write_split(path: Path, rows: list) -> int:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(rows)

    n_train = write_split(output_dir / "train.jsonl", train_rows)
    n_val = write_split(output_dir / "val.jsonl", val_rows)
    n_hard = write_split(output_dir / "hard_dev.jsonl", hard_dev_rows)

    # Persist split metadata alongside the split files so version.py can record
    # the policy + seed in MANIFEST.json without coupling to this worker's result.
    split_meta = {
        "split_policy": SPLIT_POLICY,
        "split_seed": seed,
        "source_counts": source_counts,
        "source_overlap": source_overlap,
        "text_overlap": text_overlap,
    }
    with open(output_dir / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    ended_at = datetime.now(timezone.utc).isoformat()

    h_in = hashlib.sha256()
    h_in.update(str(total).encode())
    input_hash = "sha256:" + h_in.hexdigest()[:16]

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
            str(output_dir / "split_meta.json"),
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
            "split_policy": SPLIT_POLICY,
            "n_source_groups": n_groups,
            "source_counts": source_counts,
            "source_overlap": source_overlap,
            "text_overlap": text_overlap,
            "source_distribution": {
                "train": dict(train_src_counts),
                "val": dict(val_src_counts),
                "hard_dev": dict(hard_src_counts),
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
