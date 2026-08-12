"""
workers/corpus/corpus_gate.py
=============================
Corpus freeze gate: verify the frozen split is source-disjoint.

Reads the MANIFEST.json written by workers.corpus.version and emits a single
gate metric, ``source_overlap_total``, which is the sum of all pairwise
source-overlap counts between train / val / hard_dev. A source-disjoint split
has every source_id in exactly one partition, so this total must be zero.

The gate definition lives in state/gates/corpus_source_disjoint_gate.json and
is evaluated by the corpus loop's gate runner (mirrors generalize_loop.run_gate
but is experiment-scoped rather than run-scoped).

Contract:
    Reads: config['manifest'] (MANIFEST.json path)
    Writes: result.json to config['artifact_dir']

Output schema:
{
  "worker": "workers.corpus.corpus_gate",
  "version": 1,
  "status": "success",
  "metrics": {
    "source_overlap_total": int,
    "text_overlap_total": int,
    "n_source_groups": int,
    "source_overlap": {"train_val": int, "train_hard_dev": int, "val_hard_dev": int},
    "text_overlap": {"train_val": int, "train_hard_dev": int, "val_hard_dev": int}
  }
}

Usage:
    python -m workers.corpus.corpus_gate --manifest experiments/exp-004/MANIFEST.json --result result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

WORKER_VERSION = 1


def run(config: dict) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    manifest_path = Path(config["manifest"])
    artifact_dir = Path(config.get("artifact_dir", "."))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        ended_at = datetime.now(timezone.utc).isoformat()
        err = {
            "worker": "workers.corpus.corpus_gate",
            "version": WORKER_VERSION,
            "status": "error",
            "error": f"MANIFEST.json not found: {manifest_path}",
            "started_at": started_at,
            "ended_at": ended_at,
        }
        with open(artifact_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(err, f, indent=2)
        return err

    manifest = json.loads(open(manifest_path, encoding="utf-8").read())

    source_overlap = manifest.get("source_overlap", {})
    text_overlap = manifest.get("text_overlap", {})
    source_counts = manifest.get("source_counts", {})
    max_share = manifest.get("max_source_row_share", {})

    # Coerce to ints (manifest values may be missing on a malformed freeze).
    def _sum(d: dict) -> int:
        return int(sum(int(v) for v in d.values()))

    source_overlap_total = _sum(source_overlap)
    text_overlap_total = _sum(text_overlap)
    # In a source-disjoint split, sources are partitioned, so the total number
    # of distinct source groups is the sum of per-split source counts.
    n_source_groups = _sum(source_counts)
    # Largest single source's share of any primary partition. A partition whose
    # rows come mostly from one source is not a representative held-out set.
    max_source_row_share = max(max_share.values()) if max_share else None

    ended_at = datetime.now(timezone.utc).isoformat()

    result = {
        "worker": "workers.corpus.corpus_gate",
        "version": WORKER_VERSION,
        "status": "success",
        "metrics": {
            "source_overlap_total": source_overlap_total,
            "text_overlap_total": text_overlap_total,
            "n_source_groups": n_source_groups,
            "source_overlap": source_overlap,
            "text_overlap": text_overlap,
            "max_source_row_share": max_source_row_share,
            "split_policy": manifest.get("split_policy", "unknown"),
            "split_seed": manifest.get("split_seed"),
        },
        "started_at": started_at,
        "ended_at": ended_at,
    }

    with open(artifact_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result", default=None)
    args = parser.parse_args()

    artifact_dir = Path(args.result).parent if args.result else Path(".")
    config = vars(args)
    config["artifact_dir"] = str(artifact_dir)
    result = run(config)

    if args.result:
        with open(args.result, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())