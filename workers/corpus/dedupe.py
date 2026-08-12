"""
workers/corpus/dedupe.py
======================
Normalized-text + near-duplicate dedup → dedupe report JSON.

Contract:
    Reads: config['corpus_dir']/validated.jsonl (from validate stage)
    Writes: result.json + deduped.jsonl to config['artifact_dir']

Output schema:
{
  "worker": "workers.corpus.dedupe",
  "version": 1,
  "status": "success",
  "metrics": {
    "input_rows": int,
    "exact_duplicates": int,
    "normalized_duplicates": int,
    "duplicate_pairs_found": int,
    "output_rows": int,
    "dedup_rate": float
  }
}

Usage:
    python -m workers.corpus.dedupe --corpus-dir data/raw --result result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

WORKER_VERSION = 1


def _normalize(text: str) -> str:
    """Canonical form for exact dedup."""
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()


def _ngrams(text: str, n: int = 5) -> set:
    """Character n-grams for near-duplicate detection."""
    text = text.lower()
    return set(text[i:i+n] for i in range(max(1, len(text) - n + 1)))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


MIN_NGRAM_JACCARD = 0.85
MIN_DUP_LEN = 50  # Only near-dedup texts longer than this


def run(config: dict) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    corpus_dir = Path(config["corpus_dir"])
    artifact_dir = Path(config.get("artifact_dir", "."))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    input_path = corpus_dir / "validated.jsonl"
    if not input_path.exists():
        # Fall back: validate output might be at corpus_dir root
        candidates = list(corpus_dir.glob("*.jsonl"))
        if candidates:
            input_path = candidates[0]

    rows = []
    for line in open(input_path, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    input_count = len(rows)
    exact_seen: dict[str, int] = {}
    norm_seen: dict[str, list[int]] = defaultdict(list)
    near_dup_pairs = 0
    dup_groups = []

    # Pass 1: exact dedup
    exact_dupes = 0
    unique_rows = []
    for i, row in enumerate(rows):
        text = row.get("text", "")
        if not text:
            continue
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in exact_seen:
            exact_dupes += 1
            continue
        exact_seen[key] = i
        unique_rows.append(row)

    # Pass 2: normalized dedup
    norm_dupes = 0
    final_rows = []
    for row in unique_rows:
        text = row.get("text", "")
        norm = _normalize(text)
        if not norm:
            continue
        if norm_seen[norm]:
            norm_dupes += 1
            dup_groups.append((norm, norm_seen[norm][0]))
            norm_seen[norm].append(len(final_rows))
            continue
        norm_seen[norm].append(len(final_rows))
        final_rows.append(row)

    # Pass 3: near-duplicate detection (on remaining rows)
    output_rows = []
    skip_indices = set()
    for i, row in enumerate(final_rows):
        if i in skip_indices:
            continue
        text = row.get("text", "")
        if len(text) < MIN_DUP_LEN:
            output_rows.append(row)
            continue

        row_ngrams = _ngrams(text)
        is_near_dup = False
        for j in range(i + 1, len(final_rows)):
            if j in skip_indices:
                continue
            other_text = final_rows[j].get("text", "")
            if len(other_text) < MIN_DUP_LEN:
                continue
            other_ngrams = _ngrams(other_text)
            if _jaccard(row_ngrams, other_ngrams) >= MIN_NGRAM_JACCARD:
                near_dup_pairs += 1
                is_near_dup = True
                skip_indices.add(j)

        if not is_near_dup:
            output_rows.append(row)

    total_dupes = exact_dupes + norm_dupes + near_dup_pairs

    output_path = artifact_dir / "deduped.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ended_at = datetime.now(timezone.utc).isoformat()

    h_in = hashlib.sha256()
    h_in.update(str(input_count).encode())
    input_hash = "sha256:" + h_in.hexdigest()[:16]

    h_out = hashlib.sha256()
    with open(output_path, "rb") as vf:
        for chunk in iter(lambda: vf.read(65536), b""):
            h_out.update(chunk)
    output_hash = "sha256:" + h_out.hexdigest()[:16]

    result = {
        "worker": "workers.corpus.dedupe",
        "version": WORKER_VERSION,
        "status": "success",
        "input_hash": input_hash,
        "output_hash": output_hash,
        "artifact_paths": [str(output_path)],
        "metrics": {
            "input_rows": input_count,
            "exact_duplicates": exact_dupes,
            "normalized_duplicates": norm_dupes,
            "near_duplicates": near_dup_pairs,
            "duplicate_pairs_found": near_dup_pairs,
            "output_rows": len(output_rows),
            "dedup_rate": round(len(output_rows) / input_count, 4) if input_count else 0.0,
        },
        "started_at": started_at,
        "ended_at": ended_at,
    }

    result_path = artifact_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
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
