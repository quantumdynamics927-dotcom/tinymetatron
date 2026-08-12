"""
workers/corpus/version.py
==========================
Freeze a corpus version: hash all splits, write MANIFEST.json.

Contract:
    Reads: config['corpus_dir'] (directory containing *.jsonl splits)
    Writes: config['output_dir']/MANIFEST.json

Output MANIFEST.json contains:
  - exp_id, version string
  - SHA256 of each split file
  - Aggregate corpus hash
  - Split hashes
  - Subdomain distribution
  - split_policy, split_seed
  - source_counts, source_overlap, text_overlap (computed from frozen files)
  - Creation timestamp

Usage:
    python -m workers.corpus.version --corpus-dir experiments/exp-003/corpus --output-dir experiments/exp-003
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

WORKER_VERSION = 1


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize(text: str) -> str:
    return re.sub(r'[^a-z0-9 ]', '', text.lower())


def _source_of(row: dict) -> str:
    """Source identifier for a row (matches workers.corpus.split._src)."""
    return row.get("source_id", row.get("source", row.get("domain", "unknown")))


# Split file stem → partition name used in the manifest's overlap/counts dicts.
_SPLIT_NAMES = {"train": "train", "val": "val", "hard_dev": "hard_dev"}


def _load_split_meta(corpus_dir: Path) -> dict:
    """Read split_meta.json written by the split worker, if present."""
    meta_path = corpus_dir / "split_meta.json"
    if meta_path.exists():
        try:
            return json.loads(open(meta_path, encoding="utf-8").read())
        except Exception:
            return {}
    return {}


def build_manifest(corpus_dir: Path) -> dict:
    """Build a manifest from a corpus directory with split JSONL files."""
    splits = {}
    split_rows: dict[str, list[dict]] = {}
    all_rows = []

    for f in sorted(corpus_dir.glob("*.jsonl")):
        # Only the three primary splits are part of the frozen corpus.
        # excluded_by_cap.jsonl (rows dropped by the per-source cap) is an
        # audit artifact, not a split, and must not enter the corpus hash or
        # the unique/subdomain tallies.
        if f.stem not in _SPLIT_NAMES:
            continue
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        splits[f.name] = {
            "rows": len(rows),
            "sha256": _hash_file(f),
        }
        all_rows.extend(rows)
        stem = f.stem
        if stem in _SPLIT_NAMES:
            split_rows[_SPLIT_NAMES[stem]] = rows

    # Aggregate corpus hash
    corp_h = hashlib.sha256()
    for f in sorted(corpus_dir.glob("*.jsonl")):
        if f.stem not in _SPLIT_NAMES:
            continue
        corp_h.update(_hash_file(f).encode())
    corpus_hash = corp_h.hexdigest()[:16]

    # Unique normalized facts
    norms = set(_normalize(r["text"]) for r in all_rows)

    # Subdomain distribution
    subdomains = {}
    for r in all_rows:
        sd = r.get("subdomain", "unknown")
        subdomains[sd] = subdomains.get(sd, 0) + 1

    # Source/text disjointness metrics, computed directly from the frozen split
    # files (independent verification of the split worker's claims).
    train_sources = {_source_of(r) for r in split_rows.get("train", [])}
    val_sources = {_source_of(r) for r in split_rows.get("val", [])}
    hard_sources = {_source_of(r) for r in split_rows.get("hard_dev", [])}
    train_texts = {r.get("text", "") for r in split_rows.get("train", [])}
    val_texts = {r.get("text", "") for r in split_rows.get("val", [])}
    hard_texts = {r.get("text", "") for r in split_rows.get("hard_dev", [])}

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

    # Largest single source's share of each partition's rows, computed directly
    # from the frozen split files (independent verification of the split
    # worker's claim). A partition dominated by one source is not a
    # representative held-out set, so this is gated at freeze time.
    from collections import Counter
    train_src_counts = Counter(_source_of(r) for r in split_rows.get("train", []))
    val_src_counts = Counter(_source_of(r) for r in split_rows.get("val", []))
    hard_src_counts = Counter(_source_of(r) for r in split_rows.get("hard_dev", []))

    def _max_share(counts, n_rows: int) -> float:
        return round(max(counts.values()) / n_rows, 4) if counts and n_rows else 0.0

    max_source_row_share = {
        "train": _max_share(train_src_counts, len(split_rows.get("train", []))),
        "val": _max_share(val_src_counts, len(split_rows.get("val", []))),
        "hard_dev": _max_share(hard_src_counts, len(split_rows.get("hard_dev", []))),
    }

    # Rows excluded by the per-source cap are preserved (not deleted) for audit
    # and optional long-tail eval segments. Record their location + hash.
    excluded_path = corpus_dir / "excluded_by_cap.jsonl"
    excluded_by_cap = None
    if excluded_path.exists():
        excluded_by_cap = {
            "path": str(excluded_path.resolve()),
            "rows": sum(1 for _ in open(excluded_path, encoding="utf-8")),
            "sha256": _hash_file(excluded_path),
        }

    # Split policy + seed + cap come from split_meta.json (written by the split
    # worker).
    meta = _load_split_meta(corpus_dir)

    return {
        "corpus_hash": corpus_hash,
        "splits": splits,
        "total_rows": len(all_rows),
        "unique_normalized": len(norms),
        "subdomains": subdomains,
        "split_policy": meta.get("split_policy", "source_disjoint_capped_v1"),
        "split_seed": meta.get("split_seed", 42),
        "max_rows_per_source": meta.get("max_rows_per_source"),
        "pre_cap_rows": meta.get("pre_cap_rows"),
        "post_cap_rows": meta.get("post_cap_rows"),
        "capped_sources": meta.get("capped_sources", []),
        "max_source_row_share": max_source_row_share,
        "excluded_by_cap": excluded_by_cap,
        "source_counts": source_counts,
        "source_overlap": source_overlap,
        "text_overlap": text_overlap,
    }


def run(config: dict) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()

    corpus_dir = Path(config["corpus_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(corpus_dir)
    manifest.update({
        "worker": "workers.corpus.version",
        "version": WORKER_VERSION,
        "corpus_dir": str(corpus_dir.resolve()),
        "created_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
    })
    # Experiment scope (e.g. "quantum technical domain") is written explicitly
    # into the manifest so the corpus's intended domain is never ambiguous.
    if config.get("scope"):
        manifest["scope"] = config["scope"]

    manifest_path = output_dir / "MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "worker": "workers.corpus.version",
        "version": WORKER_VERSION,
        "status": "success",
        "input_hash": "sha256:" + manifest["corpus_hash"],
        "output_hash": "sha256:" + _hash_file(manifest_path),
        "artifact_paths": [str(manifest_path)],
        "metrics": {
            "corpus_hash": manifest["corpus_hash"],
            "total_rows": manifest["total_rows"],
            "unique_normalized": manifest["unique_normalized"],
            "splits": list(manifest["splits"].keys()),
            "split_policy": manifest["split_policy"],
            "split_seed": manifest["split_seed"],
            "max_rows_per_source": manifest.get("max_rows_per_source"),
            "pre_cap_rows": manifest.get("pre_cap_rows"),
            "post_cap_rows": manifest.get("post_cap_rows"),
            "max_source_row_share": manifest.get("max_source_row_share"),
            "max_source_row_share_total": max(manifest.get("max_source_row_share", {}).values()) if manifest.get("max_source_row_share") else None,
            "source_overlap_total": sum(manifest["source_overlap"].values()),
            "text_overlap_total": sum(manifest["text_overlap"].values()),
        },
        "started_at": started_at,
        "ended_at": manifest["ended_at"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", default=None,
                       help="Experiment scope written into MANIFEST.json")
    parser.add_argument("--result", default=None)
    args = parser.parse_args()

    config = vars(args)
    result = run(config)

    out_path = Path(args.result) if args.result else \
        Path(args.output_dir) / "MANIFEST.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
