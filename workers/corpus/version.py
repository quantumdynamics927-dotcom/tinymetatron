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


def build_manifest(corpus_dir: Path) -> dict:
    """Build a manifest from a corpus directory with split JSONL files."""
    splits = {}
    all_rows = []

    for f in sorted(corpus_dir.glob("*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        splits[f.name] = {
            "rows": len(rows),
            "sha256": _hash_file(f),
        }
        all_rows.extend(rows)

    # Aggregate corpus hash
    corp_h = hashlib.sha256()
    for f in sorted(corpus_dir.glob("*.jsonl")):
        corp_h.update(_hash_file(f).encode())
    corpus_hash = corp_h.hexdigest()[:16]

    # Unique normalized facts
    norms = set(_normalize(r["text"]) for r in all_rows)

    # Subdomain distribution
    subdomains = {}
    for r in all_rows:
        sd = r.get("subdomain", "unknown")
        subdomains[sd] = subdomains.get(sd, 0) + 1

    return {
        "corpus_hash": corpus_hash,
        "splits": splits,
        "total_rows": len(all_rows),
        "unique_normalized": len(norms),
        "subdomains": subdomains,
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

    manifest_path = output_dir / "MANIFEST.json"
    with open(manifest_path, "w") as f:
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
        },
        "started_at": started_at,
        "ended_at": manifest["ended_at"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result", default=None)
    args = parser.parse_args()

    config = vars(args)
    result = run(config)

    out_path = Path(args.result) if args.result else \
        Path(args.output_dir) / "MANIFEST.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
