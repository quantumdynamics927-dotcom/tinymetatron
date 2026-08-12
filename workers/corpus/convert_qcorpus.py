"""
workers/corpus/convert_qcorpus.py
=================================
Convert the private quantum corpus DB (quantum_corpus.db) to a
source-attributed JSONL corpus for the TinyMetatron corpus loop.

This is a CONTROLLED corpus-acquisition step, not an automatic dump: every
row is attributed with a stable id, a document-level source_id, the quantum
technical domain, subdomain, license, and provenance. The source_id is the
``doc_id`` from corpus_records (``{project}:{relpath}``) — a per-file document
identity, NOT a broad topic bucket — so the downstream source-disjoint split
groups on genuinely independent documents.

Contract:
    Reads: config['db_path'] (quantum_corpus.db, built by quantum_corpus.build)
    Writes: config['output_dir']/corpus.jsonl
            config['output_dir']/convert_result.json (via --result)

Output row schema (superset of the required conversion contract):
{
  "id":            "stable content-based identity (source_identity)",
  "text":          "redacted training text",
  "source_id":     "document-or-origin-group id (doc_id = project:relpath)",
  "domain":        "quantum",
  "subdomain":     "circuits | docs | calibration | security | ...",
  "license":       "MIT | GPLv3 | proprietary | unknown",
  "provenance":    "repo URL / local path",
  "quality_score": 0.0,
  "project":       "GRE | QPyth | QAP | TMT_Quantum_Vault | ibm-quantum | ...",
  "source_type":   "repo | ibm_job | manifest | workload_csv | pdf",
  "sensitivity":   "public | internal | sensitive",
  "risk_tier":     0..3,
  "token_count":   int
}

Usage:
    python -m workers.corpus.convert_qcorpus \
        --db quantum_corpus/quantum_corpus.db \
        --output-dir experiments/exp-004/raw \
        --result experiments/exp-004/raw/convert_result.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

WORKER_VERSION = 1

# The technical domain for exp-004 (a quantum technical-domain experiment).
DOMAIN = "quantum"

# Columns exported from corpus_records, in order.
_COLS = [
    "id", "source_type", "project", "subdomain", "doc_id", "text", "split",
    "token_count", "source_license", "provenance_url", "sensitivity",
    "risk_tier", "content_hash", "source_identity",
]


def _row_to_jsonl(row: sqlite3.Row) -> dict:
    """Map a corpus_records row to the source-attributed JSONL schema."""
    return {
        "id": row["source_identity"] or f"row-{row['id']}",
        "text": row["text"],
        "source_id": row["doc_id"],
        "domain": DOMAIN,
        "subdomain": row["subdomain"] or "unknown",
        "license": row["source_license"] or "unknown",
        "provenance": row["provenance_url"] or f"{row['source_type']}:{row['project']}",
        "quality_score": 0.0,
        "project": row["project"],
        "source_type": row["source_type"],
        "sensitivity": row["sensitivity"],
        "risk_tier": row["risk_tier"],
        "token_count": row["token_count"],
    }


def inspect_db(db_path: Path) -> dict:
    """Gate inspection: schema, cardinality, null/empty/duplicate rates,
    license status, source-group count, and length distributions."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    total = conn.execute("SELECT COUNT(*) FROM corpus_records").fetchone()[0]

    # Null/empty rates on the fields that matter for attribution.
    def _null_empty(col: str) -> dict:
        n = conn.execute(
            f"SELECT COUNT(*) FROM corpus_records WHERE {col} IS NULL OR {col} = ''"
        ).fetchone()[0]
        return {"null_or_empty": n, "rate": round(n / total, 4) if total else 0.0}

    # Duplicate rates: exact text dupes and source_identity dupes.
    dup_text = conn.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT content_hash) FROM corpus_records"
    ).fetchone()[0]
    dup_si = conn.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT source_identity) FROM corpus_records"
    ).fetchone()[0]

    # Source-group cardinality: distinct doc_id (the source-disjoint group key).
    n_source_groups = conn.execute(
        "SELECT COUNT(DISTINCT doc_id) FROM corpus_records").fetchone()[0]

    # Per-project / per-source_type / per-license / per-subdomain counts.
    def _counts(col: str) -> dict:
        return {r[0]: r[1] for r in conn.execute(
            f"SELECT {col}, COUNT(*) FROM corpus_records GROUP BY {col} "
            f"ORDER BY COUNT(*) DESC").fetchall()}

    # Length distribution (chars) + tokenizer-length distribution.
    lens = [r[0] for r in conn.execute(
        "SELECT LENGTH(text) FROM corpus_records").fetchall()]
    toks = [r[0] for r in conn.execute(
        "SELECT token_count FROM corpus_records WHERE token_count > 0").fetchall()]

    def _dist(vals: list[int]) -> dict:
        if not vals:
            return {"n": 0}
        vals = sorted(vals)
        n = len(vals)
        def _pct(p: float) -> int:
            return vals[min(n - 1, int(p * n))]
        return {
            "n": n,
            "min": vals[0],
            "p25": _pct(0.25),
            "median": _pct(0.50),
            "p75": _pct(0.75),
            "p95": _pct(0.95),
            "max": vals[-1],
            "mean": round(sum(vals) / n, 1),
        }

    result = {
        "tables": tables,
        "total_rows": total,
        "null_empty": {c: _null_empty(c) for c in
                       ("text", "doc_id", "subdomain", "source_license", "provenance_url")},
        "duplicates": {
            "exact_text_dupes": dup_text,
            "source_identity_dupes": dup_si,
        },
        "n_source_groups": n_source_groups,
        "by_project": _counts("project"),
        "by_source_type": _counts("source_type"),
        "by_license": _counts("source_license"),
        "by_sensitivity": _counts("sensitivity"),
        "by_subdomain": _counts("subdomain"),
        "char_length": _dist(lens),
        "token_length": _dist(toks),
    }
    conn.close()
    return result


def run(config: dict) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    db_path = Path(config["db_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        ended_at = datetime.now(timezone.utc).isoformat()
        err = {
            "worker": "workers.corpus.convert_qcorpus",
            "version": WORKER_VERSION,
            "status": "error",
            "error": f"quantum corpus DB not found: {db_path}",
            "started_at": started_at,
            "ended_at": ended_at,
        }
        return err

    inspection = inspect_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT " + ", ".join(_COLS) + " FROM corpus_records ORDER BY id"
    ).fetchall()
    conn.close()

    out_path = output_dir / "corpus.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_row_to_jsonl(row), ensure_ascii=False) + "\n")

    ended_at = datetime.now(timezone.utc).isoformat()

    result = {
        "worker": "workers.corpus.convert_qcorpus",
        "version": WORKER_VERSION,
        "status": "success",
        "input_hash": "sha256:" + str(db_path.stat().st_size),
        "output_hash": "sha256:" + _hash_file(out_path),
        "artifact_paths": [str(out_path)],
        "metrics": {
            "total_rows": len(rows),
            "n_source_groups": inspection["n_source_groups"],
            "by_project": inspection["by_project"],
            "by_license": inspection["by_license"],
            "by_subdomain": inspection["by_subdomain"],
            "by_source_type": inspection["by_source_type"],
            "by_sensitivity": inspection["by_sensitivity"],
            "null_empty": inspection["null_empty"],
            "duplicates": inspection["duplicates"],
            "char_length": inspection["char_length"],
            "token_length": inspection["token_length"],
        },
        "inspection": inspection,
        "started_at": started_at,
        "ended_at": ended_at,
    }

    result_path = Path(config.get("result", "")) if config.get("result") else \
        output_dir / "convert_result.json"
    if str(result_path) == ".":
        result_path = output_dir / "convert_result.json"
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def _hash_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True,
                        help="Path to quantum_corpus.db")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write corpus.jsonl")
    parser.add_argument("--result", default=None)
    args = parser.parse_args()

    config = {"db_path": args.db, "output_dir": args.output_dir,
              "result": args.result}
    result = run(config)

    out_path = Path(args.result) if args.result else \
        Path(args.output_dir) / "convert_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
