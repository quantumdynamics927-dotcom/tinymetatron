"""
quantum_corpus.schema
=====================
SQLite layer for the private quantum corpus.

Deliberately SEPARATE from ``db.py`` (the TinyMetatron training DB) so private
research material can never leak into ``metatron.db`` or the public Space
image. The DB path defaults to ``quantum_corpus.db`` next to this module and is
overridable via ``TMT_QUANTUM_CORPUS_DB`` (contract rule 0.6: never hard-code a
persistent path).

Richer corpus schema (vs. the training_data trio): every record carries
provenance, license, sensitivity, a content hash for dedup, and the train /
val / test split assigned by ``quantum_corpus.split``.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Iterable, Optional


# ── Schema DDL ──────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS corpus_records (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type      TEXT    NOT NULL,   -- repo | ibm_job | manifest | workload_csv | pdf
    project          TEXT    NOT NULL,   -- repo name / "wormhole-suite" / "ibm-quantum"
    subdomain        TEXT,                -- e.g. circuits | docs | calibration | security
    doc_id           TEXT    NOT NULL,   -- group key for split-by-document
    text             TEXT    NOT NULL,
    split            TEXT    NOT NULL,   -- train | val | test
    token_count      INTEGER NOT NULL DEFAULT 0,
    source_license   TEXT,                -- MIT | GPLv3 | proprietary | unknown
    provenance_url   TEXT,                -- repo URL / local path
    sensitivity      TEXT    NOT NULL,   -- public | internal | sensitive
    risk_tier        INTEGER NOT NULL DEFAULT 0,  -- 0..3
    content_hash     TEXT    NOT NULL,   -- sha256 of final (redacted) text -> dedup
    cleaning_version TEXT    NOT NULL,
    source_identity  TEXT    NOT NULL,   -- sha256(project + doc_id + cleaning_version + content_hash + chunk_index)
    created_at       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_corpus_split ON corpus_records(split);
CREATE INDEX IF NOT EXISTS idx_corpus_project ON corpus_records(project);
CREATE INDEX IF NOT EXISTS idx_corpus_source ON corpus_records(source_type);
CREATE INDEX IF NOT EXISTS idx_corpus_hash ON corpus_records(content_hash);
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_source_identity ON corpus_records(source_identity);
"""


def default_db_path() -> str:
    """``quantum_corpus.db`` beside this module, or ``TMT_QUANTUM_CORPUS_DB``."""
    env = os.environ.get("TMT_QUANTUM_CORPUS_DB")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "quantum_corpus.db")


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(path: Optional[str] = None) -> str:
    """Create the corpus_records table if absent. Returns the resolved path."""
    path = path or default_db_path()
    conn = _connect(path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return path


# ── Record dataclass ────────────────────────────────────────────────────────

@dataclass
class Record:
    """One corpus chunk. ``split``/``token_count``/``content_hash`` are filled
    later by split.py / tokenize_count.py; callers usually leave them blank."""
    source_type: str
    project: str
    text: str
    doc_id: str
    subdomain: str = ""
    split: str = ""               # train | val | test (filled by split.py)
    token_count: int = 0          # filled by tokenize_count.py
    source_license: str = "unknown"
    provenance_url: str = ""
    sensitivity: str = "internal"
    risk_tier: int = 0
    cleaning_version: str = "v1"


def content_hash(text: str) -> str:
    """sha256 hex of the final (post-redaction) text — used for dedup."""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def source_identity(project: str, doc_id: str, cleaning_version: str,
                    content_hash_val: str, chunk_index: int = 0) -> str:
    """Stable content-based identity: sha256(project + doc_id + cleaning_version + content_hash + chunk_index).

    This is NOT derived from the SQLite row id — it survives DB rebuilds,
    deduplication, insertion order changes, and split reassignments.
    chunk_index=0 for unchunked records.
    """
    # Normalize path separators for cross-platform consistency
    normalized_doc = doc_id.replace("\\", "/")
    raw = (str(project) + "\n" + normalized_doc + "\n" +
           str(cleaning_version) + "\n" + str(content_hash_val) + "\n" +
           str(chunk_index))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def backfill_source_identities(path: str) -> int:
    """Compute and store source_identity for records that lack it (pre-migration).

    Safe to run multiple times — records that already have a non-null value are skipped.
    chunk_index is assumed 0 for all pre-migration records (no chunking metadata
    was stored separately). Returns count of records updated."""
    conn = _connect(path)
    count = 0
    try:
        rows = conn.execute(
            "SELECT id, project, doc_id, cleaning_version, content_hash "
            "FROM corpus_records WHERE source_identity IS NULL OR source_identity = ''"
        ).fetchall()
        for row in rows:
            si = source_identity(row["project"], row["doc_id"],
                               row["cleaning_version"], row["content_hash"], 0)
            conn.execute(
                "UPDATE corpus_records SET source_identity=? WHERE id=?",
                (si, row["id"]))
            count += 1
        conn.commit()
    finally:
        conn.close()
    return count


# ── Write / read ─────────────────────────────────────────────────────────────

CLEANING_VERSION = "v1"


def write_records(path: str, records: Iterable[Record]) -> tuple[int, int]:
    """Insert records, deduping on content_hash. Returns (inserted, duplicates)."""
    conn = _connect(path)
    inserted = 0
    duplicates = 0
    try:
        for r in records:
            ch = content_hash(r.text)
            exists = conn.execute(
                "SELECT 1 FROM corpus_records WHERE content_hash = ? LIMIT 1", (ch,)
            ).fetchone()
            if exists:
                duplicates += 1
                continue
            si = source_identity(r.project, r.doc_id,
                                r.cleaning_version or CLEANING_VERSION, ch, 0)
            conn.execute(
                "INSERT INTO corpus_records "
                "(source_type, project, subdomain, doc_id, text, split, "
                " token_count, source_license, provenance_url, sensitivity, "
                " risk_tier, content_hash, cleaning_version, source_identity, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r.source_type, r.project, r.subdomain, r.doc_id, r.text, r.split,
                 int(r.token_count), r.source_license, r.provenance_url,
                 r.sensitivity, int(r.risk_tier), ch,
                 r.cleaning_version or CLEANING_VERSION, si, _now()),
            )
            inserted += 1
        conn.commit()
    finally:
        conn.close()
    return inserted, duplicates


def update_split_and_tokens(path: str, rows: Iterable[dict]) -> int:
    """Bulk-set split + token_count for already-inserted rows by id. ``rows``
    are dicts with keys id/split/token_count. Returns count updated."""
    conn = _connect(path)
    n = 0
    try:
        for r in rows:
            cur = conn.execute(
                "UPDATE corpus_records SET split = ?, token_count = ? WHERE id = ?",
                (r["split"], int(r["token_count"]), int(r["id"])),
            )
            n += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return n


def fetch_all(path: str) -> list[dict]:
    conn = _connect(path)
    try:
        cur = conn.execute(
            "SELECT id, source_type, project, subdomain, doc_id, text, split, "
            "token_count, source_license, provenance_url, sensitivity, risk_tier, "
            "content_hash, source_identity "
            "FROM corpus_records ORDER BY id"
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def stats(path: str) -> dict:
    conn = _connect(path)
    try:
        total = conn.execute("SELECT COUNT(*) AS n FROM corpus_records").fetchone()["n"]
        by_source = {
            r["source_type"]: int(r["n"]) for r in conn.execute(
                "SELECT source_type, COUNT(*) AS n FROM corpus_records GROUP BY source_type"
            ).fetchall()
        }
        by_project = {
            r["project"]: int(r["n"]) for r in conn.execute(
                "SELECT project, COUNT(*) AS n FROM corpus_records GROUP BY project"
            ).fetchall()
        }
        by_split = {
            r["split"]: {"records": int(r["n"]), "tokens": int(r["t"] or 0)} for r in conn.execute(
                "SELECT split, COUNT(*) AS n, SUM(token_count) AS t "
                "FROM corpus_records GROUP BY split"
            ).fetchall()
        }
        toks = conn.execute(
            "SELECT SUM(token_count) AS t FROM corpus_records"
        ).fetchone()["t"]
    finally:
        conn.close()
    return {
        "total": int(total),
        "total_tokens": int(toks or 0),
        "by_source": by_source,
        "by_project": by_project,
        "by_split": by_split,
    }


# ── self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import tempfile

    sys.stdout.reconfigure(encoding="utf-8")

    tmp = tempfile.mkdtemp(prefix="qcorpus_")
    dbp = os.path.join(tmp, "qc.db")
    init_db(dbp)
    init_db(dbp)  # idempotent

    recs = [
        Record("repo", "GRE", "docs text one", "GRE:docs/a.md", subdomain="docs",
               source_license="MIT", sensitivity="public"),
        Record("ibm_job", "ibm-quantum", "job xyz circuit", "ibm:xyz",
               subdomain="circuits", sensitivity="internal"),
        Record("repo", "GRE", "docs text one", "GRE:docs/a.md"),  # duplicate hash -> skipped
    ]
    ins, dup = write_records(dbp, recs)
    assert ins == 2 and dup == 1, f"insert/dup got {ins}/{dup}"

    rows = fetch_all(dbp)
    assert len(rows) == 2, f"fetch_all got {len(rows)}"

    # split + token update
    upd = update_split_and_tokens(dbp, [
        {"id": rows[0]["id"], "split": "train", "token_count": 5},
        {"id": rows[1]["id"], "split": "test", "token_count": 3},
    ])
    assert upd == 2, f"updated {upd}"

    s = stats(dbp)
    assert s["total"] == 2, s
    assert s["by_split"]["train"]["records"] == 1, s
    assert s["by_split"]["train"]["tokens"] == 5, s
    assert s["total_tokens"] == 8, s
    print("schema self-test OK:", s)

    os.remove(dbp)
    os.rmdir(tmp)
    print("SELF-TEST PASSED")