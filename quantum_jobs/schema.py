"""
quantum_jobs/schema.py
=====================
SQLite schema for IBM Quantum hardware job records.

Stores raw job data from E:\\Descargas workloads zips, job zips, loose JSONs,
and all_time-workloads CSVs.

Schema:
  jobs        - one row per IBM Quantum job (info + result paired)
  workloads   - one row per workloads zip or loose dir
  csv_jobs    - one row per job row in a workload CSV
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    jid              TEXT    NOT NULL UNIQUE,
    backend          TEXT    NOT NULL,
    status           TEXT    NOT NULL,
    program          TEXT,
    cost             INTEGER,
    shots            INTEGER NOT NULL DEFAULT 0,
    created          TEXT,
    tags             TEXT,
    qasm             TEXT,
    raw_samples      TEXT,
    samples_count    INTEGER NOT NULL DEFAULT 0,
    provenance       TEXT    NOT NULL,
    workload_name    TEXT,
    source_file      TEXT,
    content_hash     TEXT    NOT NULL,
    created_at       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_jid      ON jobs(jid);
CREATE INDEX IF NOT EXISTS idx_jobs_backend ON jobs(backend);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created);

CREATE TABLE IF NOT EXISTS workloads (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT    NOT NULL,
    kind             TEXT    NOT NULL,
    source_path      TEXT    NOT NULL,
    jobs_count       INTEGER NOT NULL DEFAULT 0,
    total_shots      INTEGER NOT NULL DEFAULT 0,
    by_backend       TEXT,
    by_status        TEXT,
    created_at       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_workloads_name ON workloads(name);

CREATE TABLE IF NOT EXISTS csv_jobs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    csv_name         TEXT    NOT NULL,
    csv_path         TEXT    NOT NULL,
    workload_id      INTEGER,
    jid              TEXT    NOT NULL,
    backend          TEXT    NOT NULL,
    status           TEXT,
    qpu              TEXT,
    usage_seconds    REAL,
    user_name        TEXT,
    created_date     TEXT,
    completed_date   TEXT,
    tags             TEXT,
    account          TEXT,
    created_at       TEXT    NOT NULL,
    FOREIGN KEY (workload_id) REFERENCES workloads(id)
);

CREATE INDEX IF NOT EXISTS idx_csv_jobs_jid     ON csv_jobs(jid);
CREATE INDEX IF NOT EXISTS idx_csv_jobs_backend ON csv_jobs(backend);
"""


def default_db_path() -> str:
    """quantum_jobs.db in the quantum_jobs dir."""
    d = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(d, "quantum_jobs.db")


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(path: Optional[str] = None) -> str:
    path = path or default_db_path()
    conn = _connect(path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return path


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


# ── Write / read ──────────────────────────────────────────────────────────────

def write_jobs(path: str, jobs: Iterable[dict]) -> tuple[int, int]:
    """Insert job records, deduping on jid. Returns (inserted, skipped)."""
    conn = _connect(path)
    inserted = 0
    skipped = 0
    try:
        for j in jobs:
            ch = content_hash((j.get("qasm") or "") + str(j.get("shots", 0)))
            existing = conn.execute(
                "SELECT 1 FROM jobs WHERE jid = ? LIMIT 1", (j["jid"],)
            ).fetchone()
            if existing:
                skipped += 1
                continue
            conn.execute(
                "INSERT INTO jobs (jid, backend, status, program, cost, shots, created,"
                " tags, qasm, raw_samples, samples_count, provenance, workload_name,"
                " source_file, content_hash, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    j["jid"], j["backend"], j["status"], j.get("program"),
                    j.get("cost"), j.get("shots", 0), j.get("created"),
                    j.get("tags"), j.get("qasm"), j.get("raw_samples"),
                    j.get("samples_count", 0), j["provenance"],
                    j.get("workload_name"), j.get("source_file"),
                    ch, _now(),
                ),
            )
            inserted += 1
        conn.commit()
    finally:
        conn.close()
    return inserted, skipped


def write_workloads(path: str, workloads: Iterable[dict]) -> int:
    conn = _connect(path)
    count = 0
    try:
        for w in workloads:
            conn.execute(
                "INSERT INTO workloads (name, kind, source_path, jobs_count, total_shots,"
                " by_backend, by_status, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    w["name"], w["kind"], w["source_path"],
                    w.get("jobs_count", 0), w.get("total_shots", 0),
                    w.get("by_backend"), w.get("by_status"), _now(),
                ),
            )
            count += 1
        conn.commit()
    finally:
        conn.close()
    return count


def write_csv_jobs(path: str, csv_jobs: Iterable[dict]) -> int:
    conn = _connect(path)
    count = 0
    try:
        for cj in csv_jobs:
            conn.execute(
                "INSERT INTO csv_jobs (csv_name, csv_path, workload_id, jid, backend,"
                " status, qpu, usage_seconds, user_name, created_date, completed_date,"
                " tags, account, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cj["csv_name"], cj["csv_path"], cj.get("workload_id"),
                    cj["jid"], cj["backend"], cj.get("status"),
                    cj.get("qpu"), cj.get("usage_seconds"),
                    cj.get("user_name"), cj.get("created_date"),
                    cj.get("completed_date"), cj.get("tags"),
                    cj.get("account"), _now(),
                ),
            )
            count += 1
        conn.commit()
    finally:
        conn.close()
    return count


def stats(path: Optional[str] = None) -> dict:
    path = path or default_db_path()
    conn = _connect(path)
    try:
        jobs_total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        shots_total = conn.execute("SELECT COALESCE(SUM(shots),0) FROM jobs").fetchone()[0]
        by_backend = {
            r["backend"]: dict(r)
            for r in conn.execute(
                "SELECT backend, COUNT(*) as n, SUM(shots) as shots FROM jobs "
                "GROUP BY backend ORDER BY n DESC"
            ).fetchall()
        }
        by_status = dict(conn.execute(
            "SELECT status, COUNT(*) as n FROM jobs GROUP BY status ORDER BY n DESC"
        ).fetchall())
        workloads_count = conn.execute(
            "SELECT COUNT(*) FROM workloads"
        ).fetchone()[0]
        csv_jobs_count = conn.execute(
            "SELECT COUNT(*) FROM csv_jobs"
        ).fetchone()[0]
        return {
            "jobs": jobs_total,
            "shots_total": shots_total,
            "by_backend": by_backend,
            "by_status": by_status,
            "workloads": workloads_count,
            "csv_jobs": csv_jobs_count,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    import tempfile
    tmp = tempfile.mkdtemp(prefix="qj_")
    dbp = os.path.join(tmp, "test.db")
    init_db(dbp)

    # write + read
    ins, sk = write_jobs(dbp, [
        {
            "jid": "j1", "backend": "ibm_fez", "status": "Completed",
            "program": "sampler", "shots": 10000, "created": "2025-12-31T00:00:00Z",
            "qasm": "OPENQASM 3.0; h $0;", "provenance": "test",
            "samples_count": 10000,
        },
        {
            "jid": "j1", "backend": "ibm_fez", "status": "Completed",
            "program": "sampler", "shots": 10000, "created": "2025-12-31T00:00:00Z",
            "qasm": "OPENQASM 3.0; h $0;", "provenance": "test",
            "samples_count": 10000,  # duplicate jid -> skipped
        },
        {
            "jid": "j2", "backend": "ibm_torino", "status": "Completed",
            "program": "sampler", "shots": 0, "created": "2025-12-30T00:00:00Z",
            "provenance": "test",
        },
    ])
    assert ins == 2 and sk == 1, f"insert/skip: {ins}/{sk}"

    s = stats(dbp)
    print("stats:", s)
    assert s["jobs"] == 2
    assert s["shots_total"] == 10000

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("SCHEMA SELF-TEST PASSED")
