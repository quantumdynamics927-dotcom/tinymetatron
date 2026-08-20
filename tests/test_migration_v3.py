"""
tests/test_migration_v3.py
==========================
Automated tests for loop schema migration v3.

Migration v3 must remove the stale FOREIGN KEY on loop_events.run_id that
migration v2 failed to drop (it only handled the NOT NULL variant). It must:
  * detect the FK via PRAGMA foreign_key_list, not migration-version state
  * back up the DB before mutating
  * rebuild loop_events transactionally, preserving every row
  * recreate the index and any triggers
  * record version 3 in the ledger only after integrity checks pass
  * be idempotent (rerun on a corrected DB is a no-op)
  * fail closed and leave other tables' FKs intact

We construct a synthetic DB with the OLD loop_events schema (nullable FK on
run_id) and seed event rows — including experiment-level rows whose run_id is
an exp_id, which are the rows the FK used to break.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

import db

# Full loop schema as it existed pre-v3: identical to the canonical schema
# except loop_events has the stale nullable FK on run_id → loop_runs.
_OLD_SCHEMA = """
CREATE TABLE loop_schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
CREATE TABLE loop_experiments (
    exp_id        TEXT PRIMARY KEY,
    state         TEXT NOT NULL DEFAULT 'ACTIVE',
    hypothesis    TEXT,
    created_at    TEXT NOT NULL,
    ended_at      TEXT
);
CREATE TABLE loop_runs (
    run_id            TEXT PRIMARY KEY,
    exp_id            TEXT REFERENCES loop_experiments(exp_id),
    parent_run        TEXT,
    corpus_hash       TEXT,
    split_hash        TEXT,
    tokenizer_hash    TEXT,
    model_config_hash TEXT,
    seed              INTEGER,
    seq_len           INTEGER,
    status            TEXT NOT NULL DEFAULT 'NEW',
    promotion         TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    ended_at          TEXT
);
CREATE TABLE loop_checkpoints (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES loop_runs(run_id),
    step          INTEGER,
    file_path     TEXT,
    val_ce        REAL,
    train_ce      REAL,
    is_best       INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL
);
CREATE TABLE evaluations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES loop_runs(run_id),
    step          INTEGER,
    eval_set      TEXT NOT NULL,
    ce            REAL,
    ppl           REAL,
    total_tokens  INTEGER,
    created_at    TEXT NOT NULL
);
CREATE TABLE gate_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES loop_runs(run_id),
    gate_name     TEXT NOT NULL,
    passed        INTEGER,
    duration_s    REAL,
    stdout_path   TEXT,
    stderr_path   TEXT,
    created_at    TEXT NOT NULL
);
CREATE TABLE loop_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT REFERENCES loop_runs(run_id),
    from_state    TEXT,
    to_state      TEXT NOT NULL,
    reason_code   TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE artifact_refs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES loop_runs(run_id),
    artifact_type TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    UNIQUE(run_id, artifact_type, sha256)
);
CREATE TABLE final_test_consumed (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_sha256     TEXT NOT NULL,
    test_manifest_sha     TEXT NOT NULL,
    run_id                TEXT NOT NULL REFERENCES loop_runs(run_id),
    ce                    REAL,
    ppl                   REAL,
    consumed_at           TEXT NOT NULL,
    UNIQUE(candidate_sha256, test_manifest_sha)
);
CREATE INDEX idx_loop_events_run ON loop_events(run_id);
CREATE TRIGGER trg_loop_events_bi BEFORE INSERT ON loop_events
BEGIN
    SELECT 1;
END;
"""

# Event rows as they exist in a stale DB. Row 2 is experiment-level (run_id is
# an exp_id with no matching loop_runs row) — exactly what the FK used to break.
_EVENT_ROWS = [
    (1, "run-1", "NEW", "TRAINING", "train_start", '{"step": 0}', "2026-01-01T00:00:00Z"),
    (2, "exp-1", None, "FROZEN_CORPUS", "corpus_pipeline_complete",
     '{"corpus_hash": "d15eb2eae27587fc"}', "2026-02-02T00:00:00Z"),
    (3, None, "TRAINING", "EVALUATING", "eval_done", '{}', "2026-03-03T00:00:00Z"),
]


def _build_stale_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys=OFF")  # allow experiment-level event rows
    conn.executescript(_OLD_SCHEMA)
    conn.execute(
        "INSERT INTO loop_schema_version (version, applied_at) VALUES (1, ?)",
        ("2026-01-01T00:00:00Z",),
    )
    conn.execute(
        "INSERT INTO loop_schema_version (version, applied_at) VALUES (2, ?)",
        ("2026-01-01T00:00:00Z",),
    )
    conn.executemany(
        "INSERT INTO loop_events "
        "(id, run_id, from_state, to_state, reason_code, payload_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        _EVENT_ROWS,
    )
    conn.commit()
    conn.close()


def _loop_events_rows(path: Path) -> list[tuple]:
    conn = sqlite3.connect(str(path))
    rows = conn.execute(
        "SELECT id, run_id, from_state, to_state, reason_code, payload_json, created_at "
        "FROM loop_events ORDER BY id"
    ).fetchall()
    conn.close()
    return rows


def _stale_fk_count(path: Path) -> int:
    conn = sqlite3.connect(str(path))
    fks = conn.execute("PRAGMA foreign_key_list(loop_events)").fetchall()
    conn.close()
    return sum(1 for fk in fks if fk[2] == "loop_runs" and fk[3] == "run_id")


def _backups(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{path.name}.bak-*"))


@pytest.fixture
def stale_db(tmp_path) -> Path:
    p = tmp_path / "registry.db"
    _build_stale_db(p)
    return p


# ── migration removes the FK, preserves rows, records ledger ──────────────────

def test_migration_v3_removes_fk_and_preserves_rows(stale_db) -> None:
    assert _stale_fk_count(stale_db) == 1, "synthetic DB must start with stale FK"
    before = _loop_events_rows(stale_db)

    db.init_db(str(stale_db))

    assert _stale_fk_count(stale_db) == 0, "stale FK must be removed"
    assert _loop_events_rows(stale_db) == before, "event rows must be preserved"
    assert _backups(stale_db), "a timestamped backup must have been created"

    conn = sqlite3.connect(str(stale_db))
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    versions = {r[0] for r in conn.execute(
        "SELECT version FROM loop_schema_version").fetchall()}
    conn.close()
    assert 3 in versions, "migration v3 must be recorded in the ledger"

    # Other tables' FKs must be untouched.
    conn = sqlite3.connect(str(stale_db))
    for table in ("artifact_refs", "gate_results", "evaluations",
                  "loop_checkpoints", "final_test_consumed"):
        fks = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        assert any(fk[2] == "loop_runs" for fk in fks), (
            f"{table} must keep its loop_runs FK")
    conn.close()


# ── index + trigger recreated ─────────────────────────────────────────────────

def test_migration_v3_recreates_index_and_triggers(stale_db) -> None:
    db.init_db(str(stale_db))
    conn = sqlite3.connect(str(stale_db))
    idx = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_loop_events_run'"
    ).fetchall()
    trg = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='loop_events'"
    ).fetchall()
    conn.close()
    assert [r[0] for r in idx] == ["idx_loop_events_run"]
    assert [r[0] for r in trg] == ["trg_loop_events_bi"]


# ── idempotent: rerun on a corrected DB is a no-op ────────────────────────────

def test_migration_v3_idempotent(stale_db) -> None:
    db.init_db(str(stale_db))
    backup_count = len(_backups(stale_db))
    rows = _loop_events_rows(stale_db)

    # Rerun even with the ledger row deleted (detection-level idempotency).
    conn = sqlite3.connect(str(stale_db))
    conn.execute("DELETE FROM loop_schema_version WHERE version=3")
    conn.commit()
    conn.close()

    db.init_db(str(stale_db))

    assert _stale_fk_count(stale_db) == 0
    assert _loop_events_rows(stale_db) == rows
    assert len(_backups(stale_db)) == backup_count, "no new backup on no-op rerun"
    conn = sqlite3.connect(str(stale_db))
    versions = {r[0] for r in conn.execute(
        "SELECT version FROM loop_schema_version").fetchall()}
    conn.close()
    assert 3 in versions


# ── fresh DB (no stale FK) is a no-op: no backup, no change ──────────────────

def test_migration_v3_noop_on_fresh_db(tmp_path) -> None:
    p = tmp_path / "registry.db"
    db.init_db(str(p))  # fresh DB already has the canonical (FK-free) schema
    assert _stale_fk_count(p) == 0
    assert not _backups(p), "no backup should be created when nothing to migrate"


# ── NEW → FROZEN_CORPUS transition succeeds after migration ──────────────────

def test_migration_v3_unblocks_frozen_corpus_transition(stale_db, monkeypatch) -> None:
    db.init_db(str(stale_db))
    monkeypatch.setattr(db, "_DB_PATH", str(stale_db))

    # Pre-migration this insert violated the FK; post-migration it must persist.
    db.create_loop_experiment("exp-v3-live", hypothesis="migration v3 check")
    db.update_loop_experiment_state(
        "exp-v3-live", "FROZEN_CORPUS", "corpus_pipeline_complete",
        payload={"corpus_hash": "d15eb2eae27587fc"},
    )

    exp = db.get_loop_experiment("exp-v3-live")
    assert exp is not None and exp["state"] == "FROZEN_CORPUS"

    conn = sqlite3.connect(str(stale_db))
    events = conn.execute(
        "SELECT run_id, to_state, reason_code, payload_json FROM loop_events "
        "WHERE run_id='exp-v3-live'"
    ).fetchall()
    conn.close()
    assert len(events) == 1, "the loop event must persist"
    assert events[0][1] == "FROZEN_CORPUS"
    assert "d15eb2eae27587fc" in events[0][3]