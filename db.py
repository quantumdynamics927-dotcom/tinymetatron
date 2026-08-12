"""
db.py
=====
SQLite data layer for the TinyMetatron SLM.

Original tables:
  - training_data      : corpus rows scored by quality.score_quality
  - model_checkpoints  : one row per saved .pt (exactly one is_active=1)
  - training_sessions  : one row per training run

Loop tables (versioned, WAL mode):
  - loop_experiments   : one row per parent experiment
  - loop_runs          : one row per child run (one per seed)
  - loop_checkpoints   : checkpoints for loop runs
  - evaluations         : evaluation results against named eval sets
  - gate_results        : gate pass/fail results
  - loop_events         : state transition log
  - artifact_refs       : artifact file references with SHA256
  - final_test_consumed : one-time final-test consumption lock

All functions take an explicit ``path`` so callers (CLI, API, tests) can point
at a temp DB.  For the default registry DB, use ``get_db()`` or set
``TINYMETATRON_DB`` env var before importing.

``add_texts`` scores each text via ``quality.score_quality`` (T2 sibling module,
built in parallel). The import is done lazily inside ``add_texts`` so this
module imports cleanly even before ``quality.py`` exists.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

# Module-level default DB path; set TINYMETATRON_DB env to override
_DB_PATH = os.environ.get(
    "TINYMETATRON_DB",
    str(Path(__file__).parent / "state" / "registry.db"),
)


def get_db() -> sqlite3.Connection:
    """Return a connection to the default registry DB (WAL + FK enforcement on)."""
    return _connect(_DB_PATH)


def set_db_path(path: str) -> None:
    """Override the default registry DB path for subsequent calls."""
    global _DB_PATH
    _DB_PATH = path


# ── Schema DDL ──────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_data (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    text              TEXT    NOT NULL,
    domain            TEXT    NOT NULL,
    quality_score     REAL    NOT NULL,
    used_in_training  INTEGER NOT NULL DEFAULT 0,
    split            TEXT    NOT NULL DEFAULT 'train',
    created_at        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS model_checkpoints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    step        INTEGER,
    loss        REAL,
    val_loss    REAL,
    file_path   TEXT    NOT NULL,
    is_active   INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS training_sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    domain_filter TEXT,
    min_quality   REAL,
    start_time    TEXT    NOT NULL,
    end_time      TEXT,
    total_steps   INTEGER,
    final_loss    REAL
);

CREATE INDEX IF NOT EXISTS idx_training_data_domain_quality
    ON training_data(domain, quality_score);

CREATE INDEX IF NOT EXISTS idx_model_checkpoints_active
    ON model_checkpoints(is_active);
"""


def _connect(path: str) -> sqlite3.Connection:
    """Open a connection with row factory + foreign-enforcing pragma + WAL mode."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _now() -> str:
    """ISO-8601 UTC timestamp (deterministic, tz-aware)."""
    return datetime.now(timezone.utc).isoformat()


# ── Loop schema (versioned migrations) ─────────────────────────────────────────

_LOOP_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS loop_schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS loop_experiments (
    exp_id        TEXT PRIMARY KEY,
    state         TEXT NOT NULL DEFAULT 'ACTIVE',
    hypothesis    TEXT,
    created_at    TEXT NOT NULL,
    ended_at      TEXT
);

CREATE TABLE IF NOT EXISTS loop_runs (
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

CREATE TABLE IF NOT EXISTS loop_checkpoints (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES loop_runs(run_id),
    step          INTEGER,
    file_path     TEXT,
    val_ce        REAL,
    train_ce      REAL,
    is_best       INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES loop_runs(run_id),
    step          INTEGER,
    eval_set      TEXT NOT NULL,
    ce            REAL,
    ppl           REAL,
    total_tokens  INTEGER,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gate_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES loop_runs(run_id),
    gate_name     TEXT NOT NULL,
    passed        INTEGER,
    duration_s    REAL,
    stdout_path   TEXT,
    stderr_path   TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS loop_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT REFERENCES loop_runs(run_id),
    from_state    TEXT,
    to_state      TEXT NOT NULL,
    reason_code   TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_refs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES loop_runs(run_id),
    artifact_type TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    UNIQUE(run_id, artifact_type, sha256)
);

CREATE TABLE IF NOT EXISTS final_test_consumed (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_sha256     TEXT NOT NULL,
    test_manifest_sha     TEXT NOT NULL,
    run_id                TEXT NOT NULL REFERENCES loop_runs(run_id),
    ce                    REAL,
    ppl                   REAL,
    consumed_at           TEXT NOT NULL,
    UNIQUE(candidate_sha256, test_manifest_sha)
);

CREATE INDEX IF NOT EXISTS idx_loop_runs_exp ON loop_runs(exp_id);
CREATE INDEX IF NOT EXISTS idx_loop_checkpoints_run ON loop_checkpoints(run_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_loop_events_run ON loop_events(run_id);
CREATE INDEX IF NOT EXISTS idx_artifact_refs_run ON artifact_refs(run_id);
"""

_CURRENT_LOOP_VERSION = 1


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply any unapplied loop schema migrations."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS loop_schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL)
    """)
    applied = {r[0] for r in conn.execute(
        "SELECT version FROM loop_schema_version").fetchall()}
    if _CURRENT_LOOP_VERSION not in applied:
        conn.executescript(_LOOP_SCHEMA_SQL)
        conn.execute(
            "INSERT INTO loop_schema_version (version, applied_at) VALUES (?, ?)",
            (_CURRENT_LOOP_VERSION, _now()),
        )
        conn.commit()


def init_db(path: str) -> None:
    """
    Create the three original tables + indexes if they do not already exist.
    Also runs live schema migrations (ALTER TABLE ADD COLUMN) for new columns
    added after initial deployment so existing .db files stay current.
    Also applies loop schema migrations (versioned, transactional).
    """
    conn = _connect(path)
    try:
        conn.executescript(_SCHEMA)
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(training_data)").fetchall()}
        if "split" not in cols:
            conn.execute(
                "ALTER TABLE training_data ADD COLUMN split TEXT NOT NULL DEFAULT 'train'")
        mcols = {r[1] for r in conn.execute(
            "PRAGMA table_info(model_checkpoints)").fetchall()}
        if "val_loss" not in mcols:
            conn.execute(
                "ALTER TABLE model_checkpoints ADD COLUMN val_loss REAL")
        conn.commit()
        _apply_migrations(conn)
    finally:
        conn.close()


# ── training_data ───────────────────────────────────────────────────────────

def add_texts(path: str, texts: Iterable[str], domain: str,
              quality_threshold: float, split: str = "train") -> tuple[int, int]:
    """
    Score each text with quality.score_quality and insert those whose score
    is >= quality_threshold. Returns (added, rejected).

    ``split`` is either 'train' or 'val'; only 'train' rows are used by
    fetch_training_rows by default.

    ``quality`` is imported lazily so this module can be imported before
    quality.py exists; only add_texts actually needs the scorer.
    """
    from quality import score_quality  # T2 sibling; built in parallel

    conn = _connect(path)
    added = rejected = 0
    try:
        for text in texts:
            try:
                score = float(score_quality(text))
            except Exception:
                rejected += 1
                continue
            if score < quality_threshold:
                rejected += 1
                continue
            conn.execute(
                "INSERT INTO training_data "
                "(text, domain, quality_score, used_in_training, split, created_at) "
                "VALUES (?, ?, ?, 0, ?, ?)",
                (text, domain, score, split, _now()),
            )
            added += 1
        conn.commit()
    finally:
        conn.close()
    return added, rejected


def fetch_training_rows(path: str, domain: str, min_quality: float,
                       limit: int, used: bool = False,
                       split: str = "train") -> list[dict]:
    """
    Return up to ``limit`` training rows for ``domain`` with quality_score >=
    min_quality. ``used`` selects rows already used in training (default
    False = fetch unused rows for a fresh training run). ``split``
    filters by data split ('train' or 'val').
    """
    conn = _connect(path)
    try:
        cur = conn.execute(
            "SELECT id, text, domain, quality_score, used_in_training, created_at "
            "FROM training_data "
            "WHERE domain = ? AND quality_score >= ? AND used_in_training = ? "
            "AND split = ? "
            "ORDER BY quality_score DESC, id ASC LIMIT ?",
            (domain, min_quality, 1 if used else 0, split, int(limit)),
        )
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    return rows


def mark_used(path: str, ids: Iterable[int]) -> int:
    """Mark the given row ids as used_in_training=1. Returns count updated."""
    ids = [int(i) for i in ids]
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    conn = _connect(path)
    try:
        cur = conn.execute(
            f"UPDATE training_data SET used_in_training = 1 "
            f"WHERE id IN ({placeholders})",
            ids,
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def delete_low_quality(path: str, min_quality: float) -> int:
    """Delete training_data rows with quality_score < min_quality. Returns count."""
    conn = _connect(path)
    try:
        cur = conn.execute(
            "DELETE FROM training_data WHERE quality_score < ?",
            (min_quality,),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def stats(path: str) -> dict:
    """
    Return {total, by_domain, avg_quality, used_in_training} for training_data.
    """
    conn = _connect(path)
    try:
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM training_data"
        ).fetchone()["n"]
        used = conn.execute(
            "SELECT COUNT(*) AS n FROM training_data WHERE used_in_training = 1"
        ).fetchone()["n"]
        avg = conn.execute(
            "SELECT AVG(quality_score) AS a FROM training_data"
        ).fetchone()["a"]
        by_domain_rows = conn.execute(
            "SELECT domain, COUNT(*) AS n FROM training_data GROUP BY domain"
        ).fetchall()
    finally:
        conn.close()
    return {
        "total": int(total),
        "by_domain": {r["domain"]: int(r["n"]) for r in by_domain_rows},
        "avg_quality": float(avg) if avg is not None else 0.0,
        "used_in_training": int(used),
    }


# ── model_checkpoints ───────────────────────────────────────────────────────

def save_checkpoint(path: str, step: int, loss: float, file_path: str,
                    is_active: bool = True, val_loss: Optional[float] = None) -> int:
    """
    Insert a model_checkpoints row. When is_active=True (default), clear the
    prior active row first so at most one row has is_active=1. Returns the new
    row id. ``val_loss`` stores the held-out validation CE for overfit detection.
    """
    conn = _connect(path)
    try:
        if is_active:
            conn.execute("UPDATE model_checkpoints SET is_active = 0")
        cur = conn.execute(
            "INSERT INTO model_checkpoints "
            "(step, loss, val_loss, file_path, is_active, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (int(step), float(loss), float(val_loss) if val_loss is not None else None,
             file_path, 1 if is_active else 0, _now()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_active_checkpoint(path: str) -> Optional[dict]:
    """Return the active checkpoint row as a dict, or None."""
    conn = _connect(path)
    try:
        row = conn.execute(
            "SELECT id, step, loss, val_loss, file_path, is_active, created_at "
            "FROM model_checkpoints WHERE is_active = 1 LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def set_active_checkpoint(path: str, file_path: str) -> None:
    """
    Mark the checkpoint with the given file_path active and deactivate all
    others. No-op if no row matches file_path.
    """
    conn = _connect(path)
    try:
        row = conn.execute(
            "SELECT id FROM model_checkpoints WHERE file_path = ? LIMIT 1",
            (file_path,),
        ).fetchone()
        if row is None:
            return
        conn.execute("UPDATE model_checkpoints SET is_active = 0")
        conn.execute(
            "UPDATE model_checkpoints SET is_active = 1 WHERE id = ?",
            (row["id"],),
        )
        conn.commit()
    finally:
        conn.close()


# ── training_sessions ───────────────────────────────────────────────────────

def start_session(path: str, domain_filter: Optional[str] = None,
                  min_quality: Optional[float] = None) -> int:
    """Insert a training_sessions row with start_time=now, return its id."""
    conn = _connect(path)
    try:
        cur = conn.execute(
            "INSERT INTO training_sessions "
            "(domain_filter, min_quality, start_time) VALUES (?, ?, ?)",
            (domain_filter, min_quality, _now()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def end_session(path: str, session_id: int, total_steps: int,
                final_loss: float) -> None:
    """Set end_time, total_steps, final_loss on the given session row."""
    conn = _connect(path)
    try:
        conn.execute(
            "UPDATE training_sessions "
            "SET end_time = ?, total_steps = ?, final_loss = ? "
            "WHERE id = ?",
            (_now(), int(total_steps), float(final_loss), int(session_id)),
        )
        conn.commit()
    finally:
        conn.close()


# ── loop_experiments ──────────────────────────────────────────────────────────────

def create_loop_experiment(path: str, exp_id: str,
                          hypothesis: str = "") -> None:
    """Create a new loop_experiments row. Raises if exp_id already exists."""
    conn = _connect(path)
    try:
        conn.execute(
            "INSERT INTO loop_experiments (exp_id, hypothesis, created_at) "
            "VALUES (?, ?, ?)",
            (exp_id, hypothesis, _now()),
        )
        conn.commit()
    finally:
        conn.close()


def update_loop_experiment_state(path: str, exp_id: str, state: str,
                               reason_code: str = "",
                               payload: dict = None) -> None:
    """
    Update the state of a loop experiment. Sets ended_at if terminal.
    Records a loop_event for the experiment.
    """
    conn = _connect(path)
    try:
        conn.execute(
            "UPDATE loop_experiments SET state=?, ended_at=? WHERE exp_id=?",
            (state, _now() if state in ("FROZEN", "ARCHIVED") else None, exp_id))
        conn.execute(
            "INSERT INTO loop_events "
            "(run_id, from_state, to_state, reason_code, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (exp_id, None, state, reason_code,
             json.dumps(payload or {}), _now()),
        )
        conn.commit()
    finally:
        conn.close()


def get_loop_experiment(path: str, exp_id: str) -> Optional[dict]:
    """Return a loop experiment row, or None."""
    conn = _connect(path)
    try:
        row = conn.execute(
            "SELECT * FROM loop_experiments WHERE exp_id=?", (exp_id,)
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


# ── loop_runs ─────────────────────────────────────────────────────────────────

def create_loop_run(path: str, run_id: str, exp_id: str,
                    corpus_hash: str, split_hash: str,
                    tokenizer_hash: str, model_config_hash: str,
                    seed: int, seq_len: int,
                    parent_run: Optional[str] = None) -> None:
    """Create a new loop_runs row."""
    conn = _connect(path)
    try:
        conn.execute(
            "INSERT INTO loop_runs "
            "(run_id, exp_id, parent_run, corpus_hash, split_hash, "
            "tokenizer_hash, model_config_hash, seed, seq_len, "
            "status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'NEW', ?, ?)",
            (run_id, exp_id, parent_run, corpus_hash, split_hash,
             tokenizer_hash, model_config_hash, seed, seq_len, _now(), _now()),
        )
        conn.commit()
    finally:
        conn.close()


def update_loop_run_status(path: str, run_id: str, status: str,
                          reason_code: str, payload: dict) -> None:
    """Transition a loop run's status and record a loop_events entry."""
    conn = _connect(path)
    try:
        old = conn.execute(
            "SELECT status FROM loop_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        old_status = dict(old)["status"] if old else None

        conn.execute(
            "UPDATE loop_runs SET status=?, updated_at=?, ended_at=? "
            "WHERE run_id=?",
            (status, _now(), _now() if status in (
                "PROMOTED", "REJECTED", "ARCHIVED") else None, run_id))
        conn.execute(
            "INSERT INTO loop_events "
            "(run_id, from_state, to_state, reason_code, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, old_status, status, reason_code,
             json.dumps(payload), _now()),
        )
        conn.commit()
    finally:
        conn.close()


def set_promotion(path: str, run_id: str, promotion: str) -> None:
    """Set the promotion field on a loop run."""
    conn = _connect(path)
    try:
        conn.execute(
            "UPDATE loop_runs SET promotion=? WHERE run_id=?",
            (promotion, run_id))
        conn.commit()
    finally:
        conn.close()


def get_loop_run(path: str, run_id: str) -> Optional[dict]:
    """Return a loop run row, or None."""
    conn = _connect(path)
    try:
        row = conn.execute(
            "SELECT * FROM loop_runs WHERE run_id=?", (run_id,)
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def get_loop_runs_for_experiment(path: str, exp_id: str) -> list[dict]:
    """Return all loop_runs for an experiment."""
    conn = _connect(path)
    try:
        rows = conn.execute(
            "SELECT * FROM loop_runs WHERE exp_id=? ORDER BY created_at",
            (exp_id,)).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ── loop_checkpoints ──────────────────────────────────────────────────────────

def save_loop_checkpoint(path: str, run_id: str, step: int,
                        file_path: str, val_ce: float,
                        train_ce: float, is_best: bool = False) -> int:
    """Save a checkpoint record. Clears is_best on prior best."""
    conn = _connect(path)
    try:
        if is_best:
            conn.execute(
                "UPDATE loop_checkpoints SET is_best=0 WHERE run_id=?",
                (run_id,))
        cur = conn.execute(
            "INSERT INTO loop_checkpoints "
            "(run_id, step, file_path, val_ce, train_ce, is_best, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, step, file_path, val_ce, train_ce, 1 if is_best else 0, _now()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_best_loop_checkpoint(path: str, run_id: str) -> Optional[dict]:
    """Return the is_best=True checkpoint for a run, or None."""
    conn = _connect(path)
    try:
        row = conn.execute(
            "SELECT * FROM loop_checkpoints WHERE run_id=? AND is_best=1 LIMIT 1",
            (run_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def get_loop_checkpoints(path: str, run_id: str) -> list[dict]:
    """Return all checkpoints for a run, ordered by step."""
    conn = _connect(path)
    try:
        rows = conn.execute(
            "SELECT * FROM loop_checkpoints WHERE run_id=? ORDER BY step",
            (run_id,)).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ── evaluations ────────────────────────────────────────────────────────────────

def save_evaluation(path: str, run_id: str, step: int,
                    eval_set: str, ce: float, ppl: float,
                    total_tokens: int) -> int:
    """Save an evaluation result."""
    conn = _connect(path)
    try:
        cur = conn.execute(
            "INSERT INTO evaluations "
            "(run_id, step, eval_set, ce, ppl, total_tokens, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, step, eval_set, ce, ppl, total_tokens, _now()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_evaluations(path: str, run_id: str,
                    eval_set: Optional[str] = None) -> list[dict]:
    """Return evaluations for a run, optionally filtered by eval_set."""
    conn = _connect(path)
    try:
        if eval_set:
            rows = conn.execute(
                "SELECT * FROM evaluations WHERE run_id=? AND eval_set=? ORDER BY step",
                (run_id, eval_set)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM evaluations WHERE run_id=? ORDER BY step",
                (run_id,)).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ── gate_results ─────────────────────────────────────────────────────────────

def save_gate_result(path: str, run_id: str, gate_name: str,
                     passed: bool, duration_s: float,
                     stdout_path: str, stderr_path: str) -> int:
    """Save a gate result. Overwrites prior result for same gate/run."""
    conn = _connect(path)
    try:
        conn.execute(
            "DELETE FROM gate_results WHERE run_id=? AND gate_name=?",
            (run_id, gate_name))
        cur = conn.execute(
            "INSERT INTO gate_results "
            "(run_id, gate_name, passed, duration_s, stdout_path, stderr_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, gate_name, 1 if passed else 0, duration_s,
             stdout_path, stderr_path, _now()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_gate_results(path: str, run_id: str) -> list[dict]:
    """Return all gate results for a run."""
    conn = _connect(path)
    try:
        rows = conn.execute(
            "SELECT * FROM gate_results WHERE run_id=?", (run_id,)).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ── artifact_refs ─────────────────────────────────────────────────────────────

def save_artifact_ref(path: str, run_id: str, artifact_type: str,
                      file_path: str, sha256: str) -> None:
    """Save an artifact reference. Idempotent (UNIQUE constraint)."""
    conn = _connect(path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO artifact_refs "
            "(run_id, artifact_type, file_path, sha256, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, artifact_type, file_path, sha256, _now()),
        )
        conn.commit()
    finally:
        conn.close()


# ── final_test_consumed ───────────────────────────────────────────────────────

def try_consume_final_test(path: str, candidate_sha256: str,
                          test_manifest_sha: str, run_id: str,
                          ce: float, ppl: float) -> tuple[bool, Optional[dict]]:
    """
    Attempt to record final-test consumption.

    Returns (True, None) if successfully consumed (inserted).
    Returns (False, existing_row) if already consumed (enforces uniqueness).

    This is the database-enforced "evaluate exactly once" gate.
    """
    conn = _connect(path)
    try:
        existing = conn.execute(
            "SELECT * FROM final_test_consumed "
            "WHERE candidate_sha256=? AND test_manifest_sha=?",
            (candidate_sha256, test_manifest_sha)
        ).fetchone()
        if existing:
            conn.close()
            return False, dict(existing)
        conn.execute(
            "INSERT INTO final_test_consumed "
            "(candidate_sha256, test_manifest_sha, run_id, ce, ppl, consumed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (candidate_sha256, test_manifest_sha, run_id, ce, ppl, _now()),
        )
        conn.commit()
        conn.close()
        return True, None
    except Exception:
        conn.close()
        raise


# ── self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import types
    import tempfile

    sys.stdout.reconfigure(encoding="utf-8")

    if "quality" not in sys.modules:
        _stub = types.ModuleType("quality")

        def _score_quality(text: str) -> float:
            t = text.strip()
            if not t:
                return 0.0
            length_term = min(len(t), 200) / 200.0
            uniq_term = len(set(t.split())) / max(len(t.split()), 1)
            return round(0.6 * length_term + 0.4 * uniq_term, 4)

        _stub.score_quality = _score_quality
        sys.modules["quality"] = _stub

    tmpdir = tempfile.mkdtemp(prefix="tinymetatron_dbtest_")
    db_path = os.path.join(tmpdir, "test.db")

    def _ok(cond: bool, msg: str) -> None:
        print(("OK  " if cond else "FAIL") + " " + msg)
        assert cond, msg

    init_db(db_path)
    init_db(db_path)

    texts = [
        "Packet filtering firewall rules block unauthorized traffic by port and protocol.",
        "a",
        "Repeat repeat repeat repeat repeat repeat repeat repeat.",
        "Quantization reduces model precision to int8 to shrink memory footprint.",
    ]
    added, rejected = add_texts(db_path, texts, "cybersecurity", quality_threshold=0.4)
    print(f"add_texts: added={added} rejected={rejected}")
    _ok(added >= 1, "at least one text inserted")
    _ok(added + rejected == len(texts), "every text classified exactly once")

    rows = fetch_training_rows(db_path, "cybersecurity", 0.4, limit=10, used=False)
    _ok(len(rows) == added, f"fetched {len(rows)} == added {added}")
    _ok(isinstance(rows[0], dict) and "text" in rows[0], "rows are dicts with text")
    _ok(all(r["used_in_training"] == 0 for r in rows), "rows start unused")

    ids = [r["id"] for r in rows]
    n = mark_used(db_path, ids)
    _ok(n == len(ids), f"mark_used updated {n} == {len(ids)}")
    used_rows = fetch_training_rows(db_path, "cybersecurity", 0.4, limit=10, used=True)
    _ok(len(used_rows) == len(ids), "used=True returns the marked rows")
    _ok(len(fetch_training_rows(db_path, "cybersecurity", 0.4, 10, used=False)) == 0,
        "no unused rows remain")

    s = stats(db_path)
    print("stats:", s)
    _ok(set(s.keys()) == {"total", "by_domain", "avg_quality", "used_in_training"},
        "stats has the four contract keys")
    _ok(s["total"] == added, "stats total == added")
    _ok(s["used_in_training"] == added, "stats used == added")
    _ok("cybersecurity" in s["by_domain"], "by_domain contains cybersecurity")

    cid1 = save_checkpoint(db_path, step=10, loss=2.5,
                           file_path=os.path.join(tmpdir, "ckpt1.pt"))
    act = get_active_checkpoint(db_path)
    _ok(act is not None and act["id"] == cid1 and act["is_active"] == 1,
        "first checkpoint is active")
    cid2 = save_checkpoint(db_path, step=20, loss=1.2,
                           file_path=os.path.join(tmpdir, "ckpt2.pt"))
    act2 = get_active_checkpoint(db_path)
    _ok(act2 is not None and act2["id"] == cid2 and act2["is_active"] == 1,
        "new checkpoint becomes active")
    conn = _connect(db_path)
    n_active = conn.execute(
        "SELECT COUNT(*) AS n FROM model_checkpoints WHERE is_active = 1"
    ).fetchone()["n"]
    conn.close()
    _ok(n_active == 1, f"exactly one active checkpoint (got {n_active})")

    set_active_checkpoint(db_path, os.path.join(tmpdir, "ckpt1.pt"))
    act3 = get_active_checkpoint(db_path)
    _ok(act3 is not None and act3["id"] == cid1, "set_active_checkpoint reactivates ckpt1")
    conn = _connect(db_path)
    n_active = conn.execute(
        "SELECT COUNT(*) AS n FROM model_checkpoints WHERE is_active = 1"
    ).fetchone()["n"]
    conn.close()
    _ok(n_active == 1, "still exactly one active after set_active")

    save_checkpoint(db_path, step=30, loss=0.7,
                    file_path=os.path.join(tmpdir, "ckpt3.pt"), is_active=False)
    act4 = get_active_checkpoint(db_path)
    _ok(act4 is not None and act4["id"] == cid1,
        "is_active=False leaves the active checkpoint untouched")

    sid = start_session(db_path, domain_filter="cybersecurity", min_quality=0.4)
    _ok(isinstance(sid, int) and sid > 0, f"start_session returned id={sid}")
    end_session(db_path, sid, total_steps=200, final_loss=0.42)
    conn = _connect(db_path)
    srow = conn.execute(
        "SELECT * FROM training_sessions WHERE id = ?", (sid,)
    ).fetchone()
    conn.close()
    _ok(srow is not None and srow["end_time"] is not None
        and srow["total_steps"] == 200 and abs(srow["final_loss"] - 0.42) < 1e-9,
        "end_session wrote end_time/total_steps/final_loss")

    before = stats(db_path)["total"]
    conn = _connect(db_path)
    conn.execute(
        "INSERT INTO training_data (text, domain, quality_score, used_in_training, created_at) "
        "VALUES (?, ?, ?, 0, ?)",
        ("low", "cybersecurity", 0.05, _now()),
    )
    conn.commit()
    conn.close()
    deleted = delete_low_quality(db_path, 0.4)
    _ok(deleted >= 1, f"delete_low_quality removed {deleted} row(s)")
    after = stats(db_path)["total"]
    _ok(after == before, "delete_low_quality removed exactly the injected low row")

    try:
        os.remove(db_path)
    except OSError:
        pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    print("\nSELF-TEST PASSED")
