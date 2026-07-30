"""
db.py
=====
SQLite data layer for the TinyMetatron SLM.

Owns three tables:
  - training_data      : corpus rows scored by quality.score_quality
  - model_checkpoints  : one row per saved .pt (exactly one is_active=1)
  - training_sessions   : one row per training run

All functions take an explicit ``path`` so callers (CLI, API, tests) can point
at a temp DB; nothing here hard-codes ``metatron.db`` (contract rule 0.6: do
NOT create a persistent metatron.db during implementation).

``add_texts`` scores each text via ``quality.score_quality`` (T2 sibling module,
built in parallel). The import is done lazily inside ``add_texts`` so this
module imports cleanly even before ``quality.py`` exists.

Schema design follows the IMPLEMENTATION_CONTRACT.md section 2 interface.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Optional


# ── Schema DDL ──────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_data (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    text              TEXT    NOT NULL,
    domain            TEXT    NOT NULL,
    quality_score     REAL    NOT NULL,
    used_in_training  INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS model_checkpoints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    step        INTEGER,
    loss        REAL,
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
    """Open a connection with row factory + foreign-enforcing pragma."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _now() -> str:
    """ISO-8601 UTC timestamp (deterministic, tz-aware)."""
    return datetime.now(timezone.utc).isoformat()


def init_db(path: str) -> None:
    """Create the three tables + indexes if they do not already exist."""
    conn = _connect(path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


# ── training_data ───────────────────────────────────────────────────────────

def add_texts(path: str, texts: Iterable[str], domain: str,
              quality_threshold: float) -> tuple[int, int]:
    """
    Score each text with quality.score_quality and insert those whose score
    is >= quality_threshold. Returns (added, rejected).

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
                # A text that cannot be scored is rejected, not fatal.
                rejected += 1
                continue
            if score < quality_threshold:
                rejected += 1
                continue
            conn.execute(
                "INSERT INTO training_data "
                "(text, domain, quality_score, used_in_training, created_at) "
                "VALUES (?, ?, ?, 0, ?)",
                (text, domain, score, _now()),
            )
            added += 1
        conn.commit()
    finally:
        conn.close()
    return added, rejected


def fetch_training_rows(path: str, domain: str, min_quality: float,
                        limit: int, used: bool = False) -> list[dict]:
    """
    Return up to ``limit`` training rows for ``domain`` with quality_score >=
    min_quality. ``used`` selects rows already used in training (default
    False = fetch unused rows for a fresh training run).
    """
    conn = _connect(path)
    try:
        cur = conn.execute(
            "SELECT id, text, domain, quality_score, used_in_training, created_at "
            "FROM training_data "
            "WHERE domain = ? AND quality_score >= ? AND used_in_training = ? "
            "ORDER BY quality_score DESC, id ASC LIMIT ?",
            (domain, min_quality, 1 if used else 0, int(limit)),
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
                     is_active: bool = True) -> int:
    """
    Insert a model_checkpoints row. When is_active=True (default), clear the
    prior active row first so at most one row has is_active=1. Returns the new
    row id.
    """
    conn = _connect(path)
    try:
        if is_active:
            conn.execute("UPDATE model_checkpoints SET is_active = 0")
        cur = conn.execute(
            "INSERT INTO model_checkpoints "
            "(step, loss, file_path, is_active, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (int(step), float(loss), file_path, 1 if is_active else 0, _now()),
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
            "SELECT id, step, loss, file_path, is_active, created_at "
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


# ── self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    import types
    import tempfile

    sys.stdout.reconfigure(encoding="utf-8")  # rule 0.4 / Windows cp1252 fix

    # quality.py is built in parallel; for this self-test we register a stub
    # so add_texts can be exercised without the sibling file present.
    if "quality" not in sys.modules:
        _stub = types.ModuleType("quality")

        def _score_quality(text: str) -> float:
            # Deterministic heuristic: longer, less-repetitive text scores higher.
            t = text.strip()
            if not t:
                return 0.0
            length_term = min(len(t), 200) / 200.0
            uniq_term = len(set(t.split())) / max(len(t.split()), 1)
            return round(0.6 * length_term + 0.4 * uniq_term, 4)

        _stub.score_quality = _score_quality  # type: ignore[attr-defined]
        sys.modules["quality"] = _stub

    tmpdir = tempfile.mkdtemp(prefix="tinymetatron_dbtest_")
    db_path = os.path.join(tmpdir, "test.db")

    def _ok(cond: bool, msg: str) -> None:
        print(("OK  " if cond else "FAIL") + " " + msg)
        assert cond, msg

    # 1. init_db creates tables (idempotent)
    init_db(db_path)
    init_db(db_path)  # idempotent re-run must not error

    # 2. add_texts scores + filters
    texts = [
        "Packet filtering firewall rules block unauthorized traffic by port and protocol.",
        "a",                       # too short -> low score -> rejected
        "Repeat repeat repeat repeat repeat repeat repeat repeat.",  # repetitive
        "Quantization reduces model precision to int8 to shrink memory footprint.",
    ]
    added, rejected = add_texts(db_path, texts, "cybersecurity", quality_threshold=0.4)
    print(f"add_texts: added={added} rejected={rejected}")
    _ok(added >= 1, "at least one text inserted")
    _ok(added + rejected == len(texts), "every text classified exactly once")

    # 3. fetch_training_rows returns dicts, unused only by default
    rows = fetch_training_rows(db_path, "cybersecurity", 0.4, limit=10, used=False)
    _ok(len(rows) == added, f"fetched {len(rows)} == added {added}")
    _ok(isinstance(rows[0], dict) and "text" in rows[0], "rows are dicts with text")
    _ok(all(r["used_in_training"] == 0 for r in rows), "rows start unused")

    # 4. mark_used flips the flag; used=True then returns them
    ids = [r["id"] for r in rows]
    n = mark_used(db_path, ids)
    _ok(n == len(ids), f"mark_used updated {n} == {len(ids)}")
    used_rows = fetch_training_rows(db_path, "cybersecurity", 0.4, limit=10, used=True)
    _ok(len(used_rows) == len(ids), "used=True returns the marked rows")
    _ok(len(fetch_training_rows(db_path, "cybersecurity", 0.4, 10, used=False)) == 0,
        "no unused rows remain")

    # 5. stats keys + values
    s = stats(db_path)
    print("stats:", s)
    _ok(set(s.keys()) == {"total", "by_domain", "avg_quality", "used_in_training"},
        "stats has the four contract keys")
    _ok(s["total"] == added, "stats total == added")
    _ok(s["used_in_training"] == added, "stats used == added")
    _ok("cybersecurity" in s["by_domain"], "by_domain contains cybersecurity")

    # 6. checkpoint save swaps active exactly-one invariant
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

    # 7. set_active_checkpoint re-activates the first
    set_active_checkpoint(db_path, os.path.join(tmpdir, "ckpt1.pt"))
    act3 = get_active_checkpoint(db_path)
    _ok(act3 is not None and act3["id"] == cid1, "set_active_checkpoint reactivates ckpt1")
    conn = _connect(db_path)
    n_active = conn.execute(
        "SELECT COUNT(*) AS n FROM model_checkpoints WHERE is_active = 1"
    ).fetchone()["n"]
    conn.close()
    _ok(n_active == 1, "still exactly one active after set_active")

    # 8. save_checkpoint with is_active=False does not disturb the active row
    save_checkpoint(db_path, step=30, loss=0.7,
                    file_path=os.path.join(tmpdir, "ckpt3.pt"), is_active=False)
    act4 = get_active_checkpoint(db_path)
    _ok(act4 is not None and act4["id"] == cid1,
        "is_active=False leaves the active checkpoint untouched")

    # 9. start_session / end_session
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

    # 10. delete_low_quality removes only below-threshold rows
    before = stats(db_path)["total"]
    # insert one guaranteed-low row directly
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

    # cleanup temp db
    try:
        os.remove(db_path)
    except OSError:
        pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    print("\nSELF-TEST PASSED")