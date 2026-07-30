"""
tests/test_db.py
===============
ECOSYSTEM pytest suite for the SQLite data layer (contract section 2).

Covers (contract section 5 / test descriptions):
    * init a temp db (schema create, idempotent)
    * add_texts scores + filters rows
    * fetch_training_rows returns dicts
    * mark_used flips used_in_training
    * save_checkpoint + active swap (exactly one is_active=1)
    * stats keys present
    * delete_low_quality removes below-threshold rows
"""

from __future__ import annotations

import os
import sys
import sqlite3
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

import db


@pytest.fixture
def db_path(tmp_path) -> str:
    """A fresh temp SQLite DB path; the file is created by init_db."""
    p = str(tmp_path / "test.db")
    db.init_db(p)
    return p


_SEED_TEXTS = [
    "Packet filtering firewall rules block unauthorized traffic by port and protocol.",
    "Asymmetric encryption uses a public key pair for secure key exchange protocols.",
    "Authentication tokens validate user identity across distributed sessions safely.",
    "Sparse attention masks reduce the quadratic cost of self-attention computation.",
]


# ── schema create is idempotent ───────────────────────────────────────────────
def test_init_db_idempotent(tmp_path) -> None:
    p = str(tmp_path / "idem.db")
    db.init_db(p)
    db.init_db(p)  # second call must not error
    # all three tables present
    conn = sqlite3.connect(p)
    tabs = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert {"training_data", "model_checkpoints", "training_sessions"} <= tabs


# ── add_texts scores + filters ────────────────────────────────────────────────
def test_add_texts(db_path: str) -> None:
    texts = list(_SEED_TEXTS) + ["a", ""]  # two clearly low-quality rows
    added, rejected = db.add_texts(db_path, texts, "cybersecurity",
                                   quality_threshold=0.4)
    assert added >= 1, f"expected at least 1 added, got {added}"
    assert added + rejected == len(texts), "every text classified exactly once"
    assert rejected >= 1, "low-quality rows must be rejected"


# ── fetch_training_rows returns dicts, unused by default ──────────────────────
def test_fetch_training_rows(db_path: str) -> None:
    db.add_texts(db_path, _SEED_TEXTS, "cybersecurity", quality_threshold=0.4)
    rows = db.fetch_training_rows(db_path, "cybersecurity", 0.4, limit=10,
                                  used=False)
    assert len(rows) >= 1
    assert isinstance(rows[0], dict)
    assert "text" in rows[0] and "id" in rows[0]
    assert all(r["used_in_training"] == 0 for r in rows), (
        "rows must start unused")
    assert all(r["domain"] == "cybersecurity" for r in rows)


# ── mark_used flips the flag and used=True fetch returns them ─────────────────
def test_mark_used(db_path: str) -> None:
    db.add_texts(db_path, _SEED_TEXTS, "cybersecurity", quality_threshold=0.4)
    rows = db.fetch_training_rows(db_path, "cybersecurity", 0.4, limit=10,
                                  used=False)
    ids = [r["id"] for r in rows]
    n = db.mark_used(db_path, ids)
    assert n == len(ids), f"mark_used updated {n} != {len(ids)}"
    used_rows = db.fetch_training_rows(db_path, "cybersecurity", 0.4, 10,
                                       used=True)
    assert len(used_rows) == len(ids)
    # no unused rows remain
    unused = db.fetch_training_rows(db_path, "cybersecurity", 0.4, 10,
                                     used=False)
    assert len(unused) == 0


# ── save_checkpoint + active swap (exactly one is_active=1) ───────────────────
def test_save_checkpoint_active_swap(db_path: str, tmp_path) -> None:
    ck1 = str(tmp_path / "ckpt1.pt")
    ck2 = str(tmp_path / "ckpt2.pt")
    id1 = db.save_checkpoint(db_path, step=10, loss=2.5, file_path=ck1)
    act = db.get_active_checkpoint(db_path)
    assert act is not None and act["id"] == id1 and act["is_active"] == 1

    id2 = db.save_checkpoint(db_path, step=20, loss=1.2, file_path=ck2)
    act2 = db.get_active_checkpoint(db_path)
    assert act2 is not None and act2["id"] == id2 and act2["is_active"] == 1

    conn = sqlite3.connect(db_path)
    n_active = conn.execute(
        "SELECT COUNT(*) FROM model_checkpoints WHERE is_active = 1"
    ).fetchone()[0]
    conn.close()
    assert n_active == 1, f"exactly one active checkpoint, got {n_active}"


# ── set_active_checkpoint re-activates a previous row ─────────────────────────
def test_set_active_checkpoint(db_path: str, tmp_path) -> None:
    ck1 = str(tmp_path / "a.pt")
    ck2 = str(tmp_path / "b.pt")
    id1 = db.save_checkpoint(db_path, step=1, loss=1.0, file_path=ck1)
    db.save_checkpoint(db_path, step=2, loss=0.5, file_path=ck2)
    db.set_active_checkpoint(db_path, ck1)
    act = db.get_active_checkpoint(db_path)
    assert act is not None and act["id"] == id1, "set_active reactivated ck1"
    conn = sqlite3.connect(db_path)
    n_active = conn.execute(
        "SELECT COUNT(*) FROM model_checkpoints WHERE is_active = 1"
    ).fetchone()[0]
    conn.close()
    assert n_active == 1


# ── stats keys present ───────────────────────────────────────────────────────
def test_stats_keys(db_path: str) -> None:
    db.add_texts(db_path, _SEED_TEXTS, "cybersecurity", quality_threshold=0.4)
    s = db.stats(db_path)
    assert set(s.keys()) == {"total", "by_domain", "avg_quality",
                             "used_in_training"}
    assert s["total"] >= 1
    assert "cybersecurity" in s["by_domain"]
    assert s["used_in_training"] == 0  # nothing marked yet


# ── delete_low_quality removes below-threshold rows ───────────────────────────
def test_delete_low_quality(db_path: str) -> None:
    db.add_texts(db_path, _SEED_TEXTS, "cybersecurity", quality_threshold=0.4)
    before = db.stats(db_path)["total"]
    # inject a guaranteed-low row directly
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO training_data (text, domain, quality_score, "
        "used_in_training, created_at) VALUES (?, ?, ?, 0, ?)",
        ("low", "cybersecurity", 0.05,
         __import__("datetime").datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    deleted = db.delete_low_quality(db_path, 0.4)
    assert deleted >= 1, f"delete_low_quality removed {deleted}"
    after = db.stats(db_path)["total"]
    assert after == before, (
        f"delete_low_quality removed exactly the injected low row "
        f"(after={after} before={before})")


# ── start_session / end_session write a completed row ─────────────────────────
def test_session_lifecycle(db_path: str) -> None:
    sid = db.start_session(db_path, domain_filter="cybersecurity",
                           min_quality=0.4)
    assert isinstance(sid, int) and sid > 0
    db.end_session(db_path, sid, total_steps=5, final_loss=0.42)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM training_sessions WHERE id = ?",
                       (sid,)).fetchone()
    conn.close()
    assert row is not None
    assert row["end_time"] is not None
    assert row["total_steps"] == 5
    assert abs(row["final_loss"] - 0.42) < 1e-9