"""
tests/test_train.py
=================
ECOSYSTEM pytest suite for the incremental trainer (contract section 2).

Covers (contract section 5 / test descriptions):
    * seed a tiny temp db with a few rows via db.add_texts
    * run train_db.run_training(steps=2, ...) with a temp checkpoint_dir
    * assert the returned final_loss is finite
    * a model_checkpoints row is written with is_active=1
    * a training_sessions row has end_time set
    * consumed training_data rows have used_in_training=1
"""

from __future__ import annotations

import math
import os
import sys
import sqlite3

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

from config import CONFIG
import db
import train_db


# Technical seed texts long enough to clear the 0.4 quality threshold.
_SEED_TEXTS = [
    "Quantization reduces model precision to int8 to shrink memory footprint.",
    "Packet filtering firewall rules block unauthorized network traffic by port.",
    "Asymmetric encryption uses a public key pair for secure key exchange.",
    "Authentication tokens validate user identity across distributed sessions.",
    "Sparse attention masks reduce quadratic cost to linear complexity edges.",
    "Mixture of experts routes tokens to specialized feed forward networks.",
]


@pytest.fixture
def seeded_db(tmp_path) -> str:
    """Init a temp DB and seed it with a few technical rows."""
    p = str(tmp_path / "train_test.db")
    db.init_db(p)
    added, _ = db.add_texts(p, _SEED_TEXTS, "cybersecurity",
                            quality_threshold=0.4)
    assert added >= 1, "fixture failed to seed any rows"
    return p


def test_run_training_two_steps(seeded_db: str, tmp_path) -> None:
    ckpt_dir = str(tmp_path / "ckpt")
    torch.manual_seed(CONFIG["seed"])

    result = train_db.run_training(
        steps=2,
        domain="cybersecurity",
        min_quality=0.4,
        batch_size=2,
        learning_rate=1e-3,
        max_seq_len=CONFIG["seq_len"],
        device="cpu",
        checkpoint_dir=ckpt_dir,
        db_path=seeded_db,
        aux_loss_weight=CONFIG["moe_aux_loss_weight"],
        seed=CONFIG["seed"],
    )

    # ── final_loss is finite ──────────────────────────────────────────────────
    assert math.isfinite(result["final_loss"]), (
        f"final_loss not finite: {result['final_loss']}")
    assert result["steps"] == 2, f"steps==2, got {result['steps']}"

    # ── a checkpoint .pt file was written ──────────────────────────────────────
    assert result["checkpoint_path"] is not None
    assert os.path.isfile(result["checkpoint_path"]), (
        "checkpoint .pt file must exist on disk")

    # ── a model_checkpoints row is written with is_active=1 ────────────────────
    act = db.get_active_checkpoint(seeded_db)
    assert act is not None, "no active checkpoint row after training"
    assert act["is_active"] == 1, "active checkpoint must have is_active=1"
    assert act["file_path"] == result["checkpoint_path"], (
        "active row must point at the saved .pt")

    # ── a training_sessions row has end_time set ───────────────────────────────
    conn = sqlite3.connect(seeded_db)
    conn.row_factory = sqlite3.Row
    srow = conn.execute(
        "SELECT * FROM training_sessions WHERE id = ?",
        (result["session_id"],)).fetchone()
    conn.close()
    assert srow is not None, "training_sessions row must exist"
    assert srow["end_time"] is not None, "session end_time must be set"
    assert srow["total_steps"] == 2, "session total_steps==2"
    assert math.isfinite(float(srow["final_loss"])), (
        "session final_loss must be finite")

    # ── consumed rows have used_in_training=1 ──────────────────────────────────
    # The trainer marks ALL fetched rows used immediately (contract section 3),
    # so every row in the seeded DB must now show used_in_training=1.
    conn = sqlite3.connect(seeded_db)
    conn.row_factory = sqlite3.Row
    unused = conn.execute(
        "SELECT COUNT(*) AS n FROM training_data WHERE used_in_training = 0"
    ).fetchone()["n"]
    used = conn.execute(
        "SELECT COUNT(*) AS n FROM training_data WHERE used_in_training = 1"
    ).fetchone()["n"]
    conn.close()
    assert unused == 0, (
        f"{unused} rows still unused after training; expected 0")
    assert used >= 1, "at least one row marked used_in_training=1"


def test_run_training_resume_from_checkpoint(seeded_db: str, tmp_path) -> None:
    """A second run loads the active checkpoint (no crash) and writes a new one."""
    ckpt_dir = str(tmp_path / "ckpt2")
    torch.manual_seed(CONFIG["seed"] + 1)

    # Re-seed (rows from the first run are already used).
    db.add_texts(seeded_db, _SEED_TEXTS, "cybersecurity",
                 quality_threshold=0.4)

    result = train_db.run_training(
        steps=1,
        domain="cybersecurity",
        min_quality=0.4,
        batch_size=2,
        learning_rate=1e-3,
        max_seq_len=CONFIG["seq_len"],
        device="cpu",
        checkpoint_dir=ckpt_dir,
        db_path=seeded_db,
        aux_loss_weight=CONFIG["moe_aux_loss_weight"],
        seed=CONFIG["seed"],
    )
    assert math.isfinite(result["final_loss"]), "resume run loss finite"
    assert os.path.isfile(result["checkpoint_path"])
    act = db.get_active_checkpoint(seeded_db)
    assert act is not None and act["is_active"] == 1