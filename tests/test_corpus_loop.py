"""
tests/test_corpus_loop.py
========================
Automated smoke test for the corpus loop (Phase 3 completion).

Verifies the invariant the manual smoke run got wrong: a *source-disjoint*
split, where every source_id appears in exactly one of {train, val, hard_dev}
— not merely text-disjoint within a source.

Covers:
  * Full pipeline on the smoke fixture produces text- AND source-disjoint splits.
  * MANIFEST.json records split_policy / split_seed / source_counts /
    source_overlap / text_overlap, with every overlap count zero.
  * At least 3 independent source groups; experiment reaches FROZEN_CORPUS.
  * Deterministic: two clean runs produce the same corpus_hash.
  * Split rejects a corpus with too few source groups (< 3).
  * Corpus gate worker flags a nonzero source overlap and passes a clean one.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Point the registry at a throwaway temp DB BEFORE importing the corpus loop so
# the module-level init_db() in corpus_loop never touches state/registry.db.
_TMP_REGISTRY = Path(tempfile.mkdtemp(prefix="corpus_test_registry_")) / "registry.db"
os.environ["TINYMETATRON_DB"] = str(_TMP_REGISTRY)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import db  # noqa: E402
db.set_db_path(str(_TMP_REGISTRY))
db.init_db(str(_TMP_REGISTRY))

from loops import corpus_loop  # noqa: E402

import pytest  # noqa: E402

_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "corpus_smoke.jsonl"


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_registry(tmp_path) -> str:
    """A fresh temp registry DB per test (clean test registry)."""
    p = str(tmp_path / "registry.db")
    db.set_db_path(p)
    db.init_db(p)
    return p


@pytest.fixture
def corpus_dir(tmp_path) -> Path:
    """Temp raw-corpus dir containing a copy of the smoke fixture."""
    d = tmp_path / "raw"
    d.mkdir()
    shutil.copy(_FIXTURE, d / "corpus_smoke.jsonl")
    return d


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _texts(rows: list[dict]) -> set:
    return {r["text"] for r in rows}


def _sources(rows: list[dict]) -> set:
    return {r.get("source", r.get("domain", "unknown")) for r in rows}


# ── source-disjoint smoke ─────────────────────────────────────────────────────

def test_corpus_smoke_source_disjoint(clean_registry, corpus_dir, tmp_path) -> None:
    out = tmp_path / "out"
    summary = corpus_loop.run_corpus_pipeline({
        "exp_id": "exp-smoke-001",
        "corpus_dir": str(corpus_dir),
        "output_dir": str(out),
        "seed": 42,
        "train_pct": 0.80,
        "val_pct": 0.10,
    })

    corpus = out / "corpus"
    train = _load_jsonl(corpus / "train.jsonl")
    val = _load_jsonl(corpus / "val.jsonl")
    hard = _load_jsonl(corpus / "hard_dev.jsonl")

    tr_t, va_t, ha_t = _texts(train), _texts(val), _texts(hard)
    tr_s, va_s, ha_s = _sources(train), _sources(val), _sources(hard)

    # Every partition non-empty.
    assert train and val and hard, "all three splits must be non-empty"

    # Text-disjoint.
    assert tr_t.isdisjoint(va_t)
    assert tr_t.isdisjoint(ha_t)
    assert va_t.isdisjoint(ha_t)

    # Source-disjoint — the invariant the manual smoke run got wrong.
    assert tr_s.isdisjoint(va_s)
    assert tr_s.isdisjoint(ha_s)
    assert va_s.isdisjoint(ha_s)

    # At least 3 independent source groups overall.
    assert len(tr_s | va_s | ha_s) >= 3

    # MANIFEST.json records the split policy invariants.
    man = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert man["split_policy"] == "source_disjoint_v1"
    assert man["split_seed"] == 42
    assert man["source_counts"] == {
        "train": len(tr_s), "val": len(va_s), "hard_dev": len(ha_s),
    }
    for key in ("train_val", "train_hard_dev", "val_hard_dev"):
        assert man["source_overlap"][key] == 0, f"source_overlap[{key}] != 0"
        assert man["text_overlap"][key] == 0, f"text_overlap[{key}] != 0"
    assert sum(man["source_overlap"].values()) == 0
    assert sum(man["text_overlap"].values()) == 0

    # Gate passed → experiment frozen.
    assert summary["corpus_hash"]
    exp = db.get_loop_experiment("exp-smoke-001")
    assert exp["state"] == "FROZEN_CORPUS"


# ── determinism ───────────────────────────────────────────────────────────────

def test_corpus_smoke_deterministic(clean_registry, corpus_dir, tmp_path) -> None:
    out1 = tmp_path / "out1"
    s1 = corpus_loop.run_corpus_pipeline({
        "exp_id": "exp-smoke-det-a",
        "corpus_dir": str(corpus_dir),
        "output_dir": str(out1),
        "seed": 42,
    })
    out2 = tmp_path / "out2"
    s2 = corpus_loop.run_corpus_pipeline({
        "exp_id": "exp-smoke-det-b",
        "corpus_dir": str(corpus_dir),
        "output_dir": str(out2),
        "seed": 42,
    })
    assert s1["corpus_hash"] == s2["corpus_hash"], (
        "deterministic split must produce identical corpus_hash across runs")


# ── too-few source groups rejected ────────────────────────────────────────────

def test_split_rejects_too_few_sources(clean_registry, tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        {"text": "Alpha source sentence one about network security fundamentals.",
         "source": "alpha"},
        {"text": "Alpha source sentence two about cryptographic hash functions.",
         "source": "alpha"},
        {"text": "Beta source sentence one about machine learning neural nets.",
         "source": "beta"},
        {"text": "Beta source sentence two about gradient descent optimization.",
         "source": "beta"},
    ]
    with open(raw / "in.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="Worker error"):
        corpus_loop.run_corpus_pipeline({
            "exp_id": "exp-smoke-few",
            "corpus_dir": str(raw),
            "output_dir": str(out),
        })
    # Must not have been frozen.
    assert db.get_loop_experiment("exp-smoke-few")["state"] != "FROZEN_CORPUS"


# ── corpus gate worker: detects overlap, passes clean ─────────────────────────

def test_corpus_gate_detects_overlap(tmp_path) -> None:
    man = tmp_path / "MANIFEST.json"
    payload = {
        "split_policy": "source_disjoint_v1",
        "split_seed": 42,
        "source_counts": {"train": 2, "val": 2, "hard_dev": 1},
        "source_overlap": {"train_val": 1, "train_hard_dev": 0, "val_hard_dev": 0},
        "text_overlap": {"train_val": 0, "train_hard_dev": 0, "val_hard_dev": 0},
    }
    man.write_text(json.dumps(payload), encoding="utf-8")

    result = corpus_loop._run_worker(
        ["python", "-m", "workers.corpus.corpus_gate", "--manifest", str(man)],
        tmp_path / "gate_art", 60,
    )
    assert result["metrics"]["source_overlap_total"] == 1
    # Gate pass_condition (source_overlap_total == 0) must evaluate to False.
    assert corpus_loop._eval_condition(
        result["metrics"]["source_overlap_total"], "==", 0) is False


def test_corpus_gate_passes_clean(tmp_path) -> None:
    man = tmp_path / "MANIFEST.json"
    payload = {
        "split_policy": "source_disjoint_v1",
        "split_seed": 42,
        "source_counts": {"train": 2, "val": 1, "hard_dev": 1},
        "source_overlap": {"train_val": 0, "train_hard_dev": 0, "val_hard_dev": 0},
        "text_overlap": {"train_val": 0, "train_hard_dev": 0, "val_hard_dev": 0},
    }
    man.write_text(json.dumps(payload), encoding="utf-8")

    result = corpus_loop._run_worker(
        ["python", "-m", "workers.corpus.corpus_gate", "--manifest", str(man)],
        tmp_path / "gate_art", 60,
    )
    assert result["metrics"]["source_overlap_total"] == 0
    assert corpus_loop._eval_condition(
        result["metrics"]["source_overlap_total"], "==", 0) is True