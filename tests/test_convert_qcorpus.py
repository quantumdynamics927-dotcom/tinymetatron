"""
tests/test_convert_qcorpus.py
=============================
Tests for workers.corpus.convert_qcorpus — the controlled acquisition step
that exports the private quantum corpus DB to a source-attributed JSONL corpus.

The conversion contract requires every emitted row to carry:
  id, text, source_id (document-level, not a topic bucket), domain, subdomain,
  license, provenance, quality_score.

We build a synthetic corpus_records DB (mirroring quantum_corpus.schema) and
assert the worker emits the contract fields, preserves source groups, and
reports the gate-inspection metrics.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from workers.corpus.convert_qcorpus import inspect_db, run

_SCHEMA = """
CREATE TABLE corpus_records (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type      TEXT    NOT NULL,
    project          TEXT    NOT NULL,
    subdomain        TEXT,
    doc_id           TEXT    NOT NULL,
    text             TEXT    NOT NULL,
    split            TEXT    NOT NULL,
    token_count      INTEGER NOT NULL DEFAULT 0,
    source_license   TEXT,
    provenance_url   TEXT,
    sensitivity      TEXT    NOT NULL,
    risk_tier        INTEGER NOT NULL DEFAULT 0,
    content_hash     TEXT    NOT NULL,
    source_identity  TEXT    NOT NULL,
    created_at       TEXT    NOT NULL
);
"""

# Three genuine source groups (documents), each with multiple rows.
_ROWS = [
    # GRE repo — two documents
    ("repo", "GRE", "circuits", "GRE:src/engine.py", "Quantum engine compiles circuits to gates. " * 3,
     "MIT", "https://github.com/quantumdynamics927-dotcom/Geometric-Resonance-Engine", "public", 0),
    ("repo", "GRE", "circuits", "GRE:src/engine.py", "Quantum engine compiles circuits to gates. " * 3,
     "MIT", "https://github.com/quantumdynamics927-dotcom/Geometric-Resonance-Engine", "public", 0),
    ("repo", "GRE", "docs", "GRE:README.md", "Geometric Resonance Engine documentation. " * 3,
     "MIT", "https://github.com/quantumdynamics927-dotcom/Geometric-Resonance-Engine", "public", 0),
    # QPyth repo — one document
    ("repo", "QPyth", "security", "QPyth:src/gateway.py", "Quantum secure gateway authenticates sessions. " * 3,
     "proprietary", "https://github.com/quantumdynamics927-dotcom/QPyth", "internal", 1),
    # IBM jobs — one document
    ("ibm_job", "ibm-quantum", "calibration", "ibm-quantum:job-abc123-info.json",
     "IBM backend job sampler calibration data. " * 3,
     "unknown", "E:/Descargas/workloads.zip", "internal", 1),
]


def _build_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    for i, (st, proj, sub, doc, text, lic, prov, sens, risk) in enumerate(_ROWS):
        import hashlib
        ch = hashlib.sha256(text.encode()).hexdigest()
        si = hashlib.sha256(f"{proj}\n{doc}\n{ch}".encode()).hexdigest()
        conn.execute(
            "INSERT INTO corpus_records "
            "(source_type, project, subdomain, doc_id, text, split, token_count, "
            " source_license, provenance_url, sensitivity, risk_tier, content_hash, "
            " source_identity, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (st, proj, sub, doc, text, "train", 10, lic, prov, sens, risk, ch, si,
             "2026-08-12T00:00:00Z"),
        )
    conn.commit()
    conn.close()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


@pytest.fixture
def corpus_db(tmp_path) -> Path:
    p = tmp_path / "quantum_corpus.db"
    _build_db(p)
    return p


# ── conversion contract: every row carries the required fields ──────────────

def test_convert_emits_contract_fields(corpus_db, tmp_path) -> None:
    out = tmp_path / "out"
    result = run({"db_path": str(corpus_db), "output_dir": str(out)})

    assert result["status"] == "success"
    rows = _read_jsonl(out / "corpus.jsonl")
    assert len(rows) == len(_ROWS)

    required = {"id", "text", "source_id", "domain", "subdomain",
                "license", "provenance", "quality_score"}
    for row in rows:
        assert required <= set(row.keys()), f"missing contract fields: {required - set(row)}"
        assert row["domain"] == "quantum"
        assert row["source_id"], "source_id must be non-empty"
        assert row["text"], "text must be non-empty"
        assert row["license"] in {"MIT", "proprietary", "unknown"}
        assert row["quality_score"] == 0.0


# ── source_id is document-level, not a topic bucket ─────────────────────────

def test_source_id_is_document_level(corpus_db, tmp_path) -> None:
    out = tmp_path / "out"
    run({"db_path": str(corpus_db), "output_dir": str(out)})
    rows = _read_jsonl(out / "corpus.jsonl")

    source_ids = {r["source_id"] for r in rows}
    assert source_ids == {
        "GRE:src/engine.py", "GRE:README.md",
        "QPyth:src/gateway.py", "ibm-quantum:job-abc123-info.json",
    }, "source_id must identify the document, not a broad label"


# ── gate inspection metrics ─────────────────────────────────────────────────

def test_inspect_reports_source_groups_and_dupes(corpus_db) -> None:
    insp = inspect_db(corpus_db)
    assert insp["total_rows"] == len(_ROWS)
    assert insp["n_source_groups"] == 4
    assert insp["duplicates"]["exact_text_dupes"] == 1  # the duplicated engine row
    assert insp["by_project"]["GRE"] == 3
    assert insp["by_license"]["MIT"] == 3
    assert insp["char_length"]["n"] == len(_ROWS)
    assert insp["token_length"]["n"] == len(_ROWS)


# ── missing DB fails closed ─────────────────────────────────────────────────

def test_convert_missing_db_fails_closed(tmp_path) -> None:
    out = tmp_path / "out"
    result = run({"db_path": str(tmp_path / "nope.db"), "output_dir": str(out)})
    assert result["status"] == "error"
    assert "not found" in result["error"]
