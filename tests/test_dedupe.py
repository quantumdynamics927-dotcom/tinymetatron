"""
tests/test_dedupe.py
====================
Tests for workers.corpus.dedupe — exact, normalized, and near-duplicate
detection. The near-dup pass uses MinHash + LSH so it scales to 100k+ rows;
candidates are verified with the exact Jaccard test (no false positives).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from workers.corpus.dedupe import run, _minhash, _lsh_bands, _jaccard, _ngrams

# A base text long enough to trigger near-dup detection (>= MIN_DUP_LEN).
_BASE = ("Quantum error correction encodes logical qubits into physical qubits "
         "using stabilizer codes that detect and correct bit-flip and phase-flip "
         "errors during computation. " * 2)


def _write_corpus(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _run_dedupe(corpus_dir: Path, artifact_dir: Path) -> dict:
    return run({"corpus_dir": str(corpus_dir), "artifact_dir": str(artifact_dir)})


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


# ── MinHash + LSH helpers ────────────────────────────────────────────────────

def test_minhash_signature_is_stable_and_bounded() -> None:
    sig = _minhash(_BASE)
    assert len(sig) == 128
    assert sig == _minhash(_BASE), "signature must be deterministic"
    assert len(_lsh_bands(sig)) == 16


def test_jaccard_high_for_near_duplicate() -> None:
    near = _BASE + " A trailing sentence."
    j = _jaccard(_ngrams(_BASE), _ngrams(near))
    assert j >= 0.85, f"near-dup Jaccard too low: {j}"


# ── end-to-end dedup ─────────────────────────────────────────────────────────

def test_dedupe_catches_exact_normalized_and_near(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    artifact_dir = tmp_path / "out"
    artifact_dir.mkdir()

    rows = [
        {"id": "a", "text": _BASE, "source_id": "s1"},
        {"id": "b", "text": _BASE, "source_id": "s2"},            # exact dup
        {"id": "c", "text": _BASE.upper(), "source_id": "s3"},     # normalized dup
        {"id": "d", "text": _BASE + " A trailing sentence.", "source_id": "s4"},  # near dup
        {"id": "e", "text": "A completely different short text about qubits.", "source_id": "s5"},
    ]
    _write_corpus(corpus_dir / "validated.jsonl", rows)

    result = _run_dedupe(corpus_dir, artifact_dir)

    assert result["status"] == "success"
    assert result["metrics"]["exact_duplicates"] == 1
    assert result["metrics"]["normalized_duplicates"] == 1
    assert result["metrics"]["near_duplicates"] >= 1
    # Existing semantics: a near-dup pair drops BOTH rows (the current row is
    # not emitted when it has a near-dup later; the later row is skipped).
    assert result["metrics"]["output_rows"] == 1

    out = _read_jsonl(artifact_dir / "deduped.jsonl")
    assert {r["id"] for r in out} == {"e"}


def test_dedupe_keeps_distinct_texts(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    artifact_dir = tmp_path / "out"
    artifact_dir.mkdir()

    rows = [
        {"id": "a", "text": _BASE, "source_id": "s1"},
        {"id": "b", "text": "Entirely unrelated text about classical computing.", "source_id": "s2"},
    ]
    _write_corpus(corpus_dir / "validated.jsonl", rows)

    result = _run_dedupe(corpus_dir, artifact_dir)
    assert result["metrics"]["output_rows"] == 2
    assert result["metrics"]["near_duplicates"] == 0
