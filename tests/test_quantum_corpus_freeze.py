"""
tests/test_quantum_corpus_freeze.py
====================================
Tests for quantum_corpus.freeze — the source_disjoint_capped_v1 corpus freeze
policy exposed through the quantum-corpus package (exp-004 revision-2).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from quantum_corpus.freeze import (
    DEFAULT_MAX_SOURCE_SHARE,
    SPLIT_POLICY,
    build_manifest,
    freeze,
    gate_max_source_share,
    split_source_disjoint_capped,
    validate_manifest,
)


def _make_rows(n_sources: int = 10, rows_per_source: int = 5, seed: int = 7) -> list[dict]:
    """Deterministic synthetic corpus: n_sources source groups, each with
    rows_per_source rows. One source is oversized to exercise the cap."""
    rng = random.Random(seed)
    rows = []
    for s in range(n_sources):
        n = rows_per_source + (10 if s == 0 else 0)  # source 0 exceeds a cap of 4
        for i in range(n):
            rows.append({
                "source_id": f"src-{s:03d}",
                "text": f"source {s} row {i} {' '.join(rng.choice(['alpha','beta','gamma','delta']) for _ in range(4))}",
                "subdomain": f"domain-{s % 3}",
            })
    return rows


# ── Split ────────────────────────────────────────────────────────────────────

def test_split_source_disjoint_capped_is_disjoint():
    rows = _make_rows()
    train, val, hard, excluded, meta = split_source_disjoint_capped(
        rows, seed=42, max_rows_per_source=4)

    def _srcs(rs):
        return {r["source_id"] for r in rs}

    assert _srcs(train).isdisjoint(_srcs(val))
    assert _srcs(train).isdisjoint(_srcs(hard))
    assert _srcs(val).isdisjoint(_srcs(hard))
    assert meta["source_overlap"] == {"train_val": 0, "train_hard_dev": 0, "val_hard_dev": 0}
    assert meta["text_overlap"] == {"train_val": 0, "train_hard_dev": 0, "val_hard_dev": 0}
    assert meta["split_policy"] == SPLIT_POLICY


def test_split_cap_applied_before_split_and_excluded_preserved():
    rows = _make_rows()  # 10 sources: src-000 has 15 rows, src-001..009 have 5
    train, val, hard, excluded, meta = split_source_disjoint_capped(
        rows, seed=42, max_rows_per_source=4)

    # src-000: 15 -> 4 (11 dropped); src-001..009: 5 -> 4 (1 each) = 20 total.
    assert meta["pre_cap_rows"] == len(rows)
    assert meta["post_cap_rows"] == len(train) + len(val) + len(hard)
    assert meta["rows_dropped_by_cap"] == 20
    assert len(excluded) == 20

    capped = {c["source_id"]: c for c in meta["capped_sources"]}
    assert len(capped) == 10
    assert capped["src-000"] == {
        "source_id": "src-000", "original_rows": 15,
        "retained_rows": 4, "excluded_rows": 11,
    }
    assert capped["src-001"]["excluded_rows"] == 1

    # Retained rows are the first 4 of the seeded shuffle, so no excluded row
    # appears in any split.
    retained_texts = {r["text"] for r in train + val + hard}
    assert not (retained_texts & {r["text"] for r in excluded})


def test_split_requires_min_source_groups():
    rows = [{"source_id": "a", "text": "x"}, {"source_id": "a", "text": "y"}]
    with pytest.raises(ValueError, match="source-disjoint split requires"):
        split_source_disjoint_capped(rows, seed=42)


# ── Manifest ─────────────────────────────────────────────────────────────────

def test_build_manifest_fields(tmp_path):
    rows = _make_rows()
    train, val, hard, excluded, meta = split_source_disjoint_capped(
        rows, seed=42, max_rows_per_source=4)

    for name, rs in [("train", train), ("val", val), ("hard_dev", hard)]:
        with open(tmp_path / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(tmp_path / "excluded_by_cap.jsonl", "w", encoding="utf-8") as f:
        for r in excluded:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(tmp_path / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    manifest = build_manifest(
        tmp_path, revision=2, max_source_share_threshold=0.25, scope="test")

    assert manifest["corpus_revision"] == 2
    assert manifest["max_source_share_gate_threshold"] == 0.25
    assert manifest["max_rows_per_source"] == 4
    assert manifest["split_policy"] == SPLIT_POLICY
    assert manifest["total_rows"] == len(train) + len(val) + len(hard)
    assert manifest["source_overlap"] == {"train_val": 0, "train_hard_dev": 0, "val_hard_dev": 0}
    assert manifest["text_overlap"] == {"train_val": 0, "train_hard_dev": 0, "val_hard_dev": 0}
    assert manifest["excluded_by_cap"]["rows"] == len(excluded)
    assert len(manifest["excluded_by_cap"]["sha256"]) == 64
    assert set(manifest["splits"].keys()) == {"train.jsonl", "val.jsonl", "hard_dev.jsonl"}
    # excluded_by_cap must NOT enter the corpus hash / tallies.
    assert manifest["total_rows"] == manifest["unique_normalized"]


# ── Gate ──────────────────────────────────────────────────────────────────────

def test_gate_fails_on_dominant_source():
    # A tiny corpus (3 sources, 1 row each) has max_source_row_share = 1.0,
    # which fails the default 0.25 gate.
    rows = [{"source_id": f"s{i}", "text": f"row {i}"} for i in range(3)]
    train, val, hard, excluded, meta = split_source_disjoint_capped(
        rows, seed=42, max_rows_per_source=10)
    manifest = {"max_source_row_share": meta["max_source_row_share"]}

    passed, actual = gate_max_source_share(manifest, threshold=DEFAULT_MAX_SOURCE_SHARE)
    assert not passed
    assert actual == 1.0


def test_gate_passes_on_balanced_corpus():
    # 50 sources x 5 rows: val/hard get ~5 groups each, share = 5/25 = 0.2.
    rows = _make_rows(n_sources=50, rows_per_source=5)
    train, val, hard, excluded, meta = split_source_disjoint_capped(
        rows, seed=42, max_rows_per_source=4)
    manifest = {"max_source_row_share": meta["max_source_row_share"]}

    passed, actual = gate_max_source_share(manifest, threshold=DEFAULT_MAX_SOURCE_SHARE)
    assert passed
    assert actual <= 0.25


def test_freeze_gate_skipped_without_flag(tmp_path):
    # Without --max-source-share the gate is skipped, so a tiny corpus freezes.
    rows = [{"source_id": f"s{i}", "text": f"row {i}"} for i in range(3)]
    src = tmp_path / "src"
    src.mkdir()
    with open(src / "deduped.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out = tmp_path / "out"
    summary = freeze(src, out, max_rows_per_source=10)
    assert summary["gate"] is None
    assert (out / "MANIFEST.json").exists()


def test_freeze_gate_fails_on_tiny_corpus(tmp_path):
    rows = [{"source_id": f"s{i}", "text": f"row {i}"} for i in range(3)]
    src = tmp_path / "src"
    src.mkdir()
    with open(src / "deduped.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out = tmp_path / "out"
    with pytest.raises(ValueError, match="max-source-share gate FAILED"):
        freeze(src, out, max_rows_per_source=10, max_source_share=0.25)


# ── Validate ─────────────────────────────────────────────────────────────────

def test_validate_manifest_round_trip(tmp_path):
    rows = _make_rows(n_sources=50, rows_per_source=5)
    src = tmp_path / "src"
    src.mkdir()
    with open(src / "deduped.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out = tmp_path / "out"
    summary = freeze(src, out, max_rows_per_source=4, revision=2,
                     max_source_share=0.25, scope="test")

    result = validate_manifest(Path(out / "MANIFEST.json"), out)
    assert result["valid"] is True
    assert result["corpus_hash"]["ok"] is True
    assert result["all_split_hashes_ok"] is True
    assert result["source_overlap_total"] == 0
    assert result["text_overlap_total"] == 0
    assert result["max_source_share_gate_threshold"] == 0.25
    assert result["gate_passed"] is True


def test_validate_manifest_detects_tamper(tmp_path):
    rows = _make_rows(n_sources=50, rows_per_source=5)
    src = tmp_path / "src"
    src.mkdir()
    with open(src / "deduped.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out = tmp_path / "out"
    freeze(src, out, max_rows_per_source=4, revision=2, max_source_share=0.25)

    # Tamper with a split file.
    with open(out / "val.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"source_id": "tampered", "text": "injected"}) + "\n")

    result = validate_manifest(Path(out / "MANIFEST.json"), out)
    assert result["valid"] is False
    assert result["corpus_hash"]["ok"] is False
    assert result["split_hashes"]["val.jsonl"]["status"] == "MISMATCH"
