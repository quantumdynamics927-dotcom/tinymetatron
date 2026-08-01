"""
refresh_cards.py
================
Regenerate retrieval_failure_cards.jsonl from current dev/val evaluation results.

Run this BEFORE starting a new retrieval loop iteration to get an accurate
view of which failures remain unaddressed.

Usage::

    python loops/retrieval_loop/refresh_cards.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("TMT_QUANTUM_CORPUS_DB", r"E:\Temp\qcorpus\quantum_corpus.db")

from quantum_corpus import schema
from quantum_corpus.eval.runner import (
    _load_qa, _load_rows_split, _build_retriever, _query,
    score_item_ask, _get_structured, _load_manifest, _seed_canary_row,
)

# Threshold: gold SI must appear in top-k to be considered "addressed"
TOP_K = 5


def refresh_failure_cards():
    db_path = schema.default_db_path()

    # Load dev and val retrieval QA sets
    eval_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "quantum_corpus", "eval"
    )
    DEV_PATH = os.path.join(eval_dir, "qa_dev_retrieval.jsonl")
    VAL_RET_PATH = os.path.join(eval_dir, "qa_val_retrieval.jsonl")
    all_items = {it["id"]: it for it in _load_qa(DEV_PATH) + _load_qa(VAL_RET_PATH)}

    # Build train+val index
    print("Building train+val index...")
    rows = _load_rows_split(db_path, ("train", "val"))
    canary = _seed_canary_row()
    t0 = time.time()
    idx = _build_retriever([canary] + rows, "hybrid")
    print(f"  {len(idx)} docs in {time.time()-t0:.1f}s")

    sq = _get_structured(db_path)
    build_id, build_sha = _load_manifest()
    gates = {
        "score_floor_hybrid": 0.008,
        "score_floor_bm25": 3.0,
        "sep_ratio": 1.5,
        "sep_band": 2.0,
        "min_concepts": 1,
    }

    # Evaluate all items
    print(f"\nEvaluating {len(all_items)} items...")
    new_cards = []
    missed = 0
    for i, (mid, it) in enumerate(all_items.items()):
        hits = _query(idx, it["question"], TOP_K, "sensitive")
        rec = score_item_ask(it, hits, idx, sq, gates, True, build_id, build_sha,
                            "sensitive", score_scale="hybrid", db_path=db_path)

        retrieved_sis = rec.get("retrieved_source_identities_top5", [])
        gold_sis = it.get("gold_source_identities", [])
        gold_si_set = set(gold_sis)
        in_top5 = any(si in retrieved_sis for si in gold_si_set) if gold_si_set else None

        if not in_top5:
            missed += 1

        # Classify failure type
        failure_category = _classify_failure(it, hits, gold_sis, retrieved_sis)

        card = {
            "id": mid,
            "category": it.get("category", "unknown"),
            "question": it["question"],
            "gold_source_identities": gold_sis,
            "gold_record_ids": it.get("gold_record_ids", []),
            "retrieved_source_identities_top5": retrieved_sis,
            "in_top5": in_top5,
            "failure_category": failure_category,
            "route": rec.get("route"),
            "decision": rec.get("decision"),
            "si_recall@5": rec.get("si_recall@5"),
            "notes": "",
        }
        new_cards.append(card)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(all_items)} evaluated, {missed} missed so far")

    print(f"\nDone: {missed}/{len(all_items)} missed (in_top5=False)")

    # Categorize missed
    by_cat: dict[str, int] = {}
    for c in new_cards:
        if not c["in_top5"]:
            cat = c["failure_category"] or "unknown"
            by_cat[cat] = by_cat.get(cat, 0) + 1

    print("\nMissed by category:")
    for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        ids = [c["id"] for c in new_cards if not c["in_top5"] and c["failure_category"] == cat]
        print(f"  {cat}: {n}  ({', '.join(ids)})")

    # Write updated cards
    FAILURE_CARDS = os.path.join(eval_dir, "retrieval_failure_cards.jsonl")
    with open(FAILURE_CARDS, "w", encoding="utf-8") as f:
        for card in new_cards:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")

    print(f"\nUpdated {FAILURE_CARDS}")
    return new_cards


def _classify_failure(item: dict, hits: list, gold_sis: list,
                      retrieved_sis: list) -> str | None:
    """Classify the failure type based on question pattern and retrieval result."""
    q = item["question"].lower()
    gold_set = set(gold_sis)

    # No gold SIs — not a valid test item
    if not gold_set:
        return "no_gold_si"

    # Gold is retrieved
    if any(si in retrieved_sis for si in gold_set):
        return None  # addressed

    # Gold not retrieved — classify why
    # Structured SQL cases (job questions with specific jid)
    if "ibm quantum job" in q and any(c in q for c in "?what"):
        if item.get("route") == "structured":
            return "structured_sql"
        return "job_record_missing"

    # conscious_dna context
    if "conscious_dna" in q or "dna_agent" in q:
        if "specialization" in q:
            return "specialization_alias"
        if "phi_score" in q:
            return "phi_score_entity"
        return "conscious_dna_other"

    # Backend questions
    if "backend ibm_" in q or ("ibm_" in q and "backend" in q):
        return "backend_query"

    # General conceptual
    if item.get("category") == "conceptual":
        return "conceptual_miss"

    return "unknown"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    refresh_failure_cards()
