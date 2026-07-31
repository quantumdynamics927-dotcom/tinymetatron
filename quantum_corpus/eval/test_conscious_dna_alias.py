"""
test_conscious_dna_alias.py
===========================
Regression test for the schema-alias query expansion fix on conscious_dna misses.

Tests that the query expansion ("specialization" -> "dna_specialization")
correctly retrieves the 13 previously-failed conscious_dna gold records.

Run::

    python -m quantum_corpus.eval.test_conscious_dna_alias
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("TMT_QUANTUM_CORPUS_DB", "E:\\Temp\\qcorpus\\quantum_corpus.db")
os.environ.setdefault("TMT_QUANTUM_JOBS_DB", "E:\\Temp\\qcorpus\\quantum_jobs_structured.db")
os.environ.setdefault("TMT_DEPLOY_MODE", "private-training")

from quantum_corpus import schema
from quantum_corpus.eval.runner import (
    _load_qa, _load_rows_split, _build_retriever, _query,
    score_item_ask, _load_manifest, _get_structured, _seed_canary_row,
)


# The 13 conscious_dna gold records (all from train/val, previously missed)
CONSCIOUS_DNA_ITEMS = {
    "d049", "d050", "d051", "d052", "d053", "d055",
    "d057", "d058", "d061", "d063",
    "v036", "v037", "v038",
}

# Gold source_identities that must appear in top-5 after alias expansion
EXPECTED_SIS = {
    "d049": "273c2413cb47a298c9facd94d0181d4bfa710f800293c7e2f3e250d8099dd301",
    "d050": "273c2413cb47a298c9facd94d0181d4bfa710f800293c7e2f3e250d8099dd301",
    "d051": "42aaa7e591d6bd051a8e4ca9b97770f0217612d7c01efb6c125e1a64cee187b4",
    "d052": "42aaa7e591d6bd051a8e4ca9b97770f0217612d7c01efb6c125e1a64cee187b4",
    "d053": "009ca227a67e7f561513bb526565f5d4f634af155dee8a389d14b8913be60275",
    "d055": "69c53051fc09243b1b4cbae9ae7c5e8b14eaf8f50f6a8ea474d74e79248b826a",
    "d057": "555f420bea646b43637d5c88e74037742d6de447cc5acd4179fc5a22822833f0",
    "d058": "555f420bea646b43637d5c88e74037742d6de447cc5acd4179fc5a22822833f0",
    "d061": "82c97cbf2169271cf4d6b6def1c0c06b99e1f93fb3c99b3ecf122067f217a275",
    "d063": "e9ef339e1d6b8f8767a37c9fc08566bc513df4e75907b4d6eb82b6ff86741e5e",
    "v036": "009ca227a67e7f561513bb526565f5d4f634af155dee8a389d14b8913be60275",
    "v037": "555f420bea646b43637d5c88e74037742d6de447cc5acd4179fc5a22822833f0",
    "v038": "5a17fd8110000a5d40ddd0e225d029d03618d769a703027931544572e66a2acd",
}


def main():
    db_path = schema.default_db_path()

    # Load QA items
    eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    DEV_PATH = os.path.join(eval_dir, "qa_dev_retrieval.jsonl")
    VAL_RET_PATH = os.path.join(eval_dir, "qa_val_retrieval.jsonl")
    all_items = {it["id"]: it for it in _load_qa(DEV_PATH) + _load_qa(VAL_RET_PATH)}

    cdna_items = [all_items[mid] for mid in sorted(CONSCIOUS_DNA_ITEMS) if mid in all_items]

    # Build index
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

    # Evaluate conscious_dna items
    print(f"\nEvaluating {len(cdna_items)} conscious_dna items...")
    results = []
    for it in cdna_items:
        hits = _query(idx, it["question"], 5, "sensitive")
        rec = score_item_ask(it, hits, idx, sq, gates, True, build_id, build_sha,
                            "sensitive", score_scale="hybrid", db_path=db_path)
        results.append(rec)

    # Score
    si_recalls = [r["si_recall@5"] for r in results]
    mrrs = [r["si_mrr"] for r in results if r.get("si_mrr") == r.get("si_mrr")]
    hits = sum(1 for r in si_recalls if r == 1.0)
    misses = sum(1 for r in si_recalls if r < 1.0)

    avg_recall = sum(si_recalls) / len(si_recalls) if si_recalls else 0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0

    print(f"\n{'='*60}")
    print(f"CONSCIOUS_DNA ALIAS REGRESSION TEST")
    print(f"{'='*60}")
    print(f"  Items tested: {len(cdna_items)}")
    print(f"  Retrieval hit: {hits}/{len(cdna_items)}")
    print(f"  Retrieval miss: {misses}/{len(cdna_items)}")
    print(f"  si_recall@5: {avg_recall:.4f}")
    print(f"  si_mrr: {avg_mrr:.4f}")
    print(f"  Baseline before alias: 0.0 si_recall@5 on all 13")
    print()

    # Per-item breakdown
    print("Per-item:")
    for r in results:
        gold_si = EXPECTED_SIS.get(r.get("id", ""), "unknown")
        retrieved = r.get("retrieved_source_identities_top5", [])
        hit = "HIT" if gold_si in retrieved else "MISS"
        print(f"  [{hit}] {r.get('id')}: si_r5={r.get('si_recall@5', 0):.1f} "
              f"route={r.get('route')} decision={r.get('decision')} "
              f"gold_si={gold_si[:16]}... in_top5={gold_si in retrieved}")

    # Negative tests: ensure non-conscious_dna queries are NOT perturbed
    print(f"\nNegative tests (unrelated queries must not expand):")
    NEGATIVE_QUERIES = [
        "What is the team's specialization strategy?",
        "What specialization does the marketing team use?",
        "Describe the phi phenomenon in psychology.",
        # phi_score without conscious_dna context: no entity boost
        "What is the phi_score of the particle accelerator?",
        # Unknown agent name: no boost (name not in _CONSCIOUS_DNA_AGENTS)
        "What is the phi_score of the Hermes conscious_dna agent?",
        # Partial name: no boost (only whole-token match)
        "What is the phi_score of the Gabriel conscious_dna agent?",
    ]
    from quantum_corpus.rag import expand_query
    all_neg_pass = True
    for q in NEGATIVE_QUERIES:
        expanded = expand_query(q)
        # These queries must NOT receive agent TF boost (phi_score or specialization alias)
        has_phi_alias = "phi_score" in expanded and (
            "conscious_dna" in expanded.lower() and
            any(a in expanded for a in ["Raziel","Zadkiel","Raphael","Sandalphon","Uriel","Michael","Haniel","Jophiel"])
        )
        has_specialization_alias = "dna_specialization" in expanded and "conscious_dna" not in expanded.lower()
        bad = has_phi_alias or has_specialization_alias
        status = "BAD-EXPAND" if bad else "unchanged"
        if bad:
            all_neg_pass = False
        print(f"  [{status}] '{q}' -> '{expanded}'")

    passed = misses == 0 and hits == 13 and all_neg_pass
    print(f"\n{'='*60}")
    print(f"RESULT: {'PASS — all 13 conscious_dna records retrieved' if passed else f'FAIL — {misses} misses remain'}")
    print(f"{'='*60}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
