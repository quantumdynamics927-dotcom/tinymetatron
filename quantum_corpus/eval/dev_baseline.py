"""
Baseline eval for v0.3.4-dev retrieval improvement cycle.

Run on the train+val index to establish baseline metrics for the dev set
before any retriever changes are made.

Targets (from frozen v0.3.3 final eval):
  - si_recall@5:  0.4750  (retrieval-applicable)
  - si_mrr:       0.3696  (retrieval-applicable)
  - false_abs@5:  8.24%   (7/85 answerable)
  - fa_unans:      0.0     (0/15 unanswerable)

Dev set: qa_dev_retrieval.jsonl (21 items — 17 miss + 4 hit contrast)
Val set: qa_val_retrieval.jsonl (15 items — held-out from dev)

Run::

    python -m quantum_corpus.eval.dev_baseline
"""
from __future__ import annotations

import os, sys, json, time
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("TMT_QUANTUM_CORPUS_DB", "E:\\Temp\\qcorpus\\quantum_corpus.db")
os.environ.setdefault("TMT_QUANTUM_JOBS_DB", "E:\\Temp\\qcorpus\\quantum_jobs_structured.db")
os.environ.setdefault("TMT_DEPLOY_MODE", "private-training")

from quantum_corpus import schema
from quantum_corpus.eval.runner import (
    _load_qa, _load_rows_split, _build_retriever, _query,
    score_item_ask, _load_manifest, _get_structured, aggregate,
    _seed_canary_row,
)

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DEV_PATH = os.path.join(EVAL_DIR, "qa_dev_retrieval.jsonl")
VAL_RET_PATH = os.path.join(EVAL_DIR, "qa_val_retrieval.jsonl")
VAL_STR_PATH = os.path.join(EVAL_DIR, "qa_val_structured.jsonl")
OUT_DIR = os.path.join("E:\\Temp\\qcorpus\\reports")
os.makedirs(OUT_DIR, exist_ok=True)


def _mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def eval_items(items, idx, sq, gates, build_id, build_sha, db_path):
    records = []
    for it in items:
        hits = _query(idx, it["question"], 5, "sensitive")
        rec = score_item_ask(it, hits, idx, sq, gates, True,
                             build_id, build_sha, "sensitive",
                             score_scale="hybrid", db_path=db_path)
        records.append(rec)
    return records


def print_summary(label, records):
    si_recalls = [r["si_recall@5"] for r in records if r.get("si_recall@5") == r.get("si_recall@5")]
    mrrs = [r["si_mrr"] for r in records if r.get("si_mrr") == r.get("si_mrr")]
    non_abst = [r for r in records if not r.get("expected_abstention")]
    abst = [r for r in records if r.get("expected_abstention")]

    hit = sum(1 for r in non_abst if r.get("si_recall@5", 0) >= 1.0)
    miss = sum(1 for r in non_abst if r.get("si_recall@5", 0) < 1.0)
    false_abs = sum(1 for r in non_abst if r.get("abstained"))
    abst_recall = sum(1 for r in abst if r.get("abstained"))

    print(f"\n{'='*60}")
    print(f"{label} ({len(records)} items)")
    print(f"{'='*60}")
    print(f"  si_recall@5 : {_mean(si_recalls):.4f}  (n={len(si_recalls)})")
    print(f"  si_mrr      : {_mean(mrrs):.4f}  (n={len(mrrs)})")
    print(f"  Retrieval   : {hit} hit / {miss} miss")
    if non_abst:
        print(f"  False abst : {false_abs}/{len(non_abst)} ({false_abs/len(non_abst):.2%})")
    if abst:
        print(f"  Abst recall: {abst_recall}/{len(abst)} ({abst_recall/len(abst):.2%})")
    return {
        "label": label,
        "n": len(records),
        "si_recall@5": round(_mean(si_recalls), 4),
        "si_mrr": round(_mean(mrrs), 4),
        "hit": hit,
        "miss": miss,
        "false_abstentions": false_abs,
        "false_abstention_rate": round(false_abs / len(non_abst), 4) if non_abst else None,
        "abstention_recall": round(abst_recall / len(abst), 4) if abst else None,
    }


def main():
    db_path = schema.default_db_path()
    print(f"Loading train+val index ({len(rows) if False else '...'})...")
    rows = _load_rows_split(db_path, ("train", "val"))
    canary = _seed_canary_row()
    t0 = time.time()
    idx = _build_retriever([canary] + rows, "hybrid")
    print(f"  retriever built: {len(idx)} docs in {time.time()-t0:.1f}s")
    sq = _get_structured(db_path)
    build_id, build_sha = _load_manifest()
    gates = {
        "score_floor_hybrid": 0.008,
        "score_floor_bm25": 3.0,
        "sep_ratio": 1.5,
        "sep_band": 2.0,
        "min_concepts": 1,
    }

    # Dev set
    dev_items = _load_qa(DEV_PATH)
    dev_records = eval_items(dev_items, idx, sq, gates, build_id, build_sha, db_path)
    dev_sum = print_summary("DEV retrieval baseline", dev_records)

    # Val retrieval set
    val_items = _load_qa(VAL_RET_PATH)
    val_records = eval_items(val_items, idx, sq, gates, build_id, build_sha, db_path)
    val_sum = print_summary("VAL retrieval (held-out)", val_records)

    # Val structured set
    val_str_items = _load_qa(VAL_STR_PATH)
    val_str_records = eval_items(val_str_items, idx, sq, gates, build_id, build_sha, db_path)
    val_str_sum = print_summary("VAL structured (correct SQL)", val_str_records)

    # Combined non-abstention metrics
    all_non_abst = dev_records + val_records
    si_recalls = [r["si_recall@5"] for r in all_non_abst if r.get("si_recall@5") == r.get("si_recall@5")]
    mrrs = [r["si_mrr"] for r in all_non_abst if r.get("si_mrr") == r.get("si_mrr")]
    false_abs = sum(1 for r in all_non_abst if r.get("abstained"))

    print(f"\n{'='*60}")
    print("COMBINED (dev + val retrieval, n={})".format(len(all_non_abst)))
    print(f"  si_recall@5 : {_mean(si_recalls):.4f}")
    print(f"  si_mrr      : {_mean(mrrs):.4f}")
    print(f"  False abst : {false_abs}/{len(all_non_abst)} ({false_abs/len(all_non_abst):.2%})")
    print(f"  v0.3.3 final targets: si_recall@5=0.4750, si_mrr=0.3696")

    # Save results
    out = {
        "dev": dev_sum,
        "val_retrieval": val_sum,
        "val_structured": val_str_sum,
        "combined": {
            "n": len(all_non_abst),
            "si_recall@5": round(_mean(si_recalls), 4),
            "si_mrr": round(_mean(mrrs), 4),
            "false_abstentions": false_abs,
            "false_abstention_rate": round(false_abs / len(all_non_abst), 4),
        },
    }
    out_path = os.path.join(OUT_DIR, "dev_baseline_v034.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
