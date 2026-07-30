"""
quantum_corpus.eval.tune
========================
Tune the v0.3 gate thresholds ON THE FROZEN VALIDATION SET ONLY.

The expensive part is building the hybrid retriever (embedding ~38k train+val
records on CPU, ~minutes). The gate thresholds (score floor, separation ratio,
min concepts) are POST-retrieval, so the per-item hits do not change when the
thresholds change. This script therefore:

  1. builds the hybrid retriever over train+val ONCE (and seeds the canary),
  2. fetches hits for every val item ONCE and caches them to disk,
  3. sweeps gate-threshold combinations, re-running only the cheap gate logic
     (``answer.ask(hits=...)``) per combo, and
  4. prints a table of abstention recall / precision / false-answer rate /
     false-abstention rate / answerable recall for each combo.

Test set is never touched. Run::

    TMT_QUANTUM_CORPUS_DB=... TMT_QUANTUM_STRUCTURED_DB=... \
        python -m quantum_corpus.eval.tune --sensitivity sensitive
"""

from __future__ import annotations

import os
import sys
import json
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from quantum_corpus import schema, answer as _answer, structured as _structured
from quantum_corpus.eval.runner import (
    _load_rows_split, _load_qa, _query, _seed_canary_row, _build_retriever,
    _load_manifest, _get_structured, QA_VAL_PATH, aggregate, score_item_ask,
)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
HITS_CACHE = os.path.join(EVAL_DIR, "_val_hits.json")


def _fetch_and_cache_hits(db_path, retriever_kind, max_sensitivity, use_cache):
    """Build the retriever once, fetch hits for every val item, cache to disk."""
    rows = _load_rows_split(db_path, ("train", "val"))
    canary = _seed_canary_row()
    if use_cache and os.path.isfile(HITS_CACHE):
        cache = json.load(open(HITS_CACHE, encoding="utf-8"))
        if cache.get("kind") == retriever_kind and cache.get("n_rows") == len(rows):
            print(f"using cached val hits ({len(cache['items'])} items) from {HITS_CACHE}")
            return cache["items"]
    print(f"Building {retriever_kind} retriever over {len(rows)} records (+canary)...")
    t0 = time.time()
    idx = _build_retriever([canary] + rows, retriever_kind)
    print(f"  built {len(idx)} docs in {time.time()-t0:.1f}s")
    items = _load_qa(QA_VAL_PATH)
    out = []
    for it in items:
        hits = _query(idx, it["question"], 20, max_sensitivity)
        out.append({"item": it, "hits": hits})
    json.dump({"kind": retriever_kind, "n_rows": len(rows), "items": out},
              open(HITS_CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"  cached hits for {len(out)} items -> {HITS_CACHE}")
    return out


def _evaluate(cached, sq, gates, use_structured, build_id, build_sha,
              max_sensitivity, score_scale):
    records = []
    for row in cached:
        it, hits = row["item"], row["hits"]
        rec = score_item_ask(it, hits, None, sq, gates, use_structured,
                             build_id, build_sha, max_sensitivity,
                             score_scale=score_scale)
        records.append(rec)
    return aggregate(records)


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    retriever = "hybrid"
    max_sensitivity = "sensitive"
    use_structured = True
    rebuild = False
    for i, a in enumerate(argv):
        if a == "--retriever" and i + 1 < len(argv):
            retriever = argv[i + 1]
        if a == "--sensitivity" and i + 1 < len(argv):
            max_sensitivity = argv[i + 1]
        if a == "--no-structured":
            use_structured = False
        if a == "--rebuild":
            rebuild = True
    db_path = schema.default_db_path()
    if not os.path.exists(db_path):
        print(f"FATAL: corpus DB not found at {db_path}", file=sys.stderr); return 2
    build_id, build_sha = _load_manifest()
    sq = _get_structured(db_path) if use_structured else None

    cached = _fetch_and_cache_hits(db_path, retriever, max_sensitivity, use_cache=not rebuild)
    score_scale = "hybrid" if retriever == "hybrid" else "bm25"

    # Sweep grid. score_floor_hybrid is on the RRF scale (~0.016 single, ~0.033
    # dual contribution). sep_ratio / min_concepts are secondary.
    floors = [0.008, 0.012, 0.016, 0.020, 0.024, 0.030]
    sep_ratios = [1.1, 1.2, 1.5]
    min_concepts = [1, 2]

    print(f"\nSweeping gates (retriever={retriever}, structured={use_structured}, "
          f"max_sensitivity={max_sensitivity})...\n")

    def _f(x):
        """Format a metric that may be None or NaN."""
        if x is None:
            return "  -"
        try:
            if x != x:  # NaN
                return "  -"
        except TypeError:
            pass
        return f"{x:>6.3f}" if isinstance(x, float) else f"{x:>6}"

    hdr = (f"{'floor':>7} {'sep':>5} {'minc':>5} | {'abstRec':>7} {'abstPrec':>8} "
           f"{'faUnans':>8} {'fabAns':>7} | {'R@5':>6} {'rubric':>6}")
    print(hdr)
    print("-" * len(hdr))
    best = None
    for floor in floors:
        for sep in sep_ratios:
            for mc in min_concepts:
                gates = {"score_floor_hybrid": floor, "score_floor_bm25": 3.0,
                         "sep_ratio": sep, "sep_band": 2.0, "min_concepts": mc}
                s = _evaluate(cached, sq, gates, use_structured, build_id, build_sha,
                              max_sensitivity, score_scale)
                row = (f"{floor:>7} {sep:>5} {mc:>5} | "
                       f"{_f(s['abstention_recall']):>7} "
                       f"{_f(s['abstention_precision']):>8} "
                       f"{_f(s['false_answer_rate_on_unanswerable']):>8} "
                       f"{_f(s['false_abstention_rate_on_answerable']):>7} | "
                       f"{_f(s['recall@5']):>6} {_f(s['rubric_correctness']):>6}")
                print(row)
                # score: high abstRec, low faUnans, low fabAns, high R@5
                def _n(v):
                    return v if (v is not None and v == v) else 0
                score = (_n(s["abstention_recall"]) - _n(s["false_answer_rate_on_unanswerable"])
                         - 0.5 * _n(s["false_abstention_rate_on_answerable"])
                         + 0.2 * _n(s["recall@5"]))
                if best is None or score > best[0]:
                    best = (score, floor, sep, mc, s)
    print("-" * len(hdr))
    b = best[1], best[2], best[3]
    print(f"\nBEST: floor={b[0]} sep={b[1]} min_concepts={b[2]}  (score={best[0]:.3f})")
    bs = best[4]
    print(f"  abstRec={bs['abstention_recall']} abstPrec={bs['abstention_precision']} "
          f"faUnans={bs['false_answer_rate_on_unanswerable']} "
          f"fabAns={bs['false_abstention_rate_on_answerable']} "
          f"R@5={bs['recall@5']} rubric={bs['rubric_correctness']}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())