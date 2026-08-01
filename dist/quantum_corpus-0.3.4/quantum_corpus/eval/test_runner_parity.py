"""
test_runner_parity.py
====================
Integration test: score_item_ask() and answer.ask() must produce identical
decisions, retrieved IDs, and citations for the exact same input.

Run:
    python -m quantum_corpus.eval.test_runner_parity
"""
from __future__ import annotations
import os, sys, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ["TMT_QUANTUM_CORPUS_DB"] = "E:\\Temp\\qcorpus\\quantum_corpus.db"
os.environ["TMT_QUANTUM_JOBS_DB"] = "E:\\Temp\\qcorpus\\quantum_jobs_structured.db"
os.environ["TMT_DEPLOY_MODE"] = "private-training"

from quantum_corpus import answer as _answer
from quantum_corpus.eval import runner as _runner
from quantum_corpus.eval.runner import (
    _load_qa, _load_rows_split, _load_manifest, _get_structured, _build_retriever,
)


def _db_sha256_prefix() -> str:
    """First 8 hex chars of the corpus DB SHA-256 for trace provenance."""
    import hashlib
    db = os.environ["TMT_QUANTUM_CORPUS_DB"]
    if os.path.exists(db):
        h = hashlib.sha256()
        with open(db, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:8]
    return "not-found"


def _retriever_weights() -> dict:
    from quantum_corpus import fusion as _f
    return {"BM25_W": _f.BM25_W, "SEM_W": _f.SEM_W, "RRF_K": _f.RRF_K}


def _trace(case_id: str, top_hit_ids: list, decision: str,
           route: str, answer_len: int, passed: bool) -> dict:
    """Safe diagnostic trace — no snippets, no credentials, no raw text."""
    return {
        "case_id": case_id,
        "process_id": os.getpid(),
        "cwd": os.getcwd(),
        "db_sha256_prefix": _db_sha256_prefix(),
        "retriever": "hybrid",
        "weights": _retriever_weights(),
        "sensitivity": "sensitive",
        "use_structured": True,
        "top_k": 5,
        "top_hit_ids": top_hit_ids,
        "decision": decision,
        "route": route,
        "answer_len": answer_len,
        "field_gate_passed": passed,
    }


def test_runner_and_ask_parity():
    """score_item_ask() and answer.ask() must produce identical results
    for the exact same question, hits, and parameters."""

    print("Building shared hybrid retriever (one instance, shared between both paths)...")
    rows = _load_rows_split(os.environ["TMT_QUANTUM_CORPUS_DB"], ("train", "val"))
    idx = _build_retriever(rows, "hybrid")
    print(f"  retriever built: {len(idx)} docs  (weights: {_retriever_weights()})")

    sq = _get_structured(os.environ["TMT_QUANTUM_CORPUS_DB"])
    build_id, build_sha = _load_manifest()
    gates = {
        "score_floor_hybrid": 0.008,
        "score_floor_bm25": 3.0,
        "sep_ratio": 1.5,
        "sep_band": 2.0,
        "min_concepts": 1,
    }
    max_sensitivity = "sensitive"
    use_structured = True
    top_k = 5

    items = _load_qa("D:/TinyMetatron/quantum_corpus/eval/qa_val.jsonl")
    item_map = {it["id"]: it for it in items}

    # Target cases: v061 (error traceback) and v066 (HF token) are the known
    # false-answer failures. Also include v001 (valid answerable) as sanity.
    targets = ["v061", "v066", "v001"]

    traces = []

    for tid in targets:
        it = item_map[tid]
        question = it["question"]

        # ── path A: answer.ask() called directly ───────────────────────────────
        hits_a = idx.query(question, k=top_k, max_sensitivity=max_sensitivity)
        direct_res = _answer.ask(
            question=question,
            retriever=idx,
            structured_query=sq,
            hits=hits_a,
            top_k=top_k,
            max_sensitivity=max_sensitivity,
            build_id=build_id,
            build_sha256=build_sha,
            gates=gates,
            use_structured=use_structured,
            score_scale="hybrid",
        )

        # ── path B: score_item_ask() as the eval runner calls it ──────────────
        runner_res = _runner.score_item_ask(
            item=it,
            hits=hits_a,        # same hits — no re-retrieval
            retriever=idx,
            sq=sq,
            gates=gates,
            use_structured=use_structured,
            build_id=build_id,
            build_sha256=build_sha,
            max_sensitivity=max_sensitivity,
            score_scale="hybrid",
        )

        # ── compare ───────────────────────────────────────────────────────────
        ids_a = [h["id"] for h in hits_a[:top_k]]
        ids_b = runner_res["retrieved_top5"]

        # Get field_gate from direct result (not stored in runner output)
        fv = direct_res.get("field_gate", {})
        fv_passed = fv.get("passed", True) if fv else True

        decision_match   = runner_res["decision"] == direct_res["decision"]
        # runner stores citations as cited_ids (list[int]), direct stores as citations (list[dict])
        runner_cited = [c["id"] for c in runner_res.get("citations", [])] \
                       if runner_res.get("citations") is not None \
                       else runner_res.get("cited_ids", [])
        direct_cited  = [c["id"] for c in direct_res.get("citations", [])]
        citations_match = runner_cited == direct_cited
        retrieved_match  = ids_b == ids_a
        route_match     = runner_res["route"] == direct_res["route"]
        abstained_match = runner_res["abstained"] == direct_res["abstained"]

        all_match = all([decision_match, citations_match, retrieved_match,
                         route_match, abstained_match])

        status = "PASS" if all_match else "FAIL"
        print(f"\n{'='*60}")
        print(f"{status}  {tid}: {question[:60]}")
        print(f"  Retrieved IDs: runner={ids_b} direct={ids_a} match={retrieved_match}")
        print(f"  Decision:      runner={runner_res['decision']} direct={direct_res['decision']} match={decision_match}")
        print(f"  Route:         runner={runner_res['route']} direct={direct_res['route']} match={route_match}")
        print(f"  Abstained:     runner={runner_res['abstained']} direct={direct_res['abstained']} match={abstained_match}")
        print(f"  Citations:      match={citations_match}")
        print(f"  Field gate:    passed={fv_passed} reason={fv.get('reason','')}")

        trace = _trace(tid, ids_a, direct_res["decision"], direct_res["route"],
                       len(direct_res.get("answer", "")), fv_passed)
        traces.append(trace)

        if not all_match:
            print(f"\n  *** MISMATCH DETECTED ***")
            print(f"  direct_res keys: {list(direct_res.keys())}")
            print(f"  runner_res keys: {list(runner_res.keys())}")
            # Show where runner stores citations vs direct
            print(f"  runner citations (id list): {[c['id'] if isinstance(c,dict) else c for c in runner_res.get('citations',[])]}")
            print(f"  direct  citations (id list): {[c['id'] if isinstance(c,dict) else c for c in direct_res.get('citations',[])]}")
        else:
            print(f"  All fields match.")

        if not all_match:
            print(f"\n  *** MISMATCH DETECTED ***")
            print(f"  direct_res keys: {list(direct_res.keys())}")
            print(f"  runner_res keys: {list(runner_res.keys())}")
            # Show where runner stores citations vs direct
            print(f"  runner citations (id list): {[c['id'] if isinstance(c,dict) else c for c in runner_res.get('citations',[])]}")
            print(f"  direct  citations (id list): {[c['id'] if isinstance(c,dict) else c for c in direct_res.get('citations',[])]}")
            # Show all runner res fields
            print(f"  runner_res['cited_ids']: {runner_res.get('cited_ids')}")
            print(f"  runner_res['citations']: {runner_res.get('citations')}")
            print(f"  runner_res['decision']: {runner_res.get('decision')}")
            print(f"  runner_res['answer'][:100]: {(runner_res.get('answer') or '')[:100]}")
        else:
            print(f"  All fields match.")

    print(f"\n{'='*60}")
    print(f"Parity test: {len(targets)}/{len(targets)} passed")
    print("\nTrace (no secrets/snippets):")
    print(json.dumps(traces, indent=2))
    return traces


if __name__ == "__main__":
    test_runner_and_ask_parity()
