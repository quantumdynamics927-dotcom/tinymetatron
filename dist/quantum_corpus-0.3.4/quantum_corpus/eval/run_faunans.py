"""
quantum_corpus.eval.run_faunans
==============================
Regression runner for the field-verification gate (v0.3.2).

Runs faunans_regression.jsonl through answer.ask() and verifies each item
produces the expected decision and abstention state.

Usage::

    python -m quantum_corpus.eval.run_faunans
"""
from __future__ import annotations

import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from quantum_corpus import answer as _answer, rag, schema, secrets as _secrets
from quantum_corpus.fusion import HybridRetriever


FAUNANS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faunans_regression.jsonl")


def _load_faunans():
    return [json.loads(l) for l in open(FAUNANS_PATH, encoding="utf-8")]


def _build_retriever():
    """Build a minimal hybrid retriever over a small in-memory corpus that
    contains both records that SHOULD be found (valid questions) and records
    that won't help the unanswerable ones."""
    # Real job records (extracted from the frozen corpus patterns).
    # These are enough for the valid question (faunans-08) to retrieve correctly.
    records = [
        {
            "id": 36913,
            "project": "ibm-quantum",
            "source_type": "ibm_job",
            "doc_id": "ibm:d4mfq9l74pkc7388v73g",
            "sensitivity": "internal",
            "text": (
                "IBM Quantum job d4mfq9l74pkc7388v73g on backend ibm_fez, "
                "status Completed, program sampler, tags [], cost 600, "
                "created 2025-12-31T12:00:00Z. Measurement samples: 8192."
            ),
        },
        {
            "id": 36935,
            "project": "ibm-quantum",
            "source_type": "ibm_job",
            "doc_id": "ibm:d4r51hcfitbs739hjn9g",
            "sensitivity": "internal",
            "text": (
                "IBM Quantum job d4r51hcfitbs739hjn9g on backend ibm_fez, "
                "status Completed, program sampler, tags ['Composer'], cost 600, "
                "created 2026-01-02T09:30:00Z. Measurement samples: 4096."
            ),
        },
        {
            "id": 1,
            "project": "wormhole",
            "source_type": "manifest",
            "doc_id": "w:1",
            "sensitivity": "public",
            "text": (
                "Circuit 1: OTOC Lyapunov Exponent Measurement on ibm_kingston. "
                "Measures out-of-time-ordered correlator with readout error rate "
                "estimation."
            ),
        },
        {
            "id": 2,
            "project": "ibm-quantum",
            "source_type": "ibm_job",
            "doc_id": "ibm:decoy",
            "sensitivity": "internal",
            "text": (
                "IBM Quantum job decoy on backend ibm_fez, status Completed, "
                "program sampler, tags [], cost 600, created 2025-12-31T18:58:00Z. "
                "Measurement samples: 100."
            ),
        },
    ]
    return HybridRetriever.build(records)


def run():
    items = _load_faunans()
    retriever = _build_retriever()

    passed = 0
    failed = 0
    results = []

    for item in items:
        t0 = time.time()
        res = _answer.ask(
            item["question"],
            retriever,
            top_k=5,
            max_sensitivity="internal",
            gates=dict(_answer.DEFAULT_GATES),
            use_structured=False,  # field gate is retrieval-path only
        )
        elapsed = round((time.time() - t0) * 1000, 1)

        expected = item["expected"]
        actual = res["decision"]
        abstained = bool(res.get("abstained"))

        # "abstain" means any abstention via any gate; "not_established" requires
        # the field-verification gate specifically.
        if expected == "abstain":
            ok = abstained
        elif expected == "not_established":
            ok = (actual == "not_established" and abstained)
        elif expected == "answered":
            ok = (actual == "answered" and not abstained)
        else:
            ok = (actual == expected)

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        results.append({
            "id": item["id"],
            "failure_type": item.get("failure_type", ""),
            "expected": expected,
            "actual": actual,
            "abstained": abstained,
            "ok": ok,
            "latency_ms": elapsed,
            "notes": item.get("notes", ""),
        })

        print(
            f"{status}  {item['id']:12s}  expected={expected:20s}  "
            f"actual={actual:20s}  abstained={str(abstained):5s}  "
            f"{item.get('notes', '')[:60]}"
        )

    print()
    print(f"Results: {passed}/{len(items)} passed, {failed} failed")
    if failed:
        print("FAILED items:")
        for r in results:
            if not r["ok"]:
                print(f"  {r['id']}: expected={r['expected']} actual={r['actual']}")
        raise SystemExit(1)
    else:
        print("All faunans regression cases passed.")
    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run()
