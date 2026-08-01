"""
Diagnostic: confirm the two hybrid fusion bugs.
Run: python -m quantum_corpus.eval.test_fusion_sensitivity
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_bm25_ignores_max_sensitivity():
    """BM25 RAGIndex.query does not accept max_sensitivity — it silently ignores it."""
    from quantum_corpus.rag import RAGIndex
    import inspect
    sig = inspect.signature(RAGIndex.query)
    params = list(sig.parameters.keys())
    has_max_sens = "max_sensitivity" in params or "max_sens" in params
    print(f"RAGIndex.query params: {params}")
    print(f"has max_sensitivity param: {has_max_sens}")
    assert not has_max_sens, "BUG: RAGIndex.query should NOT have max_sensitivity param"
    print("  PASS: BM25 ignores max_sensitivity (no param)")
    return True


def test_sensitive_gold_filtered_by_hybrid():
    """Sensitive gold records should appear in hybrid top-5 with sensitive cap, but
    disappear with internal cap even though BM25 finds them at rank 1."""
    from quantum_corpus.fusion import HybridRetriever

    # 3 docs: doc1=internal (has JID keyword), doc2=internal (other job),
    #          doc3=sensitive (gold record, also has JID keyword)
    recs = [
        {
            "id": 1, "project": "ibm-quantum", "source_type": "ibm_job",
            "doc_id": "ibm:d4bu2gsi51bc738jfacg",
            "sensitivity": "internal",
            "text": "IBM Quantum job d4bu2gsi51bc738jfacg on backend ibm_fez, status Completed.",
        },
        {
            "id": 2, "project": "ibm-quantum", "source_type": "ibm_job",
            "doc_id": "ibm:d4dubr1eg65s738lsbug",
            "sensitivity": "internal",
            "text": "IBM Quantum job d4dubr1eg65s738lsbug on backend ibm_fez, status Completed.",
        },
        {
            "id": 3, "project": "ibm-quantum", "source_type": "ibm_job",
            "doc_id": "ibm:d4bu2gsi51bc738jfacg",
            "sensitivity": "sensitive",  # gold is sensitive
            "text": "IBM Quantum job d4bu2gsi51bc738jfacg on backend ibm_torino, status Completed.",
        },
    ]
    idx = HybridRetriever.build(recs)
    q = "What backend did IBM Quantum job d4bu2gsi51bc738jfacg run on?"

    # With internal cap: doc3 (sensitive gold) must NOT appear
    hits_internal = idx.query(q, k=5, max_sensitivity="internal")
    ids_internal = [h["id"] for h in hits_internal]
    print(f"  Internal cap hits: {ids_internal}")
    assert 3 not in ids_internal, "BUG: sensitive gold should be filtered with internal cap"
    print("  PASS: sensitive gold filtered with internal cap")

    # With sensitive cap: doc3 (gold) MUST appear
    hits_sens = idx.query(q, k=5, max_sensitivity="sensitive")
    ids_sens = [h["id"] for h in hits_sens]
    print(f"  Sensitive cap hits: {ids_sens}")
    assert 3 in ids_sens, "BUG: sensitive gold should appear with sensitive cap"
    print("  PASS: sensitive gold appears with sensitive cap")


def test_bm25_recall_with_internal_cap():
    """BM25 with internal cap should NOT find sensitive gold records (contrast with hybrid)."""
    from quantum_corpus.rag import RAGIndex

    recs = [
        {
            "id": 1, "project": "ibm-quantum", "source_type": "ibm_job",
            "doc_id": "ibm:d4bu2gsi51bc738jfacg",
            "sensitivity": "internal",
            "text": "IBM Quantum job d4bu2gsi51bc738jfacg on backend ibm_fez, status Completed.",
        },
        {
            "id": 2, "project": "ibm-quantum", "source_type": "ibm_job",
            "doc_id": "ibm:d4bu2gsi51bc738jfacg",
            "sensitivity": "sensitive",
            "text": "IBM Quantum job d4bu2gsi51bc738jfacg on backend ibm_torino, status Completed.",
        },
    ]
    idx = RAGIndex.build(recs)
    q = "What backend did IBM Quantum job d4bu2gsi51bc738jfacg run on?"

    # BM25 ignores max_sensitivity — it returns everything
    hits = idx.query(q, k=5)
    ids = [h["id"] for h in hits]
    print(f"  BM25 hits (ignores cap): {ids}")
    assert 2 in ids, "BM25 should find sensitive doc"
    print("  NOTE: BM25 ignores max_sensitivity — returns sensitive docs")


if __name__ == "__main__":
    print("=== Fusion sensitivity bug diagnostic ===\n")

    print("Test 1: BM25 ignores max_sensitivity parameter")
    try:
        test_bm25_ignores_max_sensitivity()
    except Exception as e:
        print(f"  FAIL: {e}\n")
        raise

    print()
    print("Test 2: Sensitive gold with hybrid internal cap")
    try:
        test_sensitive_gold_filtered_by_hybrid()
    except Exception as e:
        print(f"  FAIL: {e}\n")
        raise

    print()
    print("Test 3: BM25 vs hybrid contrast")
    test_bm25_recall_with_internal_cap()

    print()
    print("=== DIAGNOSIS ===")
    print("BUG 1: RAGIndex.query() has no max_sensitivity parameter.")
    print("  BM25-only mode silently ignores max_sensitivity, returning ALL records.")
    print("  This inflates Recall@5 in bm25 mode but breaks hybrid fusion.")
    print()
    print("BUG 2: When max_sensitivity='internal' in hybrid fusion, sensitive gold")
    print("  records are filtered from BOTH the BM25 AND semantic pools before fusion.")
    print("  If the semantic index doesn't return the sensitive gold record, BM25's")
    print("  contribution is discarded and the gold disappears from fused results.")
    print()
    print("FIXES NEEDED:")
    print("  1. Add max_sensitivity support to RAGIndex.query() for consistency")
    print("  2. Apply sensitivity filter AFTER fusion (post-filter), not before")
    print("  3. Or: for hybrid mode, always retrieve sensitive records in the pool")
    print("     even when max_sensitivity='internal', then filter post-fusion")
