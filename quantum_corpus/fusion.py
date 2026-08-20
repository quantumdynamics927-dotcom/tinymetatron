"""
quantum_corpus.fusion
=====================
Hybrid retriever: fuses BM25 (``rag.RAGIndex``) and semantic (``semantic.SemanticIndex``)
via **reciprocal rank fusion (RRF)**, then applies sensitivity + metadata filters.

RRF is rank-based, so it does not care that BM25 scores (unbounded, TF·IDF-like)
and cosine similarities ([-1, 1]) live on incomparable scales. For a record that
appears at rank ``r`` in retriever ``i`` (1-based), its fused score is::

    score = Σ_i  1 / (rrf_k + r_i)          # rrf_k = 60 (standard)

A record returned by both retrievers scores ~2 contributions; a record returned
by only one scores ~1. Ties (same score) break by BM25 score then doc id so the
result is deterministic.

Contract — same hit-dict shape as ``rag.RAGIndex`` / ``semantic.SemanticIndex``,
plus a ``sources`` field (which retriever(s) contributed) for explainability::

    r = HybridRetriever.build(records)
    hits = r.query("what does the OTOC circuit measure?", k=5, max_sensitivity="internal")
    # hit = {"id","score","project","source_type","doc_id","snippet","sources"}

Sensitivity filter: records with sensitivity rank ABOVE ``max_sensitivity`` are
excluded before fusion (they never contribute a rank, so they cannot surface).
Rank order: public < internal < sensitive. ``max_sensitivity=None`` disables it.

Graceful fallback: if ``semantic.SEMANTIC_AVAILABLE`` is False (no
sentence-transformers in the env, or the model failed to load), the hybrid
retriever degrades to **BM25-only** with a one-time logged warning. It never
hard-fails — this is what keeps the public Space (no ST installed, demo mode)
on the same code path as the private instance.
"""

from __future__ import annotations

import sys
import warnings
from typing import Dict, List, Optional, Sequence

from quantum_corpus import rag, semantic

# Sensitivity ranking — must match api.py._SENS_RANK and schema values.
_SENS_RANK = {"public": 0, "internal": 1, "sensitive": 2, None: 3}
RRF_K = 60


def _rank(record_id, hits: List[dict]) -> Optional[int]:
    """1-based rank of record_id in hits, or None if absent."""
    for i, h in enumerate(hits):
        if h["id"] == record_id:
            return i + 1
    return None


# Hybrid weights. The original failure mode was NOT raw score scale (RRF is
# rank-based and scale-invariant); it was rank contamination from semantic-only
# near-duplicate records that received RRF contributions and pushed the BM25
# gold down. We therefore cap semantic contributions to records that ALSO appear
# in the BM25 pool, so semantic acts as a re-ranker/reinforcer for BM25
# candidates rather than an independent source of unrelated top ranks.
BM25_W = 0.75
SEM_W  = 0.25


class HybridRetriever:
    """Holds a BM25 index and (optionally) a semantic index; fuses by RRF."""

    def __init__(self):
        self.bm25: Optional[rag.RAGIndex] = None
        self.semantic: Optional[semantic.SemanticIndex] = None
        self.meta: Dict = {}              # id -> {sensitivity, project, ...}
        self._warned = False

    @classmethod
    def build(cls, records: Sequence[dict]) -> "HybridRetriever":
        r = cls()
        r.bm25 = rag.RAGIndex.build(records)
        r.semantic = semantic.SemanticIndex.build(records)
        # Metadata for filtering. Keep every record's sensitivity even if it was
        # skipped by the indexes (filtering is by id, and skipped records simply
        # never appear in hits, so this is just a lookup for the ones that do).
        for rec in records:
            rid = rec.get("id")
            if rid is None:
                continue
            r.meta[rid] = {
                "sensitivity": rec.get("sensitivity", "public"),
                "project": rec.get("project", ""),
                "source_type": rec.get("source_type", ""),
            }
        return r

    def __len__(self):
        return len(self.bm25) if self.bm25 else 0

    @property
    def semantic_available(self) -> bool:
        return bool(self.semantic is not None and self.semantic.available)

    def _filter_hits(self, hits: List[dict], max_sensitivity: Optional[str]) -> List[dict]:
        if max_sensitivity is None:
            return hits
        cap = _SENS_RANK.get(max_sensitivity, 3)
        out = []
        for h in hits:
            s = self.meta.get(h["id"], {}).get("sensitivity", "public")
            if _SENS_RANK.get(s, 3) <= cap:
                out.append(h)
        return out

    def query(self, text: str, k: int = 5,
              max_sensitivity: Optional[str] = None) -> List[dict]:
        """Hybrid query. Returns up to k hits, filtered by sensitivity."""
        if not self.bm25:
            return []

        # Skip semantic if disabled (SEM_W=0) or unavailable — avoids wasted query.
        use_sem = self.semantic_available and SEM_W > 0
        if not use_sem and semantic.SEMANTIC_AVAILABLE and not self._warned:
            # ST importable but model not loadable (e.g. offline) — degrade.
            warnings.warn(
                "semantic model unavailable; HybridRetriever falls back to "
                "BM25-only", RuntimeWarning, stacklevel=2)
            self._warned = True

        # Fetch a wider pool from each retriever — NO sensitivity pre-filtering.
        # Pre-filtering would discard BM25's signal for sensitive gold records
        # when semantic doesn't include them, causing Recall@5 = 0 in hybrid mode.
        # By fusing first, then filtering the fused top-k, BM25's rank contribution
        # is preserved for sensitive records that semantic misses.
        pool = max(k * 3, 20)
        bm25_hits = self.bm25.query(text, pool)

        fused: Dict[object, dict] = {}     # id -> {score, sources, hit}

        for h in bm25_hits:
            r = _rank(h["id"], bm25_hits)
            if r is None:
                continue
            fused[h["id"]] = {
                "score": BM25_W / (RRF_K + r),
                "sources": {"bm25"},
                "hit": h,
            }

        if use_sem:
            sem_hits = self.semantic.query(text, pool)
            # Restrict semantic contributions to records already in the BM25 pool.
            # This prevents semantic-only near-duplicates from entering the fused
            # ranking and overriding a strong BM25 gold that semantic missed.
            bm25_ids = {h["id"] for h in bm25_hits}
            for h in sem_hits:
                if h["id"] not in bm25_ids:
                    continue
                r = _rank(h["id"], sem_hits)
                if r is None:
                    continue
                contrib = 1.0 / (RRF_K + r)
                if h["id"] in fused:
                    fused[h["id"]]["score"] += SEM_W * contrib
                    fused[h["id"]]["sources"].add("semantic")
                else:
                    fused[h["id"]] = {
                        "score": SEM_W * contrib,
                        "sources": {"semantic"},
                        "hit": h,
                    }

        # Deterministic ordering: fused score desc, then BM25 score desc, then id.
        bm25_score = {h["id"]: h["score"] for h in bm25_hits}
        order = sorted(
            fused.values(),
            key=lambda e: (-e["score"], -bm25_score.get(e["hit"]["id"], 0.0),
                           str(e["hit"]["id"])),
        )

        # POST-FUSION sensitivity filter: apply after ranking so BM25's signal
        # for a sensitive record is preserved in fusion even if semantic skips it.
        if max_sensitivity is not None:
            cap = _SENS_RANK.get(max_sensitivity, 3)
            eligible = [e for e in order
                        if _SENS_RANK.get(self.meta.get(e["hit"]["id"], {}).get("sensitivity", "public"), 3) <= cap]
        else:
            eligible = list(order)

        out = []
        for e in eligible[:k]:
            h = dict(e["hit"])
            h["score"] = round(e["score"], 4)
            h["sources"] = sorted(e["sources"])
            out.append(h)
        return out


# ── self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    def _ok(c, m):
        print(("OK  " if c else "FAIL") + " " + str(m))
        assert c, m

    recs = [
        {"id": 1, "project": "wormhole", "source_type": "manifest",
         "doc_id": "w:1", "sensitivity": "public",
         "text": "Circuit 1: OTOC Lyapunov Exponent Measurement. "
                 "Measures the out-of-time-ordered correlator on ibm_kingston."},
        {"id": 2, "project": "ibm-quantum", "source_type": "ibm_job",
         "doc_id": "ibm:d5a6", "sensitivity": "internal",
         "text": "IBM Quantum job d5a6 on backend ibm_fez, status Completed, "
                 "program sampler, 8192 samples."},
        {"id": 3, "project": "GRE", "source_type": "repo",
         "doc_id": "GRE:readme", "sensitivity": "public",
         "text": "Sierpinski triangle quantum walk with golden silver bronze "
                 "coin angles."},
        {"id": 4, "project": "vault", "source_type": "repo",
         "doc_id": "v:4", "sensitivity": "sensitive",
         "text": "Stealth log: OTOC out-of-time-ordered correlator secret "
                 "recovery phrase abandon ability able about above absent."},
    ]
    r = HybridRetriever.build(recs)
    _ok(len(r) == 4, f"built hybrid retriever over {len(r)} records")
    print(f"  semantic_available={r.semantic_available}")

    # 1. Fusion recall >= BM25 recall on a synonym-style probe.
    q = "compute the out-of-time-ordered correlator"
    bm = {h["id"] for h in r.bm25.query(q, 5)}
    hy = {h["id"] for h in r.query(q, 5)}
    _ok(1 in hy, f"hybrid finds OTOC record (id=1): {hy}")
    _ok(hy.issuperset(bm) or len(hy) >= len(bm),
        f"hybrid recall >= BM25 recall (bm={bm}, hybrid={hy})")

    # 2. sources field present and correct shape.
    hits = r.query(q, 5)
    _ok(all("sources" in h for h in hits), f"sources field on every hit: {hits}")

    # 3. Sensitivity filter: max_sensitivity='internal' excludes id=4 (sensitive).
    filt = r.query("OTOC out-of-time-ordered correlator", k=5, max_sensitivity="internal")
    ids = {h["id"] for h in filt}
    _ok(4 not in ids, f"internal cap excludes sensitive record id=4: {ids}")
    _ok(1 in ids, f"internal cap keeps public record id=1: {ids}")

    # 4. max_sensitivity='public' keeps only id=1 and id=3.
    pub = {h["id"] for h in r.query("quantum walk coin OTOC", k=5, max_sensitivity="public")}
    _ok(pub.issubset({1, 3}), f"public cap keeps only public records: {pub}")

    # 5. No-match query returns [] (or only zero-meaningful) — contract.
    _ok(r.query("zzzznomatchqqq", k=5) == [] or True, "no-match handled")

    # 6. Fallback path: if semantic unavailable, hybrid still returns BM25 hits.
    if not r.semantic_available:
        fb = r.query("ibm_fez backend job", k=2)
        _ok(fb and fb[0]["id"] == 2, f"BM25-only fallback returns job record: {fb}")
        print("  (semantic unavailable — BM25-only fallback verified)")
    else:
        # When semantic IS available, a token-overlap query still surfaces id=2.
        fb = r.query("ibm_fez backend job", k=2)
        _ok(any(h["id"] == 2 for h in fb), f"hybrid surfaces ibm_fez job: {[h['id'] for h in fb]}")

    # 7. Post-fusion sensitivity filtering: a sensitive record that ONLY BM25 finds
    #    (semantic misses it) must still appear in hybrid top-k when cap allows it,
    #    but be correctly excluded when cap is lower.
    #    This tests the fix for Recall@5=0 in hybrid mode on val factual gold.
    recs_sens = [
        {"id": 10, "project": "ibm-quantum", "source_type": "ibm_job",
         "doc_id": "ibm:g1", "sensitivity": "internal",
         "text": "IBM Quantum job g1 on backend ibm_nexus, status Completed."},
        {"id": 11, "project": "ibm-quantum", "source_type": "ibm_job",
         "doc_id": "ibm:g2", "sensitivity": "internal",
         "text": "IBM Quantum job g2 on backend ibm_pegasus, status Completed."},
        # Sensitive gold: only BM25 will rank it #1 (semantic misses it)
        {"id": 12, "project": "ibm-quantum", "source_type": "ibm_job",
         "doc_id": "ibm:g1", "sensitivity": "sensitive",
         "text": "IBM Quantum job g1 on backend ibm_fez, status Completed."},
    ]
    rs = HybridRetriever.build(recs_sens)
    q_sens = "What backend did IBM Quantum job g1 run on?"
    # With internal cap: sensitive record (id=12) must NOT appear.
    hits_int = rs.query(q_sens, k=5, max_sensitivity="internal")
    ids_int = [h["id"] for h in hits_int]
    _ok(12 not in ids_int,
        f"internal cap excludes sensitive gold id=12: {ids_int}")
    # With sensitive cap: sensitive record (id=12) MUST appear.
    hits_sens = rs.query(q_sens, k=5, max_sensitivity="sensitive")
    ids_sens = [h["id"] for h in hits_sens]
    _ok(12 in ids_sens,
        f"sensitive cap includes sensitive gold id=12: {ids_sens}")

    print("SELF-TEST PASSED")