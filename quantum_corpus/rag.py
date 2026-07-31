"""
quantum_corpus.rag
==================
Dependency-free BM25 retrieval over the quantum corpus records.

This is the RAG track: structured records (IBM jobs, manifest, repo corpus) are
retrieved at query time and can be prepended to a generation prompt. The model
never *learns* the 569 job dumps; it retrieves the relevant record. Retrieval
uses a normal word tokenizer (``\\w+``), NOT the project's 291-token char
tokenizer, so query terms like ``ibm_fez`` and ``OTOC`` match naturally.

Pure stdlib (math + re + collections). No sklearn/whoosh — keeps the public
Space dependency surface unchanged (this index is local-only anyway).

Usage::

    from quantum_corpus import schema, rag
    recs = schema.fetch_all(db_path)
    idx = rag.RAGIndex.build(recs)
    for hit in idx.query("what does the OTOC circuit measure?", k=3):
        print(hit["score"], hit["project"], hit["snippet"])
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

_WORD = re.compile(r"[a-z0-9_]+")
_STOP = set("the a an of to in on for and or is are was with what does how which "
            "this that these those it its from by as at be can you your our their "
            "we i you".split())


def _tok(text: str) -> List[str]:
    return [w for w in _WORD.findall((text or "").lower()) if w not in _STOP]


class RAGIndex:
    def __init__(self):
        self.docs: List[dict] = []          # {id, project, source_type, doc_id, text}
        self.term_freqs: List[Dict[str, int]] = []
        self.doc_len: List[int] = []
        self.df: Dict[str, int] = defaultdict(int)   # document frequency
        self.avgdl: float = 0.0
        self.k1 = 1.5
        self.b = 0.75

    @classmethod
    def build(cls, records: Sequence[dict]) -> "RAGIndex":
        idx = cls()
        for r in records:
            terms = _tok(r.get("text", ""))
            if not terms:
                continue
            idx.docs.append({
                "id": r.get("id"), "project": r.get("project", ""),
                "source_type": r.get("source_type", ""), "doc_id": r.get("doc_id", ""),
                "text": r.get("text", ""),
                "source_identity": r.get("source_identity", ""),
            })
            tf = Counter(terms)
            idx.term_freqs.append(dict(tf))
            idx.doc_len.append(len(terms))
            for t in tf:
                idx.df[t] += 1
        idx.avgdl = (sum(idx.doc_len) / len(idx.doc_len)) if idx.doc_len else 0.0
        return idx

    def __len__(self):
        return len(self.docs)

    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def query(self, text: str, k: int = 5) -> List[dict]:
        qterms = _tok(text)
        scores: List[float] = [0.0] * len(self.docs)
        for qt in qterms:
            idf = self._idf(qt)
            if idf == 0.0:
                continue
            for i, tf_map in enumerate(self.term_freqs):
                f = tf_map.get(qt, 0)
                if not f:
                    continue
                dl = self.doc_len[i] or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        order = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            d = self.docs[i]
            out.append({
                "id": d["id"], "score": round(scores[i], 4),
                "project": d["project"], "source_type": d["source_type"],
                "doc_id": d["doc_id"],
                "source_identity": d.get("source_identity", ""),
                "snippet": d["text"][:200].replace("\n", " "),
            })
            if len(out) >= k:
                break
        return out


# ── self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    def _ok(c, m):
        print(("OK  " if c else "FAIL") + " " + m)
        assert c, m

    recs = [
        {"id": 1, "project": "wormhole", "source_type": "manifest",
         "doc_id": "w:1", "text": "Circuit 1: OTOC Lyapunov Exponent Measurement. "
                                  "Measures C(t) = -<[W(t), V]^2> on ibm_kingston. 8192 shots."},
        {"id": 2, "project": "ibm-quantum", "source_type": "ibm_job",
         "doc_id": "ibm:d5a6", "text": "IBM Quantum job d5a6 on backend ibm_fez, "
                                       "status Completed, program sampler, 8192 samples."},
        {"id": 3, "project": "GRE", "source_type": "repo",
         "doc_id": "GRE:readme", "text": "Sierpinski triangle quantum walk with "
                                         "golden silver bronze coin angles."},
    ]
    idx = RAGIndex.build(recs)
    _ok(len(idx) == 3, f"indexed {len(idx)} docs")

    hits = idx.query("what does the OTOC circuit measure?", k=2)
    _ok(hits and hits[0]["id"] == 1, f"OTOC query -> manifest record: {hits}")
    _ok("OTOC" in hits[0]["snippet"], "snippet contains OTOC")

    hits = idx.query("ibm_fez backend job")
    _ok(hits and hits[0]["id"] == 2, f"ibm_fez query -> job record: {hits}")

    hits = idx.query("sierpinski golden coin")
    _ok(hits and hits[0]["id"] == 3, f"sierpinski query -> GRE record: {hits}")

    # empty / no-match query returns []
    _ok(idx.query("zzzznomatch") == [], "no-match query returns empty")

    print("SELF-TEST PASSED")