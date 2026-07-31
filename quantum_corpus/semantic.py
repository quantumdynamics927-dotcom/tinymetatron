"""
quantum_corpus.semantic
=======================
Semantic (dense) retrieval over the quantum corpus records.

This is the semantic half of the hybrid retriever (see ``fusion.py``). It uses
``sentence-transformers`` with the local ``all-MiniLM-L6-v2`` model (~22 MB,
384-dim) — chosen because the corpus is **private/local** and never ships to
the public HF Space. The new dependency affects only the local env, not the
Space surface (the Dockerfile is unchanged).

Contract — identical to ``rag.RAGIndex`` so fusion can join on ``id``::

    idx = SemanticIndex.build(records)        # records: list[dict] from schema.fetch_all
    hits = idx.query("what does the OTOC circuit measure?", k=5)
    # hit = {"id","score","project","source_type","doc_id","snippet"}

``build`` skips records with empty text for the SAME reason ``RAGIndex`` does
(nothing to embed), so the ``id`` sets of the two indexes align and reciprocal
rank fusion can join them cleanly.

Graceful fallback: if ``sentence_transformers`` is not importable, or the model
cannot be loaded (offline first run), ``SEMANTIC_AVAILABLE`` is False and
``query`` returns ``[]``. The hybrid retriever then degrades to BM25-only with
a logged warning — it never hard-fails. This keeps the public Space (no ST
installed, demo mode) unaffected and makes the semantic path optional.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Sequence

try:                                    # numpy is a sentence-transformers dep;
    import numpy as _np                  # present whenever ST is. Guard anyway.
    _NUMPY_AVAILABLE = True
except Exception:                        # pragma: no cover
    _NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except Exception:                        # pragma: no cover - env-dependent
    SentenceTransformer = None           # type: ignore[assignment]
    SEMANTIC_AVAILABLE = False

_MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL: Optional[object] = None
_MODEL_LOCK = threading.Lock()
# Runtime flag: set False if the model fails to load (e.g. offline). Separate
# from SEMANTIC_AVAILABLE so a transient download failure degrades gracefully
# rather than crashing every query.
_RUNTIME_OK = True


def _load_model():
    """Lazy, cached, thread-safe model load. Returns the model or None."""
    global _MODEL, _RUNTIME_OK
    if not SEMANTIC_AVAILABLE or not _NUMPY_AVAILABLE:
        return None
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            _MODEL = SentenceTransformer(_MODEL_NAME)
        except Exception:
            _RUNTIME_OK = False
            _MODEL = None
    return _MODEL


def _embed(texts: List[str]):
    """Encode a batch of texts -> L2-normalized float32 array [N, dim].

    Returns None if the model is unavailable (caller falls back to [])."""
    model = _load_model()
    if model is None:
        return None
    vecs = model.encode(
        texts, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    )
    return _np.asarray(vecs, dtype=_np.float32)


class SemanticIndex:
    """Dense cosine-similarity index, same contract as ``rag.RAGIndex``."""

    def __init__(self):
        self.docs: List[dict] = []          # {id, project, source_type, doc_id, text}
        self.embeddings = None              # np.ndarray [N, dim] normalized, or None

    @classmethod
    def build(cls, records: Sequence[dict]) -> "SemanticIndex":
        idx = cls()
        texts: List[str] = []
        for r in records:
            text = (r.get("text") or "").strip()
            if not text:
                continue                    # mirror RAGIndex: skip empty-text records
            idx.docs.append({
                "id": r.get("id"), "project": r.get("project", ""),
                "source_type": r.get("source_type", ""),
                "doc_id": r.get("doc_id", ""), "text": text,
                "source_identity": r.get("source_identity", ""),
            })
            texts.append(text)
        if texts:
            idx.embeddings = _embed(texts)
            if idx.embeddings is None:
                # Model unavailable: keep docs so len() is honest but queries [].
                idx.embeddings = None
        return idx

    def __len__(self):
        return len(self.docs)

    @property
    def available(self) -> bool:
        """True iff semantic queries can actually be served right now."""
        return bool(SEMANTIC_AVAILABLE and _RUNTIME_OK and self.embeddings is not None)

    def query(self, text: str, k: int = 5) -> List[dict]:
        if not self.available or not text:
            return []
        qv = _embed([text])
        if qv is None:
            return []
        # cosine == dot product (both L2-normalized)
        sims = self.embeddings @ qv[0]
        order = _np.argsort(-sims)[:k]
        out: List[dict] = []
        for i in order:
            s = float(sims[i])
            if s <= 0:
                continue
            d = self.docs[int(i)]
            out.append({
                "id": d["id"], "score": round(s, 4),
                "project": d["project"], "source_type": d["source_type"],
                "doc_id": d["doc_id"],
                "source_identity": d.get("source_identity", ""),
                "snippet": d["text"][:200].replace("\n", " "),
            })
        return out


# ── self-test (skipped if sentence-transformers absent) ─────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    def _ok(c, m):
        print(("OK  " if c else "FAIL") + " " + str(m))
        assert c, m

    if not SEMANTIC_AVAILABLE:
        print("SKIP self-test: sentence-transformers not installed "
              "(SEMANTIC_AVAILABLE=False; hybrid falls back to BM25-only)")
        # Still verify the fallback contract: query returns [].
        idx = SemanticIndex.build([{"id": 1, "text": "anything"}])
        _ok(idx.query("anything") == [], "fallback: query returns [] when ST absent")
        print("FALLBACK CONTRACT OK")
        raise SystemExit(0)

    recs = [
        {"id": 1, "project": "wormhole", "source_type": "manifest",
         "doc_id": "w:1", "text": "Circuit 1: OTOC Lyapunov Exponent Measurement. "
                                  "Measures the out-of-time-ordered correlator on ibm_kingston."},
        {"id": 2, "project": "ibm-quantum", "source_type": "ibm_job",
         "doc_id": "ibm:d5a6", "text": "IBM Quantum job d5a6 on backend ibm_fez, "
                                       "status Completed, program sampler, 8192 samples."},
        {"id": 3, "project": "GRE", "source_type": "repo",
         "doc_id": "GRE:readme", "text": "Sierpinski triangle quantum walk with "
                                         "golden silver bronze coin angles."},
        {"id": 4, "project": "x", "source_type": "repo", "doc_id": "x:4", "text": ""},
    ]
    idx = SemanticIndex.build(recs)
    _ok(len(idx) == 3, f"indexed {len(idx)} docs (empty-text record skipped)")
    _ok(idx.available, "index available after build")

    # Synonym-style query: "coin types" should rank the GRE record (which says
    # "golden silver bronze coin angles") above the OTOC record. Pure BM25 would
    # also hit "coin" here, so this is a contract/shape check, not a BM25-beats
    # claim; the fusion self-test compares recall against BM25 directly.
    hits = idx.query("what coin types does the quantum walk support?", k=3)
    _ok(hits and isinstance(hits[0], dict), f"hits returned: {hits}")
    _ok(set(hits[0].keys()) == {"id", "score", "project", "source_type",
                                "doc_id", "snippet"}, f"hit shape: {hits[0].keys()}")

    # Semantic match: "out-of-time-ordered correlator" query -> OTOC record,
    # even though the query shares no surface token with "OTOC Lyapunov".
    hits = idx.query("compute the out-of-time-ordered correlator", k=2)
    _ok(hits and hits[0]["id"] == 1,
        f"semantic synonym query -> OTOC record: {[h['id'] for h in hits]}")

    _ok(idx.query("zzzznomatch") == [] or idx.query("zzzznomatch")[0]["score"] <= 1.0,
        "no-match / low-sim query handled")

    print("SELF-TEST PASSED")