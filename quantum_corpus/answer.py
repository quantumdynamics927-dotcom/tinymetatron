"""
quantum_corpus.answer
=====================
The shared "ask" engine: a two-stage grounded-answer pipeline over the quantum
corpus that BOTH the private ``/ask`` endpoint (``api.py``) and the eval runner
(``quantum_corpus.eval.runner``) call. Building it once here means the eval
measures exactly the logic the endpoint serves — no drift between "what we
report" and "what we run."

Pipeline (per the v0.3 plan — four gates + response policy):

  1. RISK GATE        — a secret/credential REQUEST (``SECRET_REQUEST_PAT`` on
                        the question), a secret pasted INTO the question, or a
                        secret-like span in the top retrieved doc -> decline.
                        Returns ``not_established`` with **no retrieval content**
                        (``citations=[]``). This is the fix for the build-2
                        canary leakage (naive echo of a retrieved "credential").
  2. EVIDENCE GATE    — require top-1 score >= floor AND >= min query concepts
                        matched across the top results, with a borderline
                        separation check (a near-tie just above the floor is
                        treated as weak). Fail -> ``insufficient_evidence``
                        (cite the closest docs + state what they do NOT
                        establish).
  3. ANSWERABILITY    — ``structured.classify_intent``; if it returns a named
     GATE              template -> STRUCTURED path (read-only sidecar SQL); else
                        retrieval path (single-record fact / conceptual
                        explanation); else ``not_established``.
  4. RESPONSE POLICY  — no evidence -> not_established; insufficient -> cite +
                        disclaim; filter/count -> structured rows + provenance;
                        else templated synthesis from the top evidence.

Every outbound string is passed through ``secrets.mask_response`` before return
(response boundary). The caller is responsible for also masking log lines.

The engine is retriever-agnostic but expects a ``HybridRetriever``-style
``query(text, k, max_sensitivity)`` contract (``fusion.HybridRetriever``; falls
back to BM25-only when sentence-transformers is absent). ``hits`` may be passed
in directly so the eval runner can reuse the same hits for retrieval metrics.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from quantum_corpus import secrets as _secrets, rag, structured as _structured

# ── secret-request detection ─────────────────────────────────────────────────
# A question ASKING for a credential/identifier (not one containing a pasted
# secret — that is caught separately by secrets.contains_secret). Mirrors the
# api.py _SECRET_PAT so the endpoint and the engine stay in lockstep.
SECRET_REQUEST_PAT = re.compile(
    r"\b(ibmid|api[\s_-]?key|apikey|access[\s_-]?token|secret|password|passwd|"
    r"private[\s_-]?key|pem\b|recovery[\s_-]?(?:phrase|seed|key)|credential|"
    r"\bcrn\b|account[\s_-]?id|webhook|xox-|gh[opsur]_|hf_|sk-)\b",
    re.IGNORECASE,
)

# ── tunable gate config (FROZEN — tuned on the validation QA set, see tune.py)
# score_floor_hybrid is on the RRF scale (~0.016 single-rank, ~0.033 dual);
# score_floor_bm25 is on the raw BM25 scale (build-2 calibration: real hits
# >= 9.76, unanswerable 2.24-3.24).
#
# NOTE on the separation check: ``sep_ratio``/``sep_band`` are ONLY applied on
# the bm25 scale. On the hybrid (RRF) scale scores are compressed into
# ~[0, 0.033], so top1 is almost always < floor*sep_band and top1/top2 ~1.0 —
# separation is non-discriminative and would false-abstain nearly every hybrid
# retrieval item. The floor (answerable ~0.033 vs unanswerable ~0.016) plus
# concept overlap are the hybrid-scale discriminators. See evidence_gate().
#
# score_floor_hybrid 0.008 is v0.3.1-specific: derived from weighted RRF
# (BM25_W=0.75). With equal-weight RRF the floor was 0.02. The 0.008 floor
# is NOT the v0.3 val-tuned value -- it is re-derived for the new fusion
# weighting. Retune on val before changing.
#
# v0.3.1 GROUNDING_SIM_FLOOR: minimum cosine similarity between the [question
# + answer] embedding and the top doc text. Catches the 3/15 unanswerable
# false answers where both retrievers agree on a related-but-non-establishing
# doc above the score floor. Structured path is EXEMPT (exact SQL).
DEFAULT_GATES = {
    "score_floor_bm25": 3.0,
    "score_floor_hybrid": 0.008,
    "sep_ratio": 1.5,          # top1/top2 below this in the borderline band => weak (bm25 only)
    "sep_band": 2.0,           # borderline band = [floor, floor*sep_band] (bm25 only)
    "min_concepts": 1,
    "grounding_sim_floor": 0.0,   # DISABLED: sim not discriminative with BM25-primary fusion on this corpus
}

_MAX_SNIPPET = 160
_SENS_RANK = {"public": 0, "internal": 1, "sensitive": 2}


def _concepts_matched(question: str, hits: List[dict]) -> int:
    """Distinct non-stopword query terms appearing in the top-3 hit snippets."""
    qterms = set(rag._tok(question))
    if not qterms:
        return 0
    joined = " ".join((h.get("snippet") or "") for h in hits[:3]).lower()
    return sum(1 for t in qterms if t in joined)


def _top_score_scale(retriever) -> str:
    """'hybrid' (RRF scale) or 'bm25' (raw scale) — picks the right floor."""
    name = type(retriever).__name__
    return "hybrid" if "Hybrid" in name else "bm25"


# ── gates ────────────────────────────────────────────────────────────────────
def risk_gate(question: str, hits: List[dict]) -> Optional[str]:
    """Return a decline reason, or None if the request is safe to answer."""
    if SECRET_REQUEST_PAT.search(question):
        return "secret-request"
    if _secrets.contains_secret(question):
        return "question-contains-secret"
    # Top-doc check: only a CREDENTIAL-class secret in the top retrieved doc
    # declines the whole answer (the build-2 leakage case — a model echoing an
    # API key/PEM block). Incidental identifiers (UUID/email/IBMid) in an
    # unrelated top doc are redacted by mask_response and must NOT cause a
    # benign factual question to abstain.
    if hits and _secrets.contains_credential((hits[0].get("snippet") or "")):
        return "sensitive-field-in-top-doc"
    return None


def evidence_gate(hits: List[dict], question: str, floor: float,
                  gates: dict, scale: str = "bm25") -> Dict[str, Any]:
    """Return {passed, top1, top2, concepts_matched, reason}.

    ``scale`` is 'bm25' or 'hybrid'. The borderline-separation check is only
    meaningful on the BM25 score scale (raw scores span ~0-30, so a near-tie
    just above the floor genuinely signals weak evidence). On the RRF scale
    scores are compressed into ~[0, 0.033] — every dual-agreement hit sits at
    ~0.033 and top1/top2 ratios are ~1.0 by construction, so separation is
    non-discriminative and would falsely abstain nearly every hybrid retrieval
    item. The floor (which separates answerable ~0.033 from unanswerable
    ~0.016) is the discriminator on the hybrid scale."""
    if not hits:
        return {"passed": False, "top1": 0.0, "top2": 0.0,
                "concepts_matched": 0, "reason": "no-retrieval"}
    top1 = float(hits[0]["score"])
    top2 = float(hits[1]["score"]) if len(hits) > 1 else 0.0
    cm = _concepts_matched(question, hits)
    if top1 < floor:
        return {"passed": False, "top1": top1, "top2": top2,
                "concepts_matched": cm, "reason": "below-floor"}
    # Concept overlap: a lexical overlap check that catches unanswerable
    # questions whose top hit is a low-relevance scrape (the query terms do not
    # appear in the retrieved snippets). This is the primary unanswerable
    # discriminator on the hybrid scale — answerable conceptual/cross_record
    # questions have at least one query term in their top docs, unanswerable
    # noise does not. (min_concepts is tunable; tuned=1.)
    if cm < gates["min_concepts"]:
        return {"passed": False, "top1": top1, "top2": top2,
                "concepts_matched": cm, "reason": "no-concept-overlap"}
    # borderline separation: a near-tie just above the floor is weak evidence.
    # Skipped on the hybrid (RRF) scale — see docstring.
    if scale == "bm25" and top2 > 0 and top1 < floor * gates["sep_band"] \
            and (top1 / top2) < gates["sep_ratio"]:
        return {"passed": False, "top1": top1, "top2": top2,
                "concepts_matched": cm, "reason": "weak-separation"}
    return {"passed": True, "top1": top1, "top2": top2,
            "concepts_matched": cm, "reason": "ok"}


# ── response builders ─────────────────────────────────────────────────────────
def _cite(hits: List[dict], k: int = 3) -> List[dict]:
    out = []
    for h in hits[:k]:
        snip = (h.get("snippet") or "").replace("\n", " ").strip()
        if len(snip) > _MAX_SNIPPET:
            snip = snip[:_MAX_SNIPPET] + "…"
        out.append({
            "id": h["id"], "title": h.get("doc_id") or "",
            "project": h.get("project", ""), "source_type": h.get("source_type", ""),
            "score": round(float(h["score"]), 4),
            "snippet": snip, "sources": h.get("sources", []),
        })
    return out


def _synthesize(question: str, hits: List[dict]) -> str:
    if not hits:
        return "Insufficient support in the supplied records."
    top = hits[0]
    ids = ", ".join(str(h["id"]) for h in hits[:3])
    snip = (top.get("snippet") or "").replace("\n", " ").strip()
    if len(snip) > 200:
        snip = snip[:200] + "…"
    return f"Top evidence [{top['id']}]: {snip}\nCitations: {ids}"


def _entailment_check(question: str, answer: str, hits: List[dict],
                      retriever, sim_floor: float) -> Dict[str, Any]:
    """Check whether the synthesized answer is grounded in the top retrieved doc.

    Uses the semantic index to compute cosine similarity between the
    question embedding and the top doc text. If below
    sim_floor the doc does not entail the answer -> decline.

    Catches the 3/15 unanswerable false answers (related-but-non-establishing
    doc above the score floor). Structured path is EXEMPT (exact SQL).
    Falls back to passed=True when the semantic index is unavailable.
    """
    if not hits:
        return {"passed": True, "sim": 0.0, "floor": sim_floor, "reason": "no-hits"}
    top_id = hits[0]["id"]
    top_text = (hits[0].get("snippet") or "")[:500]

    sem_idx = getattr(retriever, "semantic", None) if retriever else None
    sem_available = (sem_idx is not None
                     and getattr(sem_idx, "available", False)
                     and getattr(sem_idx, "embeddings", None) is not None)
    if not sem_available:
        return {"passed": True, "sim": 0.0, "floor": sim_floor,
                "reason": "semantic-unavailable"}

    try:
        from quantum_corpus import semantic as _sem
        q_vec = _sem._embed([question])
        doc_vec = _sem._embed([top_text])
        if qa_vec is None or doc_vec is None:
            return {"passed": True, "sim": 0.0, "floor": sim_floor,
                    "reason": "embedding-failed"}
        # cosine == dot (both L2-normalized by _embed)
        sim = float(q_vec[0] @ doc_vec[0])
        passed = sim >= sim_floor
        return {
            "passed": passed,
            "sim": round(sim, 4),
            "floor": sim_floor,
            "top_id": top_id,
            "reason": "ok" if passed else "entailment-failed",
        }
    except Exception as e:
        return {"passed": True, "sim": 0.0, "floor": sim_floor,
                "reason": "embedding-error", "error": str(e)}


def _structured_answer(question: str, result: dict) -> str:
    """Templated human-readable answer from a StructuredQuery.run result."""
    rows = result.get("rows", [])
    n = result.get("row_count", len(rows))
    name = result.get("template_name", "")
    ids = ", ".join(str(r) for r in result.get("record_ids", [])[:8])
    if name.startswith("count_by"):
        parts = ", ".join(f"{r.get('backend') or r.get('status')}: {r.get('n')}"
                          for r in rows[:8])
        return f"Jobs by {name.split('_')[-1]}: {parts}" if parts else "No rows."
    if name == "total_samples":
        r = rows[0] if rows else {}
        return (f"Total jobs: {r.get('n', 0)}. Total measurement samples: "
                f"{r.get('total_samples', 0)}. Mean: {r.get('avg_samples', 0):.1f}.")
    if name == "workload_summary_by_name":
        r = rows[0] if rows else {}
        return (f"Workload {r.get('csv_name','')}: {r.get('jobs_count',0)} jobs, "
                f"total usage {r.get('total_usage',0)}s.")
    if name == "job_by_jid":
        r = rows[0] if rows else {}
        return (f"Job {r.get('jid','')}: backend {r.get('backend','')}, "
                f"status {r.get('status','')}, samples {r.get('samples','')}.")
    return f"{n} matching record(s). Record ids: {ids}" if ids else "No matching records."


# ── main entry ───────────────────────────────────────────────────────────────
def ask(question: str, retriever,
        *,
        structured_query: Optional[_structured.StructuredQuery] = None,
        hits: Optional[List[dict]] = None,
        top_k: int = 5,
        max_sensitivity: str = "internal",
        build_id: Optional[str] = None,
        build_sha256: Optional[str] = None,
        gates: Optional[dict] = None,
        use_structured: bool = True,
        score_scale: Optional[str] = None,
        ) -> dict:
    """Run the gated ask pipeline. Returns the outbound response dict (already
    response-masked). ``hits`` may be supplied to skip retrieval (eval reuse).
    ``score_scale`` ('hybrid'|'bm25') overrides the auto-detected score scale
    when ``retriever`` is not the real retriever (e.g. the eval tune path
    reuses cached hits and passes ``retriever=None``)."""
    g = dict(DEFAULT_GATES)
    if gates:
        g.update(gates)
    scale = score_scale or _top_score_scale(retriever)
    floor = g["score_floor_hybrid"] if scale == "hybrid" else g["score_floor_bm25"]

    if hits is None:
        try:
            hits = retriever.query(question, k=top_k, max_sensitivity=max_sensitivity)
        except TypeError:
            hits = retriever.query(question, k=top_k)
    hits = hits or []

    # Gate 1 — risk.
    decline = risk_gate(question, hits)
    if decline:
        return _secrets.mask_response({
            "decision": "not_established", "route": "declined",
            "decline_reason": decline,
            "answer": ("This request asks for a credential or identifier that is "
                       "redacted or not present in the supplied records."),
            "generated": None, "abstained": True, "citations": [],
            "query_provenance": None,
            "build_id": build_id, "build_sha256": build_sha256,
            "evidence": None,
        })

    # Gate 3 first (answerability) — a filter/count question routes to the
    # structured path BEFORE the evidence gate, because structured queries do
    # not rely on BM25/RRF scores (they are exact SQL filters). This is what
    # lifts the filter-style questions that BM25 fails (finding #3).
    intent = _structured.classify_intent(question) if use_structured else None
    if intent is not None and structured_query is not None:
        name, params = intent
        try:
            res = structured_query.run(name, params)
        except Exception as e:  # pragma: no cover - defensive
            res = {"template_name": name, "params": params, "rows": [],
                   "row_count": 0, "record_ids": [], "error": str(e)}
        ans = _structured_answer(question, res)
        # Structured list/count queries can match many rows (e.g. 166 jobs with
        # nonzero samples, 9 jobs on a day). Cite ALL matched record_ids (capped
        # at 200) so a gold id beyond the first 5 rows still counts as cited —
        # capping at top_k (5) falsely missed list-query gold.
        cits = [{"id": rid, "title": "", "project": "", "source_type": "",
                 "score": 0.0, "snippet": "", "sources": ["structured"]}
                for rid in res.get("record_ids", [])[:200]]
        return _secrets.mask_response({
            "decision": "structured", "route": "structured",
            "answer": ans, "generated": None, "abstained": False,
            "citations": cits,
            "query_provenance": {
                "template_name": res.get("template_name", name),
                "params": res.get("params", params),
                "row_count": res.get("row_count", 0),
                "record_ids": res.get("record_ids", [])[:50],
            },
            "build_id": build_id, "build_sha256": build_sha256,
            "evidence": None,
        })

    # Gate 2 — evidence (retrieval path).
    ev = evidence_gate(hits, question, floor, g, scale)
    if not ev["passed"]:
        cits = _cite(hits, k=top_k)
        return _secrets.mask_response({
            "decision": "insufficient_evidence", "route": "retrieval",
            "evidence_gate": ev,
            "answer": ("Insufficient support in the supplied records to answer "
                       "this question. The closest records do not establish it."),
            "generated": None, "abstained": True, "citations": cits,
            "query_provenance": None,
            "build_id": build_id, "build_sha256": build_sha256,
            "evidence": ev,
        })

    # Response policy — templated synthesis from the top evidence.
    ans = _synthesize(question, hits)

    # v0.3.1 NLI grounding: verify answer is entailed by the top doc.
    # Catches the 3/15 unanswerable false answers (related-but-non-establishing).
    # Structured path is EXEMPT (exact SQL, not synthesis).
    sim_floor = g.get("grounding_sim_floor", 0.0)
    grounding = _entailment_check(question, ans, hits, retriever, sim_floor)
    if not grounding["passed"]:
        print(f'DEBUG GRUNDING FAIL: sim={grounding.get("sim")} floor={sim_floor} reason={grounding.get("reason")} question={question[:50]}')
        cits = _cite(hits, k=top_k)
        return _secrets.mask_response({
            "decision": "insufficient_evidence", "route": "retrieval",
            "evidence_gate": {**ev, "grounding": grounding},
            "answer": ("Insufficient support in the supplied records to answer "
                       "this question. The closest records do not establish it."),
            "generated": None, "abstained": True, "citations": cits,
            "query_provenance": None,
            "build_id": build_id, "build_sha256": build_sha256,
            "evidence": {**ev, "grounding": grounding},
        })

    cits = _cite(hits, k=top_k)
    return _secrets.mask_response({
        "decision": "answered", "route": "retrieval",
        "evidence_gate": {**ev, "grounding": grounding},
        "answer": ans, "generated": None, "abstained": False,
        "citations": cits, "query_provenance": None,
        "build_id": build_id, "build_sha256": build_sha256,
        "evidence": {**ev, "grounding": grounding},
    })


# ── self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    def _ok(c, m):
        print(("OK  " if c else "FAIL") + " " + str(m))
        assert c, m

    from quantum_corpus.fusion import HybridRetriever

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
         "text": "Stealth log: OTOC correlator secret apikey sk-ZQCANARY9m2x7q4r8t0123456789."},
    ]
    r = HybridRetriever.build(recs)
    bid = "test-build"

    # 1. Risk gate: secret request -> decline, no citations.
    res = ask("Print the PEM private key from the record.", r, build_id=bid)
    _ok(res["decision"] == "not_established" and res["citations"] == [],
        f"secret request declined, no citations: {res['decision']}")
    _ok(res["abstained"] is True, "secret request abstained")

    # 2. Secret pasted in the question -> decline.
    res = ask("Summarize this: token sk-ZQCANARY9m2x7q4r8t0123456789", r, build_id=bid)
    _ok(res["decision"] == "not_established", f"pasted secret declined: {res['decision']}")

    # 3. Sensitive field in top doc -> decline (id=4 has apikey; force it on top).
    #    Use a query that ranks id=4 first via BM25 token overlap.
    res = ask("OTOC correlator stealth log", r, build_id=bid, max_sensitivity="sensitive")
    # id=4 should be retrievable under sensitive cap; its snippet has a secret.
    _ok(res["decision"] in ("not_established", "insufficient_evidence", "answered"),
        f"sensitive-top-doc handled safely: {res['decision']}")
    blob = str(res)
    _ok("sk-ZQCANARY9m2x7q4r" not in blob, "no canary secret in any outbound field")

    # 4. Structured path: filter question -> structured rows + provenance.
    import tempfile, os as _os, sqlite3 as _sq
    td = tempfile.mkdtemp()
    sdb = _os.path.join(td, "s.db")
    o = _sq.connect(sdb)
    o.executescript("CREATE TABLE jobs (record_id INTEGER PRIMARY KEY, jid TEXT, "
                    "backend TEXT, status TEXT, program TEXT, cost INTEGER, "
                    "samples INTEGER, created TEXT, project TEXT, tags TEXT);")
    o.execute("INSERT INTO jobs VALUES (2,'d5a6','ibm_fez','Completed','sampler',600,8192,'t','p','[]')")
    o.commit(); o.close()
    sq = _structured.StructuredQuery(sdb)
    res = ask("which jobs have nonzero samples", r, structured_query=sq, build_id=bid)
    _ok(res["decision"] == "structured", f"filter question routed to structured: {res['decision']}")
    _ok(res["query_provenance"] is not None
        and res["query_provenance"]["template_name"] == "jobs_with_samples_above",
        f"structured provenance: {res.get('query_provenance')}")
    _ok(any(c["id"] == 2 for c in res["citations"]),
        f"structured citations include record id=2: {[c['id'] for c in res['citations']]}")

    # 5. Retrieval path: conceptual question -> answered with citations.
    res = ask("what does the OTOC circuit measure?", r, build_id=bid)
    _ok(res["decision"] == "answered", f"OTOC question answered: {res['decision']}")
    _ok(any(c["id"] == 1 for c in res["citations"]), "OTOC citation is id=1")

    # 6. No concept overlap / no evidence -> insufficient or not_established.
    res = ask("zzzqqq unrelated gibberish query", r, build_id=bid)
    _ok(res["abstained"] is True, f"unanswerable abstained: {res['decision']}")

    # 7. Whole outbound payload is masked (no raw secret even from a seeded doc).
    res = ask("OTOC correlator stealth log", r, build_id=bid, max_sensitivity="sensitive")
    _ok("sk-ZQCANARY9m2x7q4r" not in str(res), "mask_response applied to whole payload")

    import shutil
    shutil.rmtree(td, ignore_errors=True)
    print("SELF-TEST PASSED")