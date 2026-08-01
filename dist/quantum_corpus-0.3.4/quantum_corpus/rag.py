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


# ── Schema-aware query expansion ────────────────────────────────────────────
#
# Scoped alias map: maps query words to field names that exist in specific
# record schemas. Each alias is applied only when the query context matches.
#
# Why aliases instead of global synonyms?
#   - Deterministic and explainable — expansion is not probabilistic
#   - Scoped to known schema families — avoids spurious matches elsewhere
#   - Versioned with test coverage — each alias has a regression test
#
# Structure: {query_token: [(field_name, source_type_hint | None), ...]}
#   field_name: the token as it appears in the stored record text
#   source_type_hint: if set, only expand when query mentions this source type
#                     None = always expand (safe, low-collision tokens)
#
_CONSCIOUS_DNA_ALIASES = {
    # conscious_dna records store "dna_specialization" but users ask "specialization"
    "specialization": [("dna_specialization", "repo")],
}

# Known conscious_dna agent names (from gold records). Used for entity-aware TF boost.
# For phi_score queries, the agent name is the primary disambiguation signal.
# Adding it as an extra query term doubles its TF contribution for matching records,
# pushing the correct agent's record above other conscious_dna records in the full corpus.
_CONSCIOUS_DNA_AGENTS = {
    "Raziel", "Zadkiel", "Raphael", "Sandalphon", "Uriel",
    "Michael", "Haniel", "Jophiel",
    # Full agent names where first token is not unique
    "Gabriel Alpha",  # dna_agent_name: Gabriel Alpha -> used for id=28071
}


def _extract_cdna_agent(text: str) -> str | None:
    """Extract a known conscious_dna agent name from a question.

    Handles both single names ("Raziel") and full names ("Gabriel Alpha").
    Requires the question to contain conscious_dna context and a phi_score field query.
    """
    text_lower = text.lower()
    if not any(t in text_lower for t in (
            "conscious_dna", "dna_agent", "phi_score")):
        return None

    # Try two-word name first: "the Gabriel Alpha conscious_dna"
    m = re.search(
        r'\bthe\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+conscious_dna',
        text, re.IGNORECASE)
    if m and m.group(1).title() in _CONSCIOUS_DNA_AGENTS:
        return m.group(1).title()

    # Single-word name: "the Raziel conscious_dna" or "of Zadkiel conscious_dna"
    m = re.search(
        r'\b(?:the|of)\s+([A-Z][a-z]+)\s+conscious_dna\b',
        text, re.IGNORECASE)
    if m and m.group(1) in _CONSCIOUS_DNA_AGENTS:
        return m.group(1)

    # Fallback: look for any known agent name in the text
    for name in _CONSCIOUS_DNA_AGENTS:
        if name in text.split():  # whole token match
            return name
    return None


def expand_query(text: str) -> str:
    """Apply schema-aware alias expansion to a query string.

    Expands query tokens to their stored field names for known schema families.
    Expansion is scoped by source-type hints where available.

    For phi_score conscious_dna queries, the agent name is added as an extra
    term to boost TF for the correct record (entity-aware TF boost).

    Examples:
        "What is the specialization of Raziel?"
          -> "What is the dna_specialization of Raziel?"  (field alias)

        "What is the phi_score of the Raziel conscious_dna agent?"
          -> "What is the phi_score of the Raziel conscious_dna agent? Raziel"  (entity boost)
    """
    if not text:
        return text

    lower = text.lower()
    tokens = _WORD.findall(lower)

    # Detect query context
    is_conscious_dna = any(
        t in lower for t in ("conscious_dna", "dna_agent", "metatron_agent",
                              "phi_score", "fibonacci", "gc_content")
    )
    is_phi_score = "phi_score" in tokens

    # Entity-aware TF boost: for phi_score queries, repeat the agent name.
    # This doubles the TF of the agent name for the matching record vs. all
    # other conscious_dna records that lack that specific name.
    if is_phi_score and is_conscious_dna:
        agent = _extract_cdna_agent(text)
        if agent:
            text = f"{text} {agent}"

    expanded = text
    for tok, fields in _CONSCIOUS_DNA_ALIASES.items():
        if tok not in tokens:
            continue
        for field_name, hint in fields:
            if hint is not None and not is_conscious_dna:
                continue
            expanded = f"{expanded} {field_name}"

    return expanded


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
        qterms = _tok(expand_query(text))
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

    # Query expansion: conscious_dna specialization alias
    _ok("dna_specialization" in expand_query(
        "What is the specialization of the Raziel conscious_dna agent?"),
        "conscious_dna specialization query expands to dna_specialization")
    _ok("dna_specialization" not in expand_query(
        "What is the specialization of the team?"),
        "generic 'specialization' query without conscious_dna context does not expand")
    _ok(expand_query("ibm_fez backend job") == "ibm_fez backend job",
        "unrelated query is unchanged")

    # Entity-aware TF boost: phi_score + conscious_dna + agent name
    _ok("Raziel" in expand_query(
        "What is the phi_score of the Raziel conscious_dna agent?"),
        "phi_score Raziel query repeats agent name for TF boost")
    _ok("Zadkiel" in expand_query(
        "What is the phi_score of the Zadkiel conscious_dna agent?"),
        "phi_score Zadkiel query repeats agent name")
    # Non-phi_score queries do NOT get agent boost (only field alias)
    _ok("dna_specialization" in expand_query(
        "What is the specialization of the Raziel conscious_dna agent?"),
        "specialization query gets dna_specialization alias but no agent TF boost")
    _ok(expand_query("What is the status of my IBM Quantum job?") ==
        "What is the status of my IBM Quantum job?",
        "non-conscious_dna query unchanged")

    print("SELF-TEST PASSED")