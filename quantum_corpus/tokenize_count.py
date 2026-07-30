"""
quantum_corpus.tokenize_count
=============================
Token counting with the project's own 291-token greedy tokenizer.

Counts body tokens (no BOS/EOS) using a PREFIX-PRUNED greedy matcher that is
*exact* (identical boundaries to ``tokenizer.Tokenizer._greedy_match``) but
~100x faster: at each position it only tries vocab tokens that start with the
current character, instead of scanning all 291 tokens. Over ~15k corpus records
the naive matcher would take many minutes; this stays in seconds.

Because quantum text (``rx(pi/2)``, ``ibm_fez``, ``Lyapunov``) is out-of-vocab
for the Slovak/English-curated 291-token vocab, much of it tokenizes to
single-char + UNK pieces — exactly why the roadmap puts structured job data in
RAG (retrieval is tokenizer-insensitive) and reserves fine-tuning for prose.
The counts here quantify that honestly.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Dict, List

# repo root must be importable for `from tokenizer import ...`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_TOK = None
_PREFIX_GROUPS: Dict[str, List[str]] = {}


def get_tokenizer():
    """Lazy-load + cache the project tokenizer (tokenizer.default_tokenizer)."""
    global _TOK, _PREFIX_GROUPS
    if _TOK is None:
        from tokenizer import default_tokenizer
        _TOK = default_tokenizer()
        # Build first-char -> tokens (longest-first) for the pruned greedy match.
        groups: Dict[str, List[str]] = defaultdict(list)
        for tok in _TOK.vocab:
            if tok:
                groups[tok[0]].append(tok)
        for g in groups.values():
            g.sort(key=len, reverse=True)
        _PREFIX_GROUPS = dict(groups)
    return _TOK


def count_tokens(text: str) -> int:
    """Body-token count for ``text`` (exact greedy match, no BOS/EOS)."""
    if not text:
        return 0
    get_tokenizer()  # ensure prefix groups built
    norm = text.lower()
    groups = _PREFIX_GROUPS
    n = 0
    i = 0
    L = len(norm)
    startswith = norm.startswith
    while i < L:
        ch = norm[i]
        cands = groups.get(ch)
        matched = 0
        if cands:
            for t in cands:
                if startswith(t, i):
                    matched = len(t)
                    break
        # matched>0 -> emit one token; else UNK (char not a single-char token)
        i += matched if matched else 1
        n += 1
    return n


def count_many(records) -> int:
    """Sum body tokens across an iterable of record dicts (with 'text')."""
    return sum(count_tokens(r["text"]) for r in records)


# ── self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    def _ok(c, m):
        print(("OK  " if c else "FAIL") + " " + m)
        assert c, m

    t = get_tokenizer()
    _ok(t.vocab_size == 291, f"vocab 291, got {t.vocab_size}")

    en = "the model uses attention and expert routing for inference."
    n = count_tokens(en)
    print("  EN prose tokens:", n)
    _ok(n > 0, "english prose tokenizes to >0")

    qasm = "OPENQASM 3.0; rz(pi/2) $0; rx(pi/2) $0;"
    nq = count_tokens(qasm)
    print("  QASM tokens:", nq)
    _ok(nq > 0, "qasm tokenizes (mostly char/UNK) to >0")
    # QASM is symbol-heavy -> many UNKs -> token count >> char-efficient
    _ok(nq >= len(qasm) // 4, "qasm token count is non-trivial")

    # determinism
    _ok(count_tokens(en) == count_tokens(en), "deterministic count")

    # cached tokenizer reused
    _ok(get_tokenizer() is t, "tokenizer cached")

    print("SELF-TEST PASSED")