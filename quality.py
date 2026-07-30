"""
quality.py
==========
Heuristic text-quality scorer for the TinyMetatron SLM data pipeline.

Public interface (per IMPLEMENTATION_CONTRACT.md section 2):
    score_quality(text: str) -> float   # in [0.0, 1.0]

The scorer is a deterministic, pure-Python blend of four weak signals:

1. Length adequacy      -- too-short and too-long texts are penalized;
                           a moderate length (a few dozen words) scores best.
2. Repetition penalty   -- repeated unigrams / bigrams / trigrams lower the
                           score (a proxy for "aaaa ..." or looped output).
3. Keyword density      -- technical-domain keywords (cybersecurity /
                           software engineering) give a small boost, so a
                           coherent technical sentence outscores gibberish.
4. Language plausibility -- vowel/consonant balance, mean word length and
                           lexical diversity act as a cheap "looks like real
                           language" filter.

No external dependencies. No randomness. The result is clamped to [0, 1].

References (patent context): the Metatron architecture routes tokens through
13 polyhedral experts; this scorer filters which texts are worth learning from
before they ever reach a metatron_moe / sparse-attention forward pass.
"""

from __future__ import annotations

from config import CONFIG


# ── Tunable constants (read from CONFIG where a natural analogue exists) ─────
# Length band: texts with a word count inside [_LEN_LO, _LEN_HI] get the full
# length score; outside this band the score decays linearly toward 0.
_LEN_LO = 6          # words: below this is "too short"
_LEN_HI = 60         # words: above this is "too long"
_LEN_FLOOR = 1       # words: below this -> length score 0 (empty / 1 token)

# Keyword sets (lowercased). Technical-domain vocabulary that, when present
# with reasonable density, indicates a useful training text.
_TECH_KEYWORDS = {
    # cybersecurity
    "firewall", "encryption", "cipher", "auth", "authentication", "authorization",
    "vulnerability", "exploit", "malware", "ransomware", "phishing", "botnet",
    "zero-day", "patch", "hardening", "intrusion", "ids", "ips", "siem",
    "pki", "tls", "ssl", "cert", "certificate", "hash", "salt", "nonce",
    "threat", "attack", "defender", "hacker", "red-team", "blue-team",
    "forensics", "anomaly", "protocol", "credential", "token", "oauth",
    "sso", "mfa", "2fa", "zero-trust", "segmentation", "payload",
    # software engineering
    "function", "method", "class", "module", "library", "framework", "api",
    "compiler", "runtime", "memory", "cache", "buffer", "queue", "thread",
    "process", "concurrency", "async", "await", "promise", "callback",
    "iterator", "generator", "recursion", "algorithm", "data-structure",
    "binary", "tree", "graph", "node", "pointer", "reference", "scope",
    "namespace", "polymorphism", "inheritance", "encapsulation", "abstraction",
    "refactor", "debug", "regression", "unit-test", "coverage", "ci", "cd",
    "pipeline", "deploy", "container", "docker", "kubernetes", "microservice",
    "schema", "migration", "orm", "query", "index", "transaction", "acid",
    "gradient", "tensor", "layer", "attention", "transformer", "embedding",
    "tokenizer", "logits", "softmax", "dropout", "optimizer", "loss",
}


def _word_list(text: str) -> list[str]:
    """Split text into a normalized lowercased word list (no punctuation)."""
    out: list[str] = []
    cur: list[str] = []
    for ch in text:
        if ch.isalnum():
            cur.append(ch.lower())
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def _length_score(n_words: int) -> float:
    """Score the word-count adequacy in [0, 1]."""
    if n_words <= _LEN_FLOOR:
        return 0.0
    if n_words < _LEN_LO:
        # linear ramp from _LEN_FLOOR -> _LEN_LO
        return (n_words - _LEN_FLOOR) / max(1, (_LEN_LO - _LEN_FLOOR))
    if n_words <= _LEN_HI:
        return 1.0
    # too long: linear decay from _LEN_HI to 2*_LEN_HI (0 beyond that)
    over = n_words - _LEN_HI
    span = max(1, _LEN_HI)
    return max(0.0, 1.0 - over / span)


def _repetition_score(words: list[str]) -> float:
    """
    Penalize repeated n-grams (n=1,2,3). Returns a score in [0, 1] where 1
    means no repetition and lower values indicate heavy repetition.
    """
    n = len(words)
    if n < 2:
        # too short to meaningfully measure repetition; treat as neutral
        return 1.0

    # unique-token ratio (type/token ratio) over unigrams
    unique_ratio = len(set(words)) / n

    # bigram repetition: fraction of bigrams that occur more than once
    bigrams = [tuple(words[i:i + 2]) for i in range(n - 1)]
    if bigrams:
        seen: dict[tuple, int] = {}
        for bg in bigrams:
            seen[bg] = seen.get(bg, 0) + 1
        repeated_bg = sum(1 for c in seen.values() if c > 1)
        bg_rep = repeated_bg / len(seen)
    else:
        bg_rep = 0.0

    # trigram repetition
    trigrams = [tuple(words[i:i + 3]) for i in range(n - 2)]
    if trigrams:
        seen3: dict[tuple, int] = {}
        for tg in trigrams:
            seen3[tg] = seen3.get(tg, 0) + 1
        repeated_tg = sum(1 for c in seen3.values() if c > 1)
        tg_rep = repeated_tg / len(seen3)
    else:
        tg_rep = 0.0

    # Combine: unigram diversity dominates, n-gram repetition adds penalty.
    # Clamp each component to [0, 1].
    score = unique_ratio * 0.5 + (1.0 - bg_rep) * 0.3 + (1.0 - tg_rep) * 0.2
    return max(0.0, min(1.0, score))


def _keyword_score(words: list[str]) -> float:
    """
    Technical-domain keyword density. Returns a small boost score in [0, 1]:
    a couple of technical keywords in a reasonable-length text give a high
    score; keyword stuffing (very high density) saturates and does not over-
    reward.
    """
    n = len(words)
    if n == 0:
        return 0.0
    hits = sum(1 for w in words if w in _TECH_KEYWORDS)
    density = hits / n
    # Saturation curve: y = 1 - exp(-k*density). A density of ~5% (1 keyword
    # per 20 words) already yields a strong boost; higher density asymptotes
    # to 1 without runaway reward.
    import math
    score = 1.0 - math.exp(-20.0 * density)
    return score


def _plausibility_score(words: list[str], text: str) -> float:
    """
    Cheap "looks like real language" check: vowel/consonant balance, mean
    word length, and lexical diversity. Returns a score in [0, 1].
    """
    if not words:
        return 0.0

    # vowel ratio over alphabetic characters
    alpha_chars = [c for c in text.lower() if c.isalpha()]
    if alpha_chars:
        vowels = sum(1 for c in alpha_chars if c in "aeiou")
        vowel_ratio = vowels / len(alpha_chars)
        # natural languages: ~0.35-0.55 vowel ratio. Penalize extremes.
        if vowel_ratio < 0.2 or vowel_ratio > 0.7:
            vowel_score = 0.4
        else:
            vowel_score = 1.0
    else:
        vowel_score = 0.0

    # mean word length: natural words average ~3-8 chars
    mean_len = sum(len(w) for w in words) / len(words)
    if 3.0 <= mean_len <= 8.0:
        len_score = 1.0
    elif 2.0 <= mean_len < 3.0 or 8.0 < mean_len <= 12.0:
        len_score = 0.6
    else:
        len_score = 0.2

    # lexical diversity (type/token ratio), smoothed
    ttr = len(set(words)) / len(words)
    # blend
    return max(0.0, min(1.0, 0.4 * vowel_score + 0.3 * len_score + 0.3 * ttr))


def score_quality(text: str) -> float:
    """
    Score a text's quality for inclusion in the TinyMetatron training set.

    Returns a float in [0.0, 1.0]. The score is a weighted blend of length
    adequacy, repetition penalty, technical-keyword density and basic
    language plausibility. Deterministic and pure-Python.

    Parameters
    ----------
    text : str
        Raw input text. Empty / whitespace-only input scores 0.0.

    Returns
    -------
    float
        Quality score clamped to [0.0, 1.0].
    """
    if not isinstance(text, str):
        return 0.0
    if not text.strip():
        return 0.0

    words = _word_list(text)
    n_words = len(words)
    if n_words == 0:
        return 0.0

    length = _length_score(n_words)
    repetition = _repetition_score(words)
    keyword = _keyword_score(words)
    plausibility = _plausibility_score(words, text)

    # Weighted blend. Length is a gating signal (a 1-word text can't be good
    # even if it has a keyword); plausibility and repetition are core quality;
    # keyword density is a bonus that lets technical prose beat gibberish of
    # equal length.
    score = (
        0.30 * length
        + 0.30 * plausibility
        + 0.25 * repetition
        + 0.15 * keyword
    )

    # Clamp to [0, 1] and guard against float rounding artifacts.
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


# ── Self-test / demo ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    samples = [
        # (label, text)
        ("empty", ""),
        ("short_gibberish", "xqw zzz"),
        ("gibberish_long", "qwz zxq qzz xzq zqx qzx xqq zzz qzq zxq "
                           "qqz xzz zqx qxz zxx qqq zzz zxq qxz xzz"),
        ("good_tech_cyber", "The firewall enforces a zero-trust policy with "
                            "mfa, tls encryption and anomaly detection to "
                            "block phishing and ransomware payloads."),
        ("good_tech_sw", "The transformer model uses attention, embeddings "
                         "and a softmax over logits with dropout and an "
                         "optimizer minimizing the loss across layers."),
        ("plain_general", "The quick brown fox jumps over the lazy dog "
                          "near the river bank every morning."),
        ("repeated", "the the the the the the the the the the the the the "
                     "the the the the the the the the the the the the the"),
        ("keyword_stuff", "firewall firewall firewall encryption encryption "
                          "tls tls tls tls malware malware"),
    ]

    print("quality.py self-test")
    print("-" * 64)
    for label, text in samples:
        s = score_quality(text)
        # sanity: result must be a float in [0, 1]
        assert isinstance(s, float), f"{label}: non-float score {s!r}"
        assert 0.0 <= s <= 1.0, f"{label}: score {s} out of [0,1]"
        print(f"  {label:18s}: {s:.4f}   ({len(text):4d} chars)")

    # Contract assertion: a good technical sentence must outscore short
    # gibberish.
    good_tech = score_quality(samples[3][1])
    short_gib = score_quality(samples[1][1])
    assert good_tech > short_gib, (
        f"contract violated: good tech ({good_tech}) <= short gibberish "
        f"({short_gib})")

    # Also require the long gibberish to be below the good technical text.
    long_gib = score_quality(samples[2][1])
    assert good_tech > long_gib, (
        f"good tech ({good_tech}) <= long gibberish ({long_gib})")

    # Empty must be 0.0
    assert score_quality("") == 0.0

    print("-" * 64)
    print(f"good_tech     = {good_tech:.4f}")
    print(f"short_gib     = {short_gib:.4f}")
    print(f"long_gib      = {long_gib:.4f}")
    print("ALL ASSERTIONS PASSED")