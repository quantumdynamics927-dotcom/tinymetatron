"""
tokenizer.py
============
Deterministic greedy longest-match tokenizer for the TinyMetatron SLM.

Vocabulary layout (total 291 ids, per IMPLEMENTATION_CONTRACT.md §2 and
config.py):
    0..4    : special tokens  (pad, bos, eos, unk, sep)
    5..68   : 64 single-char slots  (a-z, most SK diacritics, digits, punctuation)
    69..290 : curated SK+EN technical word/piece tokens — plus three extra
              single-char Slovak diacritics (ď ĺ ŕ) repurposed from the tail
              of the curated range (ids 288, 289, 290) so that EVERY Slovak
              diacritic has a dedicated single-char slot.

Encoding normalises to lowercase, then greedily matches the longest in-vocab
token at each position, falling back to UNK for any character not in the
vocab. Because every Slovak diacritic (including ď ĺ ŕ) and every common
punctuation character has a dedicated single-char slot, in-vocab-character
text round-trips losslessly through encode->decode even when a whole word is
not a curated token.

The tokenizer is a pure-Python component (patent component: deterministic
"Metatron" symbol layer feeding the polyhedral attention / MoE core).
"""

from __future__ import annotations

import json
from typing import Dict, List

from config import CONFIG, get_config


# ── Special-token names mapped to their CONFIG keys ─────────────────────────
_SPECIAL_KEYS = ("pad_id", "bos_id", "eos_id", "unk_id", "sep_id")


class Tokenizer:
    """Greedy longest-match BPE-free tokenizer over a fixed 291-token vocab.

    Public interface (contract §2):
        encode(text)   -> list[int]   # [BOS] + ids + [EOS]
        decode(ids)    -> str         # drop specials, join pieces
        vocab_size     -> int         # 291
        special_ids    -> dict        # {pad,bos,eos,unk,sep: id}
        Tokenizer.from_file(path) -> Tokenizer
    """

    def __init__(self, vocab: Dict[str, int]):
        # Defensive copy so callers cannot mutate the loaded mapping.
        self.vocab: Dict[str, int] = dict(vocab)
        # Inverse map id -> token string (last writer wins; ids are unique).
        self._inv: Dict[int, str] = {v: k for k, v in self.vocab.items()}

        # Special ids derived from CONFIG (never hardcoded).
        c = get_config()
        self.special_ids: Dict[str, int] = {name: c[name] for name in _SPECIAL_KEYS}
        self._bos_id = c["bos_id"]
        self._eos_id = c["eos_id"]
        self._unk_id = c["unk_id"]
        self._special_set = set(self.special_ids.values())

        # Greedy match requires scanning candidate tokens longest-first.
        # Pre-compute once; this is the hot path for encode().
        self._sorted_tokens: List[str] = sorted(self.vocab.keys(), key=len, reverse=True)

    # ── Properties ──────────────────────────────────────────────────────────
    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary (must be 291)."""
        return len(self.vocab)

    # ── Normalisation ───────────────────────────────────────────────────────
    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase. Deterministic, locale-independent."""
        return text.lower()

    # ── Core greedy longest-match encoder ────────────────────────────────────
    def _greedy_match(self, text: str) -> List[int]:
        """Return ids for `text` (already normalised) with UNK fallback."""
        ids: List[int] = []
        i = 0
        n = len(text)
        vocab = self.vocab
        sorted_tokens = self._sorted_tokens
        while i < n:
            matched_id = -1
            matched_len = 0
            # The candidate list is longest-first, so the first hit is the
            # longest match at position i — the greedy invariant.
            for tok in sorted_tokens:
                if tok and text.startswith(tok, i):
                    matched_id = vocab[tok]
                    matched_len = len(tok)
                    break
            if matched_len > 0:
                ids.append(matched_id)
                i += matched_len
            else:
                # No single-char slot for this codepoint -> UNK.
                ids.append(self._unk_id)
                i += 1
        return ids

    # ── Public API ───────────────────────────────────────────────────────────
    def encode(self, text: str) -> List[int]:
        """Tokenise `text` -> [BOS] + piece_ids + [EOS]."""
        normalised = self._normalize(text)
        body = self._greedy_match(normalised)
        return [self._bos_id] + body + [self._eos_id]

    def decode(self, ids: List[int]) -> str:
        """Inverse of encode(): drop specials, join the remaining pieces.

        Spacing is handled implicitly because the space character is itself a
        single-char token in the vocab; a plain concatenation reproduces the
        original spacing of the normalised source text.
        """
        inv = self._inv
        specials = self._special_set
        pieces: List[str] = []
        for i in ids:
            if i in specials:
                continue
            piece = inv.get(i)
            if piece is not None:
                pieces.append(piece)
        return "".join(pieces)

    @classmethod
    def from_file(cls, path: str) -> "Tokenizer":
        """Load a vocab.json (token string -> int id) and build a Tokenizer."""
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        # Sanity: ensure ids are ints (JSON keys are always strings).
        vocab = {str(k): int(v) for k, v in vocab.items()}
        return cls(vocab)


# ── Default loader (uses CONFIG["vocab_path"] when present) ─────────────────
def default_tokenizer() -> Tokenizer:
    """Build a Tokenizer from CONFIG["vocab_path"] relative to repo root."""
    c = get_config()
    return Tokenizer.from_file(c["vocab_path"])


# ── Self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 fix (rule 4)

    tok = default_tokenizer()
    c = get_config()
    assert tok.vocab_size == c["vocab_size"], (
        f"vocab_size {tok.vocab_size} != CONFIG vocab_size {c['vocab_size']}"
    )
    assert tok.vocab_size == 291, f"expected 291 tokens, got {tok.vocab_size}"

    # Slovak technical sentence — every char is in-vocab -> lossless.
    sk = "šifrovanie a autentifikácia hesla znižujú zraniteľnosť proti útoku phishing."
    enc_sk = tok.encode(sk)
    dec_sk = tok.decode(enc_sk)
    assert dec_sk == sk.lower(), (
        f"SK round-trip failed:\n  in : {sk.lower()}\n  out: {dec_sk}"
    )

    # Every Slovak diacritic — including the previously-missing ď ĺ ŕ — must
    # round-trip losslessly (no UNK) since each has a dedicated single-char slot.
    diac = "ďalší ĺudia ŕaden"
    enc_d = tok.encode(diac)
    dec_d = tok.decode(enc_d)
    assert tok.special_ids["unk_id"] not in enc_d, "ď/ĺ/ŕ must be in-vocab"
    assert dec_d == diac.lower(), (
        f"diacritic round-trip failed:\n  in : {diac.lower()}\n  out: {dec_d}"
    )

    # English technical sentence — lossless round-trip.
    en = "the model uses attention and expert routing for inference."
    enc_en = tok.encode(en)
    dec_en = tok.decode(enc_en)
    assert dec_en == en.lower(), (
        f"EN round-trip failed:\n  in : {en.lower()}\n  out: {dec_en}"
    )

    # BOS / EOS wrapping.
    assert enc_sk[0] == tok.special_ids["bos_id"], "BOS missing at front"
    assert enc_sk[-1] == tok.special_ids["eos_id"], "EOS missing at end"

    # UNK fallback: a character outside the char slots (e.g. emoji) -> UNK id.
    weird = tok.encode("xé")  # x, é both in-vocab -> no UNK expected
    assert tok.special_ids["unk_id"] not in weird, "é/x should be in-vocab"
    unk_test = tok.encode("")  # BEL control char -> UNK
    assert unk_test[1] == tok.special_ids["unk_id"], "UNK fallback failed"

    # from_file round-trip on a temp copy
    import json as _json, os as _os, tempfile as _tf
    with _tf.NamedTemporaryFile("w", suffix=".json", delete=False,
                                encoding="utf-8") as tf:
        _json.dump(tok.vocab, tf, ensure_ascii=False)
        tmp_path = tf.name
    try:
        tok2 = Tokenizer.from_file(tmp_path)
        assert tok2.vocab_size == tok.vocab_size
        assert tok2.encode(sk) == enc_sk
    finally:
        _os.unlink(tmp_path)

    print("tokenizer self-test OK")
    print(f"vocab_size         : {tok.vocab_size}")
    print(f"special_ids        : {tok.special_ids}")
    print(f"SK round-trip      : {dec_sk}")
    print(f"EN round-trip      : {dec_en}")
    print(f"SK encoded length  : {len(enc_sk)}")