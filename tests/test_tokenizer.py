"""
tests/test_tokenizer.py
=======================
ECOSYSTEM pytest suite for the TinyMetatron tokenizer (contract section 2).

Covers (contract section 5 / test descriptions):
    * vocab_size == 291
    * encode/decode round-trip lossless for a Slovak (SK) and an English (EN)
      technical sentence whose characters are all in-vocab
    * BOS / EOS are present in the encoded sequence
    * UNK fallback for an out-of-vocab character
"""

from __future__ import annotations

import os
import sys

# Ensure repo root is importable when pytest runs from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

from config import CONFIG
from tokenizer import Tokenizer, default_tokenizer


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    """Build the default tokenizer from CONFIG['vocab_path']."""
    return default_tokenizer()


# ── vocab_size == 291 ─────────────────────────────────────────────────────────
def test_vocab_size_is_291(tok: Tokenizer) -> None:
    assert tok.vocab_size == 291
    assert tok.vocab_size == CONFIG["vocab_size"]


# ── round-trip SK technical sentence (in-vocab chars) ─────────────────────────
def test_roundtrip_slovak_technical(tok: Tokenizer) -> None:
    sk = "šifrovanie a autentifikácia hesla znižujú zraniteľnosť proti útoku phishing."
    enc = tok.encode(sk)
    dec = tok.decode(enc)
    # decode drops specials and joins pieces -> must reproduce lowercased text
    assert dec == sk.lower(), f"SK round-trip failed:\n  in : {sk.lower()}\n  out: {dec}"


# ── round-trip EN technical sentence (in-vocab chars) ─────────────────────────
def test_roundtrip_english_technical(tok: Tokenizer) -> None:
    en = "the model uses attention and expert routing for inference."
    enc = tok.encode(en)
    dec = tok.decode(enc)
    assert dec == en.lower(), f"EN round-trip failed:\n  in : {en.lower()}\n  out: {dec}"


# ── BOS / EOS wrapping ─────────────────────────────────────────────────────────
def test_bos_eos_wrapping(tok: Tokenizer) -> None:
    enc = tok.encode("hello")
    assert enc[0] == tok.special_ids["bos_id"], "BOS must be the first token"
    assert enc[-1] == tok.special_ids["eos_id"], "EOS must be the last token"
    # The body sits strictly between BOS and EOS.
    assert len(enc) >= 3, "encode must produce BOS + body + EOS (>=3 tokens)"


# ── UNK fallback for an out-of-vocab character ────────────────────────────────
def test_unk_fallback(tok: Tokenizer) -> None:
    # A control character (BEL, U+0007) has no single-char slot -> UNK.
    bel = "\x07"
    enc = tok.encode(bel)
    # enc = [BOS, UNK, EOS]
    assert enc[0] == tok.special_ids["bos_id"]
    assert enc[-1] == tok.special_ids["eos_id"]
    assert tok.special_ids["unk_id"] in enc, "out-of-vocab char must yield UNK"
    # The UNK must sit in the body (not be one of the specials wrapping it).
    body = enc[1:-1]
    assert body == [tok.special_ids["unk_id"]], (
        f"body must be exactly [UNK], got {body}")


# ── in-vocab chars do NOT produce UNK ─────────────────────────────────────────
def test_in_vocab_chars_no_unk(tok: Tokenizer) -> None:
    # "xé" — both 'x' and 'é' have dedicated char slots in the vocab.
    enc = tok.encode("xé")
    assert tok.special_ids["unk_id"] not in enc, (
        "x and é are in-vocab and must not produce UNK")


# ── special_ids dict exposes all five CONFIG special ids ──────────────────────
def test_special_ids_complete(tok: Tokenizer) -> None:
    for name in ("pad_id", "bos_id", "eos_id", "unk_id", "sep_id"):
        assert name in tok.special_ids, f"special_ids missing {name}"
        assert tok.special_ids[name] == CONFIG[name], (
            f"special_ids[{name}] mismatch with CONFIG")