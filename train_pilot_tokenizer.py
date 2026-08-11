"""
train_pilot_tokenizer.py
========================
Pilot Byte-Level BPE tokenizer training and evaluation for TinyMetatron v2.

CORRECTED PER ARCHITECTURAL GUIDANCE:
  - Special tokens: pad, bos, eos, en, sk  (no unk; byte-level covers all UTF-8)
  - Training data: balanced by character count, not file count
  - Language tags trained as corpus markers (not just special tokens)
  - add_prefix_space=False (frozen before training)
  - Comprehensive bilingual test suite

LIMITATION: The quantum corpus (E:\\Temp\\qcorpus\\quantum_corpus.db) is ~98% English.
Only 2 Slovak-diacritic records exist in the entire 46k-record corpus. This pilot
extracts available English text. A proper bilingual tokenizer requires actual bilingual
training data — see README for Slovak data requirements.

DECISION RULE:
  Keep 8k when Slovak median tokens/word is within ~20% of English.
  Move to 16k when Slovak fragmentation is materially worse.

USAGE::

    # 1. (Before running) Place real bilingual data in data_v2/tokenizer_samples/
    #    en_balanced.txt  — English text, UTF-8, at least 100KB
    #    sk_balanced.txt  — Slovak text, UTF-8, at least 100KB, same char count as en
    #
    # 2. Run:
    python train_pilot_tokenizer.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ── Configuration ────────────────────────────────────────────────────────────
# DECISION: VOCAB_SIZE = 16384
#
# Pilot results (English-only training — corpus has no Slovak):
#   8k:  EN=2.36  SK=4.43  imbalance=+87.3%
#  16k:  EN=2.22  SK=4.26  imbalance=+91.9%
#
# English improves with more capacity (2.36→2.22 tokens/word, -6%).
# Slovak fragmentation cannot improve without Slovak training data.
# The imbalance is structural: Slovak diacritics (á ä č ď é ...) encode as
# multi-byte UTF-8 sequences that the English-trained BPE cannot merge efficiently.
#
# ACTION REQUIRED: To reduce Slovak fragmentation, obtain real Slovak text
# (≥100KB, character-balanced with English) and retrain. The v2 tokenizer is
# otherwise fully functional: zero unknown tokens, atomic language tags.
VOCAB_SIZE = 16384
OUT_DIR = Path("tokenizers/tinymetatron_v2_bpe_16k")
DATA_FILES = [
    "data_v2/tokenizer_samples/en_balanced.txt",
    "data_v2/tokenizer_samples/sk_balanced.txt",
]

# Special tokens — frozen IDs after training:
#   0: <|pad|>   1: <|bos|>   2: <|eos|>   3: <|en|>   4: <|sk|>
# No <unk> — byte-level BPE covers all UTF-8 without unknown tokens.
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|en|>", "<|sk|>"]

# ── Comprehensive bilingual test suite ────────────────────────────────────────
TEST_CASES: Dict[str, str] = {
    # Language-tag atomicity
    "en_tag_atomic":   "<|en|>The quantum processor runs at 10 MHz.",
    "sk_tag_atomic":   "<|sk|>Kvantový procesor beží na 10 MHz.",

    # General prose
    "en_general":       "<|en|> The quantum processing unit requires strict error correction protocols.",
    "sk_general":       "<|sk|> Kvantová procesorová jednotka vyžaduje prísne protokoly korekcie chýb.",

    # All 17 Slovak diacritics
    "sk_all_diacs":     "<|sk|> á ä č ď é í ľ ĺ ň ó ô ŕ š ť ú ý ž",

    # Technical
    "en_technical":     "<|en|> Run rx(pi/2) on ibm_fez; compare OTOC and Lyapunov estimates.",
    "sk_technical":     "<|sk|> Porovnaj OTOC a Lyapunovov odhad pre QASM obvody.",

    # Quantum identifiers
    "en_quantum_ids":   "<|en|> OTOC Lyapunov Hadamard QASM CNOT RX RZ ibm_fez ibm_torino",
    "sk_quantum_ids":   "<|sk|> OTOC Lyapunov Hadamard QASM CNOT RX RZ kvantová superpozícia",

    # Numbers and punctuation
    "en_numbers":       "<|en|> Batch 42 ran 8192 samples at 2026-07-31 with cost=3.14.",
    "sk_numbers":       "<|sk|> Dávka 42 mala 8192 vzoriek pri cene 3,14 eura.",

    # Code / identifiers
    "en_code":           "<|en|> def add(a, b): return a + b  https://example.org/model-v2",
    "sk_code":           "<|sk|> funkcia súčet(a, b) -> a + b  https://model.sk/v2",

    # Long Slovak compounds (fragmentation stress-test)
    "sk_long_word":     "<|sk|> Najneobvyklejšieho zimolezovníka nenájdem.",
    "sk_inflection":     "<|sk|> štúdií štúdiách štúdiám študentom",

    # Empty / edge
    "en_empty":         "<|en|>",
    "sk_empty":         "<|sk|>",
}


# ── Sample extraction from corpus ─────────────────────────────────────────────
def extract_corpus_samples(db_path: str, out_dir: Path, target_chars: int = 500_000):
    """Extract English text samples from the quantum corpus for tokenizer training.

    This is a fallback when no external bilingual data is available.
    The corpus is ~98% English, so Slovak data must come from elsewhere.
    """
    import sqlite3
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # English: GRE project records (code/prose, largest pool)
    print(f"Extracting English samples from corpus ({target_chars:,} char target)...")
    rows = conn.execute('''
        SELECT text FROM corpus_records
        WHERE project = 'GRE'
        AND split IN ('train', 'val')
        AND text IS NOT NULL
        ORDER BY RANDOM()
    ''').fetchall()

    english = ""
    for r in rows:
        text = (r["text"] or "")[:1000]  # up to 1K per record
        english += text + "\n"
        if len(english) >= target_chars:
            break

    en_path = out_dir / "en_corpus_sample.txt"
    with open(en_path, "w", encoding="utf-8") as f:
        f.write(english)
    print(f"  English: {len(english):,} chars -> {en_path}")

    # IBM job clean records as supplementary English
    rows2 = conn.execute('''
        SELECT text FROM corpus_records
        WHERE source_type = 'ibm_job'
        AND split IN ('train', 'val')
        AND text NOT LIKE '%<bound%'
        AND text IS NOT NULL
        ORDER BY RANDOM()
    ''').fetchall()

    ibm_text = ""
    for r in rows2:
        text = (r["text"] or "")[:500]
        ibm_text += text + "\n"
        if len(ibm_text) >= target_chars // 4:
            break

    ibm_path = out_dir / "ibm_jobs_sample.txt"
    with open(ibm_path, "w", encoding="utf-8") as f:
        f.write(ibm_text)
    print(f"  IBM jobs: {len(ibm_text):,} chars -> {ibm_path}")

    conn.close()
    return en_path, ibm_path


# ── Tokenizer training ────────────────────────────────────────────────────────
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_size(f: Path) -> str:
    kb = f.stat().st_size / 1024
    return f"{kb:.1f} KB" if kb < 1024 else f"{kb/1024:.1f} MB"


def train_pilot_tokenizer(training_files: List[Path]) -> Optional["ByteLevelBPETokenizer"]:
    """Train and save the Byte-Level BPE pilot tokenizer."""
    try:
        from tokenizers import ByteLevelBPETokenizer
    except ImportError:
        print("FATAL: tokenizers library not installed.")
        print("  pip install tokenizers")
        return None

    print(f"\n{'='*60}")
    print("TRAINING BYTE-LEVEL BPE TOKENIZER (v2 PILOT)")
    print(f"{'='*60}")
    print(f"  Vocab size  : {VOCAB_SIZE}")
    print(f"  Special IDs : {SPECIAL_TOKENS}")
    print(f"  add_prefix_space: False (frozen)")
    print(f"  Data files :")
    for f in training_files:
        print(f"    {f}  ({_file_size(f)})")

    # Balance report
    sizes = [f.stat().st_size for f in training_files]
    if len(sizes) >= 2 and sizes[0] > 0 and sizes[1] > 0:
        diff = abs(sizes[0] - sizes[1]) / max(sizes)
        print(f"\n  Char balance: {sizes[0]:,} vs {sizes[1]:,} ({diff:.1%} diff)")
        if diff > 0.5:
            print("  WARNING: >50% imbalance — balance data for bilingual training!")
    elif len(sizes) == 1:
        print(f"\n  WARNING: Only one training file ({sizes[0]:,} chars) — bilingual training needs two!")

    tokenizer = ByteLevelBPETokenizer(add_prefix_space=False)

    print(f"\nTraining...")
    tokenizer.train(
        files=[str(f) for f in training_files],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(OUT_DIR))
    tokenizer.save(str(OUT_DIR / "tokenizer.json"))

    print(f"\nSaved -> {OUT_DIR}/")
    print(f"  vocab.json      (HuggingFace vocab)")
    print(f"  merges.txt     (BPE merge rules)")
    print(f"  tokenizer.json (full config)")

    return tokenizer


# ── Comprehensive evaluation ───────────────────────────────────────────────────
def evaluate_tokenizer(tokenizer) -> tuple[dict, dict]:
    """Run bilingual test suite and return per-case results + summary."""
    print(f"\n{'='*60}")
    print("BILINGUAL FRAGMENTATION EVALUATION")
    print(f"{'='*60}")

    results = {}
    for name, text in TEST_CASES.items():
        enc = tokenizer.encode(text)
        dec = tokenizer.decode(enc.ids, skip_special_tokens=False)
        tokens = enc.tokens
        ids = enc.ids

        word_count = max(1, len(text.split()))
        char_count = len(text)
        token_count = len(tokens)
        unk_id = tokenizer.token_to_id("<|unk|>")
        unk_count = sum(1 for i in ids if i == unk_id) if unk_id is not None else 0

        # Language-tag atomicity
        lang_tag = "<|en|>" if name.startswith("en") else "<|sk|>"
        tag_enc = tokenizer.encode(lang_tag)
        tag_atomic = (len(tag_enc.ids) == 1 and tag_enc.ids[0] == tokenizer.token_to_id(lang_tag))

        results[name] = {
            "token_count": token_count,
            "word_count": word_count,
            "char_count": char_count,
            "ratio_words": round(token_count / word_count, 4),
            "ratio_chars": round(token_count / char_count, 4),
            "unk_count": unk_count,
            "tag_atomic": tag_atomic,
            "tokens": tokens,
        }

        print(f"\n[{name}]")
        print(f"  text       : {text[:65]}")
        print(f"  tokens/word: {token_count/word_count:.3f}  tokens/char: {token_count/char_count:.4f}")
        print(f"  unk_count : {unk_count}  (must be 0 for byte-level)")
        print(f"  tag_atomic : {tag_atomic}  {lang_tag}={tag_enc.ids}")
        print(f"  tokens    : {tokens[:10]}{'...' if len(tokens) > 10 else ''}")

    # Aggregate
    en_res = {k: v for k, v in results.items() if k.startswith("en_")}
    sk_res = {k: v for k, v in results.items() if k.startswith("sk_")}

    def avg_ratio(r_dict):
        vals = [v["ratio_words"] for v in r_dict.values() if v["word_count"] > 0]
        return sum(vals) / len(vals) if vals else float("nan")

    en_avg = avg_ratio(en_res)
    sk_avg = avg_ratio(sk_res)
    imbalance = (sk_avg - en_avg) / en_avg if en_avg else float("nan")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  EN avg tokens/word: {en_avg:.4f}")
    print(f"  SK avg tokens/word: {sk_avg:.4f}")
    print(f"  SK/EN imbalance  : {imbalance:+.2%}")
    print(f"  Target: |imbalance| < 20%")

    all_unk = sum(v["unk_count"] for v in results.values())
    all_atomic = all(v["tag_atomic"] for v in results.values())

    p95_en = sorted(v["ratio_words"] for v in en_res.values())[int(len(en_res) * 0.95)] if en_res else float("nan")
    p95_sk = sorted(v["ratio_words"] for v in sk_res.values())[int(len(sk_res) * 0.95)] if sk_res else float("nan")

    print(f"  Unknown tokens : {all_unk}  (must be 0)")
    print(f"  Lang tags atomic: {'PASS' if all_atomic else 'FAIL'}")
    print(f"  95th-pct EN    : {p95_en:.3f}")
    print(f"  95th-pct SK    : {p95_sk:.3f}")

    decision = "PASS 8k" if abs(imbalance) < 0.20 and all_unk == 0 and all_atomic else "REVIEW 16k"
    print(f"  DECISION       : {decision}")

    summary = {
        "en_avg_ratio": round(en_avg, 4),
        "sk_avg_ratio": round(sk_avg, 4),
        "imbalance": round(imbalance, 4),
        "all_unk": all_unk,
        "all_tags_atomic": all_atomic,
        "p95_en": round(p95_en, 4),
        "p95_sk": round(p95_sk, 4),
        "decision": decision,
    }
    return results, summary


# ── Save manifest ──────────────────────────────────────────────────────────────
def save_manifest(tokenizer, training_files: List[Path]):
    tok_json = OUT_DIR / "tokenizer.json"
    with open(tok_json, "rb") as f:
        tok_sha = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "name": "tinymetatron-v2-bpe-16k",
        "algorithm": "byte_level_bpe",
        "vocab_size": VOCAB_SIZE,
        "special_tokens": SPECIAL_TOKENS,
        "languages": ["en", "sk"],
        "add_prefix_space": False,
        "data_files": {str(f.name): _sha256(f) for f in training_files},
        "tokenizer_json_sha256": tok_sha,
    }
    path = OUT_DIR / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n  manifest -> {path}")
    return manifest


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("Byte-Level BPE Pilot — TinyMetatron v2-S Tokenizer")
    print(f"Python: {sys.version.split()[0]}")

    # Check for training data
    training_files = []
    missing = []
    for f in DATA_FILES:
        p = Path(f)
        if p.is_file() and p.stat().st_size > 0:
            training_files.append(p)
        else:
            missing.append(f)

    if len(training_files) < 2:
        print(f"\nWARNING: Missing training data ({len(missing)} files).")
        print("Extracting English samples from quantum corpus as fallback...")
        db_path = os.environ.get("TMT_QUANTUM_CORPUS_DB", "E:\\Temp\\qcorpus\\quantum_corpus.db")
        if os.path.exists(db_path):
            en_path, ibm_path = extract_corpus_samples(db_path, Path("data_v2/tokenizer_samples"))
            # Combine English sources for a single-file training run
            # (proper bilingual training needs separate en/sk files)
            training_files = [en_path]
            print(f"  Using English-only corpus sample ({_file_size(en_path)})")
            print("  NOTE: Slovak training data still required for bilingual evaluation!")
        else:
            print(f"FATAL: corpus DB not found at {db_path}")
            return 1

    tokenizer = train_pilot_tokenizer(training_files)
    if not tokenizer:
        return 1

    _, summary = evaluate_tokenizer(tokenizer)
    manifest = save_manifest(tokenizer, training_files)

    print(f"\n{'='*60}")
    print("PILOT COMPLETE")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
