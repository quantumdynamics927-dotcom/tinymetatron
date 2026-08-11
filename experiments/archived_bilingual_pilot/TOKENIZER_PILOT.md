# Tokenizer Pilot — TinyMetatron v2-S

**Date**: 2026-07-31
**Script**: `train_pilot_tokenizer.py`
**Corpus**: `E:\Temp\qcorpus\quantum_corpus.db` (507K chars extracted)

---

## Status

> **TinyMetatron v2 tokenizer infrastructure: validated.**
> **Bilingual tokenizer: blocked pending Slovak corpus acquisition and review.**
>
> **16k is retained as the current technical baseline artifact; it is not approved
> for bilingual model pretraining.**

---

## What was tested

| Run | Vocab | Training data | EN tokens/word | SK tokens/word | Imbalance |
|-----|-------|---------------|----------------|----------------|-----------|
| 1   | 8,192 | English only (507K chars) | 2.3639 | 4.4272 | **+87.3%** |
| 2   | 16,384 | English only (507K chars) | 2.2179 | 4.2550 | **+91.9%** |

Decision rule: keep 8k when |SK/EN imbalance| < 20%.

**Both runs → REVIEW** (imbalance >> 20%)

---

## Root cause analysis

The corpus (`quantum_corpus.db`) contains **2 Slovak-diacritic records** out of 46,579 total.
The tokenizer was trained on English text only.

Slovak diacritics (á ä č ď é í ľ ĺ ň ó ô ŕ š ť ú ý ž) encode as multi-byte UTF-8
sequences (2–3 bytes each). An English-trained BPE has no merge rules for these bytes,
so it falls back to individual bytes — producing 2–3× more tokens per word than English.

**Increasing vocab size does not fix this.** The extra 16k capacity improves English
compression (2.36→2.22 tokens/word, −6%) but the imbalance barely changes because
the Slovak merge rules were never in the training data.

---

## What passes

| Criterion | Result |
|-----------|--------|
| Unknown tokens (`<\|unk\|>` count) | **0** — byte-level BPE covers all UTF-8 |
| Language tag atomicity (`<\|en\|>`, `<\|sk\|>`) | **PASS** — both encode as single tokens (IDs 3, 4) |
| English fragmentation (p95) | 3.11 tokens/word at 16k |
| No `<\|unk\|>` tokens anywhere | ✅ |

---

## What fails

| Criterion | Threshold | Result |
|-----------|-----------|--------|
| SK/EN tokens/word imbalance | < 20% | **+91.9%** (fails) |
| Slovak long-word fragmentation | — | 8.75 tokens/word (e.g. "Najneobvyklejšieho") |

---

## Retraining gate

When Slovak data is available, retrain **both** 8k and 16k from scratch and select
the smallest model meeting all conditions:

```
- Slovak median tokens/word <= English median * 1.20
- Slovak 95th-percentile fragmentation <= 3.5 tokens/word
- Unknown-token count = 0
- Exact or documented-normalized encode/decode round-trip passes
- Slovak diacritics, language tags, numbers, URLs, paths, and code pass
```

Place approved data only in:

```
D:\TinyMetatron\data_v2\approved\sk\
D:\TinyMetatron\data_v2\approved\en\
```

Keep the private Quantum RAG corpus out of training data:

```
E:\Temp\qcorpus\quantum_corpus.db  (private retrieval corpus, not LM training)
```

---

## Saved artifacts

| Path | Description |
|------|-------------|
| `tokenizers/tinymetatron_v2_bpe_8k/` | 8k pilot (historical) |
| `tokenizers/tinymetatron_v2_bpe_16k/` | 16k — current technical baseline |
| `tokenizers/tinymetatron_v2_bpe_16k/manifest.json` | SHA256-tracked manifest |
| `data_v2/tokenizer_samples/en_corpus_sample.txt` | 500K char English corpus sample |
| `data_v2/tokenizer_samples/ibm_jobs_sample.txt` | 125K char IBM jobs sample |
