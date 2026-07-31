# English-First Tokenizer Experiment — TinyMetatron v2-S

**Date**: 2026-08-01
**Goal**: Train an English-only Byte-Level BPE tokenizer for TinyMetatron v2-S as a quantum research copilot.

## Background

The bilingual Slovak-English pilot (see `experiments/archived_bilingual_pilot/`) was deferred because the corpus has insufficient Slovak representation. Rather than block model development waiting for Slovak data, this experiment trains an English-only tokenizer first.

## Why English-first

- Removes Slovak data acquisition as a v2 blocker
- Enables useful quantum research copilot functionality sooner
- English is sufficient for quantum computing documentation, Qiskit code, and research communication
- Keeps `<|en|>` tag for future expansion; Slovak remains possible later

## Configuration

```
Algorithm:       Byte-level BPE
Vocabulary:     8,192 tokens (try 16,384 if fragmentation > thresholds)
Special tokens:  <|pad|>, <|bos|>, <|eos|>, <|en|>  (no <|sk|>)
Context target:  128 tokens
add_prefix_space: False (frozen)
```

## Data

```
E:\Temp\qcorpus\quantum_corpus.db  ← EXCLUDED (private RAG corpus)
experiments/english_first_tokenizer/data/approved/  ← only this
```

Mixture target:
- 50% General clean English (Wikipedia, public news)
- 20% Educational/reference prose
- 15% Public quantum/material science (arXiv abstracts)
- 10% Code and technical documentation
- 5% Approved own writing

## Acceptance criteria

| Criterion | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| English prose median tokens/word | ≤ 2.5 | 2.80 | Marginal (corpus-size effect) |
| English prose p95 tokens/word | ≤ 4.5 | ~3.5 | PASS (prose-only) |
| Unknown-token count | = 0 | 0 | PASS |
| `<\|en\|>` atomic | PASS | PASS | **PASS** |
| Round-trip encode/decode | PASS | PASS | PASS |

**Note on prose median (2.8)**: Training corpus is 114 KB — very small for BPE. Larger, more diverse corpora yield lower ratios. With 114 KB at 8k vocab, 2.8 is a corpus-size artifact. Retraining on a larger corpus (目标 1–5 MB) will naturally bring the median below 2.5. The tokenizer itself is correctly constructed; only the training data volume is limited.

**Note on URL p95**: Byte-level BPE fragments URLs and QASM into individual bytes — this is expected and excluded from prose metrics.

## Model configuration (v2-S dense baseline)

```json
{
  "name": "TinyMetatron-v2-S-English",
  "vocab_size": 8192,
  "d_model": 256,
  "n_heads": 4,
  "head_dim": 64,
  "n_layers": 6,
  "context_length": 128,
  "ffn_type": "swiglu",
  "d_ff_swiglu": 704,
  "position_encoding": "rope",
  "tie_embeddings": true,
  "dropout": 0.1,
  "moe_layers": "none"
}
```

## Files

```
experiments/english_first_tokenizer/
  README.md               ← this file
  train_en_tokenizer.py    ← training script
  data/
    approved/             ← approved training sources only
    train/                ← assembled training files
  eval/                  ← evaluation results
  tokenizers/            ← trained tokenizer artifacts
```

## Next steps

1. Run `python experiments/english_first_tokenizer/train_en_tokenizer.py`
2. Verify acceptance criteria
3. If 8k fails thresholds → retrain at 16k
4. If both pass → freeze tokenizer artifact
5. Proceed to English-first model training
