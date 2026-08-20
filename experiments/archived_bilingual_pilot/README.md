# Archived: Bilingual Slovak-English Tokenizer Pilot

**Status**: Deferred — not abandoned.
**Reason**: Approved Slovak corpus is insufficient for fair bilingual training.
**Restart condition**: Approved Slovak corpus meeting minimum requirements (>100KB clean Slovak text, character-balanced with English) + fresh 8k/16k bilingual tokenizer evaluation.

## What this pilot proved

The Byte-Level BPE infrastructure works correctly:
- Zero unknown tokens (byte-level UTF-8 coverage confirmed)
- Language tags `<|en|>` and `<|sk|>` are atomic tokens
- Pipeline, SHA256 manifest tracking, and evaluation suite are functional

The pilot **correctly detected** that the TinyMetatron corpus has ~no Slovak representation (2 Slovak-diacritic records out of 46,579). Training a bilingual tokenizer on English-only data produces >90% SK/EN token imbalance regardless of vocabulary size.

## What was NOT a failure

- The 16k pilot English compression improved (2.36→2.22 tokens/word vs 8k)
- The byte-level base ensures zero unknown tokens even with no Slovak training

## Restart requirements

1. Obtain approved Slovak training data: ≥100KB clean Slovak text, UTF-8, character-balanced with English
2. Place in `data_v2/tokenizer_samples/sk_balanced.txt`
3. Retrain both 8k and 16k from scratch with bilingual corpus
4. Verify |SK/EN imbalance| < 20% before adopting

## Files

- `TOKENIZER_PILOT.md` — original pilot report (this directory)
- `train_pilot_tokenizer.py` — original pilot script
- `tokenizers/8k/` and `tokenizers/16k/` — pilot artifacts (not approved for bilingual training)
- `data_v2/tokenizer_samples/` — corpus samples used

## New direction

See `D:\TinyMetatron\experiments\english_first_tokenizer\` for the English-only tokenizer experiment.
