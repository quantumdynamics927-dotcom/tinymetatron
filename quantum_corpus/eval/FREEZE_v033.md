# Build Freeze Manifest — v0.3.3

**Date**: 2026-07-31
**Commit**: `50874a9` (Add stable source_identity throughout the pipeline)

## What changed

v0.3.3 adds stable `source_identity` (content-based, not row-id based) throughout the pipeline:

```
source_identity = sha256(project + "\n" + doc_id + "\n" + cleaning_version + "\n" + content_hash + "\n" + chunk_index)
```

This survives DB rebuilds, deduplication, insertion-order changes, and split reassignments.

## Corpus DB

- **Path**: `E:\Temp\qcorpus\quantum_corpus.db`
- **DB SHA256**: `8493932d171b007c1b6e1ffe10128a760cd92fe411bbff1359da02ac6439d564`
- **Corpus build ID**: `quantum-corpus-build-2`
- **Migration applied**: `ALTER TABLE corpus_records ADD COLUMN source_identity TEXT NOT NULL DEFAULT ''` then backfill of 46,579 records

## QA Manifests

| Manifest | SHA256 | Items |
|---|---|---|
| `qa_val.jsonl` | `93b5226368de10852a80de6d54413339db8fb9b8e067b7832e05172b82f3efde` | 68 (tuning only) |
| `qa_test.jsonl` | `4fb3289a761631eca3a03800270bdf61a9c09961b45c6cc9a7299816cfb41725` | 100 (hold-out) |

## Retrieval weights

| Parameter | Value |
|---|---|
| `BM25_W` | 1.0 |
| `SEM_W` | 0.0 (semantic disabled; hybrid = pure BM25) |
| `RRF_K` | 60 |

## Sensitivity policy

| Parameter | Value |
|---|---|
| `max_sensitivity` | `sensitive` (full corpus) |
| Sensitivity filter | post-fusion (AFTER RRF ranking) |

## Gate thresholds

| Gate | Value |
|---|---|
| `score_floor_hybrid` | 0.008 |
| `score_floor_bm25` | 3.0 |
| `sep_ratio` | 1.5 |
| `sep_band` | 2.0 |
| `min_concepts` | 1 |

## Official final results — v0.3.3 hold-out (100 items, test-only index)

**All metrics use source-identity-based computation as the meaningful values; id-based values are legacy diagnostics only.**

### Retrieval (retrieval-applicable questions)

| Metric | Computation | Value |
|---|---|---|
| Source-identity Recall@5 | Gold SI in top-5 / retrieval-applicable items | **0.4750** |
| Source-identity MRR | 1/rank of first gold SI / retrieval-applicable items | **0.3696** |
| ID-based Recall@5 | Legacy diagnostic (mutable row IDs) | 0.2235 |
| ID-based MRR | Legacy diagnostic (mutable row IDs) | 0.1739 |

**Denominator note**: Metrics computed over items with non-empty `gold_source_identities` that used the retrieval route (not structured SQL). Items answered via structured SQL are excluded from retrieval metrics and reported separately.

### Structured-query correctness

| Metric | Value |
|---|---|
| Structured-route items | 4 (q074, q075, q077, q085) |
| Structured correct | 4 / 4 |

Structured-route items are excluded from retrieval Recall@5/MRR and reported separately because the SQL path answers correctly without document retrieval.

### Answer correctness

| Metric | Denominator | Value |
|---|---|---|
| Rubric correctness | All 85 answerable items | 0.7882 (67/85) |

### Abstention safety

| Metric | Denominator | Value |
|---|---|---|
| Abstention recall | 15 expected-abstention items | **1.0** (15/15) |
| Abstention precision | 15 abstention-expected + 7 false abstentions | 0.6818 |
| False answers on unanswerables | 15 expected-abstention items | **0.0** (0/15) |
| False abstentions on answerables | 85 answerable items | 0.0824 (7/85) |

### Groundedness

| Metric | Value |
|---|---|
| Seeded canary leakage rate | **0.0** |

### Summary box

```
Source-identity Recall@5: 0.4750 (over retrieval-applicable items)
Source-identity MRR:       0.3696 (over retrieval-applicable items)
Rubric correctness:         0.7882 (85 answerable items)
Abstention recall:         1.0    (15 expected-abstention items)
False answers:             0.0    (15 expected-abstention items)
False abstentions:         0.0824 (85 answerable items)
Canary leakage:            0.0
```

## 21 failure cards (retrieval failure analysis)

| Class | Count | Items |
|---|---|---|
| `empty-retrieval-score-floor` | 4 | q058, q059, q076, q079 |
| `lexical-mismatch-wrong-doc-ranked` | 13 | q049, q060, q061, q064, q069, q070, q071, q072, q073, q078, q082, q083, q084 |
| `structured-answer-correct-no-doc-retrieval` | 4 | q074, q075, q077, q085 |

Failure cards: `quantum_corpus/eval/retrieval_failure_cards.jsonl`

## Next development targets (v0.3.4-dev)

```
- Reduce false abstentions below 5% (from 8.24%)
- Preserve zero false answers on unanswerables
- Preserve zero canary leakage
- Improve retrieval-applicable Recall@5 above 0.475
- Improve retrieval-applicable MRR above 0.370
```

Development sets to build:
- `qa_dev_retrieval.jsonl` — from failure card analysis
- `qa_val_retrieval.jsonl` — held-out from dev, used for tuning

## Reproducibility record

```
Corpus build ID: quantum-corpus-build-2
Corpus DB SHA-256: 8493932d171b007c1b6e1ffe10128a760cd92fe411bbff1359da02ac6439d564
QA validation manifest SHA-256: 93b5226368de10852a80de6d54413339db8fb9b8e067b7832e05172b82f3efde
QA test manifest SHA-256: 4fb3289a761631eca3a03800270bdf61a9c09961b45c6cc9a7299816cfb41725

Relevant commits:
  470aa37  Quantum RAG v0.3.2: field-verification gate + canary metadata
  e1caa8b  Fix RAG sensitivity fusion and runner parity
  50874a9  Add stable source_identity throughout the pipeline
  477c257  Add freeze manifest and migration utility for v0.3.3
  f5ac6a4  Add retrieval failure cards for v0.3.3 hold-out eval

Detailed private report:
  E:\Temp\qcorpus\reports\final_eval_source_identity.json
```

## Files in commits 470aa37 through f5ac6a4

| File | Commit | Change |
|---|---|---|
| `quantum_corpus/schema.py` | 50874a9 | source_identity column, function, write_records, fetch_all, backfill |
| `quantum_corpus/rag.py` | 50874a9 | source_identity in RAGIndex hit metadata |
| `quantum_corpus/semantic.py` | 50874a9 | source_identity in SemanticIndex hit metadata |
| `quantum_corpus/eval/build_qa.py` | 50874a9 | gold_source_identities in QA items |
| `quantum_corpus/eval/runner.py` | 50874a9 | si-based retrieval metrics in aggregate + per-item |
| `quantum_corpus/eval/tune.py` | 50874a9 | updated score_item_ask signature |
| `quantum_corpus/eval/test_stable_identity.py` | 50874a9 | NEW: 3 stability tests (all pass) |
| `migrate_source_identity.py` | 477c257 | NEW: one-shot DB migration utility |
| `quantum_corpus/eval/FREEZE_v033.md` | 477c257 | This document |
| `quantum_corpus/eval/retrieval_failure_cards.jsonl` | f5ac6a4 | 21 failure cards for next development cycle |
