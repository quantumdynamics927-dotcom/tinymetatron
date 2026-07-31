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

## Retrieval metrics (final eval, test-only index, hybrid+structured)

| Metric | ID-based | source_identity-based |
|---|---|---|
| Recall@1 | 0.1471 | 0.3125 |
| Recall@5 | 0.2235 | **0.4750** |
| MRR | 0.1739 | **0.3696** |

The discrepancy between id-based and si-based reflects DB rebuilds between when gold IDs were assigned and the current index. `source_identity` metrics reflect true retrieval quality.

## Answer quality

| Metric | Value |
|---|---|
| Rubric correctness (ask) | 0.7882 |
| Abstention recall | 1.0 |
| False answer rate on unanswerable | 0.0 |
| False abstention rate on answerable | 0.0824 |
| Seeded canary leakage rate | 0.0 |

## Stability tests (all 3/3 PASS)

```
REBUILD STABILITY: PASS
ROW-ID INDEPENDENCE: PASS
CHANGE DETECTION: PASS
```

## Files in commit `50874a9`

- `quantum_corpus/schema.py` — source_identity column, function, write_records, fetch_all, backfill
- `quantum_corpus/rag.py` — source_identity in RAGIndex hit metadata
- `quantum_corpus/semantic.py` — source_identity in SemanticIndex hit metadata
- `quantum_corpus/eval/build_qa.py` — gold_source_identities in QA items
- `quantum_corpus/eval/runner.py` — si-based retrieval metrics in aggregate + per-item
- `quantum_corpus/eval/tune.py` — updated score_item_ask signature
- `quantum_corpus/eval/test_stable_identity.py` — NEW: 3 stability tests
- `quantum_corpus/eval/qa_val.jsonl` — rebuilt with gold_source_identities
- `quantum_corpus/eval/qa_test.jsonl` — rebuilt with gold_source_identities
