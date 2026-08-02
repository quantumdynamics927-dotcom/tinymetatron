**Note**: v0.3.4 is frozen. All future retrieval experiments belong under v0.3.5-dev in FREEZE_v033.md.

# Retrieval-Improvement Loop

## Purpose

Iteratively improve TinyMetatron RAG retrieval by classifying failures,
applying one bounded change at a time, and running full regression gates
before any commit.

## Loop States

```
NEW -> OBSERVED -> HYPOTHESIS_CREATED -> VALIDATED -> AWAITING_APPROVAL -> EXECUTED -> MEASURED -> ACCEPTED | REJECTED | ROLLED_BACK -> ARCHIVED
```

## Input

- `quantum_corpus/eval/retrieval_failure_cards.jsonl` — failure cards with
  root-cause classification, gold source_identity, and one-line hypothesis

## Allowed Actions (bounded)

Only one change per loop iteration:

1. Add a scoped schema alias (`rag.py:expand_query`)
2. Add an entity-aware TF boost (`rag.py:expand_query`)
3. Adjust a documented retrieval weight (`BM25_W`, `SEM_W`, `RRF_K`)
4. Adjust a documented gate threshold (`score_floor_bm25`, `sep_ratio`, etc.)
5. Fix a confirmed bug in the retrieval pipeline

**Forbidden**: semantic model changes, changing the corpus schema,
modifying canary records, touching the QSG directory.

## Documented Retrieval Rules

### Circuit-Header Chunking (`rag.py:_chunk_circuit_headers`)

**Trigger condition** — all three must be true:
1. `source_type == "ibm_job"`
2. `text.find("OPENQASM 3.0;") > 0`  (OPENQASM marker exists and is not at position 0)
3. `text.find("OPENQASM 3.0;") < len(text) * 0.50`  (circuit body is >50% of text)

**Action**: `text = text[:text.find("OPENQASM 3.0;")].strip()` — replaces the record
text with header-only (JID, backend, status, program, cost, timestamps), keeps
the original `id`, sets `is_chunked=True`.

**Why the 50% threshold**: IBM Quantum job records have two structural patterns:
- **Circuit-dominant** (6–33% OPENQASM position): The circuit body is 67–94% of text,
  diluting JID term density below that of short provenance stub records → chunking needed.
- **Metadata-dominant** (91–94% OPENQASM position): JID already near the top with high
  term density → no chunking needed; the circuit text is already a trailing fragment.

The 50% boundary cleanly separates these two populations.

**Retrievability of circuit body**: The full record remains accessible via `doc_id`
lookup. Chunking only affects BM25 indexing; it does not delete data.

**Exp-001 reference**: Resolves v044–v048 (JID `d5a6s6jht8fs73a5d46g`, record 38202).
Gates passed: 5/5 affected questions retrieve gold in top-5; no regression on 106/106 dev+val.

## Mandatory Gates (all must pass)

| Gate | Command | Pass condition |
|------|---------|---------------|
| Retrieval dev eval | `python -m quantum_corpus.eval.runner dev --mode ask --retriever hybrid` | si_recall@5 ≥ previous |
| Canary suite | `python -m quantum_corpus.eval.runner run_canaries` | leakage rate = 0.0 |
| Regression test | `python -m quantum_corpus.eval.test_conscious_dna_alias` | 13/13 PASS |
| Runner parity | `python -m quantum_corpus.eval.runner dev --mode bm25` | no crash |
| No test-holdout access | Loop code never reads `qa_test.jsonl` during development | Code audit |

## Output

- `loops/retrieval_loop/experiments/` — one subdir per experiment containing:
  - `experiment.json` — hypothesis, action taken, validation results
  - `report_dev_*.json` — per-run dev eval report
  - `candidate_patch.diff` — the actual code change
- `loops/retrieval_loop/archive/` — experiments that were rejected or rolled back
- Commit only after ALL gates pass and human approval is obtained

## Workflow

1. **Observe**: `python loops/retrieval_loop/refresh_cards.py` — regenerate failure cards from live eval (NEVER rely on stale cards)
2. **Orient**: `python loops/retrieval_loop/run_loop.py list` — review open failures and their categories; pick ONE root cause
3. **Propose**: `python loops/retrieval_loop/run_loop.py propose --hypothesis "..." --action "..." --root-cause "..."`
4. **Validate**: `python loops/retrieval_loop/run_loop.py gates --exp XXX` — runs all mandatory gates
5. **Approve**: Show results to user, await explicit approval
6. **Execute**: `python loops/retrieval_loop/run_loop.py approve --exp XXX --note "..."` — commits only after approval
7. **Measure**: Record metrics in experiment JSON
8. **Learn**: Move to archive; repeat

## Experiment naming

`exp-001/`, `exp-002/`, ... — sequential integers, no reset.

## State transitions

- `NEW`: Experiment directory created with `experiment.json`
- `OBSERVED`: Failure cards reviewed, root cause identified
- `HYPOTHESIS_CREATED`: Hypothesis written and plausible
- `VALIDATED`: All mandatory gates passed
- `AWAITING_APPROVAL`: Waiting for human sign-off
- `EXECUTED`: Change committed to repository
- `MEASURED`: Metrics recorded in `experiment.json`
- `ACCEPTED` / `REJECTED` / `ROLLED_BACK`: Terminal state
- `ARCHIVED`: Moved to `archive/`
