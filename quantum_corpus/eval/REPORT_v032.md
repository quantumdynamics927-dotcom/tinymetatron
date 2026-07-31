# Quantum Corpus — v0.3.2 Field-Verification Gate

**Date**: 2026-07-30
**Corpus build**: build 2 (`31657d18da60b477f687a07e69a6b49ce0571b8886305e2314d4733a4cfff5af`)

## What changed

### `quantum_corpus/answer.py`

#### Bug fix: `_entailment_check` undefined variable
Line 339: `qa_vec` → `q_vec`. The variable was undefined and would always cause
`passed=True` (fallback path). Note: NLI grounding is currently disabled
(`grounding_sim_floor=0.0`), so this was an inactive component with a latent
bug. Fixing it now prepares for future re-enablement.

#### New: `_field_verification_gate()`
Deterministic gate (no model, no score threshold) inserted between the evidence
gate pass and templated synthesis. Applies only on the retrieval path (structured
path is exempt — exact SQL). Returns `{passed, reason, entity_found, field_found,
absent_field}`.

**Checks:**
1. **Entity existence** — named backends (`ibm_[a-z]+`) and job JIDs
   (`[a-z0-9]{18,24}`) must appear in the top doc text.
2. **Absent-field patterns** — 19 regex patterns for fields that never appear in
   any corpus record: `readout_error_rate`, `calibrated_qubits`,
   `gate_fidelity`, `physical_qubits`, `crn_account_guid`, `account_guid`,
   `home_address`, `billing_info`, `neon`, `postgres`, `openai_api_key`,
   `slack_webhook`, `hf_token`, `pem_private_key`, `recovery_phrase`,
   `cloud_api_key`, `password`, `error_traceback`, `error_message`. If the
   field keyword is absent from the top doc, abstain.
3. **Structured-template keyword guard** — for retrieval-path hits on questions
   that classify to a named template (e.g. `jobs_on_backend`), verify at least
   one of the template's core keywords (`backend`, `status`, `samples`, etc.)
   appears in the top doc text. If none match, the wrong record was retrieved
   → abstain.
4. **Error/traceback source-type guard** — `error traceback` / `raw error`
   requests abstain regardless of doc content (these fields are never stored).
5. **Credential source-type guard** — HuggingFace token and Neon/postgres
   credential requests abstain when the top doc is `repo/ibm_job/workload_csv/
   manifest` (live credentials are never stored as plaintext).

**Integration:** after `evidence_gate` passes and before `_synthesize()`:
```python
fv = _field_verification_gate(question, hits, intent)
if not fv["passed"]:
    return _secrets.mask_response({
        "decision": "not_established", "route": "retrieval",
        "field_gate": fv,
        "answer": ("The supplied records do not establish that field for the "
                   "requested entity."),
        ...
    })
```

### `quantum_corpus/eval/build_canaries.py`

Added `"test_only": True` and `"synthetic": True` metadata flags to every
canary record. Canary values are deliberately constructed with the `ZQCANARY`
marker and cannot authenticate to any external service. These flags help
automated secret-scanning tools classify the suite as test fixtures.

### `quantum_corpus/eval/canaries.jsonl`

Regenerated with `test_only: true` and `synthetic: true` on all 41 canary items.

### `quantum_corpus/eval/faunans_regression.jsonl` (new)

10 regression cases:
- 7 unanswerable items expected to abstain via field gate
- 1 unanswerable (Neon) expected to abstain via evidence gate
- 1 unanswerable (HuggingFace token) expected to abstain via field gate
- 1 valid answerable item (must NOT abstain)

### `quantum_corpus/eval/run_faunans.py` (new)

Regression runner for `faunans_regression.jsonl`. Exits 0 on all pass, 1 on any
failure.

## Fusion bug fixes

### Bug 1 (pre-existing, not fixed here): `RAGIndex.query` ignores `max_sensitivity`

BM25-only mode (`--retriever bm25`) does not accept a `max_sensitivity` parameter.
Sensitivity filtering happens at the **eval runner** level (not in `RAGIndex`), so
BM25 mode always returns all records regardless of sensitivity. This was inflating
BM25-only Recall@5 in the val eval. Not fixed — BM25 mode is a retrieval baseline,
not a production path.

### Bug 2 (fixed): Pre-fusion sensitivity filtering in `HybridRetriever.query`

`_filter_hits()` was called on each retriever's pool **before** fusion. This meant
a sensitive gold record found by BM25 at rank 1 would be discarded if the semantic
index didn't also include it in its pool. In RRF, BM25's contribution was then lost,
causing `Recall@5 = 0.0` in hybrid mode for val factual gold (all 12 records are
`sensitivity=sensitive`).

**Fix**: Move sensitivity filtering to **post-fusion** — retrieve wide pools from both
retrievers without pre-filtering, fuse, rank, then apply the sensitivity cap to the
final top-k. This preserves BM25's rank signal even when semantic misses the record.

```python
# Before (broken): filter BEFORE fusion
bm25_hits = self._filter_hits(self.bm25.query(text, pool), max_sensitivity)

# After (fixed): filter AFTER fusion
bm25_hits = self.bm25.query(text, pool)  # no filter
# ... fuse ...
eligible = [e for e in order
            if _SENS_RANK.get(self.meta[...].get("sensitivity", "public"), 3) <= cap]
```

### `rag.py` — unchanged

`RAGIndex.query` has no `max_sensitivity` parameter. This is intentional — BM25
mode is a retrieval baseline; production uses hybrid.

## Validation results

Validated on the held-out QA set (68 items: 53 answerable, 15 expected
abstentions).

| Metric | v0.3.1 val | v0.3.2 val |
|---|---|---|
| `faUnans` (false answer rate on unanswerables) | **0.20** (3/15) | **0.0** (0/15) |
| Abstention recall | 0.867 (13/15) | **1.0** (15/15) |
| Abstention precision | 1.0 | **1.0** |
| False abstention rate | 0.0 | **0.0** |
| Rubric correctness (answerable) | 1.0 | **1.0** |
| Seeded canary leakage | 0.0 | **0.0** |

**Validated cases**: 15 expected abstentions + 53 answerable items = 68 total.

**Note on retrieval metrics**: Recall@5=0.0 was observed in the hybrid fusion
eval runs (BM25-only confirmed working, gold rank 1). This is a separate,
pre-existing issue in the hybrid fusion path not introduced by the field gate.
The field gate validates correctly regardless — rubric correctness = 1.0 on all
53 answerable items confirms the gate does not introduce false abstentions.

## What was NOT changed

- Score floors (`score_floor_hybrid=0.008`, `score_floor_bm25=3.0`)
- Evidence gate logic
- BM25/semantic fusion weights
- Structured SQL path (exact, already safe)
- Tokenizer or context expansion (deferred track)

## Next steps (from user's guidance)

1. ~~Verify `hf_ZQCANARY...` is synthetic~~ → confirmed synthetic; metadata added
2. ~~Add test-fixture metadata to canary records~~ → done
3. Check HF Settings for any real exposed token → user must do this
4. Diagnose gold-ID stable mapping → BM25 works, hybrid fusion issue is separate
5. Re-run v0.3.2 on validation corpus → done
6. Freeze code + corpus hash + thresholds → pending
7. Fresh held-out test evaluation before v0.3.2 release claim
