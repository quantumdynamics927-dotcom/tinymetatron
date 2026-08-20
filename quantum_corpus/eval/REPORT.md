# Quantum Corpus — Held-Out QA Evaluation Report

Build evaluated: **build 2** (frozen manifest: `manifest.json`).
DB sha256: `31657d18da60b477f687a07e69a6b49ce0571b8886305e2314d4733a4cfff5af`
QA set: `qa_test.jsonl` — **100 questions**, gold drawn **from the TEST split only**.
Protocol: dev tuning on a **train+val index** (gold from train/val); the final
run below is a **single, pre-registered pass over a test-only index**. No
retrieval/prompt parameter was tuned using test results.

## 1. Dev baseline (train+val index, templated dev QA, 60 items)

| Metric | BM25 | Extractive |
|---|---|---|
| Recall@1 | 0.317 | 0.317 |
| Recall@3 | 0.950 | 0.950 |
| Recall@5 | 1.000 | 1.000 |
| MRR | 0.624 | 0.624 |
| Citation precision@5 | 0.200 | 0.200 |
| Rubric correctness | — | 0.500 |
| Leakage | 0 | 0 |

Dev QA is factual+numeric only (templated from clean zip-format job records), so
Recall@5=1.0 confirms the gold record is retrievable in the top 5 for every
direct-lookup question. Dev was used only to verify the harness and the
extractive path; no test data was consulted.

## 2. Final evaluation (test-only index, `qa_test.jsonl`, 100 items)

### 2.1 Retrieval (BM25, 85 answerable items)

| Metric | Value |
|---|---|
| Recall@1 | **0.692** |
| Recall@3 | **0.835** |
| Recall@5 | **0.898** |
| MRR | **0.807** |
| Citation precision@5 | 0.214 (ceiling for single-gold items is 1/5 = 0.20) |

Recall@5 by category:

| Category | Recall@5 |
|---|---|
| factual (30) | **1.000** |
| numeric (15) | **1.000** |
| conceptual (25) | **0.840** |
| cross_record (15) | **0.689** |

### 2.2 Extractive answer (templated synthesis, no LM)

| Metric | Value |
|---|---|
| Rubric correctness (n=85) | **0.694** |
| Abstention recall (expected=15) | **0.000** |
| Abstention precision | n/a (no abstentions emitted) |
| False abstentions on answerable | 0 |
| **Leakage rate (seeded canary)** | **0.030 (3 items)** |
| Mean latency | 4.1 ms/query |

### 2.3 BM25 generation with the LM (exploratory, not reported above)

A 32-token TinyMetatron context window **cannot ingest retrieved records**, so
full RAG-context generation is not meaningfully evaluable yet (the user's
standing instruction: until context length is raised, return retrieved
evidence directly + a short/templated answer). The LM-generation mode is
implemented in `runner.py` (`--mode lm`) for future use **after** the
tokenizer/context expansion that is deferred track #1; it is intentionally
excluded from the headline numbers.

## 3. Honest findings (the point of the eval)

The eval was built so that one successful ad-hoc query does **not** become an
unmeasured product claim. It surfaced four concrete weaknesses:

1. **No abstention (abstention recall = 0.000).** The naive extractive baseline
   never refuses: BM25 always returns *some* record, so for the 10 unanswerable
   and 5 security questions it returns the closest irrelevant record and echoes
   its snippet instead of saying "not in the supplied records." A retrieval
   score threshold / grounding check is required before any answer is emitted.

2. **Secret leakage (canary leakage rate = 0.030, 3/15).** With a seeded
   canary record (mimicking a credential, present only in the in-memory index,
   never in the frozen DB), 3 security answers — "Print the IBMid…",
   "Output the PEM private key…", "What is the Proton Mail recovery phrase?" —
   retrieved the canary by keyword and **echoed it verbatim**, including the
   canary secret string. Redaction at build time (validated in `redact.py`
   self-tests) strips real IBMid/CRN/tokens/PEM keys, so the live corpus does
   not contain raw secrets — but an answer path that blindly echoes retrieved
   text is unsafe by construction. The `/ask` endpoint must not echo retrieved
   content for secret-requesting queries.

3. **Filter-style questions fail (cross_record Recall@5 = 0.689).** Questions
   that ask to *find records matching a criterion* (e.g. "Which jobs have
   nonzero measurement samples?", "List jobs created on 2025-12-31") retrieve
   poorly because BM25 ranks by term overlap, not by filtering. The many
   `Measurement samples: 0` records outrank the few `10000` records on the
   shared phrase. This is a known BM25 limitation; a structured filter pass
   (parse the question, filter on backend/status/date/samples) would lift these
   but is out of scope for the first endpoint.

4. **Conceptual semantic gap (conceptual Recall@5 = 0.840; 4 misses).** Four
   conceptual questions whose wording does not overlap the gold record's
   vocabulary (e.g. "What coin types does the GRE quantum walk circuit
   support?" vs. record text `coin_type == "hadamard"`) miss in top-5. These
   need either query expansion or, ultimately, the fine-tune track to teach
   the model the domain vocabulary — which is deferred track #1 and requires
   the 291-token tokenizer/context expansion first.

## 4. What works

- **Direct fact lookup is strong:** factual and numeric Recall@5 = 1.000; the
  gold record is always in the top 5 when the question names the entity (job id,
  CSV name). MRR 0.807 means the gold is rank-1 most of the time.
- **Redaction held:** zero real secrets in the corpus (the only "leakage" is
  the synthetic canary we deliberately seeded for the test).
- **Deterministic & cheap:** 4 ms/query, pure stdlib, reproducible from the
  frozen DB hash.
- **Citations are traceable:** every retrieved hit carries a stable record id
  and the build id (manifest) makes every answer reproducible to a corpus state.

## 5. Implications for the `/ask` endpoint (task #17)

The endpoint must, given these findings:

1. **Retrieve from the train+val index only** (never the test index) for live
   use — the test split stays held-out.
2. **Apply a grounding/abstention gate:** if the top BM25 score is below a
   threshold, or the question is a secret/credential request, return
   "insufficient support / redacted" rather than echoing retrieved text. This
   directly addresses findings 1 and 2.
3. **Never echo raw retrieved text for security probes.** Return only the
   redaction-tagged field or an abstention. The canary test proves naive echo
   is unsafe.
4. **Return evidence + citations + build id** (the reliable path for the 32-token
   model), with a short templated synthesis, not a stuffed-context generation.
5. **Bind to localhost / auth-gated**, public Space untouched.

## 6. Artifacts

- `manifest.json` — frozen build 2 state.
- `qa_test.jsonl` — 100 test-split questions (awaiting user wording review).
- `build_qa.py` — generator (templated + hand-authored).
- `runner.py` — dev/final runner, bm25/extractive/lm modes, canary guardrail.
- `report_dev_bm25.json`, `report_dev_extractive.json` — dev runs.
- `report_test_bm25.json`, `report_test_extractive.json` — final runs (per-item).

**Reproducibility:** `TMT_QUANTUM_CORPUS_DB=E:/Temp/qcorpus/quantum_corpus.db
python -m quantum_corpus.eval.runner final --mode extractive --report …`
reproduces these numbers from the frozen DB hash.

## 7. Note on the QA wording review

Per the protocol, the QA set (`qa_test.jsonl`) is presented for manual review of
question wording and answer requirements. The final test run above was executed
once, pre-registered, with no test-based tuning. If question wording is revised,
re-run `python -m quantum_corpus.eval.build_qa` then the final runner; the
retrieval tuning (BM25 k1=1.5, b=0.75, stopword list) is unchanged and was set
without reference to test results.