"""
quantum_corpus.eval
===================
Held-out QA evaluation for the frozen quantum corpus (build 2).

Contents:
  * ``manifest.json``  — frozen build state (DB sha256, splits, redaction/BM25 params).
  * ``build_qa.py``     — generates ``qa_test.jsonl`` from TEST-split gold records.
  * ``qa_test.jsonl``   — 100 questions, 6 categories, gold ids from test split only.
  * ``runner.py``       — dev (train/val index) + final (test-only index) evaluation.

Nothing here touches the public HF Space. The corpus DB is private/local.
"""