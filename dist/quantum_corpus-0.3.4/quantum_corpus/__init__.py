"""
quantum_corpus
==============
RAG pipeline and corpus tools for quantum research copilot.

This package provides:
  - ``schema``    : SQLite corpus schema with stable source_identity
  - ``redact``    : PII redaction (IBMid, keys, emails, secrets)
  - ``split``     : Deterministic train/val/test split by project
  - ``rag``       : BM25 RAG index with schema-aware query expansion
  - ``fusion``    : BM25 + optional semantic hybrid retrieval
  - ``structured``: Structured SQL query layer over corpus
  - ``answer``    : Evidence-gated answer synthesis
  - ``extract``    : Ingest from repo dirs, IBM job zips, PDFs

Corpus data lives in a SQLite database whose path is controlled by the
``TMT_QUANTUM_CORPUS_DB`` environment variable (default: ``quantum_corpus.db``
beside the module). No data is shipped with the package.
"""

from __future__ import annotations

__all__ = [
    "schema",
    "redact",
    "split",
    "rag",
    "fusion",
    "structured",
    "answer",
    "extract",
]
