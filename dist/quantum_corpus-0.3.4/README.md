# quantum-corpus

**Version 0.3.4** — RAG pipeline and corpus tools for quantum research copilot.

A local-only corpus pipeline that ingests quantum research material from multiple
sources, applies PII redaction, splits records into train/val/test sets, builds
a versioned SQLite corpus, and constructs a BM25 + optional semantic hybrid
RAG index.

## Installation

```bash
pip install quantum-corpus          # core only (BM25 RAG, no ML deps)
pip install "quantum-corpus[advanced]"  # + sentence-transformers + PyMuPDF
pip install "quantum-corpus[dev]"   # + pytest
```

## Core Modules

| Module | Description | Dependencies |
|--------|-------------|-------------|
| `quantum_corpus.schema` | SQLite schema + source_identity | sqlite3 (stdlib) |
| `quantum_corpus.redact` | PII redaction (IBMid, keys, emails) | re (stdlib) |
| `quantum_corpus.split` | Train/val/test split by project | hashlib (stdlib) |
| `quantum_corpus.tokenize_count` | Token counting with HuggingFace tokenizers | tokenizers |
| `quantum_corpus.rag` | BM25 RAG index + schema-aware query expansion | — |
| `quantum_corpus.fusion` | BM25 + semantic hybrid fusion | quantum_corpus.rag, quantum_corpus.semantic |
| `quantum_corpus.structured` | Structured SQL query layer | sqlite3 (stdlib) |
| `quantum_corpus.answer` | Evidence-gated answer synthesis | quantum_corpus.rag, quantum_corpus.structured |
| `quantum_corpus.extract` | Ingest from repo dirs, IBM job zips, PDFs | fitz (optional) |
| `quantum_corpus.build` | End-to-end build orchestrator | all above |

## Quick Start

```python
from quantum_corpus import schema, rag

# Use existing corpus
db_path = schema.default_db_path()  # or set TMT_QUANTUM_CORPUS_DB
idx = rag.RAGIndex.load(db_path)

# Query
hits = idx.query("ibm_fez backend job OTOC", k=5)
for hit in hits:
    print(hit["snippet"][:120])
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TMT_QUANTUM_CORPUS_DB` | `quantum_corpus.db` | Path to corpus SQLite DB |
| `TMT_QUANTUM_JOBS_DB` | — | Path to IBM job structured DB |
| `TMT_DEPLOY_MODE` | `private-training` | `private-training` or `public-demo` |

## Corpus Schema

Each record has:
- `id`, `source_type`, `project`, `subdomain`, `doc_id`, `text`
- `split` (train/val/test), `token_count`
- `sensitivity` (public/internal/restricted/sensitive)
- `risk_tier` (public/standard/elevated/critical)
- `source_identity` (stable SHA-256 content hash)
- `provenance_url`, `source_license`

## License

MIT. See LICENSE file.
