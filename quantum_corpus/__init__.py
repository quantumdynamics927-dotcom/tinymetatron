"""
quantum_corpus
==============
Private, local-only corpus pipeline that turns the user's quantum research
(4 local repos + real IBM Quantum hardware job exports + the wormhole
experiment manifest) into a versioned, metadata-tagged, split-aware corpus
for TinyMetatron's domain-adaptation + RAG layer.

This package NEVER touches metatron.db and is NEVER uploaded to the public
Hugging Face Space. All output lives in a separate ``quantum_corpus.db`` next
to this module (overridable via the ``TMT_QUANTUM_CORPUS_DB`` env var) and a
plain-text RAG index. The public Space's read-only cybersecurity demo is
unaffected.

Two-track use (see the two-layer roadmap):
  * RAG track        : structured records (IBM jobs, manifest, repo corpus)
                      retrieved at query time -> ``quantum_corpus.rag``.
  * Fine-tune track  : prose-only records (docs/READMEs/docstrings/research
                      text) used for domain adaptation -> the ``train`` split.

Per the user's decisions (2026-07-29):
  * TMT_Quantum_Vault- is the user's own repo (GPLv3 does not constrain the
    copyright holder) -> usable for both tracks.
  * "Include everything, redact identifiers" -> no subdirs are excluded on
    sensitivity grounds; the redaction layer strips IBM account identifiers
    and credential token values, but keeps names/emails/research content.
  * Scope: all four repos + E:\\Descargas IBM jobs & manifest.

See [[verify-beyond-selftests]]: the whole pipeline is validated end-to-end
locally (ingest -> redact -> schema -> split -> token count -> RAG smoke)
before anything is wired toward the Space.
"""

from __future__ import annotations

__all__ = ["schema", "redact", "extract", "split", "tokenize_count", "rag", "build"]