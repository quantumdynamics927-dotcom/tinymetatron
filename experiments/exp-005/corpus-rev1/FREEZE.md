# exp-005 Corpus Revision 1 — Freeze

**Frozen:** 2026-08-14
**Corpus revision:** 1
**Corpus hash:** `338ef735e1e36438`
**Scope:** Quantum-software niche corpus for CPU-only, 10M–20M parameter exp-005 training.

## Ingestion pipeline

Reused the `workers/corpus/` pipeline pattern:

1. `experiments/exp-005/corpus-rev1/scripts/stage_sources.py` — staged four source categories into `raw/`.
2. `workers.corpus.validate` — validated rows (min/max length, JSON integrity).
3. `workers.corpus.dedupe` — exact + normalized + MinHash/LSH near-duplicate removal.
4. `workers.corpus.split` — source-disjoint 80/10/10 split with deterministic per-source cap.
5. `workers.corpus.version` — froze `MANIFEST.json` with hashes, category shares, and per-source provenance.

All file opens use `encoding="utf-8"`.

## Source categories and provenance

| Category | Rows (frozen) | Share | Source | License | Provenance |
|---|---|---|---|---|---|
| Quantum-domain technical | 14,190 | 72.38% | exp-004 frozen corpus (`train.jsonl` + `val.jsonl` + `hard_dev.jsonl`) | Mixed (see exp-004 `MANIFEST.json`) | `experiments/exp-004/corpus` |
| Quantum-specific code | 2,338 | 11.92% | `qiskit/qiskit-tutorials` notebooks + `.py` file + TinyMetatron own quantum/Qiskit `.py` snippets | Apache-2.0 (Qiskit); N/A (own) | `https://github.com/Qiskit/qiskit-tutorials` + own repo paths |
| Tool-use traces | 2,279 | 11.62% | Synthetic structured JSON from Qiskit / IBM Runtime / TinyMetatron signatures | N/A | `experiments/exp-005/corpus-rev1/scripts/stage_sources.py` |
| General English | 800 | 4.08% | Project Gutenberg #1342: *Pride and Prejudice* | Public domain (US) | `https://www.gutenberg.org/files/1342/1342-0.txt` |
| **Total** | **19,607** | **100%** | | | |

## Split summary

| Split | Rows | Share | Max source row share |
|---|---|---|---|
| train | 14,924 | 76.12% | 5.36% |
| val | 1,712 | 8.73% | 23.36% |
| hard_dev | 2,971 | 15.15% | 15.15% |

- **Source-disjoint:** all cross-split source overlaps = 0.
- **Text-disjoint:** all cross-split text overlaps = 0.
- **Per-source cap:** 800 rows (one source capped: `gutenberg:1342`, 385 rows excluded).
- **Pre-cap rows:** 19,992; **post-cap rows:** 19,607.
- **Max-source-share gate threshold:** 0.25 — all splits pass.

## Deviation from SCOPE.md target shares

The SCOPE.md target was 45/30/12/13. The landed shares are 72/12/12/4 because `qiskit/qiskit-tutorials` plus own snippets did not provide enough quantum-code volume to reach 30%. Per the approved guardrail, no additional public code sources (The Stack, CodeParrot, etc.) were pulled in; the freeze reports the honest achieved composition.

## Next steps (not started)

- Final tokenizer training (16k byte-level BPE) on the frozen corpus.
- Model training and 3-set evaluation.

Do not change the frozen split files without creating a new corpus revision.
