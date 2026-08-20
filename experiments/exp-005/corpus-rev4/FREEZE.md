# exp-005 Corpus Revision 4 — Freeze

**Frozen:** 2026-08-16
**Corpus revision:** 4
**Corpus hash:** `4f9d10d3dfc9ff45`
**Scope:** Quantum-software niche corpus for CPU-only, 10M–20M parameter exp-005 training.

## Ingestion pipeline

Reused the `workers/corpus/` pipeline pattern:

1. `experiments/exp-005/corpus-rev4/scripts/stage_sources.py` — staged Qiskit docs, Cirq docs, QuTiP v5 docs and loaded frozen rev3 base.
2. `workers.corpus.validate` — validated rows (min/max length, JSON integrity).
3. `workers.corpus.dedupe` — exact + normalized + MinHash/LSH near-duplicate removal.
4. `workers.corpus.split` — source-disjoint 80/10/10 split with deterministic per-source cap (400).
5. `workers.corpus.version` — froze `MANIFEST.json` with hashes, category shares, and per-source provenance.

All file opens use `encoding="utf-8"`.

## Source categories and provenance

| Category | Rows (frozen) | Share | Target | Source | License |
|---|---|---|---|---|---|
| Quantum-domain technical | 19,704 | 29.3% | 45% | exp-004 frozen base (14,190) + PennyLane docs (924) + Qiskit tutorials (549) + Qiskit Textbook (17) + **Qiskit docs (193) + Cirq docs (1,745) + QuTiP v5 (2,327)** | Mixed / Apache-2.0 / BSD-3 |
| Quantum-specific code | 41,924 | 62.4% | 30% | FROZEN from revision 2/3: qiskit-tutorials + pennylane + qiskit-textbook + own snippets | Apache-2.0 |
| Tool-use traces | 2,832 | 4.2% | 12% | FROZEN from revision 2: synthetic | N/A |
| General English | 2,685 | 4.0% | 13% | FROZEN from revision 2: Project Gutenberg (7 texts, capped) | Public domain (US) |
| **Total** | **67,145** | **100%** | | | |

## New quantum-domain technical sources (revision 4)

| Source | Files | Rows (pre-cap) | License |
|---|---|---|---|
| Qiskit main documentation (`docs/` RST) | 31 | 193 | Apache-2.0 |
| Cirq documentation (notebooks + markdown) | 88 | 1,745 | Apache-2.0 |
| QuTiP v5 documentation (RST) | 76 | 2,327 | BSD-3-Clause |
| **Total new** | **195** | **4,265** | |

Rev3 had 15,652 quantum-domain technical rows; rev4 has 19,704 (+4,052 net after dedupe).

## Split summary

| Split | Rows | Share | Max source row share |
|---|---|---|---|
| train | 53,562 | 79.8% | 0.75% |
| val | 5,809 | 8.7% | 6.89% |
| hard_dev | 7,774 | 11.6% | 5.15% |

- **Source-disjoint:** all cross-split source overlaps = 0.
- **Text-disjoint:** all cross-split text overlaps = 0.
- **Per-source cap:** 400 rows — no sources were capped (new docs are small per-source).
- **Pre-cap rows:** 67,145; **post-cap rows:** 67,145.
- **Max-source-share gate threshold:** 0.25 — all splits pass (max 6.89%).

## Distance from SCOPE.md targets

| Category | Target | Rev3 | Rev4 | Δ |
|---|---|---|---|---|
| Quantum-domain technical | 45% | 24.81% | **29.3%** | +4.5 pp |
| Quantum-code | 30% | 66.45% | 62.4% | −4.0 pp |
| Tool traces | 12% | 4.49% | 4.2% | −0.3 pp |
| General English | 13% | 4.26% | 4.0% | −0.3 pp |

Rev4 adds Cirq + QuTiP + Qiskit docs (~4k new technical rows). The quantum-domain share improved from 24.8% to 29.3%. The quantum-code share dropped slightly (62.4%) because the new technical rows dilute it without changing the frozen code count.

## Why still below 45%

The frozen quantum-code base (41,924 rows) is 62% of the corpus. At the 400-row per-source cap, each new technical doc source contributes at most 400 rows:

- To reach 45% quantum-domain with 41,924 frozen code rows:
  - Required quantum-domain = 0.45 / 0.55 × 41,924 ≈ 34,268 rows
  - Currently have: 19,704
  - **Deficit: ~14,564 rows ≈ ~37 more new technical sources @ 400 cap each**

Qiskit docs yielded only 193 rows because most content is API reference (RST with heavy markup). Cirq notebooks (88 files) yielded 1,745 rows. QuTiP (76 RST files) yielded 2,327 rows — the densest source per-file.

## Next steps

- **Decision needed:** add more quantum-domain documentation sources (Q# docs, Amazon Braket, Xanadu Codebook, IBM Quantum Learning, arXiv papers), **or** accept the achievable ceiling (~30% quantum-domain with current policy).
- The 41,924 frozen quantum-code rows are the dominant factor. To push quantum-domain above 40%, would need ~15k more technical rows from ~37 additional distinct sources.
- Final tokenizer training (16k byte-level BPE) on whichever corpus revision is approved.
- Model training and 3-set evaluation.

Revisions 1 (`bab31ed`), 2 (`11c5d75`), and 3 (`98a3344`) remain untouched.
