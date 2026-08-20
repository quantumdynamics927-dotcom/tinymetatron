# exp-005 Corpus Revision 3 — Freeze

**Frozen:** 2026-08-15
**Corpus revision:** 3
**Corpus hash:** `82ed267da70534f5`
**Scope:** Quantum-software niche corpus for CPU-only, 10M–20M parameter exp-005 training.

## Ingestion pipeline

Reused the `workers/corpus/` pipeline pattern:

1. `experiments/exp-005/corpus-rev3/scripts/stage_sources.py` — staged quantum-domain technical documentation (new) and copied frozen revision 2 sources.
2. `workers.corpus.validate` — validated rows (min/max length, JSON integrity).
3. `workers.corpus.dedupe` — exact + normalized + MinHash/LSH near-duplicate removal.
4. `workers.corpus.split` — source-disjoint 80/10/10 split with deterministic per-source cap (400).
5. `workers.corpus.version` — froze `MANIFEST.json` with hashes, category shares, and per-source provenance.

All file opens use `encoding="utf-8"`.

## Source categories and provenance

| Category | Rows (frozen) | Share | Target | Source | License |
|---|---|---|---|---|---|
| Quantum-domain technical | 15,652 | 24.81% | 45% | exp-004 frozen corpus (14,190) + PennyLane docs (924) + Qiskit tutorials docs (549) + Qiskit Textbook narrative (17) | Mixed / Apache-2.0 |
| Quantum-specific code | 41,924 | 66.45% | 30% | FROZEN from revision 2: qiskit-tutorials + pennylane + qiskit-textbook + own snippets | Apache-2.0; N/A |
| Tool-use traces | 2,832 | 4.49% | 12% | FROZEN from revision 2: synthetic | N/A |
| General English | 2,685 | 4.26% | 13% | FROZEN from revision 2: Project Gutenberg (7 texts, capped) | Public domain (US) |
| **Total** | **63,093** | **100%** | | | |

## New quantum-domain technical sources (revision 3)

| Source | Files/Notebooks | Rows (pre-cap) | License |
|---|---|---|---|
| PennyLane documentation (`doc/`) | 73 `.rst` files | 924 | Apache-2.0 |
| Qiskit tutorials markdown cells | 26 notebooks | 549 | Apache-2.0 |
| Qiskit Textbook narrative content | 12 `.md` files | 17 | Apache-2.0 |
| **Total new** | **111 source groups** | **1,490** | |

## Split summary

| Split | Rows | Share | Max source row share |
|---|---|---|---|
| train | 48,534 | 76.92% | 0.82% |
| val | 8,197 | 12.99% | 4.88% |
| hard_dev | 6,362 | 10.08% | 6.29% |

- **Source-disjoint:** all cross-split source overlaps = 0.
- **Text-disjoint:** all cross-split text overlaps = 0.
- **Per-source cap:** 400 rows (14 sources capped).
- **Pre-cap rows:** 68,110; **post-cap rows:** 63,093.
- **Max-source-share gate threshold:** 0.25 — all splits pass (max 6.29%).

## Distance from SCOPE.md targets

| Category | Target | Landed | Δ |
|---|---|---|---|
| Quantum-domain technical | 45% | 24.81% | −20.19 pp |
| Quantum-code | 30% | 66.45% | +36.45 pp |
| Tool traces | 12% | 4.49% | −7.51 pp |
| General English | 13% | 4.26% | −8.74 pp |

Only general-English is within ±10 pp of target if we consider the cap effect. Quantum-domain technical improved from 21.29% (rev2) to 24.81% but remains far below the 45% target.

## Why the quantum-domain share is still low

Revision 3 added ~1,490 new quantum-domain rows from documentation sources, but the frozen quantum-code base (41,924 rows) dominates the corpus. At the 400-row per-source cap:

- To reach 45% quantum-domain with 41,924 frozen code rows:
  - Required quantum-domain = 0.45 / 0.55 × 41,924 ≈ 34,268 rows
  - Currently have: 15,652
  - **Deficit: ~18,616 rows ≈ 47 new sources @ 400 cap each**

The general-English and tool-traces shares also dropped due to per-source capping of the synthetic and Gutenberg sources.

## Honest achievable ceiling assessment

**If we keep the frozen 41,924 quantum-code rows and 400-row cap:**
- Maximum achievable quantum-domain share ≈ 25–30% (adding ~10-20k more technical rows from docs)
- Tool traces and general English will stay suppressed by caps unless we increase their source diversity

**Options to reach 45% target:**
1. **Add ~47 more permissive-license quantum documentation sources** (Qiskit main docs, Cirq docs, QuTiP docs, IBM Quantum Learning, Q# docs, Braket docs, Xanadu Codebook, arXiv quant-ph, etc.)
2. **Relax the cap for specific categories** (but this violates the source-disjoint capped policy)
3. **Accept a code-heavy corpus** and update SCOPE.md targets to reflect reality (~25/65/5/5)

## Next steps

- **Decision needed:** add more quantum-domain documentation sources and create revision 4, **or** adjust SCOPE.md targets.
- If adding sources: prioritize Qiskit main documentation, Cirq documentation, QuTiP documentation, IBM Quantum Learning, Microsoft Q# docs, Amazon Braket docs.
- Final tokenizer training (16k byte-level BPE) on whichever corpus revision is approved.
- Model training and 3-set evaluation.

Revisions 1 (`bab31ed`) and 2 (`11c5d75`) remain untouched.