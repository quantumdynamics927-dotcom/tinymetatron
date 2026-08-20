# exp-005 Corpus Revision 2 — Freeze

**Frozen:** 2026-08-15
**Corpus revision:** 2
**Corpus hash:** `455e4ed9885140ae`
**Scope:** Quantum-software niche corpus for CPU-only, 10M–20M parameter exp-005 training.

## Ingestion pipeline

Reused the `workers/corpus/` pipeline pattern:

1. `experiments/exp-005/corpus-rev2/scripts/stage_sources.py` — staged four source categories into `raw/`.
2. `workers.corpus.validate` — validated rows (min/max length, JSON integrity).
3. `workers.corpus.dedupe` — exact + normalized + MinHash/LSH near-duplicate removal.
4. `workers.corpus.split` — source-disjoint 80/10/10 split with deterministic per-source cap.
5. `workers.corpus.version` — froze `MANIFEST.json` with hashes, category shares, and per-source provenance.

All file opens use `encoding="utf-8"`.

## Source categories and provenance

| Category | Rows (frozen) | Share | Target | Source | License |
|---|---|---|---|---|---|
| Quantum-domain technical | 14,190 | 21.29% | 45% | exp-004 frozen corpus (`train`+`val`+`hard_dev`) | Mixed (see exp-004 `MANIFEST.json`) |
| Quantum-specific code | 41,981 | 62.99% | 30% | `qiskit-tutorials` (503 rows) + `pennylane` (49,374 rows pre-cap) + `qiskit-textbook` (2,673 rows) + own snippets (2,854 rows) | Apache-2.0 (public repos); N/A (own) |
| Tool-use traces | 4,223 | 6.34% | 12% | Synthetic structured JSON from Qiskit / IBM Runtime / PennyLane / TinyMetatron signatures | N/A |
| General English | 6,256 | 9.39% | 13% | Project Gutenberg (7 public-domain texts) | Public domain (US) |
| **Total** | **66,650** | **100%** | | | |

## Public code sources (permissive only)

| Repo | License | Rows contributed (pre-cap) | Note |
|---|---|---|---|
| `qiskit/qiskit-tutorials` | Apache-2.0 | 503 | 26 notebooks + 1 `.py` file |
| `PennyLaneAI/pennylane` | Apache-2.0 | 49,374 | 677 `.py` files; skipped `tests/`, `doc/`, `.github/`, `__pycache__` |
| `qiskit-community/qiskit-textbook` | Apache-2.0 | 2,673 | 143 notebooks + 14 `.py` files |

## Split summary

| Split | Rows | Share | Max source row share |
|---|---|---|---|
| train | 49,928 | 74.91% | 2.40% |
| val | 10,502 | 15.76% | 11.43% |
| hard_dev | 6,220 | 9.33% | 9.65% |

- **Source-disjoint:** all cross-split source overlaps = 0.
- **Text-disjoint:** all cross-split text overlaps = 0.
- **Per-source cap:** 1,200 rows (two sources capped: `gutenberg:2701` excluded 524 rows; `gutenberg:98` excluded 182 rows).
- **Pre-cap rows:** 67,356; **post-cap rows:** 66,650.
- **Max-source-share gate threshold:** 0.25 — all splits pass.

## Distance from SCOPE.md targets

| Category | Target | Landed | Δ |
|---|---|---|---|
| Quantum-domain technical | 45% | 21.29% | −23.71 pp |
| Quantum-code | 30% | 62.99% | +32.99 pp |
| Tool traces | 12% | 6.34% | −5.66 pp |
| General English | 13% | 9.39% | −3.61 pp |

Only quantum-code and general-English are within the requested ±10 percentage-point band. Quantum-domain technical is far below target, and quantum-code is far above.

## Why the quantum-domain share dropped

Revision 2 kept the same 14,190-row quantum-domain base from exp-004 but added large quantum-code sources (especially PennyLane). The absolute quantum-domain row count is unchanged from revision 1; the share dropped because the total corpus grew.

## Honest achievable ceiling assessment

Adding PennyLane and Qiskit Textbook dramatically increases quantum-code volume. Reaching the 45/30/12/13 target now requires either:

1. **More quantum-domain technical text** — either by relaxing the exp-004-only constraint and supplementing with new IBM/Qiskit/PennyLane docs, or by accepting a smaller quantum-code share by subsampling the available code sources.
2. **Accepting the current composition** as a **quantum-code-heavy** corpus and updating `SCOPE.md` targets to reflect reality.

If the policy stays "permissive-only, qiskit-tutorials preferred for code, exp-004 for domain text," the achievable honest composition is closer to **20–25% quantum-domain / 60–65% quantum-code / 6–12% traces / 8–13% English**.

## Next steps (not started)

- **Decision needed:** adjust `SCOPE.md` targets to match revision 2's honest composition, **or** add more quantum-domain sources and create revision 3.
- Final tokenizer training (16k byte-level BPE) on whichever corpus revision is approved.
- Model training and 3-set evaluation.

Revision 1 (`bab31ed`) remains untouched.
