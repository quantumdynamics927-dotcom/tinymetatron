# exp-005 Corpus — Source Intake Checklist (Updated post-revision 3)

**Goal:** Add quantum-domain technical sources to reach ~45% share while keeping all 41,924 quantum-code rows frozen from revision 2/3.

**Constraints:**
- Per-source cap: 400 rows (applied before split)
- Max source share gate: ≤ 0.25 (opt-in)
- Source-disjoint splits (no source leaks across train/val/hard_dev)
- Permissive licenses only (Apache-2.0, MIT, BSD, CC-BY, public domain)
- Do not modify frozen exp-004 corpus or revision 1/2/3

---

## Current State (Revision 3 Frozen)

| Category | Rows | Share | Target |
|---|---|---|---|
| Quantum-domain technical | 15,652 | 24.81% | 45% |
| Quantum-specific code | 41,924 | 66.45% | 30% |
| Tool-use traces | 2,832 | 4.49% | 12% |
| General English | 2,685 | 4.26% | 13% |
| **Total** | **63,093** | **100%** | |

**Deficit to 45%:** Need ~18,616 more technical rows → **47 new sources @ 400 rows each** (with frozen 41,924 code rows)

---

## Revision 3 — COMPLETED Sources

| # | Source | License | Rows Added | Status |
|---|---|---|---|---|
| 1 | **PennyLane Documentation** (doc/*.rst) | Apache-2.0 | 924 | ✅ Done (73 files) |
| 2 | **Qiskit Tutorials Markdown** (notebook markdown cells) | Apache-2.0 | 549 | ✅ Done (26 notebooks) |
| 3 | **Qiskit Textbook Narrative** (content/*.md) | Apache-2.0 | 17 | ✅ Done (12 files) |
| **Total new** | | | **1,490** | |

---

## Candidate Sources for Revision 4 — Quantum-Domain Technical Text

| # | Source | License | Est. Rows (pre-cap) | Access Method | Status |
|---|---|---|---|---|---|
| 4 | **Qiskit Main Documentation** (qiskit/qiskit docs/) | Apache-2.0 | ~5,000+ | GitHub: qiskit/qiskit (docs/) | ⬜ Not started |
| 5 | **Cirq Documentation** (quantumlib/Cirq) | Apache-2.0 | ~2,000+ | GitHub: quantumlib/Cirq (docs/) | ⬜ Not started |
| 6 | **QuTiP Documentation** (qutip/qutip) | BSD-3-Clause | ~1,500+ | GitHub: qutip/qutip (doc/) | ⬜ Not started |
| 7 | **IBM Quantum Learning** | Need verification | ~1,000+ | GitHub: qiskit-community/ibm-quantum-learning | ⬜ Need license check |
| 8 | **Microsoft Q# Documentation** | MIT | ~1,000+ | GitHub: microsoft/qsharp-runtime (docs/) | ⬜ Not started |
| 9 | **Amazon Braket Documentation** | Apache-2.0 | ~1,000+ | GitHub: aws/amazon-braket-sdk-python (docs/) | ⬜ Not started |
| 10 | **Xanadu Quantum Codebook** | Apache-2.0 | ~500+ | GitHub: XanaduAI/quantum-codebook | ⬜ Not started |
| 11 | **arXiv Quantum Physics Papers** (quant-ph) | Varies (mostly CC-BY) | ~unlimited | arXiv API bulk download | ⬜ Need license filtering |
| 12 | **University Course Materials** (MIT, Stanford, etc.) | CC-BY / custom | ~500-2,000 each | GitHub course repos | ⬜ Need discovery |

---

## License Compatibility Notes

| License | Compatible? | Notes |
|---|---|---|
| Apache-2.0 | ✅ Yes | Used by Qiskit, PennyLane, Cirq, Braket, Xanadu |
| MIT | ✅ Yes | Q#, some university materials |
| BSD-3-Clause | ✅ Yes | QuTiP |
| CC-BY 4.0 | ✅ Yes | arXiv (most), some course materials |
| CC-BY-SA 4.0 | ⚠️ Conditional | Share-alike — can use but must propagate; prefer CC-BY |
| Public Domain | ✅ Yes | Gutenberg (already used) |
| Proprietary / No License | ❌ No | Skip |

---

## Intake Process Per Source

For each candidate source:

1. **License Verification** — confirm permissive license file in repo or documented policy
2. **Access & Extraction** — clone repo or download docs; extract text (markdown, rst, html → plain text)
3. **Chunking** — split into ~20-100k char rows (matching validation pipeline)
4. **Source ID Assignment** — unique `source_id` per document/repo section
5. **Provenance Record** — URL, license, commit hash, extraction date
6. **Staging** — write to `corpus-rev4/raw/quantum_domain_technical.jsonl`
7. **Validation** — run `workers.corpus.validate` (length, syntax, quality)
8. **Deduplication** — run `workers.corpus.dedupe` (exact + normalized + MinHash/LSH)
9. **Split** — run `workers.corpus.split` (source-disjoint, 400 cap)
10. **Version** — run `workers.corpus.version` (manifest with categories, sources[])

---

## Priority Order (High Yield / Low Effort)

1. **Qiskit Main Documentation** — Apache-2.0, largest volume, core ecosystem
2. **Cirq Documentation** — Apache-2.0, Google-backed, substantial technical depth
3. **QuTiP Documentation** — BSD-3, mature library, different paradigm (quantum optics)
4. **Microsoft Q# Docs** — MIT, Microsoft-backed, quantum programming language focus
5. **Amazon Braket Docs** — Apache-2.0, AWS-backed, hardware-agnostic content
6. **Xanadu Quantum Codebook** — Apache-2.0, tutorial-style technical content
7. **IBM Quantum Learning** — if license permits
8. **arXiv quant-ph** — high volume but needs license filtering per paper

---

## Target Calculation (with frozen 41,924 code rows)

| Source | Est. Rows (capped @ 400) | Cumulative Technical | Total Corpus | Technical % |
|---|---|---|---|---|
| Base (rev3) | 15,652 | 15,652 | 63,093 | 24.8% |
| + Qiskit Main Docs | 400 | 16,052 | 63,493 | 25.3% |
| + Cirq Docs | 400 | 16,452 | 63,893 | 25.7% |
| + QuTiP Docs | 400 | 16,852 | 64,293 | 26.2% |
| + Q# Docs | 400 | 17,252 | 64,693 | 26.7% |
| + Braket Docs | 400 | 17,652 | 65,093 | 27.1% |
| + Xanadu Codebook | 400 | 18,052 | 65,493 | 27.6% |
| **+ 40 more sources** | **16,000** | **34,052** | **81,493** | **41.8%** |
| **+ 47 more sources** | **18,800** | **36,852** | **84,293** | **43.7%** |
| **+ 52 more sources** | **20,800** | **38,852** | **86,293** | **45.0%** |

**Need ~52 more distinct technical sources** at 400 cap each to reach 45%.

---

## Next Actions for Revision 4

1. [ ] Clone/access Qiskit main repo (qiskit/qiskit) for documentation
2. [ ] Clone/access Cirq repo (quantumlib/Cirq) for documentation
3. [ ] Clone/access QuTiP repo (qutip/qutip) for documentation
4. [ ] Write extraction scripts for RST/Sphinx docs (common pattern)
5. [ ] Stage into `experiments/exp-005/corpus-rev4/raw/`
6. [ ] Run validation → dedupe → split → version pipeline
7. [ ] Verify manifest: categories, source shares, disjointness
8. [ ] Update SCOPE.md if composition lands within ±10pp bands