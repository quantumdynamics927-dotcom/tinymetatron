:construction: **DRAFT — awaiting human approval before any corpus/tokenizer/training work begins.**

# exp-005 Scope Document

## Tools/Skills Definition (Step 0 — resolved)

**Primary corpus category:** code capability — general programming plus own-domain code in Python, TypeScript, PowerShell, and Bash.

**Layered on top:** a modest function-calling / tool-use trace format (structured invocations and results).

**Explicitly out of scope for exp-005:** general instruction-following / chat-style compliance. Revisit in a later experiment.

## Tokenizer Decision (Step 1 — complete, byte-fallback probe done)

### Sample sourcing provenance

| Sample | Source | How obtained | License |
|---|---|---|---|
| General English | Project Gutenberg: *Pride and Prejudice* by Jane Austen | Fetched as plain text from Project Gutenberg (`https://www.gutenberg.org/files/1342/1342-0.txt`) | Public domain in the US |
| Quantum-domain | `experiments/exp-004/dedupe/deduped.jsonl` | Read-only random sample of 1,500 rows (seed 42) from `text` field | Same as exp-004 corpus (already frozen, internal) |
| Code | TinyMetatron's own `.py` files under `workers/`, `loops/`, `tests/`, `tools/` | Read-only concatenation of 42 source files | N/A — own codebase |
| Tool-use traces | Synthetic | 350 trace blocks generated from real CLI/function signatures in this repo (`compute_ce`, `quantum_corpus_freeze`, `quantum_corpus_validate`, corpus workers) | N/A — own codebase |

### Plain BPE probe results (combined eval holdout = 20%, 282,941 chars, 26,731 words)

| Vocab size | Train time (s) | Tokens/word | UNK count | UNK rate |
|---|---|---|---|---|
| 8,000 | 0.454 | 2.9385 | 19 | 0.000242 |
| 16,000 | 0.568 | 2.7643 | 19 | 0.000257 |
| 24,000 | 0.682 | 2.7158 | 19 | 0.000262 |
| 32,000 | 0.735 | 2.6945 | 19 | 0.000264 |

### Byte-level BPE probe results (same eval holdout)

Byte-level pre-tokenizer (GPT-2/RoBERTa style) — UNK rate is zero by construction.

| Vocab size | Train time (s) | Tokens/word |
|---|---|---|
| 8,000 | 0.645 | 3.7043 |
| 16,000 | 0.747 | 3.5212 |
| 24,000 | 0.803 | 3.4689 |
| 32,000 | 0.928 | 3.4389 |

### Recommendation

**Recommended vocab size: 16,000.**

Both probes show a sharp compression gain from 8k to 16k, then diminishing returns:
- Plain BPE: 2.94 → 2.76 tokens/word (–6.0%), then only 2.72/2.69.
- Byte-level BPE: 3.70 → 3.52 tokens/word (–4.9%), then only 3.47/3.44.

The 16k point captures most of the benefit while keeping embedding lookup fast and model memory small. The final tokenizer will use **byte-level BPE** to guarantee zero UNK on arbitrary Unicode/code symbols.

## Corpus Composition Plan (Step 2 — proposal, awaiting approval)

Based on the tools/skills definition (code capability primary, function-calling layered on top, general instruction-following out of scope).

### Compute constraint

- **CPU-only on current machine, few days maximum wall-clock** for a full training run.
- No GPU access assumed unless explicitly stated later.
- Target total corpus: **20,000–60,000 rows**.
- Target architecture: **10M–20M parameters** (not the earlier 20M–60M hypothesis, which assumed more compute).

### Row-count targets by category (against 20k–60k total)

| Category | Target share | Row target range | Description | Source approach |
|---|---|---|---|---|
| Code — general + own-domain | 50–60% | 10,000–36,000 rows | Python, TypeScript, PowerShell, Bash | Public permissive-only (MIT / Apache-2.0 / BSD) code subsets; **exclude GPL/AGPL entirely**. If using The Stack / CodeParrot / similar, filter to their permissive-license subsets and record the exact filter in the manifest. TinyMetatron own code (`workers/`, `loops/`, `tests/`, `tools/`) is a **flavor/domain-alignment signal**, not a volume driver — keep it as a minority within this category. |
| Quantum-domain technical | 20–30% | 4,000–18,000 rows | Quantum computing, IBM Quantum, Qiskit, error correction, hardware docs | Reuse exp-004 source corpus under its existing provenance; supplement with new source-disjoint IBM/Qiskit docs. |
| Function-calling / tool-use traces | 10–20% | 2,000–12,000 rows | Structured JSON invocations and results | Synthetic/templated from actual CLI and function signatures in this repo. |
| General English | 5–10% | 1,000–6,000 rows | Prose coverage only; not instruction-following | Public-domain texts (Project Gutenberg), clean web archives, or own writing. |

### License and sourcing rules

- **Permissive only:** MIT, Apache-2.0, BSD. No GPL/AGPL in the code corpus.
- **Filtered public corpora only:** no wholesale unfiltered The Stack / CodeParrot; record exact license filter in `MANIFEST.json`.
- **No AGI-model content or imports.**
- **No instruction-following/chat data** for this experiment.
- **Provenance per source:** source URL/identifier, license, retrieval date, content hash — same discipline as exp-004.
- **Source cap + source-disjoint splits:** cap per source before splitting; no source leaks across train/val/hard_dev.

### Tool-use trace format

Structured JSON only:
```json
{"tool": "name", "args": {...}, "result": {...}}
```
Not natural-language phrasing. Derived from real function/CLI signatures in the repo.

## Target Architecture Size (proposed, not final)

Under CPU-only / few-days constraint: **10M–20M parameters** as the realistic exp-005 scale. Final choice depends on actual corpus token count and measured steps/second once the tokenizer and corpus are fixed.

## Open Questions for Human Review

1. **Approve 16k vocab size with byte-level BPE?**
2. **Approve 20k–60k total rows and the per-category row targets above?**
3. **Confirm CPU-only / few-days compute budget?** (If you have GPU access, say so — it changes the scale.)
4. **Approve permissive-only code sourcing rule?** Any preferred public code dataset (e.g., a specific permissive subset of The Stack) or domain emphasis?
