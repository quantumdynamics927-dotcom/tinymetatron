:construction: **DRAFT — awaiting human approval before any corpus/tokenizer/training work begins.**

# exp-005 Scope Document

## Tools/Skills Definition (Step 0 — resolved)

**Primary corpus category:** code capability — general programming plus own-domain code in Python, TypeScript, PowerShell, and Bash.

**Layered on top:** a modest function-calling / tool-use trace format (structured invocations and results).

**Explicitly out of scope for exp-005:** general instruction-following / chat-style compliance. Revisit in a later experiment.

## Tokenizer Decision (Step 1 — complete, byte-fallback probes done)

### Sample sourcing provenance

| Sample | Source | How obtained | License |
|---|---|---|---|
| General English | Project Gutenberg: *Pride and Prejudice* by Jane Austen | Fetched as plain text from Project Gutenberg (`https://www.gutenberg.org/files/1342/1342-0.txt`) | Public domain in the US |
| Quantum-domain technical | `experiments/exp-004/dedupe/deduped.jsonl` | Read-only weighted sample (seed 42) matching target share from `text` field | Same as exp-004 corpus (already frozen, internal) |
| Quantum-specific code | `qiskit/qiskit-tutorials` | Code cells extracted from official tutorial notebooks + any `.py` files | Apache-2.0 for Qiskit tutorials |
| Tool-use traces | Synthetic, quantum-tool-specific | 1,200 trace blocks generated from Qiskit / IBM Quantum / TinyMetatron CLI and function signatures | N/A — own codebase |

### Byte-level BPE probe A — initial broad mix (for reference)

Sample mix roughly equal by available raw size: general English heavy. Eval holdout = 20%, 282,941 chars.

| Vocab size | Train time (s) | Tokens/word | UNK rate |
|---|---|---|---|
| 8,000 | 0.645 | 3.7043 | 0 (by construction) |
| 16,000 | 0.747 | 3.5212 | 0 |
| 24,000 | 0.803 | 3.4689 | 0 |
| 32,000 | 0.928 | 3.4389 | 0 |

### Byte-level BPE probe B — quantum-focused mix (decisive)

Sample mix weighted to match the proposed corpus composition: 45% quantum-domain, 30% Qiskit code, 13% general English, 12% tool traces. Eval holdout = 20%, 282,942 chars, 33,681 words.

| Vocab size | Train time (s) | Tokens/word | UNK rate |
|---|---|---|---|
| 8,000 | 0.674 | 2.8906 | 0 (by construction) |
| 16,000 | 0.739 | 2.6937 | 0 |
| 24,000 | 0.861 | 2.6363 | 0 |
| 32,000 | 1.015 | 2.6227 | 0 |

### Recommendation

**Recommended vocab size: 16,000 with byte-level BPE.**

The quantum-focused probe shows the same pattern as the broad mix: a sharp compression gain from 8k to 16k, then diminishing returns. Specifically:
- Quantum-focused: 2.89 → 2.69 tokens/word (–6.8%), then only 2.64/2.62.

16k captures most of the benefit while keeping embedding lookup fast for a 10M–20M parameter CPU-trained model. Larger vocabs add training time and memory for marginal compression gains.

## Corpus Composition Plan (Step 2 — proposal, awaiting approval)

### Strategic rationale: niche depth over broad shallow competition

exp-005 keeps the same small compute budget as exp-004 (CPU-only, few days). It cannot compete with general-purpose models on breadth, and it should not try. The pivot is to **deepen the model's usefulness in the quantum-software niche**: quantum-domain text, Qiskit/quantum SDK code, and tool-use traces for quantum job submission and circuit building. A 13% general-English slice preserves basic fluency without diluting the niche signal.

### Compute constraint

- **CPU-only on current machine, few days maximum wall-clock** for a full training run.
- No GPU access assumed unless explicitly stated later.
- Target total corpus: **20,000–60,000 rows**.
- Target architecture: **10M–20M parameters**.

### Row-count targets by category (against 20k–60k total)

| Category | Share | Row target range | Description | Source approach |
|---|---|---|---|---|
| Quantum-domain technical | 45% | 9,000–27,000 rows | Quantum computing concepts, IBM Quantum docs, Qiskit, error correction, hardware docs | Reuse exp-004 source corpus under its existing provenance; supplement with new source-disjoint IBM/Qiskit docs |
| Quantum-specific code | 30% | 6,000–18,000 rows | Qiskit, Cirq, and other quantum SDK code | `qiskit/qiskit-tutorials` (Apache-2.0) as primary source; TinyMetatron own Qiskit snippets as flavor signal only |
| Function-calling / tool-use traces | 12% | 2,400–7,200 rows | Structured invocations for quantum tools: circuit builders, job submission, result parsers | Synthetic/templated from real Qiskit/IBM Quantum/TinyMetatron CLI and function signatures |
| General English | 13% | 2,600–7,800 rows | Prose coverage only; preserves fluency, not instruction-following | Public-domain texts (Project Gutenberg), clean web archives, or own writing |

### License and sourcing rules

- **Permissive only for public code:** MIT, Apache-2.0, BSD. **No GPL/AGPL** in the code corpus.
- **Filtered public corpora only:** no wholesale unfiltered The Stack / CodeParrot. If used, filter to permissive-license subsets and record exact filter in `MANIFEST.json`.
- **No AGI-model content or imports.**
- **No instruction-following/chat data** for this experiment.
- **Provenance per source:** source URL/identifier, license, retrieval date, content hash — same discipline as exp-004.
- **Source cap + source-disjoint splits:** cap per source before splitting; no source leaks across train/val/hard_dev.

### Tool-use trace format

Structured JSON only:
```json
{"tool": "name", "args": {...}, "result": {...}}
```
Not natural-language phrasing. Derived from real Qiskit/IBM Quantum/function signatures in the repo.

## Target Architecture Size (proposed, not final)

Under CPU-only / few-days constraint: **10M–20M parameters** as the realistic exp-005 scale. Final choice depends on actual corpus token count and measured steps/second once the tokenizer and corpus are fixed.

## Open Questions for Human Review

1. **Approve 16k vocab size with byte-level BPE?**
2. **Approve the quantum-focused 45/30/12/13 composition shares and row targets?**
3. **Confirm CPU-only / few-days compute budget?** (If you have GPU access, say so — it changes the scale.)
4. **Approve permissive-only code sourcing rule?** Any preferred public code dataset or domain emphasis?
