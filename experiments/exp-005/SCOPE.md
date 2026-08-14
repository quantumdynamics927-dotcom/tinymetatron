:construction: **DRAFT — awaiting human approval before any corpus/tokenizer/training work begins.**

# exp-005 Scope Document

## Tools/Skills Definition (Step 0 — resolved)

**Primary corpus category:** code capability — general programming plus own-domain code in Python, TypeScript, PowerShell, and Bash.

**Layered on top:** a modest function-calling / tool-use trace format (structured invocations and results).

**Explicitly out of scope for exp-005:** general instruction-following / chat-style compliance. Revisit in a later experiment.

:warning: **Step 0 resolved. Step 1 complete. No real tokenizer training or corpus ingestion has started.**

## Tokenizer Decision (Step 1 — complete)

### Sample sourcing provenance

| Sample | Source | How obtained | License |
|---|---|---|---|
| General English | Project Gutenberg: *Pride and Prejudice* by Jane Austen | Fetched as plain text from Project Gutenberg (`https://www.gutenberg.org/files/1342/1342-0.txt`) | Public domain in the US |
| Quantum-domain | `experiments/exp-004/dedupe/deduped.jsonl` | Read-only random sample of 1,500 rows (seed 42) from `text` field | Same as exp-004 corpus (already frozen, internal) |
| Code | TinyMetatron's own `.py` files under `workers/`, `loops/`, `tests/`, `tools/` | Read-only concatenation of 42 source files | N/A — own codebase |
| Tool-use traces | Synthetic | 350 trace blocks generated from real CLI/function signatures in this repo (`compute_ce`, `quantum_corpus_freeze`, `quantum_corpus_validate`, corpus workers) | N/A — own codebase |

### Probe results (combined eval holdout = 20%, 282,941 chars, 26,731 words)

| Vocab size | Train time (s) | Tokens/word | UNK count | UNK rate |
|---|---|---|---|---|
| 8,000 | 0.454 | 2.9385 | 19 | 0.000242 |
| 16,000 | 0.568 | 2.7643 | 19 | 0.000257 |
| 24,000 | 0.682 | 2.7158 | 19 | 0.000262 |
| 32,000 | 0.735 | 2.6945 | 19 | 0.000264 |

### Recommendation

**Recommended vocab size: 16,000.**

Reasoning: tokens/word drops sharply from 8k (2.94) to 16k (2.76), then gains diminish (2.72 at 24k, 2.69 at 32k). The 16k point captures most of the compression benefit while keeping the vocabulary small enough for a modest model and fast embedding lookup. UNK rate is essentially flat across all sizes (~0.00025, 19 tokens total) because the probe uses plain BPE without byte fallback and the eval text is heavily ASCII/English/code.

**Caveat:** this probe measures relative vocab-size efficiency on the sample mix, not absolute robustness to arbitrary unicode, rare code identifiers, or out-of-domain tokens. The final exp-005 tokenizer should use **byte-level BPE or BPE with `byte_fallback` enabled** so that no input can produce an UNK. A follow-up probe with byte-fallback should be run before finalizing.

## Corpus Composition Plan (Step 2 — proposal, awaiting approval)

Based on the tools/skills definition (code capability primary, function-calling layered on top, general instruction-following out of scope):

| Category | Target share | Description | Source approach |
|---|---|---|---|
| Code — general + own-domain | 50–60% | Python, TypeScript, PowerShell, Bash; mix of public clean-license corpora and own repo code | Public: The Stack/Stack Overflow subsets with clear licenses; own: TinyMetatron `workers/`, `loops/`, `tests/`, `tools/` |
| Quantum-domain technical | 20–30% | Quantum computing, IBM Quantum, Qiskit, error correction, hardware docs | Reuse exp-004 source corpus under its existing provenance; supplement with new source-disjoint IBM/Qiskit docs |
| Function-calling / tool-use traces | 10–20% | Structured invocations and results derived from real functions/APIs in the codebase | Synthetic/templated from actual CLI signatures and function docs |
| General English | 5–10% | Prose coverage only; not instruction-following | Public-domain texts (Project Gutenberg), clean web archives, or own writing |

**Exclusions (mirroring exp-004 discipline):**
- No AGI-model content or imports.
- No instruction-following/chat data for this experiment.
- No ambiguous-license scraped content.
- All sources recorded with license + provenance; cap per source to prevent dominance; source-disjoint splits.

## Target Architecture Size (proposed, not final)

Hold until corpus size and compute budget are fixed. As a starting hypothesis only: if the final corpus reaches ~100k–500k rows, a 20M–60M parameter model may be worth testing, but this must be validated by compute-budget estimation before any architecture decision.

## Open Questions for Human Review

1. **Approve 16k vocab target?** (with byte-fallback follow-up probe before final tokenizer training)
2. **Approve corpus composition shares?** Especially: what public code corpora are acceptable, and how much own-repo code vs. external code?
3. **Target compute budget / acceptable training wall-clock?** This gates architecture size and whether GPU access is required.
4. **What is the concrete definition of "tool-use traces"?** e.g., JSON-style `{"tool": "...", "args": {...}, "result": {...}}`, or natural-language `"Call X with Y"` format?

## Target Architecture Size (proposed, not final)

To be discussed after vocabulary and corpus plan are fixed.

## Open Questions for Human Review

1. What general-English sample should be used for the tokenizer probe? (clean public-domain or your-own text; a few MB is enough)
2. What specific code/tool-use sources are acceptable? (licenses, own repos, public corpora)
3. Target compute budget and acceptable training wall-clock time?
