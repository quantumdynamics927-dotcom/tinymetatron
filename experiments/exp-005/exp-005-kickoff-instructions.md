# exp-005 Kickoff Instructions — Coding Agent

## Hard constraints (read first)

- **Do not touch `experiments/exp-004/` in any way.** No edits, no re-splits, no gate changes,
  no re-training. It is frozen at revision 2 (`027c61b`) and stays the clean baseline.
- **Do not touch `quantum_corpus/` package code or the public-release branch.** This is a
  separate, unrelated workstream.
- **This phase is scoping and probing only.** No full training run starts until the human
  has reviewed and approved `experiments/exp-005/SCOPE.md` (produced in Step 3 below).
- Every file open (`open(...)`) must explicitly set `encoding="utf-8"`.
- All outputs go under `experiments/exp-005/` (new directory, untracked, same convention as
  `exp-003`/`exp-004`). Never write outside that directory during this phase.
- If anything here conflicts with a decision already recorded in memory
  (`exp-004-corpus-policy.md`, `public-release-strategy.md`, or the scaling-recommendation
  memory), stop and flag the conflict — do not silently resolve it either way.

## Step 0 — Resolve the "tools/skills" definition (blocking, ask the human)

Before any data or tokenizer work, get an explicit answer to: what does "skills/tools" mean
for exp-005? Candidate interpretations (not mutually exclusive, but they determine sourcing
priority):

1. Function-calling / tool-use traces (structured tool-invocation examples).
2. Code capability (general programming corpora).
3. Instruction-following (general instruction/response pairs).

Do not guess. Present these three options (or others the human raises) and record the answer
in `experiments/exp-005/SCOPE.md` before proceeding to Step 1.

## Step 1 — Tokenizer scoping probe (non-destructive)

Goal: pick a vocabulary size in the 16k–32k range using real measurements, not a guess.

1. Assemble three small **read-only** text samples (a few MB each is enough for this probe):
   - General English sample (any clean public-domain or your-own text; do not reuse
     exp-004 rows as "general English" — that would bias the measurement).
   - Quantum-domain sample: read (do not modify) a subset of
     `experiments/exp-004/dedupe/deduped.jsonl` — this is read-only reference use, not a
     corpus mutation.
   - Code/instruction sample, sized according to the Step 0 answer.
2. Run `tools/exp005_tokenizer_probe.py` (script below) across candidate vocab sizes
   `[8000, 16000, 24000, 32000]` on the combined sample.
3. Record for each size: unk-rate, average tokens/word, and training time.
4. Recommend one vocab size with a one-paragraph justification citing the actual numbers
   (not vocabulary-scaling folklore) — write this into `experiments/exp-005/SCOPE.md`.

## Step 2 — Corpus composition plan (planning artifact, not real ingestion yet)

Produce a written plan, not actual collected data:

- Target row/token counts per category (general English / quantum-domain / code-or-tools /
  other), based on the Step 0 answer.
- Source candidates per category, with license notes (mirror the diligence used for the
  earlier external-eval spec — no ambiguous-license scraping).
- An explicit statement of what is **excluded** and why (mirrors exp-004's
  `excluded_by_cap.jsonl` discipline — every exclusion should be a decision, not an accident).

Write this into the same `SCOPE.md`.

## Step 3 — Write `experiments/exp-005/SCOPE.md`

Consolidate Steps 0–2 into one document with sections:
`Tools/Skills Definition`, `Tokenizer Decision`, `Corpus Composition Plan`,
`Target Architecture Size (proposed, not final)`, `Open Questions for Human Review`.

**Stop here.** Do not create the tokenizer, do not collect the real corpus, do not train
anything. Present `SCOPE.md` for review.

## Step 4 — After human approval only

Once the human explicitly approves `SCOPE.md`:
- Corpus ingestion follows the same dedupe/source-disjoint/cap pipeline pattern as exp-004
  (reuse `workers/corpus/*` patterns; do not silently redesign them).
- Tokenizer training becomes final and versioned (record vocab size, training corpus hash,
  and the probe results that justified it).
- A new gate/freeze cycle for exp-005 follows the same pattern as exp-004's FREEZE.md —
  corpus freeze first, then model/training/eval freeze, each with a manifest recording its
  revision.

## Reporting format

At the end of each step, report back in this shape (no step-by-step narration, just the
decision-relevant facts):
- What was measured / decided.
- The concrete numbers or file paths produced.
- What remains blocked on human input.
