:construction: **DRAFT — awaiting human approval before any corpus/tokenizer/training work begins.**

# exp-005 Scope Document

## Tools/Skills Definition (Step 0 — blocking question)

Before any data sourcing or tokenizer probing, the "tools/skills" ambition for exp-005 must be defined explicitly. The following interpretations are not mutually exclusive, but each determines a different corpus-sourcing priority:

1. **Function-calling / tool-use traces** — structured examples of invoking external tools or APIs (e.g., `"Call function X with args Y", "Result: Z"`).
2. **Code capability** — general programming corpora (Python, etc.) plus code-completion / code-instruction pairs.
3. **Instruction-following** — general instruction/response pairs for chat-style compliance and task completion.

**Human decision required:** Which of these (or another interpretation) is the primary "tools/skills" goal for exp-005? This answer must be recorded here before Step 1 (tokenizer probe) and Step 2 (corpus composition plan) can proceed.

## Tokenizer Decision (Step 1 — pending Step 0)

To be filled after the human answers the tools/skills question and the tokenizer probe is run.

## Corpus Composition Plan (Step 2 — pending Step 0)

To be filled after the tokenizer probe and tools/skills definition.

## Target Architecture Size (proposed, not final)

To be discussed after vocabulary and corpus plan are fixed.

## Open Questions for Human Review

1. What does "tools/skills" mean for exp-005? (Step 0)
2. What general-English and code/instruction sources are acceptable? (licenses, own data vs. public corpora)
3. Target compute budget and acceptable training wall-clock time?
