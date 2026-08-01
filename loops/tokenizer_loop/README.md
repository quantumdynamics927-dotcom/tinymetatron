# Tokenizer-Improvement Loop

## Purpose

Iteratively improve the TinyMetatron English tokenizer by classifying
fragmentation issues, applying one bounded change at a time, and running
full regression gates before committing any change.

## Loop States

```
NEW -> OBSERVED -> HYPOTHESIS_CREATED -> VALIDATED -> AWAITING_APPROVAL -> EXECUTED -> MEASURED -> ACCEPTED | REJECTED | ARCHIVED
```

## Allowed Actions (one per iteration)

Only one change per loop iteration:

1. Change corpus composition (add/remove approved source categories)
2. Change vocabulary size (8k -> 16k, etc.)
3. Change training data volume (target chars)
4. Change min_frequency threshold
5. Retrain tokenizer

**Forbidden**: changing special token set, changing add_prefix_space,
touching the quantum_corpus private DB, modifying test cases.

## Mandatory Gates (all must pass)

| Gate | Pass Condition |
|------|----------------|
| Train tokenizer | No crash |
| Run evaluation suite | No crash |
| `<\|en\|>` atomic | `len(encode("<|en|>").ids) == 1` |
| Unknown token count | `== 0` on TEST_CASES |
| prose_median tokens/word | `≤ 4.0` (soft — note if `> 2.5`) |
| Round-trip encode/decode | All TEST_CASES pass |

## Input Artifacts (freeze per iteration)

- `experiments/english_first_tokenizer/data/approved/` — approved training sources with SHA256 hashes in manifest
- `experiments/english_first_tokenizer/tokenizers/tinymetatron_v2_en_8k/manifest.json` — tokenizer artifact hashes
- `experiments/english_first_tokenizer/eval/summary.json` — evaluation summary

## Output

- `loops/tokenizer_loop/experiments/exp-XXX/` — one subdir per experiment containing:
  - `experiment.json` — hypothesis, action taken, validation results
  - `evaluation_summary.json` — per-run metrics
  - `candidate_patch.diff` — the actual change (corpus, config, or retrain)
- `loops/tokenizer_loop/archive/` — experiments that were rejected or rolled back
- Commit only after ALL gates pass and human approval is obtained

## Workflow

1. **Observe**: `python loops/tokenizer_loop/refresh.py` — measure current tokenizer fragmentation
2. **Orient**: `python loops/tokenizer_loop/run_loop.py list` — review current metrics; pick ONE issue
3. **Propose**: `python loops/tokenizer_loop/run_loop.py propose --hypothesis "..." --action "..."`
4. **Validate**: `python loops/tokenizer_loop/run_loop.py gates --exp XXX` — runs all mandatory gates
5. **Approve**: Show results to user, await explicit approval
6. **Execute**: `python loops/tokenizer_loop/run_loop.py approve --exp XXX --note "..."` — commits only after approval
7. **Measure**: Record metrics in experiment JSON
8. **Learn**: Move to archive; repeat

## Experiment naming

`exp-001/`, `exp-002/`, ... — sequential integers, no reset.

## State transitions

- `NEW`: Experiment directory created with `experiment.json`
- `OBSERVED`: Current metrics reviewed, issue identified
- `HYPOTHESIS_CREATED`: Hypothesis written and plausible
- `VALIDATED`: All mandatory gates passed
- `AWAITING_APPROVAL`: Waiting for human sign-off
- `EXECUTED`: Change committed to repository
- `MEASURED`: Metrics recorded in `experiment.json`
- `ACCEPTED` / `REJECTED`: Terminal state
- `ARCHIVED`: Moved to `archive/`
