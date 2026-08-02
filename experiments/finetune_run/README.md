# Fine-Tuning Loop

## Purpose

Fine-tune Qwen2.5-3B-Instruct (via QLoRA) on quantum_corpus QA data to improve
faithful RAG-grounded answer synthesis versus the zero-shot base model.

## Loop States

```
NEW -> OBSERVED -> HYPOTHESIS_CREATED -> VALIDATED -> AWAITING_APPROVAL -> EXECUTED -> MEASURED -> ACCEPTED | REJECTED | ARCHIVED
```

## Experiment Lifecycle

```
experiments/finetune_run/experiments/exp-XXX/
  experiment.json      # metadata, hypothesis, status, gates
  config/lora.yaml    # QLoRA + training hyperparameters
  data_pipeline/      # SFT data formatting
  baseline_eval.json  # zero-shot eval results (pre-training)
  finetuned_eval.json # post-training eval results
  adapters/          # saved LoRA adapter weights (not committed to repo)
```

## Loop CLI

```bash
# List experiments
python loops/finetune_run/run_loop.py list

# Show current metrics (baseline scores, GPU availability)
python loops/finetune_run/run_loop.py status

# Propose a new experiment
python loops/finetune_run/run_loop.py propose --hypothesis "..."

# Run validation gates (GPU required)
python loops/finetune_run/run_loop.py gates --exp exp-XXX

# Approve / reject after gates pass
python loops/finetune_run/run_loop.py approve --exp exp-XXX
python loops/finetune_run/run_loop.py reject --exp exp-XXX
```

## Mandatory Gates (all must pass before training)

| Gate | Pass Condition |
|------|----------------|
| GPU available | CUDA device count > 0 |
| Zero-shot baseline | Mandatory before any training |
| Model loads | QLoRA applies cleanly, no OOM |
| Training tokenizes | No crash on first training batch |
| Loss converges | Training loss decreases over epochs |
| Eval improves | Post-train eval > baseline on primary metric |

## Current Status

- `exp-001`: **blocked: awaiting GPU** — zero-shot baseline pending

## Data Contract

Fine-tuning data is derived from `quantum_corpus/eval/qa_val.jsonl`
filtered to non-abstention records, formatted as instruction-tuning samples.

Format: `{"messages": [{"role": "system"|"user"|"assistant", "content": str}], "metadata": {...}}`
