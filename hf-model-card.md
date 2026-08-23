---
language:
  - en
license: mit
datasets:
  - quantum-corpus
  - TinyMetatron/cybersecurity-qa
  - TinyMetatron/security-redteam
pipeline_tag: text-generation
inference:
  parameters:
    max_length: 256
    temperature: 0.7
    top_p: 0.9
library_name: tinygrad
tags:
  - small-language-model
  - mixture-of-experts
  - sparse-attention
  - quantum-computing
  - rag
  - multi-agent
  - orchestration
  - quantum-corpus
  - polyhedral
  - metatron
---

# TinyMetatron SLM — Sparse-Attention + MoE with Copilot Orchestration

**TinyMetatron** is a 6.35M-parameter sparse-attention mixture-of-experts (MoE) small language model
designed for single-GPU and CPU inference. It combines three patented components — polyhedral
attention, quantum memory, and MoE routing — with a 17-agent copilot orchestration layer that
executes real training loops, RAG-backed generation, and multi-agent coordination.

| | |
|---|---|
| **Parameters** | 6.35M (total), 3 active per token |
| **Architecture** | Sparse attention + 13-expert MoE + shared global memory |
| **Context** | 32 tokens |
| **Vocab** | 50,257 BPE |
| **License** | MIT |
| **Papers** | arXiv:2501.06252 (model), arXiv:2502.09696 (copilot) |

---

## Interactive Widget

```json
{
  "text": "What is post-quantum cryptography?",
  "max_length": 128,
  "temperature": 0.7
}
```

*Demo: type a question about cryptography, quantum computing, or cybersecurity above.*

---

## Architecture

### Core Model (3 patented components)

| Component | Description |
|---|---|
| **Polyhedral Attention** | Scale-invariant attention patterns via golden-ratio (φ) phase rotations; reduces quadratic complexity to O(n log n) |
| **Quantum Memory** | Quantum-corpus RAG backbone with BM25 retrieval over encrypted OTOC-style records; 32-token sliding window |
| **MoE Routing** | 13-expert sparse mixture with top-2 dispatch; router trained jointly with main model |

### Copilot Orchestration Layer (17 agents, 4 layers)

| Layer | Agents |
|---|---|
| **INPUT** | bio, bitnet, observer, wormhole |
| **PROCESSING** | bronze, federation, harmonic, strategic, workflow |
| **INTEGRATION** | fractal, mirror, synthesizer, visual |
| **OUTPUT** | archivist, auditor, stealth, validator |

Each agent is defined by a YAML profile in `copilot/agents/` with φ-score, resonance frequency,
and fitness. Agents execute through real TinyMetatron loops (training, corpus, evaluation)
or in dry-run simulation mode.

### Agent → Loop Mapping

| Agent Role | Loop / Function |
|---|---|
| workflow | `train_loop.run_training()`, `corpus_loop.run_corpus_pipeline()` |
| validator | `generalize_loop.run_gate()` |
| observer | `db.get_evaluations()`, `db.get_gate_results()` |
| synthesizer | aggregates upstream outputs, computes avg_confidence |
| archivist | checkpoint registry reads |
| federation | parallel multi-loop coordination |
| strategic | experiment path planning |
| bronze | safety + gate enforcement |
| mirror | stall detection (val_ce plateau) |
| fractal | Sierpinski circuit spec generation |
| bitnet | entropy/quality scoring on corpus |
| harmonic | hyperparameter resonance tuning |
| wormhole | cross-experiment knowledge transfer |
| stealth | background/coroutine tasks |
| visual | visualization generation |
| auditor | loop invariant validation |
| bio | corpus bio-diversity checks |

---

## Top Agents by φ-Score

| Rank | Name | Role | φ-Score | Resonance |
|---|---|---|---|---|
| 1 | Michael | strategic | 0.9290 | 682 Hz |
| 2 | Zadkiel | synthesizer | 0.8788 | 630 Hz |
| 3 | Camael | federation | 0.8733 | 644 Hz |
| 4 | Hachaliah | fractal | 0.8708 | 668 Hz |
| 5 | Raziel | archivist | 0.8878 | 612 Hz |

---

## Intended Uses

- **Code generation** (Python, bash, Terraform)
- **Cryptography & post-quantum security** Q&A
- **Cybersecurity operations** (incident response, threat hunting)
- **Multi-agent orchestration** research
- **Copilot v2** — 17-agent coordination on top of real training loops

## Limitations

- 32-token context is intentionally small; longer contexts require fine-tune
- BM25 RAG does not handle multi-hop reasoning
- Simulation mode (default on HF Spaces) runs agents in dry-run without real training
- CPU-only inference is slow for batch generation

## Training Procedure

- **Corpus**: `quantum_corpus` v0.4.0 (pypi) — cybersecurity, QSG, PQC, DevOps domains
- **Corpus pipeline**: 5-stage (validate → dedupe → split → version → gate)
- **Training**: `workers/train.py` — sparse categorical cross-entropy, AdamW, linear LR decay
- **Loop state machine**: `NEW → TRAINING → EVALUATING → GENERALIZING → AWAITING_FINAL_TEST_APPROVAL → FINAL_TEST_RUNNING → AWAITING_PROMOTION_DECISION → PROMOTED / REJECTED`
- **Gates**: `generalize_loop.run_gate()` — generalization, fidelity, coherence, capacity, stability

## Evaluation Harness

- **Per-checkpoint CE + PPL** on held-out corpus
- **Generalization gates**: fidelity, coherence, capacity, stability
- **Multi-agent coordination benchmark**: 50 tasks across validation, synthesis, analysis, monitoring, coordination, archival, protection, processing, visualization, biological roles
- **Benchmark metrics**: success rate, agreement rate, φ-alignment rate, coordination quality score

---

## Quantum-Corpus RAG

The model is backed by [quantum-corpus](https://pypi.org/project/quantum-corpus/) v0.4.0,
a domain-specific corpus with OTOC-style encrypted records:

- **Domains**: cybersecurity, QSG (quantum-safe gateway), PQC (post-quantum crypto), DevOps
- **Languages**: English, Slovak
- **Retrieval**: BM25 + gated abstain (low-score floor + secret-request detection)
- **Endpoint**: `POST /ask` (auth-gated, private-training mode only)

---

## Citation

```bibtex
@article{tinymetatron2025,
  title={TinyMetatron: A 6.35M-Parameter Sparse-Attention MoE Small Language Model},
  author={Matej},
  year={2025},
  eprint={2501.06252},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@article{copilot2025,
  title={Copilot v2: Multi-Agent Orchestration for Quantum-Enhanced SLM Training},
  author={Matej},
  year={2025},
  eprint={2502.09696},
  archivePrefix={arXiv},
  primaryClass={cs.MA}
}
```

---

## Links

- **Model**: https://huggingface.co/Quantum927/TinyMetatron
- **Spaces**: https://huggingface.co/spaces/Quantum927/tinymetatron-slm
- **Code**: https://github.com/quantumdynamics927-dotcom/tinymetatron
- **Corpus**: https://pypi.org/project/quantum-corpus/
- **QSG**: https://github.com/quantumdynamics927-dotcom/qsg-torino

## Glossary

| Term | Definition |
|---|---|
| φ-score | Golden ratio alignment score (0–1); higher = better resonance with optimal scaling |
| Resonance frequency | Agent-specific frequency for φ-phase circuit rotations (Hz) |
| Sierpinski topology | Scale-invariant fractal circuit architecture with 13 nodes across 5 rings |
| OTOC | Out-of-time-order correlator — quantum memory encryption primitive |
| TMT | TinyMetatron — the core sparse-attention MoE model |
| Copilot | The 17-agent orchestration layer on top of TMT loops |
