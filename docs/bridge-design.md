# TinyMetatron ↔ QuantumResearchLab Bridge

## Relationship

TinyMetatron (D:\TinyMetatron) and QuantumResearchLab (D:\QuantumResearchLab)
are **sibling projects**. They share no code, database, secrets, or runtime.
Each has its own git repository and loop infrastructure.

## What TinyMetatron knows about QuantumResearchLab

TinyMetatron's RAG corpus (`E:\Temp\qcorpus\quantum_corpus.db`) may contain
references to experiment outcomes from QuantumResearchLab. The following rules govern
what enters the corpus:

### Allowed fields (sanitized summaries only)

When a QuantumResearchLab experiment completes, the following fields MAY be
extracted and stored in `quantum_corpus.db`:

| Field | Example |
|---|---|
| experiment_id | `qrl-2026-001` |
| circuit_family | `teleport`, `variational`, `basis_bench` |
| circuit_qasm_hash | `sha256:8accaf20...` (hash only, not full QASM) |
| n_qubits | `2` |
| backend_name | `ibm_fez` |
| experiment_status | `completed`, `failed` |
| experiment_conclusion | Free-text summary of what was measured |
| experiment_date | `2026-08-01` |

### Forbidden fields (never enter TinyMetatron corpus)

These fields remain exclusively in QuantumResearchLab:

| Field | Reason |
|---|---|
| Full QASM source | May contain proprietary gate sequences |
| job_id (hardware) | IBM job identifiers — not for RAG |
| raw counts / counts_hash | Device-specific result data |
| Backend calibration metadata | Live backend data changes over time |
| Cost / billing information | Not relevant to corpus |
| Transpiler seed | Build artifact, not semantic content |
| User identity / IBMid | Privacy |

### Safe evidence contract

```
QuantumResearchLab experiment
        │
        ▼
  [Extract allowed fields]
        │
        ▼
  quantum_corpus.db (TinyMetatron RAG corpus)
        │
        ▼
  Queried via TinyMetatron RAG as quantum research evidence
```

**Process**: When a QuantumResearchLab experiment is ready to share,
a human exports the sanitized summary fields into a JSONL record and
adds it to the corpus ingestion pipeline. No automated pipeline shares
raw results.

## What QuantumResearchLab knows about TinyMetatron

QuantumResearchLab may use TinyMetatron as a copilot for:
- Finding prior experiment context (querying the RAG corpus)
- Drafting QASM or analysis scripts

It must NOT:
- Write to `E:\Temp\qcorpus\quantum_corpus.db`
- Access TinyMetatron training data or model weights
- Modify TinyMetatron code or configuration

## Loop cross-talk

Each project has its own loop:

```
D:\TinyMetatron\loops\
  retrieval_loop/   ← TinyMetatron RAG improvement
  tokenizer_loop/   ← TinyMetatron tokenizer improvement

D:\QuantumResearchLab\loops\
  experiment_loop/   ← Quantum hardware experiment management
```

These loops run independently. A retrieval improvement in TinyMetatron does not
trigger any action in QuantumResearchLab, and vice versa.

## Evidence hierarchy

```
Level 0: Raw device output (job results, counts)
  ↳ QuantumResearchLab only (D:\QuantumResearchLab\results\)

Level 1: Canonical experiment manifest (experiment_id, qasm_hash, status)
  ↳ Both projects (via sanitized summary)

Level 2: RAG corpus evidence (circuit_family, n_qubits, conclusion)
  ↳ TinyMetatron RAG (quantum_corpus.db)

Level 3: Aggregate analysis (benchmarks, learning)
  ↳ Either project, published externally
```

## Adding a QuantumResearchLab experiment to TinyMetatron corpus

1. Complete the experiment in QuantumResearchLab
2. Human reviews the manifest — strip forbidden fields
3. Human writes sanitized summary JSONL
4. Add to `E:\Temp\qcorpus\` ingestion pipeline (outside TinyMetatron repo)
5. Rebuild `quantum_corpus.db` — the experiment is now queryable via RAG

This manual step ensures no proprietary device data or job IDs enter the RAG corpus.
