# QPyth → TinyMetatron Corpus Bridge

**Status:** design spec — no Python import coupling between QPyth and TinyMetatron.  
**Date:** 2026-08-20  
**Owner:** TMT Audit / TinyMetatron team

## Purpose

This document defines how QPyth lab results enter TinyMetatron without making QPyth a dependency of TinyMetatron.

The rule is **producer → file → ingest → corpus → `/ask`**:

1. QPyth executes a circuit or experiment locally (its own Qiskit, IBM credentials, noise profiles, web UI).
2. QPyth writes a structured, redacted job record to a watched directory.
3. TinyMetatron's `quantum_corpus` pipeline ingests the record (BM25 + semantic index, train/val/test split, provenance).
4. TinyMetatron `/ask` retrieves the record through the field-gated answer engine.

TinyMetatron never imports `quantumpytho`, never instantiates QPyth classes, and never calls QPyth functions.  QPyth never imports TinyMetatron.  The only contract is a JSON file schema.

## Why not a pip dependency

- QPyth pulls Qiskit, IBM authentication, VQE, QEC, a web UI, and hardware credentials.  Adding it would inflate TinyMetatron's Docker image, widen the attack surface, and violate the audit principle that TinyMetatron is a **retrieval + refusal** copilot, not a circuit executor.
- The QRL folders inside TinyMetatron (`qrl-2026-*`) are **records of prior lab work**, not a live circuit runner.  The product is RAG over those records, not a re-run button.
- QPyth and TinyMetatron share only subject matter (teleport, CHSH, QEC, Sierpinski, IBM jobs).  The overlap is corpus content, not code.

## JSON job record schema

A QPyth run produces one file per experiment, named

```
<project>_<yyyy>-<mm>-<dd>T<HH><MM><SS>_<run_id>.json
```

where `<project>` is a lower-kebab QPyth module name, e.g. `teleport-bridge`, `qec-surface`, `tmt-sierpinski`, `hardware-ibm`, `chsh`.

### Required top-level fields

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | `"1.0"` |
| `record_type` | string | `"qpyth_job"` |
| `created_at` | ISO-8601 string | UTC timestamp of the QPyth run |
| `project` | string | QPyth project/namespace, e.g. `"QPyth"` |
| `source_type` | string | One of `ibm_job`, `manifest`, `repo`, `workload_csv`, `pdf` — same enum as `quantum_corpus.schema` |
| `doc_id` | string | Stable document key: `{project}:{run_id}` |
| `provenance_url` | string | Path to QPyth run directory or repository URL.  No absolute local paths that leak user home directories. |
| `sensitivity` | string | `public`, `internal`, or `sensitive` |
| `risk_tier` | integer | 0–3, assigned by QPyth redaction pass |
| `redacted` | boolean | `true` if PII/IBM tokens/job IDs were scrubbed |
| `text` | string | Human-readable summary for BM25 indexing.  Must be token-counted and English-leaning. |
| `structured` | object | Machine-readable fields for field verification and SQL queries. |

### Structured object (required keys)

| Key | Type | Description |
|---|---|---|
| `experiment_family` | string | `teleport`, `chsh`, `ghz`, `qec`, `vqe`, `hardware`, `noise`, `sierpinski`, `other` |
| `backend` | string | Backend identifier, e.g. `"aer_simulator"`, `"ibm_fez"`, `"ibm_kingston"`, `"ideal"` |
| `shots` | integer or null | Number of shots; `null` for ideal statevector runs |
| `success` | boolean | Whether the run completed without exception |
| `metrics` | object | Free-form key/value numbers/strings relevant to the experiment (e.g. `bell_score`, `fidelity`, `lyapunov_exponent`). |
| `circuit_sha256` | string | SHA-256 of the transpiled circuit QASM (for reproducibility, optional but recommended). |
| `raw_data_path` | string or null | Relative path to the QPyth raw output file.  TinyMetatron stores only metadata; it does not ingest multi-megabyte JSON by default. |

### Example: CHSH Bell-test record

```json
{
  "schema_version": "1.0",
  "record_type": "qpyth_job",
  "created_at": "2026-08-20T14:32:11Z",
  "project": "QPyth",
  "source_type": "manifest",
  "doc_id": "qpyth:chsh:20260820-143211-a1b2",
  "provenance_url": "https://github.com/quantumdynamics927-dotcom/QPyth/tree/main/demos/chsh",
  "sensitivity": "public",
  "risk_tier": 0,
  "redacted": true,
  "text": "CHSH Bell inequality test executed via QPyth. Aer simulator, 8192 shots. Estimated S parameter 2.82 ± 0.04. Result status success. Circuit family chsh, backend aer_simulator. No hardware tokens present.",
  "structured": {
    "experiment_family": "chsh",
    "backend": "aer_simulator",
    "shots": 8192,
    "success": true,
    "metrics": {
      "s_parameter": 2.82,
      "s_std": 0.04,
      "win_rate": 0.91
    },
    "circuit_sha256": "a1b2c3...",
    "raw_data_path": null
  }
}
```

### Example: IBM hardware job record

```json
{
  "schema_version": "1.0",
  "record_type": "qpyth_job",
  "created_at": "2026-08-19T09:15:00Z",
  "project": "QPyth",
  "source_type": "ibm_job",
  "doc_id": "qpyth:ibm:20260819-091500-d5a6",
  "provenance_url": "jobs/ibm/2026-08-19",
  "sensitivity": "internal",
  "risk_tier": 1,
  "redacted": true,
  "text": "IBM Quantum job executed through QPyth hardware_ibm module. Backend ibm_fez, 8192 shots, program sampler, status Completed. Job identifier redacted.",
  "structured": {
    "experiment_family": "hardware",
    "backend": "ibm_fez",
    "shots": 8192,
    "success": true,
    "metrics": {
      "status": "Completed",
      "program": "sampler"
    },
    "circuit_sha256": null,
    "raw_data_path": "raw/ibm/d5a6.json"
  }
}
```

## Mapping QPyth modules to `experiment_family`

| QPyth module / folder | `experiment_family` | Notes |
|---|---|---|
| `vqe_*` | `vqe` | Molecular / variational eigensolver runs |
| `qec_shor`, `qec_steane`, `qec_surface` | `qec` | Quantum error correction |
| `teleport_bridge` | `teleport` | Teleportation experiments |
| `hardware_ibm` | `hardware` | IBM Quantum job submissions |
| `noise_builder` | `noise` | Realistic noisy simulation |
| `tmt_sierpinski` | `sierpinski` | Sierpinski quantum walks |
| CHSH demos | `chsh` | Bell inequality tests |
| GHZ / Mermin demos | `ghz` | GHZ-state / Mermin inequality tests |

## Ingest pipeline (TinyMetatron side)

A new small adapter, `workers/corpus/convert_qpyth.py` or a subcommand of `quantum_corpus/build.py`, performs these steps:

1. **Read** the JSON file from an import directory (default `quantum_jobs/`, configurable via `TMT_QUANTUM_JOBS_DIR`).
2. **Validate** against the schema above (reject unknown `schema_version`, missing required keys).
3. **Normalize** `provenance_url` to a relative or public URL; refuse absolute Windows home paths.
4. **Redact** any residual IBM token, key, or email using `quantum_corpus.redact`.
5. **Set** `source_type`, `sensitivity`, `risk_tier`, `split` via the existing `quantum_corpus.split` rules.
6. **Compute** `content_hash` and `source_identity` via `quantum_corpus.schema`.
7. **Write** to `quantum_corpus.db` with `quantum_corpus.schema.write_records`.

No import of QPyth is required.  The adapter only needs `json`, `pathlib`, `hashlib`, and the existing `quantum_corpus` modules.

## Security / contract rules

- **No live tokens in `text` or `provenance_url`.**  The record must pass `quantum_corpus.redact.redact_text` with zero replacements or be marked `redacted: true`.
- **No raw job dumps by default.**  TinyMetatron indexes metadata only.  Large raw JSON files stay in QPyth storage.
- **Sensitivity is QPyth's responsibility.**  TinyMetatron trusts the `sensitivity` field but still runs its own risk gate in `/ask`.
- **`/ask` never routes to QPyth.**  The answer engine may cite a QPyth record, but it never invokes QPyth functions or triggers a new circuit run.

## Suggested directory layout

```text
/mnt/d/TinyMetatron/
├── quantum_jobs/                    # drop zone for QPyth JSON records
│   ├── chsh_2026-08-20T143211_a1b2.json
│   └── ibm_2026-08-19T091500_d5a6.json
├── quantum_corpus/
│   └── build.py                     # add --qpyth-dir flag
├── workers/corpus/convert_qpyth.py  # adapter (no QPyth import)
└── docs/qpyth-corpus-bridge.md      # this document
```

## Open questions for QPyth maintainers

1. Is `quantumpytho/` going to be renamed to `qpyth/`?  If so, the record schema stays stable — only QPyth's internal packaging changes.
2. Can QPyth commit and push the post-March 2026 git work?  The PyPI `0.4.0` package is ahead of the public repo if the last push was 2026-07-26 after the 2026-03-27 upload.
3. Can QPyth stop tracking `node_modules/` and add a root `.gitignore`?
4. Should QPyth emit the record schema natively, or should TinyMetatron maintain a post-run formatter script in the QPyth repo?

## Related TinyMetatron work

- `quantum_corpus.schema` — `Record` dataclass and `source_identity` hashing.
- `quantum_corpus.redact` — PII / IBM token / email scrubbing.
- `quantum_corpus.split` — train/val/test assignment by project.
- `quantum_corpus.build` — ingestion orchestration.
- `api.py::ask` — the `/ask` endpoint that consumes the indexed records.

## Decision

Keep QPyth and TinyMetatron as **siblings connected by a file contract**, not as a Python dependency graph.  Highest leverage remains P0 model correctness and hybrid retrieval; this bridge is a P2 integration spec to be implemented only after those are stable.
