---
title: TinyMetatron SLM
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# TinyMetatron SLM

A small, CPU-runnable language model built on the TMT (Tensor Metatron
Technology) architecture: **sparse polyhedral attention** + **Mixture-of-Experts
(MoE)** routing + a shared **global memory**. The project ships as a flat Python
repo (no package) with a FastAPI server, a SQLite-backed training pipeline, and a
CLI for data management.

- Total parameters: **~6.3M** (inside the 5-7M spec budget; `d_ff=112` is the
  tuning knob in `config.py`).
- Active parameters per token: **~1M** via top-2/13 MoE routing.
- Vocabulary: **291 tokens** (custom Slovak + English technical sub-word /
  character-level vocabulary; see `vocab.json` and `tokenizer.py`).
- Sequence length: 32. Hidden size: 256. Layers: 6. Heads: 4.

## IP housekeeping

- Patent drafts and diagrams were moved to `~/TMT_Private/`, outside Git-tracked paths.
- `.gitignore` now excludes IP-sensitive and local-artifact paths:
  - `data_v2/`
  - `tokenizers/`
  - `qsg_static_posthoc_validation.py`
  - `qsg_layer3_validation.py`
  - `ckpt/`
  - `data/`
  - `models/`
  - `*.pt`
- `data_v2/` remains local (~660 KB) and is not committed.
- No patent-related files are exposed by the checked remote paths.

## Architecture

The model (`tinymetatron_model.py`) composes three patented components:

1. **Sparse polyhedral attention** (`metatron_sparse_attention.py`) — each
   attention head uses a fixed polyhedral connectivity mask (tetrahedron,
   hexahedron, octahedron, dodecahedron, icosahedron) instead of dense
   QK^T. Edges of the polyhedron define which tokens attend to which, giving
   O(edges) attention work with provable geometric structure. The default
   solid is the icosahedron.

2. **Mixture of Experts** (`metatron_moe.py`) — Metatron's Cube of 13 spheres
   gives 13 experts per layer with top-2 routing. A Switch-Transformer-style
   auxiliary balancing loss (`aux_loss = n_e * sum(fractions^2)`) keeps expert
   load even. Token routing is decided per token by a learned gate.

3. **Global memory** (`metatron_global_memory.py`) — a shared 13-node
   learnable memory tensor (one node per Metatron's-Cube sphere) read by every
   layer via per-token gating. The memory is a `nn.Parameter` that persists
   across forward passes; it is written by training, not by the forward path.

A note on scope: the **polyhedral sharding planner**
(`metatron_shard.py`) is a **distributed-only patent component**. It plans how
to shard a sequence across devices along polyhedral-face boundaries. It is **not
imported by the single-CPU forward path** (`tinymetatron_model.py` never imports
`metatron_shard`) and is exercised only in isolation by `tests/test_shard.py`.

## Installation

Requires Python 3.13 (the contract target) and a CPU-only PyTorch wheel.

```bash
pip install -r requirements.txt
```

For a guaranteed CPU-only torch install (smaller, no CUDA runtime):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn pydantic pytest
```

## Running the API

The FastAPI app is defined in `api.py` and exposes a module-level `app`:

```bash
uvicorn api:app --port 8010
```

The server listens on `0.0.0.0:8010` by default. Interactive docs are at
`http://localhost:8010/docs`.

### Endpoints

| Method | Path            | Description                                                      |
|--------|-----------------|------------------------------------------------------------------|
| POST   | `/generate`      | Load the active checkpoint, tokenize `prompt`, run `model.generate`, decode. |
| POST   | `/train/start`   | Launch training in a background thread (returns 202).           |
| GET    | `/train/status`  | Module-global training state: `is_training`, `current_step`, `total_steps`, `current_loss`, `session_id`. |
| POST   | `/data/add`      | Insert texts via `db.add_texts` with `quality.score_quality`.   |
| GET    | `/data/stats`    | Dataset stats via `db.stats`.                                    |
| GET    | `/model/info`    | Returns `CONFIG` and the active checkpoint metadata.            |

## Training

`train_db.py` reads rows from the `training_data` SQLite table where
`domain = ?` and `quality >= ?` and `used_in_training = 0`, marks them used,
opens a training session, trains, and writes a checkpoint row plus a `.pt`
file. Optimizer is Adam; loss is cross-entropy over LM logits plus
`aux_loss_weight * moe_aux`.

```bash
python train_db.py --steps 200 --domain cybersecurity --min_quality 0.8
```

Full flag set (defaults in parentheses):

| Flag                 | Default     | Meaning                                  |
|----------------------|-------------|------------------------------------------|
| `--steps`            | 200         | Number of training steps.                |
| `--domain`           | general     | Domain filter for training rows.         |
| `--min_quality`       | 0.5         | Minimum quality score for rows.          |
| `--batch_size`        | 16          | Mini-batch size.                         |
| `--learning_rate`     | 1e-3        | Adam learning rate.                      |
| `--max_seq_len`       | 32          | Sequence length.                         |
| `--device`            | cpu         | `cpu` or `cuda`.                         |
| `--checkpoint_dir`    | ckpt        | Where to write `.pt` checkpoints.        |
| `--db_path`           | metatron.db | SQLite database path.                    |
| `--aux_loss_weight`   | 0.01        | MoE balancing loss weight.               |
| `--seed`              | 42          | RNG seed.                                |

Progress is printed in Slovak: `Krok N/M: loss=x.xxx` and finally
`Tréning dokončený. Finálna strata: x.xxx`.

## Managing data

`manage_data.py` has five subcommands:

```bash
# Synthesize N deterministic rows for a domain (cybersecurity/software/general)
python manage_data.py generate --domain cybersecurity --count 50

# Read lines from a file, score quality, insert into the DB
python manage_data.py import --file corpus.txt --domain software

# Export matching rows
python manage_data.py export --domain cybersecurity --min_quality 0.8 --out out.jsonl

# Print dataset statistics
python manage_data.py stats

# Delete rows below a quality threshold
python manage_data.py clean --min_quality 0.5
```

## Corpus freeze policy

The `quantum-corpus-freeze` command implements the `source_disjoint_capped_v1`
split policy:

- **Source-disjoint**: every source group is assigned wholly to exactly one of
  train / val / hard_dev; no source is sliced across partitions.
- **Per-source cap before split**: each source is deterministically capped at
  `max_rows_per_source` (default 400) *before* allocation, so retained rows
  don't depend on which split a source lands in. Rows beyond the cap are
  preserved in `excluded_by_cap.jsonl` (recorded, not deleted) for audit and
  optional long-tail eval segments.
- **Manifest provenance**: `MANIFEST.json` records `corpus_revision`,
  `max_source_share_gate_threshold`, `max_rows_per_source`, `capped_sources`,
  `excluded_by_cap` (path/rows/sha256), and per-split `max_source_row_share`.
- **Opt-in dominance gate**: `max_source_row_share <= 0.25` is enforced only
  when `--max-source-share` is passed.

Reproduce the reference freeze from a deduped corpus:

```bash
quantum-corpus-freeze --corpus-dir <deduped> --output-dir <out> \
    --seed 42 --max-rows-per-source 400 --max-source-share 0.25 --revision 2
quantum-corpus-validate --manifest <out>/MANIFEST.json --corpus-dir <out>
```

The reference revision-2 manifest ships as package data at
`quantum_corpus/data/exp-004-rev2-MANIFEST.json`.

> **GRE long-tail eval segment is NOT part of the primary corpus.** The
> `eval_segments/gre_runtime_jobdata_longtail.jsonl` segment is a separate
> challenge metric for long-tail GRE runtime job data. It is excluded from the
> frozen train / val / hard_dev corpus and is evaluated independently.

## Docker

A `Dockerfile` (Python 3.13-slim, CPU torch) and a `docker-compose.yml` are
provided. The compose file mounts `./data`, `./ckpt`, and `./metatron.db` so
state persists across container restarts.

```bash
docker-compose up
```

The service exposes port 8010. The compose service is named `slm` and uses the
image tag `metatron-slm-slm:latest`. Environment inside the container sets
`PYTHONPATH=/app` and `MODEL_PATH=/app/ckpt`.

> Note: the Dockerfile installs CPU torch from
> `https://download.pytorch.org/whl/cpu`, so no CUDA runtime is pulled. If
> `docker` is not available on your machine, the project still runs natively
> via `pip install -r requirements.txt` and `uvicorn api:app --port 8010`.

## Copilot v2 — 17-Agent Orchestration

The `copilot/` directory implements a 17-agent ensemble that coordinates
real TinyMetatron loop execution. Each agent has a YAML profile defining its
role, φ-score (golden-ratio alignment), resonance frequency, and fitness.

### Architecture

Agents are organized in 4 layers, each executing through real loops or
dry-run simulation:

| Layer | Agents | Primary loop |
|---|---|---|
| **INPUT** | bio, bitnet, observer, wormhole | `corpus_loop.run_corpus_pipeline()` |
| **PROCESSING** | bronze, federation, harmonic, strategic, workflow | `train_loop.run_training()` |
| **INTEGRATION** | fractal, mirror, synthesizer, visual | aggregation + routing |
| **OUTPUT** | archivist, auditor, stealth, validator | `generalize_loop.run_gate()` |

### Execution Modes

| Mode | Behaviour |
|---|---|
| `simulation` (default) | Dry-run — agents return mock outputs, no loops executed |
| `live` | Real loop execution with actual training/corpus/evaluation |
| `hybrid` | Simulation with Qiskit IBM Quantum hardware fallback |

### Copilot API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/copilot/agents` | List all 17 agent profiles with φ-scores |
| `GET` | `/copilot/agents/{id}` | Single agent profile |
| `POST` | `/copilot/agents/{id}/invoke` | Invoke agent with task contract |
| `GET` | `/copilot/topology` | Sierpinski fractal network topology |
| `GET` | `/copilot/protocols` | Execution modes |
| `GET` | `/copilot/benchmark/results` | Latest benchmark run |
| `GET` | `/copilot/telemetry` | SSE telemetry stream (1s intervals) |
| `WS` | `/copilot/ws` | Bidirectional WebSocket dispatch |
| `POST` | `/copilot/auth/token` | Issue WS auth token (P2) |
| `POST` | `/copilot/agents/{id}/approve` | Resolve hi-tl approval (P2) |
| `GET` | `/copilot/benchmark/run` | Run benchmark suite (Phase 4) |
| `GET` | `/copilot/benchmark/ablation` | Systematic ablation study (Phase 4) |
| `GET` | `/copilot/benchmark/baseline` | Baseline comparison (Phase 4) |

### Security (P0–P3)

| Priority | Feature | Implementation |
|---|---|---|
| P0 | CORS + rate limiting | `slowapi` — 10/min invoke, 30/min SSE |
| P0 | Input validation | Pydantic `Field(max_length)` on all request models |
| P1 | Prompt injection guard | `copilot/security/prompt_guard.py` — 25+ regex patterns |
| P1 | Output redaction | `copilot/security/output_guard.py` — API key/token redaction |
| P1 | Least-privilege capabilities | `copilot/security/capabilities.py` — 17 role × bounded action sets |
| P1 | RAG content isolation | `<untrusted_content>` tags on all corpus data |
| P1 | CSP headers | `Content-Security-Policy` in `index.html` |
| P2 | Message signing | HMAC-SHA256 — verify on deliver, drop forged/stale (30s TTL) |
| P2 | Human-in-the-loop approval | `HIGH_RISK_ACTIONS` gate + `/approve` endpoint |
| P2 | WS token auth | Two-step JWT — issue token via `/auth/token`, present on WS connect |
| P2 | Corpus integrity | SHA-256 verification + `state/corpus_hashes.jsonl` |
| P2 | Provenance tracking | Append-only `state/provenance.jsonl` per data event |
| P3 | Non-root Docker | `USER metatron` (uid 1000) in both Dockerfiles |
| P3 | Security audit CI | `pip-audit` + `npm audit` + CodeQL on every push/PR |
| P3 | In-memory token storage | `ApprovalManager` singleton — process-local, no disk |

### Benchmark Suite (Phase 4)

Three benchmark adapters + ablation study:

- **τ-bench** (`tau_bench.py`): multi-turn dialogue routing evaluation;
  maps customer_support→synthesizer, technical_support→validator, etc.
- **SWE-bench** (`swebench.py`): software engineering task planning via
  analysis routing; returns `success_likelihood` per instance
- **Ablation study** (`ablation.py`): removes each agent/layer/feature
  individually and measures the Δ in coordination quality score

## Hugging Face Docker Space

This repo deploys as a **Hugging Face Docker Space** (`sdk: docker`,
`app_port: 7860`). Use `Dockerfile-hf-space` (not the root `Dockerfile`)
for Space deployment — it includes all Phase 1–4 components, non-root user,
and HF metadata labels.

**Read-only demo by default.** The API runs in `TMT_DEPLOY_MODE=demo` mode:

- Public: `/health`, `/generate`, `/model/info`, `/data/stats`, `/train/status`.
- Locked (return `403`): `/train/start` and `/data/add`. These are enabled only
  when `TMT_DEPLOY_MODE=private-training` **and** a valid `X-API-Key` header
  matching the `TMT_API_KEY` Secret is supplied.

Set these in the Space **Settings → Variables and secrets**:

| Name | Kind | Purpose |
|------|------|---------|
| `TMT_DEPLOY_MODE` | Variable | `demo` (default) or `private-training`. |
| `TMT_API_KEY` | Secret | Required for `/train/start` + `/data/add` in private mode. |
| `TMT_DB_PATH` | Variable (optional) | Override the SQLite path (e.g. a mounted `/data` volume). |
| `TMT_CHECKPOINT_DIR` | Variable (optional) | Override the checkpoint directory. |

> The demo stores its SQLite DB and checkpoint in the image layer (`/app`),
> so it works with no attached storage and resets to the build state on
> restart. For a stateful deployment, set `TMT_DB_PATH` and
> `TMT_CHECKPOINT_DIR` to a mounted persistent volume and switch to
> `private-training`.
>
> Demo output quality: the build-time checkpoint is trained for ~300 steps on
> ~400 synthesized rows — enough to showcase the architecture serving a real
> checkpoint, not fluent generation.

### Push to Hugging Face Space

```bash
# Install HF CLI and login
pip install huggingface_hub
huggingface-cli login

# Push this repo as a Space (creates tinymetatron-copilot Space)
huggingface-cli repo create tinymetatron-copilot --repo-type space --sdk docker

# Set the Space SDK to docker in the repo
git clone https://huggingface.co/spaces/<username>/tinymetatron-copilot
cd tinymetatron-copilot
# Replace the README.md content with the README from this repo
# Push — HF automatically builds the Docker image from Dockerfile-hf-space
git add .
git commit -m "init copilot v2"
git push
```

The Space `README.md` only needs the YAML front-matter; all implementation
lives in `Dockerfile-hf-space` which HF reads automatically.

## Testing

```bash
python -m pytest -q
```

Tests cover sparse attention (octa/dodeca face counts, CSR sorted, forward
shape), MoE (load sum, aux loss, reset, device), global memory (persistence,
set_state device, local attention), the sharding planner (in isolation),
the compiler (Morton bijection, CSR sorted, int8, bandwidth), tokenizer
(vocab 291, round-trip), model (forward shape, 5-7M params, generate range,
backward), DB, API, and training.

## Configuration

All model constants live in `config.py` and are frozen. Every module imports
`from config import CONFIG` (or `get_config()`). `d_ff=112` is tuned so the
total parameter count lands in the 5-7M spec budget; the model agent may
adjust `d_ff` (and only `d_ff`) if the realized count falls outside that
range. Run `python config.py` to print the parameter breakdown.