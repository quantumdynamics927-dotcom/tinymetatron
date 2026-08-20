# TMT Implementation Contract (FROZEN)

This document is the binding contract every implementation agent reads before
writing code. `config.py` is already written and FROZEN — import from it; never
redefine model constants. `d_ff=112` is tuned so total params ≈ 6.26M (in the
5–7M spec budget); the model agent may adjust `d_ff` ONLY if the actual count
falls outside [5e6, 7e6].

Project root: `D:/Users/Documents/TMT (Tensor Metatron Technology)` (flat repo,
no package). Python 3.13, torch 2.13 CPU, pytest 9. Run code from repo root.

## 0. Global rules for ALL agents

1. Read `config.py` and THIS contract before writing any code.
2. `from config import CONFIG` (or `get_config`) for every model constant. No
   hardcoded `256`/`4`/`6`/`32`/`291`/`13`/`2` in logic — read `CONFIG[...]`.
3. Only create/edit files listed in your task below. Do NOT touch other files.
4. Every `if __name__ == "__main__":` block starts with
   `import sys; sys.stdout.reconfigure(encoding="utf-8")` (Windows cp1252 fix).
   No ✓/✗/accented literals in print without that reconfigure.
5. Derive device from input tensors (`x.device`); never assume CPU. Derive dtype
   from inputs where relevant. Registered buffers must follow `.to(device)`.
6. After writing, SELF-TEST by importing your module and (where applicable)
   running its `__main__` or a quick `python -c "import <mod>"`. Fix until clean.
7. Use type hints. Keep docstrings referencing the patent claims (existing style).
8. Do NOT create a persistent `metatron.db` or `ckpt/` during implementation —
   tests and the verify phase handle real runs. Use temp paths in any scratch.

## 1. Interface contract (post-fix)

### config.py (DONE — frozen)
`CONFIG` dict, `get_config() -> dict`, `estimate_total_params(d_ff=None)`.

### metatron_sparse_attention.py — API CHANGES
- Fix POLYHEDRA: octahedron → 8 correct triangular faces
  `[[0,1,2],[0,2,3],[0,3,4],[0,4,1],[5,2,1],[5,3,2],[5,4,3],[5,1,4]]` (V=6,
  apexes 0 top / 5 bottom, equator 1-2-3-4). dodecahedron → 12 correct
  pentagonal faces (standard labeling; every face 5 vertices, total 12).
- `geometric_chords`: add `cross` to the set before discarding idx.
- `MetatronSparseMask.csr`: delete dead `indptr=[0]; for r in rows: pass`; build
  indices SORTED within each row (CSR invariant for torch.sparse_csr_tensor).
- `measure_sparsity`: rename `ops_reduction_pct` → `mask_sparsity_pct` (do not
  claim realized FLOP reduction; the forward is masked-dense reference).
- **`MetatronSparseAttention` forward contract**:
  ```python
  class MetatronSparseAttention(nn.Module):
      def __init__(self, d_model, n_heads, mask=None, dropout=0.1, bias=False): ...
      def forward(self, x, mask=None) -> torch.Tensor:
          # x: (B, L, d_model) self-attention (q=k=v=x internally)
          # -> project with Linear(d_model, d_model) for Q,K,V,O (standard MHA)
          # -> reshape to (B, H, L, d_k), scale=1/sqrt(d_k)
          # -> apply sparse mask from active_mask.edges (set -inf elsewhere)
          # -> softmax; sparse-aware output via gather over mask edges (scatter_add)
          # -> output proj; return (B, L, d_model)
  ```
  Derive `r, c` from `active_mask.edges` inside forward (not stored buffers),
  on `x.device`. Drop `_mask_rows/_mask_cols/_mask_count` buffers entirely.
  The scatter_add index must be 4-D: `idx = r.view(1,1,n_edges,1).expand_as(weighted)`
  with `weighted` shaped `(B,H,n_edges,d_k)`.
- Module-level functions `radial_level/local_window/geometric_chords/long_range_routes`
  keep their signatures. `MetatronSparseMask.build` keeps signature.
- `__main__` smoke test must PASS: build mask for all 5 solids (print sparsity +
  edges), then `MetatronSparseAttention(d_model=256, n_heads=4, mask=mask)`;
  `x = torch.randn(B, L, 256)` (NOT pre-split heads); `out = attn(x)`; assert
  `out.shape == (B, L, 256)`.

### metatron_moe.py — API CHANGES
- Load tracking: aggregate over ALL k slots:
  `all_idx = topk_idx.reshape(-1); counts = torch.bincount(all_idx, minlength=n_e).float()`
  (do NOT reference loop-leftover `expert_ids`). `last_load.sum() == top_k*B*L`.
- Honor `return_load`: `if return_load: return out.view(B,L,D), counts`.
- Add `return_aux: bool = False` param; compute
  `aux_loss = n_e * (fractions).pow(2).sum()` where
  `fractions = counts / counts.sum()` (Switch-Transformer balancing loss);
  return `(out, counts, aux_loss)` when both flags set. Provide
  `reset_load()` zeroing `_expert_counts`.
- `expert_utilisation()` returns `self._expert_counts/(sum+1e-8)` on the module
  device (never CPU zeros).
- Decide `_expert_counts` vs `last_load`: `last_load` = per-call counts (source
  of truth for aux loss); `_expert_counts` = cumulative (exposed via
  `expert_utilisation`, resettable). Keep both, document.
- `capacity_factor`: keep stored; do NOT implement dropping (document as
  reserved). Keep signature.
- `__main__` smoke test must PASS: `out = moe(x)` shape (B,L,D);
  `out, load = moe(x, return_load=True)`; assert `load.sum().item() == top_k*B*L`;
  `out, load, aux = moe(x, return_load=True, return_aux=True)`; assert
  `aux.item() > 0`. Print utilization, active experts.

### metatron_global_memory.py — API CHANGES
- Fix smoke test line: `torch.round(attn[0,:3,:5], decimals=3)`.
- `MetatronMemoryEnhancedAttention.forward`: APPLY local_attn:
  `x = self.norm1(x); if self.local_attn is not None: x = x + self.local_attn(x);
  x = self.norm2(x + self.global_mem(x)); return x`. (Fixes double-residual: the
  inner `MetatronGlobalMemory.forward` keeps its own residual, so the wrapper
  must NOT also add raw x twice — use the pattern above; ensure x added once in
  the global branch via `x = self.norm2(x + self.global_mem(x))` where
  `global_mem` already does `local_x + dropout(proj(...))`. Document clearly.)
- `set_state`: `with torch.no_grad(): self.memory.copy_(new_memory.to(
  self.memory.device, self.memory.dtype))` (preserve device/dtype).
- Gating: keep per-sequence (mean over dim=1) BUT document it explicitly in the
  docstring as per-sequence gating; OR upgrade to per-token
  `gate_val = torch.sigmoid(self.gate(local_x))`. Implementer choice; per-token
  preferred for expressivity. Document the choice.
- Fix persistence smoke assertion: capture `p0 = mem.memory.data_ptr()` once,
  call forward again, compare `mem.memory.data_ptr() == p0`.
- Clean up the contradictory "buffer not learnable" comment: memory IS a learnable
  Parameter; fix the comment.
- `n_heads` arg of `MetatronMemoryEnhancedAttention`: wire it (pass to local_attn
  when set externally) or drop it. Document.
- `__main__` smoke test must PASS: prints shapes, `persists: True`.

### metatron_shard.py — fixes (tested in isolation; NOT wired into CPU model)
- `__main__`: ASCII `OK`/`FAIL` (or reconfigure stdout UTF-8). Must run clean.
- `compute_diameter`: HONOR the topology argument (add `topology=None` param; if
  None use `self.topology`). Correct diameter table: tetra=1, octa=2, cube=3,
  icosa=3, dodeca=5.
- `diameter_reduction(base_topology)`: instantiate a SEPARATE planner for the base
  and compute its diameter (do NOT use `self.topology` for both). Must return
  >0 for dodeca vs hexa (5 vs 3 → 0.4 ≥ 0.333).
- `bytes_per_token`: set to 4 (float32) in radial/ring; pass total bytes
  (`token_count*4`) to `_est_latency`. Chord already uses 4.
- `_balance_load`: recompute ALL shard boundaries from new per-shard token counts
  derived from load ratios (no independent end-grow → overlaps/gaps). Init
  `self.total_tokens=0` in `__init__`.
- Rename `ICOSADISTRIBUTED` → `ICOSA_DISTRIBUTED`; update `_strategy_name` and
  `_effective_shards`.
- Wire `_min_cut_routes` into `plan()` (use to inform chord/ring targets) OR
  remove it and drop the docstring claim. Implementer choice; if kept, make it a
  real BFS shortest-path over the face-adjacency graph.
- Remove dead `tokens_per_shard` in `_shard_to_face`.
- `__main__` smoke test must PASS: prints diameters, `reduction` ≥ 33.3% as OK,
  then `plan(1024).print_summary()` and phase breakdown.

### metatron_compiler.py — fixes
- `morton_reorder`: sort by RAW morton key (`order.append(morton)`), not
  `morton % seq_len`. Result must be a bijection (permutation of 0..seq_len-1).
  Update the docstring example to match actual output.
- `compile_to_csr`: sort columns within each row (CSR invariant). Return sorted
  (indptr, indices).
- `bandwidth_estimate`: `*4` not `*3` (Q,K,V,O = 4 transfers). Accept `d_k`
  param (default 64) instead of hardcoding.
- `flop_estimate`/`flops_per_token`/`summary`: accept `d_k` param (default 64).
- `polyhedral_ring_order`: real BFS from vertex 0 over `poly['edges']`; drop the
  `[1]` default (use `[]`).
- `Quantizer('int8').quantize`: guard scale — `amax = tensor.abs().max();
  scale = amax/127.0 if amax>0 else torch.tensor(1.0)`.
- `__main__` smoke test must PASS: prints Morton order, CSR lengths, kernel
  summary, quantized dtype.

## 2. New files (each owned by exactly ONE agent)

| File | Owner tier | Public interface |
|---|---|---|
| `tokenizer.py` + `vocab.json` | T2 | `Tokenizer` class: `encode(text)->list[int]`, `decode(ids)->str`, `vocab_size` prop, `special_ids` dict, `@classmethod from_file(path)`. `vocab.json` = token→id map. |
| `db.py` | T2 | `init_db(path)`, `add_texts(path, texts, domain, quality_threshold)->(added,rejected)`, `fetch_training_rows(path, domain, min_quality, limit, used=False)->list[dict]`, `mark_used(path, ids)`, `stats(path)->dict`, `save_checkpoint(path, step, loss, file_path, is_active=True)`, `start_session(path, domain_filter, min_quality)->int`, `end_session(path, session_id, total_steps, final_loss)`, `get_active_checkpoint(path)->dict\|None`, `set_active_checkpoint(path, file_path)`. |
| `quality.py` | T2 | `score_quality(text)->float` in [0,1] (heuristic: length, repetition, keyword density for technical domains). |
| `tinymetatron_model.py` | T2 | `TinyMetatron(config: dict)->nn.Module`; `.forward(input_ids: LongTensor[B,L])->logits[B,L,vocab]`; `.generate(input_ids, max_length, temperature)->LongTensor[B,L_out]`; `.param_count()->dict`; classmethod `from_config()`; `load_checkpoint(path)`, `save_checkpoint(path)`. |
| `train_db.py` | T3 | `main(argv)` CLI; flags per contract §3. Prints Slovak progress (`Krok N/M: loss=x.xxx`) + `Tréning dokončený. Finálna strata: x.xxx`. |
| `manage_data.py` | T3 | `main(argv)` with subcommands `generate/import/export/stats/clean`. |
| `api.py` | T3 | `app: FastAPI`; endpoints `/generate`, `/train/start`, `/train/status`, `/data/add`, `/data/stats`, `/model/info`. Training runs in a background thread; status is module-global. |
| `tests/test_sparse_attention.py` | T4a | shapes, mask-override, octa/dodeca faces correct, smoke forward, csr sorted. |
| `tests/test_moe.py` | T4a | load.sum()==top_k*B*L; return_load tuple; aux>0; reset_load; device. |
| `tests/test_global_memory.py` | T4a | persistence across 2 forwards; set_state device; local_attn applied; smoke. |
| `tests/test_shard.py` | T4a | diameter_reduction(dodeca vs hexa)>=0.333; no overlapping ranges; UTF-8 smoke; bytes_per_token==4. |
| `tests/test_compiler.py` | T4a | morton_reorder is a bijection on 0..127 & matches docstring; CSR sorted within rows; int8 zero tensor; bandwidth *4. |
| `tests/test_tokenizer.py` | T4b | vocab_size==291; round-trip SK+EN technical text (lossless for in-vocab chars); BOS/EOS wrapping; UNK fallback. |
| `tests/test_model.py` | T4b | forward (B,32)->(B,32,291); total_params in [5e6,7e6]; generate ids in [0,291); loss+backward runs. |
| `tests/test_db.py` | T4b | schema create; insert/fetch; mark_used; active checkpoint swap; stats keys. |
| `tests/test_api.py` | T4b | FastAPI TestClient: /data/add 200, /data/stats keys, /model/info config, /generate 200. |
| `tests/test_train.py` | T4b | 2-step run on tiny DB: loss decreases; checkpoint row written; session end_time set; used_in_training flips. |
| `Dockerfile` | T4c | python:3.13-slim, CPU torch wheel, fastapi/uvicorn/pydantic, /app, EXPOSE 8010, CMD uvicorn api:app. |
| `docker-compose.yml` | T4c | `slm` service, image `metatron-slm-slm:latest` (verbatim), 3 volumes (./data ./ckpt ./metatron.db), port 8010, env PYTHONPATH/MODEL_PATH. |
| `requirements.txt` | T4c | pinned: torch (cpu index url note), fastapi, uvicorn, pydantic, pytest. |
| `README.md` | T4c | build/run/train/API/CLI instructions, endpoint list, vocab note, param count. |

## 3. CLI & API details

`train_db.py` flags: `--steps`(200) `--domain`(general) `--min_quality`(0.5)
`--batch_size`(16) `--learning_rate`(1e-3) `--max_seq_len`(32) `--device`(cpu)
`--checkpoint_dir`(ckpt) `--db_path`(metatron.db) `--aux_loss_weight`(0.01)
`--seed`(42). Behavior: load active ckpt or init; fetch rows from training_data
WHERE domain=? AND quality>=? AND used_in_training=0; mark used; insert
training_sessions row; train; write model_checkpoints row + .pt (is_active=1,
clear prior); update session end_time/final_loss. Optimizer: Adam. Loss: CE over
LM logits + aux_loss_weight*moe_aux.

`manage_data.py` subcommands:
- `generate --domain X --count N`: synthesize N rows (use a small built-in
  template list per domain: cybersecurity/software/general; deterministic).
- `import --file F --domain X`: read lines, score, insert.
- `export --domain X --min_quality Q [--out F]`: print/dump matching rows.
- `stats`: print total, by_domain, avg_quality, used_in_training counts.
- `clean --min_quality Q`: delete rows with quality < Q.

`api.py`: FastAPI. Module-global `_train_state` dict {is_training, current_step,
total_steps, current_loss, session_id}. `/train/start` launches a background
thread calling `train_db`-equivalent logic (refactor a callable
`run_training(config_override) -> None` in train_db.py that api.py imports).
`/generate` loads active ckpt, tokenizes prompt, `model.generate`, decodes.
`/data/add` calls `db.add_texts` with `quality.score_quality`. `/data/stats`
calls `db.stats`. `/model/info` returns CONFIG + active checkpoint.

## 4. Coherence risks & mitigations (binding)

- **R1 config drift**: import CONFIG everywhere (rule 2). config.py frozen.
- **R2 attention contract**: forward takes (B,L,d_model) — model passes hidden,
  NOT pre-split heads. This is mandatory; the fixed module enforces it.
- **R3 device/dtype**: derive from inputs (rule 5). Tests gate on
  `cuda if available` else cpu.
- **R4 namespace**: only files in §2 may exist. No shared utils.py.
- **R5 vocab determinism**: `vocab.json` written once by the T2 tokenizer agent;
  model/tests only consume `Tokenizer`.
- **R6 aux-loss wiring**: trainer reads `return_aux=True`; test_train asserts
  `aux_loss.detach() > 0` on a step.
- **R7 UTF-8**: rule 4. Tests run with `PYTHONIOENCODING=utf-8`.
- **R8 param budget**: test_model asserts `5e6 <= total <= 7e6`; model agent
  retunes `d_ff` (edit config.py's d_ff only) if outside, then re-run
  `python config.py` to confirm.
- **R9 shard not in CPU path**: TinyMetatron never imports metatron_shard;
  shard tested in isolation only.

## 5. Self-test before handing off

Each agent must, after writing, run from repo root:
- `python -c "import <your_module>"` (must exit 0)
- `python <your_module>` for modules with `__main__` (must exit 0, no traceback)
- For test files: `python -m pytest <your_test_file> -q` must be green.
Fix until clean. Report any deviation.