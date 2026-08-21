"""
api.py
======
FastAPI REST API for the TinyMetatron SLM.

Exposes the model, the incremental trainer and the SQLite data layer over a
small HTTP surface (IMPLEMENTATION_CONTRACT.md section 2 + section 3):

    POST /generate       {prompt, max_length, temperature} -> {text, tokens_generated, step}
    POST /train/start     {steps, domain, min_quality, learning_rate?}
                          -> 202 {session_id, status, total_steps}
    GET  /train/status   -> {is_training, current_step, total_steps, current_loss}
    POST /data/add        {texts, domain, quality_threshold} -> {added, rejected, ...}
    GET  /data/stats      -> db.stats
    GET  /model/info      -> {config, active_checkpoint}

Training runs in a background thread; status is reported through a
module-global ``_train_state`` dict guarded by a ``threading.Lock`` so the
HTTP handlers and the trainer thread never race on the shared fields.

The SQLite DB path is taken from the ``TMT_DB_PATH`` env var (defaulting to
``CONFIG['db_path']``) so tests can point the API at a throwaway temp DB
without touching the real ``metatron.db`` (contract rule 0.6).

Patent references: the API is the thin serving layer over the Metatron
sparse-attention + 13-expert MoE + shared global-memory SLM described in the
IMPLEMENTATION_CONTRACT.
"""

from __future__ import annotations

import os
import secrets
import threading
from typing import List, Optional

import torch

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from config import CONFIG, get_config
import db
import quality  # noqa: F401  (db.add_texts imports it lazily; kept for clarity)
from tokenizer import default_tokenizer, Tokenizer
from tinymetatron_model import TinyMetatron
import train_db


# ── DB path resolution ───────────────────────────────────────────────────────
def _db_path() -> str:
    """
    Return the SQLite DB path for this API instance.

    ``TMT_DB_PATH`` env var overrides CONFIG['db_path'] so tests can point at
    a temp DB.  Falls back to CONFIG['db_path'] when unset.
    """
    env = os.environ.get("TMT_DB_PATH")
    if env:
        return env
    return CONFIG["db_path"]


def _ckpt_dir() -> str:
    """
    Return the checkpoint directory for this API instance.

    ``TMT_CHECKPOINT_DIR`` env var overrides CONFIG['checkpoint_dir'] so a
    stateful deployment can point at a mounted /data volume.  Falls back to
    CONFIG['checkpoint_dir'] when unset.
    """
    env = os.environ.get("TMT_CHECKPOINT_DIR")
    if env:
        return env
    return CONFIG["checkpoint_dir"]


# ── Deployment mode + admin gating ───────────────────────────────────────────
# Read at request time (not import time) so tests can flip them via env.
#   TMT_DEPLOY_MODE = "demo" (default, read-only public Space)
#                     "private-training" (enables /train/start + /data/add)
#   TMT_API_KEY      = bearer-style secret required for state-changing endpoints
def _deploy_mode() -> str:
    return os.environ.get("TMT_DEPLOY_MODE", "demo")


def _allow_training() -> bool:
    return _deploy_mode() == "private-training"


def _allow_data_writes() -> bool:
    return _deploy_mode() == "private-training"


def _require_training_enabled(
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")
) -> None:
    """Dependency: training must be enabled AND a valid X-API-Key supplied."""
    if not _allow_training():
        raise HTTPException(
            status_code=403,
            detail="training disabled (TMT_DEPLOY_MODE != 'private-training').")
    key = os.environ.get("TMT_API_KEY")
    if not key:
        raise HTTPException(
            status_code=403,
            detail="training requires TMT_API_KEY to be configured.")
    if not x_api_key or not secrets.compare_digest(str(x_api_key), str(key)):
        raise HTTPException(status_code=403,
                            detail="invalid or missing X-API-Key.")


def _require_data_writes_enabled(
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")
) -> None:
    """Dependency: data writes must be enabled AND a valid X-API-Key supplied."""
    if not _allow_data_writes():
        raise HTTPException(
            status_code=403,
            detail="data writes disabled (TMT_DEPLOY_MODE != 'private-training').")
    key = os.environ.get("TMT_API_KEY")
    if not key:
        raise HTTPException(
            status_code=403,
            detail="data writes require TMT_API_KEY to be configured.")
    if not x_api_key or not secrets.compare_digest(str(x_api_key), str(key)):
        raise HTTPException(status_code=403,
                            detail="invalid or missing X-API-Key.")


# ── Private /ask endpoint (quantum corpus RAG) ───────────────────────────────
# The /ask endpoint serves the PRIVATE quantum corpus (build 2) over BM25 RAG.
# It is auth-gated AND disabled in demo mode, so the public HF Space (which runs
# in demo mode with no key) is never exposed: a public /ask returns 403. Bind a
# private instance to localhost/VPN (HOST=127.0.0.1) — never 0.0.0.0 public.
#
# Per the held-out eval (quantum_corpus/eval/REPORT.md), the naive extractive
# path (a) never abstains and (b) leaks a seeded canary on secret-requesting
# questions. /ask therefore applies TWO gates before answering:
#   1. Secret/credential request detection -> always abstain, never echo content.
#   2. Low BM25 score floor -> abstain "insufficient support" (catches the
#      clearly-unmatched cases; calibrated below the minimum score of any real
#      gold hit so it never abstains on a genuine answer).
# The 32-token TinyMetatron window cannot ingest retrieved records, so /ask
# returns the retrieved EVIDENCE directly (citations + bounded snippets) plus a
# short templated synthesis, and only OPTIONALLY attaches a tiny bounded LM
# continuation. This is the reliable path endorsed until context length is
# raised (deferred fine-tune track #1).
import re as _re

# Secret/credential request patterns. If the question asks for any of these, the
# endpoint abstains and returns NO retrieved content (prevents echo-leakage of
# redacted/seeded material). Research terms (backend, QASM, OTOC, fidelity) are
# intentionally NOT here.
_SECRET_PAT = _re.compile(
    r"\b(ibmid|api[\s_-]?key|apikey|access[\s_-]?token|secret|password|passwd|"
    r"private[\s_-]?key|pem\b|recovery[\s_-]?(phrase|seed|key)|credential|"
    r"\bcrn\b|account[\s_-]?id|webhook|xox-|gh[opsur]_|hf_|sk-)\b",
    _re.IGNORECASE,
)

# BM25 score floor for "insufficient support" abstention. Calibrated from the
# test eval: the minimum top-1 score on any real gold hit was 9.76, while the
# clearly-unanswerable cases scored 2.24-3.24. A floor of 3.0 abstains only on
# the latter, never on a genuine hit.
_ASK_SCORE_FLOOR = 3.0

# Sensitivity ranking: public < internal < sensitive. The default query excludes
# 'sensitive' records (the TMT_Quantum_Vault sensitive subdirs) unless the caller
# explicitly opts in via sensitivity="sensitive".
_SENS_RANK = {"public": 0, "internal": 1, "sensitive": 2}
_ASK_MAX_CONTEXT_CHARS = 160  # per-snippet bound for the returned evidence


def _quantum_db_path() -> Optional[str]:
    """Return the private quantum corpus DB path, or None if not configured."""
    from quantum_corpus import schema as _qschema
    p = _qschema.default_db_path()
    return p if (p and os.path.isfile(p)) else None


# Cached train+val HYBRID retriever (BM25 + semantic, RRF fusion). Built over
# ALL train+val records once; sensitivity filtering is applied per-request at
# query time (so one warm cache serves every sensitivity level). Rebuilt only
# if the DB file mtime changes. The semantic half degrades to BM25-only when
# sentence-transformers is absent (the public Space stays unaffected).
_ask_retriever_cache: dict = {"retriever": None, "db_path": None, "mtime": None}
_ask_retriever_lock = threading.Lock()

# Tuned gate thresholds (tuned on the frozen validation set; see
# quantum_corpus/eval/tune.py + REPORT_v03.md). Override defaults from the
# answer engine with the frozen hybrid values.
_ASK_GATES = None  # cached copy of the frozen answer.DEFAULT_GATES


def _ask_gates() -> dict:
    global _ASK_GATES
    if _ASK_GATES is None:
        from quantum_corpus import answer as _qanswer
        # Frozen v0.3 gate thresholds (tuned on the val set via tune.py):
        # score_floor_hybrid=0.02, sep_ratio=1.5, min_concepts=1, sep_band=2.0,
        # score_floor_bm25=3.0. See answer.DEFAULT_GATES + REPORT_v03.md.
        _ASK_GATES = dict(_qanswer.DEFAULT_GATES)
    return _ASK_GATES


def _get_ask_retriever():
    """Return a cached HybridRetriever over the TRAIN+VAL records (never test)."""
    from quantum_corpus import schema as _qschema
    from quantum_corpus.fusion import HybridRetriever as _HR
    import sqlite3 as _sqlite3
    dbp = _quantum_db_path()
    if dbp is None:
        return None, None
    mtime = os.path.getmtime(dbp)
    with _ask_retriever_lock:
        if (_ask_retriever_cache["retriever"] is None
                or _ask_retriever_cache["db_path"] != dbp
                or _ask_retriever_cache["mtime"] != mtime):
            conn = _sqlite3.connect(dbp); conn.row_factory = _sqlite3.Row
            # Train+val ONLY — the test split stays held-out for live serving.
            rows = conn.execute(
                "SELECT * FROM corpus_records WHERE split IN ('train','val')",
            ).fetchall()
            conn.close()
            retr = _HR.build([dict(r) for r in rows])
            _ask_retriever_cache.update(retriever=retr, db_path=dbp, mtime=mtime)
        return _ask_retriever_cache["retriever"], dbp


def _get_ask_structured():
    """Return a StructuredQuery over the sidecar DB, or None if not built."""
    try:
        from quantum_corpus.structured import StructuredQuery as _SQ
        return _SQ()
    except Exception:
        return None


def _ask_query(retriever, question: str, top_k: int, max_sensitivity: str):
    """Unified query: HybridRetriever takes max_sensitivity."""
    try:
        return retriever.query(question, k=top_k, max_sensitivity=max_sensitivity)
    except TypeError:
        return retriever.query(question, k=top_k)


def _ask_build_id(dbp: str) -> dict:
    """Return the frozen build id + sha256 from the eval manifest (if present)."""
    import hashlib as _hl
    # api.py lives at the repo root; the manifest is at repo/quantum_corpus/eval.
    manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "quantum_corpus", "eval", "manifest.json")
    if os.path.isfile(manifest_path):
        try:
            import json as _json
            m = _json.load(open(manifest_path, encoding="utf-8"))
            return {"build_id": m.get("build_id"),
                    "build_sha256": m.get("db_sha256")}
        except Exception:
            pass
    # fall back to the live DB hash
    h = _hl.sha256()
    with open(dbp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"build_id": "live-db", "build_sha256": h.hexdigest()}


def _require_ask_enabled(
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")
) -> None:
    """Dependency: /ask is private-only. Disabled in demo mode (public Space
    protection) AND requires a valid X-API-Key."""
    if _deploy_mode() == "demo":
        raise HTTPException(
            status_code=403,
            detail="/ask is disabled in demo mode (public Space protection). "
                    "Run a private instance with TMT_DEPLOY_MODE=private-training.")
    key = os.environ.get("TMT_API_KEY")
    if not key:
        raise HTTPException(
            status_code=403,
            detail="/ask requires TMT_API_KEY to be configured.")
    if not x_api_key or not secrets.compare_digest(str(x_api_key), str(key)):
        raise HTTPException(status_code=403,
                            detail="invalid or missing X-API-Key.")


class AskRequest(BaseModel):
    question: str = Field(..., max_length=2000,
                          description="Natural-language question (<=2000 chars).")
    top_k: int = Field(5, ge=1, le=20,
                       description="Number of records to retrieve (1-20).")
    mode: str = Field("quantum-private",
                      description='Must be "quantum-private".')
    sensitivity: str = Field(
        "internal",
        description="Max sensitivity to include: public|internal|sensitive. "
                    "'sensitive' opts into TMT_Quantum_Vault sensitive records.")


# ── Model cache for /generate (perf on CPU) ──────────────────────────────────
# Building a TinyMetatron + loading the active checkpoint on every request is
# expensive on a small CPU container.  Cache one model per DB path; invalidate
# when a training run completes (a new active checkpoint is then on disk).
_model_cache: dict = {"model": None, "step": 0, "db_path": None}
_model_cache_lock = threading.Lock()
_model_dirty: bool = False
# Cap concurrent /generate inference so a burst of requests cannot exhaust a
# small CPU container.  No external rate-limit dependency.
_generate_sem = threading.Semaphore(2)


def _get_model() -> "tuple[TinyMetatron, int]":
    """Return the cached (model, step), loading lazily and on invalidation."""
    global _model_dirty
    path = _db_path()
    with _model_cache_lock:
        if (_model_dirty or _model_cache["model"] is None
                or _model_cache["db_path"] != path):
            model, step = _load_model_for_generate()
            _model_cache["model"] = model
            _model_cache["step"] = step
            _model_cache["db_path"] = path
            _model_dirty = False
        return _model_cache["model"], _model_cache["step"]


# ── Module-global training status (thread-safe) ──────────────────────────────
_train_state: dict = {
    "is_training": False,
    "current_step": 0,
    "total_steps": 0,
    "current_loss": 0.0,
    "session_id": None,
}
_train_lock = threading.Lock()
# A separate lock guards the "is_training" launch critical section so two
# concurrent /train/start requests cannot both pass the 409 check and spawn
# two trainer threads.
_launch_lock = threading.Lock()


def _set_training(started: bool, session_id: Optional[int] = None,
                  total_steps: int = 0) -> None:
    """Atomically flip the training flag and reset step counters."""
    with _train_lock:
        _train_state["is_training"] = started
        if started:
            _train_state["current_step"] = 0
            _train_state["total_steps"] = total_steps
            _train_state["current_loss"] = 0.0
            _train_state["session_id"] = session_id
        else:
            # Keep last-known step/loss/total on completion; clear session id.
            _train_state["session_id"] = session_id


def _on_step(step: int, total_steps: int, loss: float) -> None:
    """on_step callback passed to train_db.run_training; updates _train_state."""
    with _train_lock:
        _train_state["current_step"] = int(step)
        _train_state["total_steps"] = int(total_steps)
        _train_state["current_loss"] = float(loss)


# ── Pydantic request models ──────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=2000,
                        description="Prompt text to continue (<=2000 chars).")
    max_length: int = Field(100, ge=1, le=256,
                            description="Maximum NEW tokens to generate (<=256).")
    temperature: float = Field(0.7, gt=0.0, le=2.0,
                               description="Sampling temperature (>0).")


class TrainStartRequest(BaseModel):
    steps: int = Field(200, ge=1, le=100000,
                       description="Number of training steps.")
    domain: str = Field("general", description="Domain filter for training rows.")
    min_quality: float = Field(0.5, ge=0.0, le=1.0,
                                description="Minimum quality_score to include.")
    learning_rate: Optional[float] = Field(
        None, gt=0.0,
        description="Adam learning rate (defaults to CONFIG['learning_rate']).")


class DataAddRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1,
                              description="Texts to score and insert.")
    domain: str = Field("general", description="Domain tag for the texts.")
    quality_threshold: float = Field(0.5, ge=0.0, le=1.0,
                                     description="Minimum score to insert.")


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="TinyMetatron SLM API",
    description="REST surface for the TinyMetatron sparse-attention + MoE SLM.",
    version="0.1.0",
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_model_for_generate() -> "tuple[TinyMetatron, int]":
    """
    Build a TinyMetatron and load the active checkpoint if one is present.

    Returns (model, step) where ``step`` is the active checkpoint's training
    step (0 when no checkpoint exists / model is untrained).  Device is taken
    from CONFIG['device'] (honouring CUDA availability); the generate path
    derives the input tensor device from the model parameters (rule 5).
    """
    device = torch.device(CONFIG["device"] if CONFIG["device"] != "cuda"
                          or torch.cuda.is_available() else "cpu")

    model = TinyMetatron.from_config()
    model.to(device)

    path = _db_path()
    db.init_db(path)
    active = db.get_active_checkpoint(path)
    step = 0
    if active is not None and active.get("file_path") \
            and os.path.isfile(str(active["file_path"])):
        try:
            model.load_checkpoint(str(active["file_path"]))
            step = int(active.get("step") or 0)
        except Exception:
            # Corrupt checkpoint -> fall back to the fresh untrained model.
            model = TinyMetatron.from_config()
            model.to(device)
            step = 0
    model.eval()
    return model, step


def _trainer_thread(steps: int, domain: str, min_quality: float,
                     learning_rate: float, session_id: Optional[int]) -> None:
    """
    Background trainer thread entry point.

    Calls train_db.run_training with the on_step callback that keeps
    _train_state current, then clears the training flag on completion
    (success or failure).  Exceptions are swallowed so the thread never
    leaves ``is_training`` stuck True.
    """
    try:
        train_db.run_training(
            steps=steps,
            domain=domain,
            min_quality=min_quality,
            batch_size=CONFIG["batch_size"],
            learning_rate=learning_rate,
            max_seq_len=CONFIG["seq_len"],
            device=CONFIG["device"],
            checkpoint_dir=_ckpt_dir(),
            db_path=_db_path(),
            aux_loss_weight=CONFIG["moe_aux_loss_weight"],
            seed=CONFIG["seed"],
            on_step=_on_step,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        # The trainer thread must never crash silently leaving is_training set.
        print(f"[api] trainer thread error: {exc!r}")
    finally:
        # A new active checkpoint may now be on disk -> force /generate reload.
        global _model_dirty
        _model_dirty = True
        _set_training(False, session_id=session_id)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    """Liveness probe (used by the HF Space and the local curl gate)."""
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root() -> HTMLResponse:
    """Landing page for the Hugging Face Space embedded app view."""
    return HTMLResponse("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TinyMetatron SLM</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         max-width: 760px; margin: 0 auto; padding: 2rem; }
  h1 { margin: 0 0 .25rem; }
  .sub { color: #888; margin: 0 0 1.25rem; }
  code, .mono { font-family: ui-monospace, Consolas, monospace; }
  textarea { width: 100%; box-sizing: border-box; font: inherit; padding: .5rem; }
  row { display: flex; gap: 1rem; flex-wrap: wrap; align-items: end; }
  label { font-size: .85rem; }
  button { font: inherit; padding: .55rem 1.1rem; cursor: pointer; border: 0;
           border-radius: 6px; background: #6d28d9; color: #fff; }
  button:disabled { opacity: .5; cursor: wait; }
  .out { white-space: pre-wrap; background: rgba(125,125,125,.12);
         padding: .75rem 1rem; border-radius: 8px; min-height: 1.5rem;
         border: 1px solid rgba(125,125,125,.2); }
  .meta { color: #888; font-size: .82rem; margin-top: .4rem; }
  nav a { margin-right: 1rem; }
  hr { border: 0; border-top: 1px solid rgba(125,125,125,.25); margin: 1.5rem 0; }
</style>
</head>
<body>
<h1>🧠 TinyMetatron SLM</h1>
<p class="sub">Sparse polyhedral attention · 13-expert MoE · shared global memory · ~6.35M params · CPU demo</p>

<p>Generate text from the build-time trained checkpoint:</p>
<textarea id="prompt" rows="2">the firewall</textarea>
<div class="row" style="display:flex; gap:1rem; flex-wrap:wrap; align-items:flex-end; margin:.6rem 0;">
  <label>max new tokens <input id="maxlen" type="number" value="32" min="1" max="256" style="width:6rem"></label>
  <label>temperature <input id="temp" type="number" value="0.7" step="0.1" min="0.1" max="2.0" style="width:6rem"></label>
  <button id="go" onclick="runGen()">Generate</button>
</div>
<div class="out" id="out">—</div>
<div class="meta" id="meta"></div>

<hr>
<nav>
  <a href="/docs">API docs</a>
  <a href="/health">/health</a>
  <a href="/model/info">/model/info</a>
  <a href="/data/stats">/data/stats</a>
  <a href="/train/status">/train/status</a>
</nav>
<p class="sub" style="margin-top:1.5rem">Read-only demo · <code>/train/start</code> and <code>/data/add</code> are disabled (403). Output is from a lightly-trained smoke checkpoint, not a polished LM.</p>

<script>
async function runGen(){
  const go=document.getElementById('go'), out=document.getElementById('out'), meta=document.getElementById('meta');
  go.disabled=true; out.textContent="generating…"; meta.textContent="";
  const body={prompt:document.getElementById('prompt').value,
              max_length:parseInt(document.getElementById('maxlen').value,10),
              temperature:parseFloat(document.getElementById('temp').value)};
  try{
    const t0=performance.now();
    const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(!r.ok){ out.textContent='error '+r.status; meta.textContent=JSON.stringify(d); return; }
    out.textContent=d.text||'(empty)';
    meta.textContent=`step ${d.step} · ${d.tokens_generated} new tokens · ${Math.round(performance.now()-t0)} ms`;
  }catch(e){ out.textContent='request failed'; meta.textContent=String(e); }
  finally{ go.disabled=false; }
}
</script>
</body>
</html>""")


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    """
    Generate text from a prompt using the active checkpoint (or an untrained
    TinyMetatron when no checkpoint exists).

    Returns {text, tokens_generated, step}.
    """
    tokenizer = default_tokenizer()
    prompt_ids = tokenizer.encode(req.prompt)
    if not prompt_ids:
        prompt_ids = [CONFIG["bos_id"], CONFIG["eos_id"]]

    # Concurrency cap + cached model: the slow part (model.generate) runs
    # under the semaphore; the model is loaded once and reused.
    with _generate_sem:
        model, step = _get_model()
        device = next(model.parameters()).device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out_ids = model.generate(input_ids, max_length=req.max_length,
                                     temperature=req.temperature)

    # tokens_generated = number of NEW tokens (output length minus prompt).
    new_count = int(out_ids.shape[1]) - len(prompt_ids)
    if new_count < 0:
        new_count = 0
    text = tokenizer.decode(out_ids[0].tolist())
    return {
        "text": text,
        "tokens_generated": new_count,
        "step": int(step),
    }


# ── /ask : private quantum-corpus RAG ───────────────────────────────────────
@app.post("/ask")
def ask(req: AskRequest,
        _auth: None = Depends(_require_ask_enabled)) -> dict:
    """
    Private quantum-corpus retrieval + grounded answer (v0.3: Safe Hybrid
    Retrieval).

    Auth-gated (X-API-Key) and disabled in demo mode, so the public HF Space is
    never exposed. Retrieves from the **train+val** hybrid (BM25 + semantic, RRF
    fusion) index only (test stays held-out), then runs the shared two-stage
    answer pipeline (``quantum_corpus.answer.ask``):

      1. RISK gate      — secret/credential request or secret in top doc ->
                          decline (``not_established``), no citations.
      2. EVIDENCE gate  — top-1 score floor + concept overlap + separation.
      3. ANSWERABILITY  — filter/count questions route to the allowlisted
                          read-only structured-query path; else retrieval.
      4. RESPONSE       — every outbound string is masked (``secrets.mask_response``)
                          so no secret/injection span reaches the caller.

    A 32-token context model cannot ingest retrieved records, so evidence is
    returned directly with a short templated synthesis.  The /ask endpoint
    never attaches an LM-generated continuation to the response.

    Returns the contract shape {decision, route, evidence, answer} plus
    useful metadata (abstained, query_provenance, build_id, build_sha256,
    index, mode, latency_ms).  The 32-token TMT sampler is intentionally not
    used to produce the answer; evidence is the list of citations and answer
    is grounded synthesis from the retrieved records or the structured SQL path.
    No raw-record / bulk-export / arbitrary doc-id lookup endpoint exists
    (intentionally denied).
    """
    request_id = secrets.token_hex(8)
    try:
        import time as _time
        if req.mode != "quantum-private":
            raise HTTPException(status_code=400,
                                detail='mode must be "quantum-private".')
        if req.sensitivity not in _SENS_RANK:
            raise HTTPException(status_code=400,
                                detail="sensitivity must be public|internal|sensitive.")
        from quantum_corpus import redact as _qredact, answer as _qanswer

        t0 = _time.perf_counter()

        # Corpus must be configured + present.
        retriever, dbp = _get_ask_retriever()
        if retriever is None:
            raise HTTPException(
                status_code=503,
                detail="quantum corpus not configured (set TMT_QUANTUM_CORPUS_DB "
                       "to a built quantum_corpus.db).")

        # Fetch hits once (sensitivity-filtered at query time) so we can reuse them
        # for the optional LM hint. The engine reuses these hits (no double query).
        hits = _ask_query(retriever, req.question, req.top_k, req.sensitivity)

        build = _ask_build_id(dbp)
        res = _qanswer.ask(
            req.question, retriever,
            structured_query=_get_ask_structured(), hits=hits, top_k=req.top_k,
            max_sensitivity=req.sensitivity,
            build_id=build["build_id"], build_sha256=build["build_sha256"],
            gates=_ask_gates(), use_structured=True,
        )

        # Enforce the /ask contract: {decision, route, evidence, answer}.
        # The answer engine returns citations in "citations" and gate metadata in
        # "evidence"; we expose citations as the primary "evidence" field and move
        # the metadata to "evidence_gate" for callers that want details.
        citations = res.pop("citations", [])
        evidence_gate = res.pop("evidence", None)
        res["evidence"] = citations
        if evidence_gate is not None:
            res["evidence_gate"] = evidence_gate
        # Remove any "generated" key from the answer engine: the 32-token TMT sampler
        # is intentionally NOT used to produce the user-facing /ask answer.
        res.pop("generated", None)

        # Redact the question for telemetry logging (never log raw identifiers).
        red_q, _ = _qredact.redact_text(req.question)
        top_id = res["evidence"][0].get("id") if res["evidence"] else None
        top_score = float(hits[0]["score"]) if hits else 0.0
        _ask_log(red_q, top_id, top_score, bool(res.get("abstained")),
                 str(res.get("decision")))

        res["index"] = "train+val"
        res["mode"] = req.mode
        res["latency_ms"] = _round_ms(t0)
        return res
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled /ask failure", extra={"request_id": request_id})
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR", "request_id": request_id},
        )


def _round_ms(t0: float) -> int:
    import time as _time
    return int((_time.perf_counter() - t0) * 1000)


def _ask_citations(hits: list[dict]) -> list[dict]:
    """Bound snippets + expose id/title/project/source_type/score."""
    out = []
    for h in hits:
        snip = (h.get("snippet") or "").replace("\n", " ").strip()
        if len(snip) > _ASK_MAX_CONTEXT_CHARS:
            snip = snip[:_ASK_MAX_CONTEXT_CHARS] + "…"
        out.append({
            "id": h["id"],
            "title": h.get("doc_id") or "",
            "project": h.get("project", ""),
            "source_type": h.get("source_type", ""),
            "score": round(float(h["score"]), 3),
            "snippet": snip,
        })
    return out


def _ask_synthesize(question: str, hits: list[dict]) -> str:
    """Short templated synthesis from the top retrieved records (extractive, no
    LM). Cites record ids; pulls a one-line snippet from the top hit."""
    if not hits:
        return "Insufficient support in the supplied records."
    top = hits[0]
    ids = ", ".join(str(h["id"]) for h in hits[:3])
    snip = (top.get("snippet") or "").replace("\n", " ").strip()
    if len(snip) > 200:
        snip = snip[:200] + "…"
    return f"Top evidence [{top['id']}]: {snip}\nCitations: {ids}"


def _ask_log(redacted_question: str, top_id, top_score: float,
             abstained: bool, reason: str) -> None:
    """Redacted telemetry only: never logs raw retrieved text or secrets."""
    try:
        print(f"[ask] abstained={abstained} reason={reason} "
              f"top_id={top_id} score={top_score:.2f} "
              f"q={redacted_question[:80]!r}")
    except Exception:
        pass


@app.post("/train/start", status_code=202)
def train_start(req: TrainStartRequest,
                _admin: None = Depends(_require_training_enabled)) -> dict:
    """
    Launch a background training session.  Returns 202 with the session id and
    total_steps.  Refuses with 409 if a training run is already in progress.

    Disabled (403) unless TMT_DEPLOY_MODE=private-training and a valid
    X-API-Key (matching TMT_API_KEY) is supplied.
    """
    with _launch_lock:
        with _train_lock:
            if _train_state["is_training"]:
                raise HTTPException(status_code=409,
                                    detail="A training run is already in progress.")
        # Ensure the DB exists before launching so the trainer thread can write.
        path = _db_path()
        db.init_db(path)
        # Pre-register a session row so we can return its id immediately (the
        # trainer also calls start_session; that creates a second row, which is
        # harmless and keeps the API response decoupled from trainer timing).
        session_id = db.start_session(path, domain_filter=req.domain,
                                      min_quality=req.min_quality)
        lr = req.learning_rate if req.learning_rate is not None \
            else CONFIG["learning_rate"]
        _set_training(True, session_id=session_id, total_steps=req.steps)

        t = threading.Thread(
            target=_trainer_thread,
            args=(req.steps, req.domain, req.min_quality, lr, session_id),
            name="tinymetatron-trainer",
            daemon=True,
        )
        t.start()

    return {
        "session_id": int(session_id),
        "status": "started",
        "total_steps": int(req.steps),
    }


@app.get("/train/status")
def train_status() -> dict:
    """Return the current training status snapshot."""
    with _train_lock:
        return {
            "is_training": bool(_train_state["is_training"]),
            "current_step": int(_train_state["current_step"]),
            "total_steps": int(_train_state["total_steps"]),
            "current_loss": float(_train_state["current_loss"]),
        }


@app.post("/data/add")
def data_add(req: DataAddRequest,
             _admin: None = Depends(_require_data_writes_enabled)) -> dict:
    """
    Score and insert texts via db.add_texts (which uses quality.score_quality).

    Returns {added, rejected, domain, quality_threshold}.

    Disabled (403) unless TMT_DEPLOY_MODE=private-training and a valid
    X-API-Key (matching TMT_API_KEY) is supplied.
    """
    path = _db_path()
    db.init_db(path)
    added, rejected = db.add_texts(path, req.texts, req.domain,
                                    quality_threshold=req.quality_threshold)
    return {
        "added": int(added),
        "rejected": int(rejected),
        "domain": req.domain,
        "quality_threshold": float(req.quality_threshold),
    }


@app.get("/data/stats")
def data_stats() -> dict:
    """Return db.stats for the configured DB path."""
    path = _db_path()
    db.init_db(path)
    return db.stats(path)


@app.get("/model/info")
def model_info() -> dict:
    """Return the frozen CONFIG and the active checkpoint row (if any)."""
    path = _db_path()
    db.init_db(path)
    return {
        "config": get_config(),
        "active_checkpoint": db.get_active_checkpoint(path),
    }


# ── Self-test ─────────────────────────────────────────────────────────────────
def _serve() -> None:
    """Run the API under uvicorn (used by ``python api.py --serve``).

    Host/port honour the HOST/PORT env vars (Hugging Face sets PORT=7860) and
    fall back to CONFIG['host'] / CONFIG['port'] (0.0.0.0:8010) when unset.
    """
    import uvicorn
    host = os.environ.get("HOST", CONFIG["host"])
    port = int(os.environ.get("PORT", CONFIG["port"]))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 fix (rule 4)

    # ``python api.py --serve`` boots uvicorn on CONFIG host/port; with no
    # argument we run the in-process self-test (contract section 5).
    if "--serve" in sys.argv[1:]:
        _serve()
        raise SystemExit(0)

    import tempfile

    from fastapi.testclient import TestClient

    # Point the API at a temp DB so we never touch the real metatron.db.
    tmpdir = tempfile.mkdtemp(prefix="tinymetatron_api_")
    tmp_db = os.path.join(tmpdir, "test.db")
    os.environ["TMT_DB_PATH"] = tmp_db
    # Self-test exercises the admin endpoints (/data/add, /train/start) so it
    # runs in private-training mode with a test API key. Gating reads env at
    # request time, so this takes effect immediately.
    os.environ["TMT_DEPLOY_MODE"] = "private-training"
    os.environ["TMT_API_KEY"] = "testkey"
    _admin_headers = {"X-API-Key": "testkey"}

    client = TestClient(app)

    # 1. GET /data/stats -> 200, contract keys present.
    r = client.get("/data/stats")
    assert r.status_code == 200, f"/data/stats status {r.status_code}: {r.text}"
    stats_body = r.json()
    for k in ("total", "by_domain", "avg_quality", "used_in_training"):
        assert k in stats_body, f"/data/stats missing key {k}"
    print(f"GET /data/stats -> 200  total={stats_body['total']}")

    # 2. GET /model/info -> config present, active_checkpoint is None (fresh db).
    r = client.get("/model/info")
    assert r.status_code == 200, f"/model/info status {r.status_code}: {r.text}"
    info = r.json()
    assert "config" in info and isinstance(info["config"], dict), "config dict present"
    assert info["config"].get("vocab_size") == CONFIG["vocab_size"], "config echoed"
    assert "active_checkpoint" in info, "active_checkpoint key present"
    print(f"GET /model/info -> 200  vocab_size={info['config']['vocab_size']}")

    # 3. POST /data/add -> 200, added count returned.
    texts = [
        "The firewall enforces a zero-trust policy with mfa and tls encryption.",
        "Sparse attention masks reduce the quadratic cost of self-attention.",
        "Mixture of experts routes tokens to specialized feed forward networks.",
        "The transformer model uses attention, embeddings and a softmax over logits.",
    ]
    r = client.post("/data/add", json={
        "texts": texts, "domain": "cybersecurity", "quality_threshold": 0.4,
    }, headers=_admin_headers)
    assert r.status_code == 200, f"/data/add status {r.status_code}: {r.text}"
    add_body = r.json()
    assert add_body["added"] >= 1, f"added>=1, got {add_body['added']}"
    assert add_body["added"] + add_body["rejected"] == len(texts), \
        "added+rejected == #texts"
    assert add_body["domain"] == "cybersecurity"
    print(f"POST /data/add -> 200  added={add_body['added']} "
          f"rejected={add_body['rejected']}")

    # 3b. Admin gate: no X-API-Key -> 403 even in private-training mode.
    r_nokey = client.post("/data/add", json={
        "texts": texts, "domain": "cybersecurity", "quality_threshold": 0.4,
    })
    assert r_nokey.status_code == 403, \
        f"no-key /data/add expected 403, got {r_nokey.status_code}"
    # 3c. Demo mode: /data/add disabled even with a valid key.
    os.environ["TMT_DEPLOY_MODE"] = "demo"
    r_demo = client.post("/data/add", json={
        "texts": texts, "domain": "cybersecurity", "quality_threshold": 0.4,
    }, headers=_admin_headers)
    assert r_demo.status_code == 403, \
        f"demo /data/add expected 403, got {r_demo.status_code}"
    os.environ["TMT_DEPLOY_MODE"] = "private-training"  # restore
    print("admin gate OK: no-key -> 403, demo-mode -> 403")

    # 4. POST /generate -> 200, returns text + tokens_generated using an
    #    untrained model (no checkpoint yet).
    r = client.post("/generate", json={
        "prompt": "firewall ", "max_length": 8, "temperature": 0.7,
    })
    assert r.status_code == 200, f"/generate status {r.status_code}: {r.text}"
    gen = r.json()
    assert "text" in gen and isinstance(gen["text"], str), "text field present"
    assert "tokens_generated" in gen, "tokens_generated field present"
    assert isinstance(gen["tokens_generated"], int), "tokens_generated is int"
    assert gen["tokens_generated"] >= 0, "tokens_generated >= 0"
    assert "step" in gen and gen["step"] == 0, "step==0 for untrained model"
    print(f"POST /generate -> 200  tokens_generated={gen['tokens_generated']} "
          f"step={gen['step']}")

    # 4b. /ask gating: demo mode -> 403 (public Space protection); no-key -> 403;
    #     private + key but no corpus configured -> 503. Point the quantum DB at a
    #     nonexistent path so the 503 path is deterministic and the self-test does
    #     not depend on the real (private) corpus being built.
    os.environ["TMT_QUANTUM_CORPUS_DB"] = os.path.join(tmpdir, "no_quantum.db")
    os.environ["TMT_DEPLOY_MODE"] = "demo"
    r_ask_demo = client.post("/ask", json={"question": "what backend is ibm_fez?",
                                            "mode": "quantum-private"},
                              headers=_admin_headers)
    assert r_ask_demo.status_code == 403, \
        f"demo /ask expected 403, got {r_ask_demo.status_code}"
    os.environ["TMT_DEPLOY_MODE"] = "private-training"
    r_ask_nokey = client.post("/ask", json={"question": "what backend is ibm_fez?",
                                            "mode": "quantum-private"})
    assert r_ask_nokey.status_code == 403, \
        f"no-key /ask expected 403, got {r_ask_nokey.status_code}"
    r_ask_badmode = client.post("/ask", json={"question": "x", "mode": "wrong"},
                                headers=_admin_headers)
    assert r_ask_badmode.status_code == 400, \
        f"bad-mode /ask expected 400, got {r_ask_badmode.status_code}"
    r_ask_nocorpus = client.post("/ask",
                                 json={"question": "what backend is ibm_fez?",
                                       "mode": "quantum-private"},
                                 headers=_admin_headers)
    assert r_ask_nocorpus.status_code == 503, \
        f"no-corpus /ask expected 503, got {r_ask_nocorpus.status_code}: {r_ask_nocorpus.text}"
    print("admin gate OK: demo /ask -> 403, no-key /ask -> 403, "
          "bad-mode -> 400, no-corpus -> 503")
    del os.environ["TMT_QUANTUM_CORPUS_DB"]

    # 5. GET /train/status -> 200, not training, all fields present.
    r = client.get("/train/status")
    assert r.status_code == 200, f"/train/status status {r.status_code}: {r.text}"
    st = r.json()
    assert st["is_training"] is False, "not training at rest"
    for k in ("current_step", "total_steps", "current_loss"):
        assert k in st, f"/train/status missing {k}"
    print(f"GET /train/status -> 200  is_training={st['is_training']}")

    # 6. POST /train/start -> 202 (accept), then 409 on a second concurrent start.
    r = client.post("/train/start", json={
        "steps": 1, "domain": "nonexistent_domain_xyz", "min_quality": 0.4,
    }, headers=_admin_headers)
    assert r.status_code == 202, f"/train/start status {r.status_code}: {r.text}"
    ts = r.json()
    assert ts["status"] == "started", "status started"
    assert ts["total_steps"] == 1, "total_steps echoed"
    assert isinstance(ts["session_id"], int), "session_id is int"
    print(f"POST /train/start -> 202  session_id={ts['session_id']}")

    # Second start while the trainer thread is alive -> 409.
    r2 = client.post("/train/start", json={
        "steps": 1, "domain": "general", "min_quality": 0.4,
    }, headers=_admin_headers)
    assert r2.status_code == 409, f"expected 409, got {r2.status_code}: {r2.text}"
    print(f"POST /train/start (concurrent) -> {r2.status_code} (409 expected)")

    # Wait for the trainer thread to finish (empty domain -> it exits quickly).
    import time
    deadline = time.time() + 30.0
    while time.time() < deadline:
        sr = client.get("/train/status").json()
        if not sr["is_training"]:
            break
        time.sleep(0.2)
    assert not client.get("/train/status").json()["is_training"], \
        "trainer thread released the is_training flag"
    print("trainer thread completed; is_training=False")

    # cleanup temp db
    try:
        os.remove(tmp_db)
    except OSError:
        pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    print("\nSELF-TEST PASSED")
