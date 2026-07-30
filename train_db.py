"""
train_db.py
===========
Incremental training CLI for the TinyMetatron SLM.

Patent component: the data-driven incremental trainer. Loads the active
checkpoint (or initialises a fresh model), pulls a slice of unused rows from
the SQLite data layer, fine-tunes for N steps with Adam, writes a new
checkpoint + DB row, and updates the training session.

Public interface (IMPLEMENTATION_CONTRACT.md section 2 + section 3):
    main(argv)                          -> CLI entry point (argparse flags)
    run_training(steps, domain, min_quality, batch_size, learning_rate,
                 max_seq_len, device, checkpoint_dir, db_path,
                 aux_loss_weight, seed, on_step=None) -> dict

The ``run_training`` callable is imported by ``api.py`` and run in a
background thread; ``on_step(step, total_steps, loss)`` is invoked after each
logged step for status reporting (contract section 3).

Loss = cross-entropy over the LM logits shifted by one token plus
``aux_loss_weight`` times the MoE Switch-Transformer balancing aux loss
(contract section 3, R6 aux-loss wiring).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Callable, Optional, List, Tuple

import torch
import torch.nn.functional as F

from config import CONFIG
import db
from tokenizer import default_tokenizer, Tokenizer
from tinymetatron_model import TinyMetatron


# ── Helpers ──────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    """Seed Python-hash, torch CPU and (if present) CUDA RNG for determinism."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    """Resolve a device string to a torch.device, honouring CUDA availability."""
    d = (device or CONFIG.get("device") or "cpu").lower()
    if d == "cuda" and not torch.cuda.is_available():
        d = "cpu"
    return torch.device(d)


def _tokenize_rows(rows: List[dict], tokenizer: Tokenizer,
                   max_seq_len: int) -> Tuple[torch.LongTensor, List[int]]:
    """
    Tokenise each row's ``text`` and pad/truncate to ``max_seq_len``.

    Returns (input_ids [R, max_seq_len], row_ids).  Rows that tokenise to
    nothing useful (only specials) are still included so the DB mark_used
    matches the fetch — the trainer simply samples from the available pool.
    Padding uses CONFIG['pad_id'].
    """
    pad_id = CONFIG["pad_id"]
    R = len(rows)
    out = torch.full((R, max_seq_len), pad_id, dtype=torch.long)
    ids: List[int] = []
    for i, row in enumerate(rows):
        ids.append(int(row["id"]))
        enc = tokenizer.encode(str(row["text"]))
        # Drop BOS/EOS is NOT done — the model is trained on the wrapped
        # sequence; truncation keeps the leading BOS when possible.
        if len(enc) > max_seq_len:
            enc = enc[:max_seq_len]
        out[i, :len(enc)] = torch.tensor(enc, dtype=torch.long)
    return out, ids


# ── Callable trainer (imported by api.py) ─────────────────────────────────────

def run_training(steps: int,
                 domain: str,
                 min_quality: float,
                 batch_size: int,
                 learning_rate: float,
                 max_seq_len: int,
                 device: str,
                 checkpoint_dir: str,
                 db_path: str,
                 aux_loss_weight: float,
                 seed: int,
                 on_step: Optional[Callable[[int, int, float], None]] = None
                 ) -> dict:
    """
    Run one incremental training session and return a status dict.

    Args mirror the CLI flags (contract section 3).  ``on_step`` is called as
    ``on_step(step, total_steps, loss)`` after each *logged* step
    (every CONFIG['log_every'] steps) so api.py can update module-global
    status without coupling to the print stream.

    Returns a dict with: {steps, final_loss, session_id, checkpoint_path,
    rows_used, domain}.  Loss is the mean of the logged step losses; if no
    step was logged (steps < log_every) the final step's loss is used.
    """
    _set_seed(int(seed))
    dev = _resolve_device(device)

    # Ensure DB + checkpoint dir exist.
    db.init_db(db_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Load active checkpoint (from the real db_path) or init fresh ────────
    start_step = 0
    active = db.get_active_checkpoint(db_path)
    model = TinyMetatron.from_config()
    model.to(dev)
    if active is not None and active.get("file_path") \
            and os.path.isfile(str(active["file_path"])):
        try:
            model.load_checkpoint(str(active["file_path"]))
            # Reject checkpoints whose parameters contain NaN/Inf — a
            # poisoned state_dict would propagate corruption into the run.
            if not all(torch.isfinite(p).all().item()
                       for p in model.parameters()):
                raise ValueError("checkpoint contains non-finite parameters")
            start_step = int(active.get("step") or 0)
        except Exception:
            # Corrupt / NaN checkpoint → fall back to fresh model; do not
            # crash the API thread.  The new run starts from scratch.
            model = TinyMetatron.from_config()
            model.to(dev)
            start_step = 0

    tokenizer = default_tokenizer()

    # ── Fetch a slice of unused training rows ───────────────────────────────
    # Pull a reasonable slice: at least enough to form a few batches, capped
    # so we don't load the whole corpus into memory.  We fetch more than the
    # bare minimum so successive batches across steps see variety.
    fetch_limit = max(batch_size * max(steps, 1) * 2, batch_size * 4)
    rows = db.fetch_training_rows(db_path, domain, min_quality,
                                  limit=fetch_limit, used=False)

    if not rows:
        # Nothing to train on — still record an empty session so callers can
        # see the run happened.  final_loss = NaN sentinel.
        session_id = db.start_session(db_path, domain_filter=domain,
                                       min_quality=min_quality)
        db.end_session(db_path, session_id, total_steps=0, final_loss=float("nan"))
        print("Žiadne trénovacie dáta pre zadaný filter.")
        return {
            "steps": 0,
            "final_loss": float("nan"),
            "session_id": session_id,
            "checkpoint_path": None,
            "rows_used": 0,
            "domain": domain,
        }

    # Mark the fetched rows used immediately (contract section 3).
    row_ids = [int(r["id"]) for r in rows]
    db.mark_used(db_path, row_ids)

    # Tokenise the rows into a padded tensor pool.
    pool, _ = _tokenize_rows(rows, tokenizer, max_seq_len)
    pool = pool.to(dev)
    R = pool.shape[0]

    # ── Training session row ────────────────────────────────────────────────
    session_id = db.start_session(db_path, domain_filter=domain,
                                   min_quality=min_quality)

    optim = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    model.train()

    log_every = int(CONFIG.get("log_every", 10))
    total_steps = int(steps)
    last_loss = float("nan")
    logged_losses: List[float] = []

    for step in range(1, total_steps + 1):
        # Sample a random batch (with replacement when the pool is small so
        # short runs still see variety; contract says "sample batch").
        if R >= batch_size:
            idx = torch.randint(0, R, (batch_size,), device=dev)
        else:
            idx = torch.randint(0, R, (batch_size,), device=dev)
        batch = pool[idx]                                   # (B, L)

        logits, aux = model(batch)                          # (B, L, V), scalar
        # CE over logits shifted by one: predict token t+1 from position t.
        # Drop the last position (no next token) and the first input token
        # (BOS is the predictor target at position 0 -> nothing to learn).
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        V = CONFIG["vocab_size"]
        ce = F.cross_entropy(shift_logits.view(-1, V),
                             shift_labels.view(-1))
        aux_v = aux.to(ce.dtype) if isinstance(aux, torch.Tensor) \
            else torch.tensor(float(aux), device=dev, dtype=ce.dtype)
        loss = ce + float(aux_loss_weight) * aux_v

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        last_loss = float(loss.detach().item())
        global_step = start_step + step

        if step % log_every == 0 or step == total_steps:
            msg = f"Krok {step}/{total_steps}: loss={last_loss:.3f}"
            try:
                print(msg)
            except UnicodeEncodeError:
                # Defensive: never crash on a misconfigured stdout.
                sys.stdout.reconfigure(encoding="utf-8")
                print(msg)
            logged_losses.append(last_loss)
            if on_step is not None:
                try:
                    on_step(global_step, start_step + total_steps, last_loss)
                except Exception:
                    # Status callback must never abort training.
                    pass

    final_loss = (sum(logged_losses) / len(logged_losses)
                 if logged_losses else last_loss)

    # ── Save checkpoint (.pt) + DB row (is_active=1, clears prior) ──────────
    # Guard against NaN/Inf poisoning the persisted active checkpoint: a run
    # that diverged (e.g. unstable LR) must NOT overwrite the last good active
    # model, otherwise every subsequent run loads garbage. We still write the
    # .pt for inspection but only flip is_active=1 when the run is finite.
    params_finite = all(torch.isfinite(p).all().item()
                        for p in model.parameters())
    loss_finite = math.isfinite(final_loss)
    ckpt_name = f"tinymetatron_step{start_step + total_steps}.pt"
    ckpt_path = os.path.abspath(os.path.join(checkpoint_dir, ckpt_name))
    model.save_checkpoint(ckpt_path)
    if params_finite and loss_finite:
        db.save_checkpoint(db_path, step=start_step + total_steps,
                            loss=final_loss, file_path=ckpt_path, is_active=True)
    else:
        # Non-finite run: record the checkpoint as inactive (kept for
        # forensics) and leave the prior active row untouched.
        db.save_checkpoint(db_path, step=start_step + total_steps,
                           loss=final_loss, file_path=ckpt_path, is_active=False)
        warn = (f"Varovanie: nefinitná tréningová strata ({final_loss}) "
                f"alebo váhy — kontrolný bod uložený ako neaktívny, "
                f"predchádzajúci aktívny model zostáva zachovaný.")
        try:
            print(warn)
        except UnicodeEncodeError:
            sys.stdout.reconfigure(encoding="utf-8")
            print(warn)

    # ── End session ─────────────────────────────────────────────────────────
    db.end_session(db_path, session_id,
                   total_steps=total_steps, final_loss=final_loss)

    final_msg = f"Tréning dokončený. Finálna strata: {final_loss:.3f}"
    try:
        print(final_msg)
    except UnicodeEncodeError:
        sys.stdout.reconfigure(encoding="utf-8")
        print(final_msg)

    return {
        "steps": total_steps,
        "final_loss": final_loss,
        "session_id": session_id,
        "checkpoint_path": ckpt_path,
        "rows_used": len(row_ids),
        "domain": domain,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_db.py",
        description="Incremental trainer for the TinyMetatron SLM.")
    # Default 200 per contract section 3.
    p.add_argument("--steps", type=int, default=200,
                   help="Number of training steps (default 200).")
    p.add_argument("--domain", type=str, default="general",
                   help="Domain filter for training_data rows.")
    p.add_argument("--min_quality", type=float, default=0.5,
                   help="Minimum quality_score to include.")
    p.add_argument("--batch_size", type=int, default=CONFIG["batch_size"],
                   help="Mini-batch size.")
    p.add_argument("--learning_rate", type=float,
                   default=CONFIG["learning_rate"], help="Adam LR.")
    p.add_argument("--max_seq_len", type=int, default=CONFIG["seq_len"],
                   help="Maximum sequence length (pad/truncate).")
    p.add_argument("--device", type=str, default=CONFIG["device"],
                   help="torch device: cpu | cuda.")
    p.add_argument("--checkpoint_dir", type=str,
                   default=CONFIG["checkpoint_dir"], help="Checkpoint dir.")
    p.add_argument("--db_path", type=str, default=CONFIG["db_path"],
                   help="Path to the SQLite DB.")
    p.add_argument("--aux_loss_weight", type=float,
                   default=CONFIG["moe_aux_loss_weight"],
                   help="MoE aux-loss weight.")
    p.add_argument("--seed", type=int, default=CONFIG["seed"],
                   help="RNG seed.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.  Returns process exit code (0 on success)."""
    args = _build_parser().parse_args(argv)
    run_training(
        steps=args.steps,
        domain=args.domain,
        min_quality=args.min_quality,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        db_path=args.db_path,
        aux_loss_weight=args.aux_loss_weight,
        seed=args.seed,
    )
    return 0


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 fix (rule 4)

    # Real CLI dispatch by default; pass --selftest to run the embedded self-test.
    if "--selftest" not in sys.argv[1:]:
        sys.exit(main(sys.argv[1:]))

    import math
    import tempfile
    import types

    # quality.py is a T2 sibling built in parallel; register a deterministic
    # stub so add_texts works in the self-test without the real file present.
    if "quality" not in sys.modules:
        _stub = types.ModuleType("quality")

        def _score_quality(text: str) -> float:
            t = text.strip()
            if not t:
                return 0.0
            length_term = min(len(t), 200) / 200.0
            uniq_term = len(set(t.split())) / max(len(t.split()), 1)
            return round(0.6 * length_term + 0.4 * uniq_term, 4)

        _stub.score_quality = _score_quality  # type: ignore[attr-defined]
        sys.modules["quality"] = _stub

    torch.manual_seed(CONFIG["seed"])

    with tempfile.TemporaryDirectory(prefix="tinymetatron_train_") as td:
        db_path = os.path.join(td, "test.db")
        ckpt_dir = os.path.join(td, "ckpt")

        # Seed a few rows that score above the min_quality threshold.
        db.init_db(db_path)
        seed_texts = [
            "Quantization reduces model precision to int8 to shrink memory footprint.",
            "Packet filtering firewall rules block unauthorized network traffic by port.",
            "Asymmetric encryption uses a public key pair for secure key exchange.",
            "Authentication tokens validate user identity across distributed sessions.",
            "Sparse attention masks reduce quadratic cost to linear complexity edges.",
            "Mixture of experts routes tokens to specialized feed forward networks.",
        ]
        added, rejected = db.add_texts(db_path, seed_texts, "cybersecurity",
                                       quality_threshold=0.4)
        print(f"seeded rows: added={added} rejected={rejected}")
        assert added >= 1, "self-test needs at least one seeded row"

        # Tiny 2-step run.
        result = run_training(
            steps=2,
            domain="cybersecurity",
            min_quality=0.4,
            batch_size=2,
            learning_rate=1e-3,
            max_seq_len=CONFIG["seq_len"],
            device="cpu",
            checkpoint_dir=ckpt_dir,
            db_path=db_path,
            aux_loss_weight=CONFIG["moe_aux_loss_weight"],
            seed=CONFIG["seed"],
        )
        print("run_training result:", result)

        # ── Assertions (contract section 5) ────────────────────────────────
        assert math.isfinite(result["final_loss"]), \
            f"final_loss must be finite, got {result['final_loss']}"
        assert result["steps"] == 2, f"steps==2, got {result['steps']}"
        assert result["checkpoint_path"] is not None, "checkpoint_path set"
        assert os.path.isfile(result["checkpoint_path"]), \
            "checkpoint .pt file written"

        # DB checkpoint row written with is_active=1.
        act = db.get_active_checkpoint(db_path)
        assert act is not None and act["is_active"] == 1, \
            "active checkpoint row written"
        assert act["file_path"] == result["checkpoint_path"], \
            "active row points at the saved .pt"
        assert os.path.isfile(act["file_path"]), "active file_path exists on disk"

        # Session row ended.
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        srow = conn.execute(
            "SELECT * FROM training_sessions WHERE id = ?",
            (result["session_id"],)).fetchone()
        conn.close()
        assert srow is not None and srow["end_time"] is not None, \
            "session end_time set"
        assert srow["total_steps"] == 2, "session total_steps==2"
        assert math.isfinite(float(srow["final_loss"])), \
            "session final_loss finite"

        # Rows marked used.
        st = db.stats(db_path)
        assert st["used_in_training"] == added, \
            f"all fetched rows marked used ({st['used_in_training']} vs {added})"

        # ── on_step callback exercised with aux_loss > 0 sanity ───────────
        seen: List[Tuple[int, int, float]] = []

        def cb(step: int, total: int, loss: float) -> None:
            seen.append((step, total, loss))

        # Re-seed (rows were marked used) and run again with the callback.
        db.add_texts(db_path, seed_texts, "cybersecurity",
                     quality_threshold=0.4)
        result2 = run_training(
            steps=2, domain="cybersecurity", min_quality=0.4,
            batch_size=2, learning_rate=1e-3, max_seq_len=CONFIG["seq_len"],
            device="cpu", checkpoint_dir=ckpt_dir, db_path=db_path,
            aux_loss_weight=CONFIG["moe_aux_loss_weight"],
            seed=CONFIG["seed"], on_step=cb)
        assert result2["steps"] == 2 and math.isfinite(result2["final_loss"]), \
            "second run ok"
        assert len(seen) >= 1, "on_step callback invoked"
        assert all(math.isfinite(l) for _, _, l in seen), \
            "on_step losses finite"

        # ── Resume-from-checkpoint sanity: a fresh run should load the ckpt ─
        # we just wrote (active row points at it) and not crash.
        db.add_texts(db_path, seed_texts, "cybersecurity",
                     quality_threshold=0.4)
        result3 = run_training(
            steps=1, domain="cybersecurity", min_quality=0.4,
            batch_size=2, learning_rate=1e-3, max_seq_len=CONFIG["seq_len"],
            device="cpu", checkpoint_dir=ckpt_dir, db_path=db_path,
            aux_loss_weight=CONFIG["moe_aux_loss_weight"],
            seed=CONFIG["seed"])
        assert math.isfinite(result3["final_loss"]), \
            "resume-from-checkpoint run finite loss"

        # ── Empty-data path ─────────────────────────────────────────────────
        result_empty = run_training(
            steps=2, domain="nonexistent_domain", min_quality=0.9,
            batch_size=2, learning_rate=1e-3, max_seq_len=CONFIG["seq_len"],
            device="cpu", checkpoint_dir=ckpt_dir, db_path=db_path,
            aux_loss_weight=CONFIG["moe_aux_loss_weight"],
            seed=CONFIG["seed"])
        assert result_empty["steps"] == 0, \
            f"empty-domain run steps==0, got {result_empty['steps']}"
        assert result_empty["checkpoint_path"] is None, \
            "empty-domain run writes no checkpoint"

        # ── CLI argv path ─────────────────────────────────────────────────
        # Use a second temp DB so the CLI run is self-contained.
        cli_db = os.path.join(td, "cli_test.db")
        cli_ckpt = os.path.join(td, "ckpt_cli")
        db.init_db(cli_db)
        db.add_texts(cli_db, seed_texts, "cybersecurity",
                     quality_threshold=0.4)
        rc = main(["--steps", "2", "--domain", "cybersecurity",
                   "--min_quality", "0.4", "--batch_size", "2",
                   "--db_path", cli_db, "--checkpoint_dir", cli_ckpt,
                   "--device", "cpu"])
        assert rc == 0, "main() returned 0"
        act_cli = db.get_active_checkpoint(cli_db)
        assert act_cli is not None and act_cli["is_active"] == 1, \
            "CLI run wrote an active checkpoint"

    print("\nSELF-TEST PASSED")