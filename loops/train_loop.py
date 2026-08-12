"""
loops/train_loop.py
===================
Orchestrates single-seed training: fresh init → staged-train → checkpoint →
early-stop → select best. Owns all registry state; workers emit only result JSON.

State machine (run-level):
    NEW → TRAINING → EVALUATING → GENERALIZING → AWAITING_FINAL_TEST_APPROVAL
                                                              ↓
                                                     FINAL_TEST_RUNNING
                                                              ↓
                                              AWAITING_PROMOTION_DECISION
                                                              ↓
                                                    PROMOTED | REJECTED | ARCHIVED

Usage:
    python -m loops.train_loop new --exp-id exp-004 --seed 42 --steps 2000
    python -m loops.train_loop status --run-id exp-004-train-seed42
    python -m loops.train_loop checkpoints --run-id exp-004-train-seed42
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from db import (
    create_loop_experiment,
    create_loop_run,
    get_loop_experiment,
    get_loop_run,
    get_loop_runs_for_experiment,
    update_loop_experiment_state,
    update_loop_run_status,
    save_loop_checkpoint,
    get_best_loop_checkpoint,
    set_promotion,
    set_db_path,
)

# Registry path for this loop
_REGISTRY = str(Path(__file__).resolve().parents[1] / "state" / "registry.db")
set_db_path(_REGISTRY)


# ── Path safety ────────────────────────────────────────────────────────────────

_PROTECTED_PATHS = [
    Path("experiments/exp-003").resolve(),
]


def _check_write(path: Path) -> None:
    """Raise if path is beneath a protected experiment directory."""
    resolved = Path(path).resolve()
    for protected in _PROTECTED_PATHS:
        try:
            resolved.relative_to(protected)
            raise PermissionError(
                f"Loop code may not write beneath {protected} — exp-003 is frozen. "
                f"Attempted write: {path}"
            )
        except ValueError:
            pass  # not relative to protected


# ── State machine ─────────────────────────────────────────────────────────────

VALID_TRANSITIONS = {
    "NEW": {"FROZEN_CORPUS", "TRAINING"},
    "FROZEN_CORPUS": {"TRAINING"},
    "TRAINING": {"EVALUATING"},
    "EVALUATING": {"GENERALIZING"},
    "GENERALIZING": {"AWAITING_FINAL_TEST_APPROVAL"},
    "AWAITING_FINAL_TEST_APPROVAL": {"FINAL_TEST_RUNNING"},
    "FINAL_TEST_RUNNING": {"AWAITING_PROMOTION_DECISION"},
    "AWAITING_PROMOTION_DECISION": {"PROMOTED", "REJECTED", "ARCHIVED"},
    "PROMOTED": set(),
    "REJECTED": set(),
    "ARCHIVED": set(),
}


def _transition(run_id: str, from_state: str, to_state: str, reason_code: str,
                payload: dict = None) -> None:
    if to_state not in VALID_TRANSITIONS.get(from_state, set()):
        raise ValueError(
            f"Invalid transition {run_id}: {from_state} → {to_state} "
            f"(reason: {reason_code})"
        )
    update_loop_run_status(run_id, to_state, reason_code, payload or {})


# ── Worker invocation ────────────────────────────────────────────────────────

def _run_worker(argv: list[str], artifact_dir: Path, timeout_seconds: int = 300) -> dict:
    """Run a worker subprocess, parse its result JSON."""
    _check_write(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    result_path = artifact_dir / "result.json"
    full_argv = argv + ["--result", str(result_path)]

    completed = subprocess.run(
        full_argv,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        cwd=str(_ROOT),
    )

    if completed.returncode != 0 and not result_path.exists():
        raise RuntimeError(
            f"Worker failed with code {completed.returncode}\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        )

    if result_path.exists():
        result = json.loads(open(result_path).read())
        if result.get("status") == "error":
            raise RuntimeError(f"Worker returned error: {result.get('error')}")
        return result
    else:
        # Fallback: treat non-zero exit without result as error
        raise RuntimeError(
            f"Worker exited {completed.returncode} but produced no result.json"
        )


# ── Train loop ───────────────────────────────────────────────────────────────

def run_training(config: dict) -> dict:
    """Run the training loop for a single seed."""
    exp_id = config["exp_id"]
    run_id = config["run_id"]
    seed = config["seed"]
    steps = config.get("steps", 2000)
    artifact_dir = Path(config["artifact_dir"])
    corpus_dir = Path(config.get("corpus_dir", str(_ROOT / "experiments/exp-003/corpus")))
    tokenizer_path = config.get("tokenizer_path", str(_ROOT / "vocab.json"))

    _check_write(artifact_dir)

    started_at = datetime.now(timezone.utc).isoformat()

    # Ensure parent experiment exists
    try:
        get_loop_experiment(exp_id)
    except KeyError:
        create_loop_experiment(
            exp_id=exp_id,
            state="ACTIVE",
            hypothesis=f"exp-004 training run with seed {seed}",
        )

    # Create or resume run
    try:
        run_record = get_loop_run(run_id)
        print(f"Resuming existing run: {run_id} (state={run_record['status']})")
    except KeyError:
        create_loop_run(
            run_id=run_id,
            exp_id=exp_id,
            seed=seed,
            seq_len=32,
            status="NEW",
            corpus_hash="",  # filled after corpus_loop
            split_hash="",
            tokenizer_hash="",
            model_config_hash="",
        )
        print(f"Created new run: {run_id}")

    # Transition: NEW → TRAINING
    try:
        current = get_loop_run(run_id)
        if current and current["status"] == "NEW":
            _transition(run_id, "NEW", "TRAINING", "manual_start")
    except ValueError as e:
        print(f"Skipping transition: {e}")

    # Build training worker config
    train_artifact_dir = artifact_dir / "train"
    train_config_path = train_artifact_dir / "config.json"
    _check_write(train_artifact_dir)
    train_artifact_dir.mkdir(parents=True, exist_ok=True)

    train_worker_config = {
        "seed": seed,
        "steps": steps,
        "val_every": config.get("val_every", 50),
        "log_every": config.get("log_every", 25),
        "patience": config.get("patience", 3),
        "batch_size": config.get("batch_size", 16),
        "lr": config.get("lr", 1e-3),
        "weight_decay": config.get("weight_decay", 1e-4),
        "small_model": config.get("small_model", False),
        "artifact_dir": str(train_artifact_dir),
        "corpus_dir": str(corpus_dir),
        "tokenizer_path": tokenizer_path,
    }

    with open(train_config_path, "w") as f:
        json.dump(train_worker_config, f, indent=2)

    # Run training worker
    print(f"[{run_id}] Starting training: {steps} steps, seed={seed}")
    result = _run_worker(
        argv=[
            "python", "-m", "workers.train.train",
            "--config", str(train_config_path),
        ],
        artifact_dir=train_artifact_dir,
        timeout_seconds=config.get("timeout_seconds", 3600),
    )

    metrics = result["metrics"]
    best_step = metrics["best_step"]

    # Save checkpoints to registry
    checkpoint_dir = train_artifact_dir / "checkpoints"
    for ckpt_file in sorted(checkpoint_dir.glob("step*.pt")):
        step_num = int(ckpt_file.stem.replace("step", "").replace("_final", ""))
        is_best = (step_num == best_step)
        save_loop_checkpoint(
            run_id=run_id,
            step=step_num,
            file_path=str(ckpt_file),
            val_ce=metrics.get("best_val_ce") if is_best else None,
            train_ce=None,
            is_best=is_best,
        )

    print(
        f"[{run_id}] Training complete: best_step={best_step}, "
        f"best_val_ce={metrics['best_val_ce']:.4f}, "
        f"total_steps={metrics['total_steps']}"
    )

    # Transition: TRAINING → EVALUATING
    _transition(run_id, "TRAINING", "EVALUATING", "training_complete")

    return result


# ── Evaluate best checkpoint ─────────────────────────────────────────────────

def run_evaluation(run_id: str, artifact_dir: Path, corpus_dir: Path,
                  tokenizer_path: str) -> dict:
    """Evaluate the best checkpoint against val set."""
    best = get_best_loop_checkpoint(run_id)
    if not best:
        raise RuntimeError(f"No checkpoints found for {run_id}")

    ckpt_path = best["file_path"]
    step = best["step"]

    eval_artifact_dir = artifact_dir / "evaluate" / f"step{step:06d}"
    _check_write(eval_artifact_dir)

    print(f"[{run_id}] Evaluating step {step} on val set")

    result = _run_worker(
        argv=[
            "python", "-m", "workers.evaluate.compute_ce",
            "--run-id", run_id,
            "--step", str(step),
            "--eval-set", "val",
            "--checkpoint-path", ckpt_path,
            "--corpus-dir", str(corpus_dir),
            "--tokenizer-path", tokenizer_path,
        ],
        artifact_dir=eval_artifact_dir,
        timeout_seconds=120,
    )

    from db import save_evaluation
    metrics = result["metrics"]
    save_evaluation(
        run_id=run_id,
        step=step,
        eval_set="val",
        ce=metrics["ce"],
        ppl=metrics["ppl"],
        total_tokens=metrics["total_tokens"],
    )

    print(f"[{run_id}] Val CE={metrics['ce']:.4f} PPL={metrics['ppl']:.2f} at step {step}")
    return result


# ── Status commands ──────────────────────────────────────────────────────────

def cmd_status(run_id: str) -> None:
    run = get_loop_run(run_id)
    if run is None:
        print(f"Run not found: {run_id}")
        return
    checkpoints = get_best_loop_checkpoint(run_id)

    print(f"Run: {run_id}")
    print(f"  Experiment: {run['exp_id']}")
    print(f"  Seed: {run['seed']}")
    print(f"  Status: {run['status']}")
    print(f"  Created: {run['created_at']}")
    print(f"  Updated: {run['updated_at']}")
    if checkpoints:
        print(f"  Best step: {checkpoints['step']} (val_ce={checkpoints['val_ce']:.4f})")

    from db import get_evaluations
    evals = get_evaluations(run_id)
    if evals:
        print("  Evaluations:")
        for e in evals:
            print(f"    step={e['step']} {e['eval_set']}: CE={e['ce']:.4f} PPL={e['ppl']:.2f}")


def cmd_checkpoints(run_id: str) -> None:
    from db import get_loop_checkpoints
    ckpts = get_loop_checkpoints(run_id)
    print(f"Checkpoints for {run_id}:")
    for c in ckpts:
        best_marker = " *" if c["is_best"] else ""
        print(f"  step={c['step']:6d}  val_ce={c['val_ce']:.4f}{best_marker}")


def cmd_list(exp_id: str) -> None:
    runs = get_loop_runs_for_experiment(exp_id)
    if not runs:
        print(f"No runs found for experiment {exp_id}")
        return
    for r in runs:
        print(f"  {r['run_id']}: seed={r['seed']} status={r['status']}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TinyMetatron training loop")
    sub = parser.add_subparsers(dest="cmd")

    # new
    p_new = sub.add_parser("new", help="Start a new training run")
    p_new.add_argument("--exp-id", required=True)
    p_new.add_argument("--seed", type=int, required=True)
    p_new.add_argument("--steps", type=int, default=2000)
    p_new.add_argument("--val-every", type=int, default=50)
    p_new.add_argument("--log-every", type=int, default=25)
    p_new.add_argument("--patience", type=int, default=3)
    p_new.add_argument("--batch-size", type=int, default=16)
    p_new.add_argument("--lr", type=float, default=1e-3)
    p_new.add_argument("--weight-decay", type=float, default=1e-4)
    p_new.add_argument("--small-model", action="store_true")
    p_new.add_argument("--artifact-dir", required=True)
    p_new.add_argument("--corpus-dir",
                       default=str(_ROOT / "experiments/exp-003/corpus"))
    p_new.add_argument("--tokenizer-path", default=str(_ROOT / "vocab.json"))

    # status
    p_status = sub.add_parser("status", help="Show run status")
    p_status.add_argument("--run-id", required=True)

    # checkpoints
    p_ckpts = sub.add_parser("checkpoints", help="List checkpoints")
    p_ckpts.add_argument("--run-id", required=True)

    # list
    p_list = sub.add_parser("list", help="List runs for an experiment")
    p_list.add_argument("--exp-id", required=True)

    args = parser.parse_args()

    if args.cmd == "new":
        config = vars(args)
        config["run_id"] = f"{args.exp_id}-train-seed{args.seed}"
        result = run_training(config)
        print(json.dumps(result["metrics"], indent=2))

    elif args.cmd == "status":
        cmd_status(args.run_id)

    elif args.cmd == "checkpoints":
        cmd_checkpoints(args.run_id)

    elif args.cmd == "list":
        cmd_list(args.exp_id)

    else:
        parser.print_help()


if __name__ == "__main__":
    raise SystemExit(main())
