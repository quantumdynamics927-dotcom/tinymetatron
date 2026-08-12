"""
loops/evaluate_loop.py
=====================
Orchestrates per-checkpoint CE/PPL + per-row loss evaluation.
Reads checkpoint list from registry; writes evaluations table.

Usage:
    python -m loops.evaluate_loop run --run-id exp-004-train-seed42
    python -m loops.evaluate_loop status --run-id exp-004-train-seed42
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from db import (
    get_loop_run,
    get_loop_checkpoints,
    save_evaluation,
)


_PROTECTED_PATHS = [Path("experiments/exp-003").resolve()]


def _check_write(path: Path) -> None:
    resolved = Path(path).resolve()
    for protected in _PROTECTED_PATHS:
        try:
            resolved.relative_to(protected)
            raise PermissionError(
                f"Loop code may not write beneath {protected} — exp-003 is frozen. "
                f"Attempted write: {path}"
            )
        except ValueError:
            pass


def _run_worker(argv: list[str], artifact_dir: Path, timeout_seconds: int = 120) -> dict:
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
    if not result_path.exists():
        raise RuntimeError(
            f"Worker produced no result.json; exit={completed.returncode}\n"
            f"stderr: {completed.stderr}"
        )
    result = json.loads(open(result_path).read())
    if result.get("status") == "error":
        raise RuntimeError(f"Worker error: {result.get('error')}")
    return result


def run_evaluation_for_checkpoints(
    run_id: str,
    artifact_dir: Path,
    corpus_dir: Path,
    tokenizer_path: str,
    eval_sets: list[str] = None,
    checkpoint_filter: str = None,
) -> dict:
    """Evaluate all checkpoints (or best only) against named eval sets."""
    if eval_sets is None:
        eval_sets = ["val", "hard_dev", "novel_eval"]

    run_record = get_loop_run(run_id)
    checkpoints = get_loop_checkpoints(run_id)

    if not checkpoints:
        raise RuntimeError(f"No checkpoints found for {run_id}")

    # If checkpoint_filter == "best", only evaluate best
    if checkpoint_filter == "best":
        best = next((c for c in checkpoints if c["is_best"]), None)
        checkpoints = [best] if best else []

    results = {}
    total_checkpoints = len(checkpoints)

    for idx, ckpt in enumerate(checkpoints):
        step = ckpt["step"]
        ckpt_path = ckpt["file_path"]
        print(f"[{run_id}] [{idx+1}/{total_checkpoints}] Evaluating step {step}")

        for eval_set in eval_sets:
            eval_artifact_dir = artifact_dir / f"step{step:06d}" / eval_set
            try:
                result = _run_worker(
                    argv=[
                        "python", "-m", "workers.evaluate.compute_ce",
                        "--run-id", run_id,
                        "--step", str(step),
                        "--eval-set", eval_set,
                        "--checkpoint-path", ckpt_path,
                        "--corpus-dir", str(corpus_dir),
                        "--tokenizer-path", tokenizer_path,
                    ],
                    artifact_dir=eval_artifact_dir,
                    timeout_seconds=120,
                )
                metrics = result["metrics"]
                save_evaluation(
                    run_id=run_id,
                    step=step,
                    eval_set=eval_set,
                    ce=metrics["ce"],
                    ppl=metrics["ppl"],
                    total_tokens=metrics["total_tokens"],
                )
                key = f"step{step}_{eval_set}"
                results[key] = {"ce": metrics["ce"], "ppl": metrics["ppl"]}
                print(f"  {eval_set}: CE={metrics['ce']:.4f} PPL={metrics['ppl']:.2f}")

            except Exception as exc:
                print(f"  {eval_set}: FAILED — {exc}")

    return results


def run_per_row_loss(
    run_id: str,
    artifact_dir: Path,
    corpus_dir: Path,
    tokenizer_path: str,
    eval_set: str = "hard_dev",
) -> dict:
    """Compute per-row CE breakdown for the best checkpoint."""
    checkpoints = get_loop_checkpoints(run_id)
    best = next((c for c in checkpoints if c["is_best"]), None)
    if not best:
        raise RuntimeError(f"No best checkpoint found for {run_id}")

    step = best["step"]
    ckpt_path = best["file_path"]
    eval_artifact_dir = artifact_dir / f"step{step:06d}" / eval_set / "per_row"

    result = _run_worker(
        argv=[
            "python", "-m", "workers.evaluate.per_row_loss",
            "--run-id", run_id,
            "--step", str(step),
            "--eval-set", eval_set,
            "--checkpoint-path", ckpt_path,
            "--corpus-dir", str(corpus_dir),
            "--tokenizer-path", tokenizer_path,
        ],
        artifact_dir=eval_artifact_dir,
        timeout_seconds=180,
    )
    return result


def cmd_status(run_id: str) -> None:
    from db import get_evaluations
    run = get_loop_run(run_id)
    checkpoints = get_loop_checkpoints(run_id)
    evals = get_evaluations(run_id)

    print(f"Run: {run_id}  status={run['status']}")
    best = next((c for c in checkpoints if c["is_best"]), None)
    if best:
        print(f"  Best: step={best['step']}  val_ce={best['val_ce']:.4f}")

    if evals:
        print("  Evaluations:")
        for e in sorted(evals, key=lambda x: (x["step"], x["eval_set"])):
            print(f"    step={e['step']:6d}  {e['eval_set']:10s}  CE={e['ce']:.4f}  PPL={e['ppl']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run evaluation for all checkpoints")
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument("--artifact-dir", required=True)
    p_run.add_argument("--corpus-dir",
                       default=str(_ROOT / "experiments/exp-003/corpus"))
    p_run.add_argument("--tokenizer-path", default=str(_ROOT / "vocab.json"))
    p_run.add_argument("--eval-sets", nargs="+",
                       default=["val", "hard_dev", "novel_eval"],
                       choices=["val", "hard_dev", "novel_eval", "test_final"])
    p_run.add_argument("--best-only", action="store_true",
                       help="Evaluate only the best checkpoint")

    p_per_row = sub.add_parser("per-row", help="Run per-row loss on best checkpoint")
    p_per_row.add_argument("--run-id", required=True)
    p_per_row.add_argument("--artifact-dir", required=True)
    p_per_row.add_argument("--corpus-dir",
                           default=str(_ROOT / "experiments/exp-003/corpus"))
    p_per_row.add_argument("--tokenizer-path", default=str(_ROOT / "vocab.json"))
    p_per_row.add_argument("--eval-set", default="hard_dev",
                           choices=["val", "hard_dev", "novel_eval", "test_final"])

    p_status = sub.add_parser("status", help="Show evaluation status")
    p_status.add_argument("--run-id", required=True)

    args = parser.parse_args()

    if args.cmd == "run":
        artifact_dir = Path(args.artifact_dir)
        _check_write(artifact_dir)
        results = run_evaluation_for_checkpoints(
            run_id=args.run_id,
            artifact_dir=artifact_dir,
            corpus_dir=Path(args.corpus_dir),
            tokenizer_path=args.tokenizer_path,
            eval_sets=args.eval_sets,
            checkpoint_filter="best" if args.best_only else None,
        )
        print(json.dumps(results, indent=2))

    elif args.cmd == "per-row":
        artifact_dir = Path(args.artifact_dir)
        _check_write(artifact_dir)
        result = run_per_row_loss(
            run_id=args.run_id,
            artifact_dir=artifact_dir,
            corpus_dir=Path(args.corpus_dir),
            tokenizer_path=args.tokenizer_path,
            eval_set=args.eval_set,
        )
        print(json.dumps(result["metrics"], indent=2))

    elif args.cmd == "status":
        cmd_status(args.run_id)

    else:
        parser.print_help()


if __name__ == "__main__":
    raise SystemExit(main())
