"""
loops/promote_loop.py
====================
One-time final-test → package → promote/reject.
Enforces database uniqueness on candidate_sha256 + test_manifest_sha.
Explicit AWAITING_FINAL_TEST_APPROVAL state before test_final is touched.

State transitions:
  AWAITING_FINAL_TEST_APPROVAL → FINAL_TEST_RUNNING (human approved)
  FINAL_TEST_RUNNING → AWAITING_PROMOTION_DECISION (test complete)
  AWAITING_PROMOTION_DECISION → PROMOTED | REJECTED

Usage:
    python -m loops.promote_loop approve --run-id exp-004-train-seed42
    python -m loops.promote_loop run --run-id exp-004-train-seed42
    python -m loops.promote_loop promote --run-id exp-004-train-seed42
    python -m loops.promote_loop reject --run-id exp-004-train-seed42 --reason "CE regression"
    python -m loops.promote_loop status --run-id exp-004-train-seed42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from db import (
    get_loop_run,
    get_best_loop_checkpoint,
    update_loop_run_status,
    save_evaluation,
    try_consume_final_test,
    set_promotion,
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


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_worker(argv: list[str], artifact_dir: Path, timeout_seconds: int = 120) -> dict:
    _check_write(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifact_dir / "result.json"
    full_argv = argv + ["--result", str(result_path)]
    completed = subprocess.run(
        full_argv, capture_output=True, text=True,
        timeout=timeout_seconds, cwd=str(_ROOT),
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


def cmd_approve(run_id: str) -> dict:
    """
    Transition from AWAITING_FINAL_TEST_APPROVAL → FINAL_TEST_RUNNING.
    Records human approval in loop_events.
    """
    run = get_loop_run(run_id)
    if run["status"] != "AWAITING_FINAL_TEST_APPROVAL":
        raise ValueError(
            f"Cannot approve: run {run_id} is {run['status']}, "
            f"expected AWAITING_FINAL_TEST_APPROVAL"
        )

    update_loop_run_status(
        run_id,
        "AWAITING_FINAL_TEST_APPROVAL",
        "human_approved_final_test",
        {"approved_at": datetime.now(timezone.utc).isoformat()},
    )

    print(f"[{run_id}] Human approved — transitioned to FINAL_TEST_RUNNING")
    return {"run_id": run_id, "new_state": "FINAL_TEST_RUNNING"}


def cmd_run(run_id: str, artifact_dir: Path, corpus_dir: Path,
            tokenizer_path: str) -> dict:
    """
    Run the final test: one-time evaluation on test_final.
    Enforces uniqueness via try_consume_final_test.
    """
    _check_write(artifact_dir)
    run = get_loop_run(run_id)

    if run["status"] != "FINAL_TEST_RUNNING":
        raise ValueError(
            f"Cannot run final test: run {run_id} is {run['status']}, "
            f"expected FINAL_TEST_RUNNING. Use 'approve' first."
        )

    best = get_best_loop_checkpoint(run_id)
    if not best:
        raise RuntimeError(f"No best checkpoint for {run_id}")

    ckpt_path = best["file_path"]
    step = best["step"]

    # Compute hashes for uniqueness enforcement
    candidate_hash = _hash_file(Path(ckpt_path))[:16]
    test_manifest_path = corpus_dir / "test_final.jsonl"
    test_hash = _hash_file(test_manifest_path)[:16]

    # Try to consume — enforces uniqueness constraint
    consumed = try_consume_final_test(
        candidate_sha256=candidate_hash,
        test_manifest_sha=test_hash,
        run_id=run_id,
        ce=None,  # filled after evaluation
        ppl=None,
    )

    if not consumed:
        raise RuntimeError(
            f"Final test for candidate {candidate_hash} against test manifest {test_hash} "
            f"has already been consumed. Each (candidate, test_manifest) pair can only "
            f"be evaluated once."
        )

    print(f"[{run_id}] Running final test on test_final.jsonl")

    eval_artifact_dir = artifact_dir / "final_test"
    result = _run_worker(
        argv=[
            "python", "-m", "workers.evaluate.compute_ce",
            "--run-id", run_id,
            "--step", str(step),
            "--eval-set", "test_final",
            "--checkpoint-path", ckpt_path,
            "--corpus-dir", str(corpus_dir),
            "--tokenizer-path", tokenizer_path,
        ],
        artifact_dir=eval_artifact_dir,
        timeout_seconds=180,
    )

    metrics = result["metrics"]
    ce, ppl = metrics["ce"], metrics["ppl"]

    # Record the consumption with actual CE
    from db import get_db
    conn = get_db()
    conn.execute(
        """
        UPDATE final_test_consumed
        SET ce = ?, ppl = ?, consumed_at = ?
        WHERE candidate_sha256 = ? AND test_manifest_sha = ?
        """,
        (ce, ppl, datetime.now(timezone.utc).isoformat(), candidate_hash, test_hash),
    )
    conn.commit()

    # Save evaluation
    save_evaluation(
        run_id=run_id,
        step=step,
        eval_set="test_final",
        ce=ce,
        ppl=ppl,
        total_tokens=metrics["total_tokens"],
    )

    # Transition to AWAITING_PROMOTION_DECISION
    update_loop_run_status(
        run_id,
        "FINAL_TEST_RUNNING",
        "final_test_complete",
        {"test_ce": ce, "test_ppl": ppl},
    )

    print(f"[{run_id}] Final test complete: CE={ce:.4f} PPL={ppl:.2f}")
    print(f"[{run_id}] Transitioned to AWAITING_PROMOTION_DECISION")

    return {
        "run_id": run_id,
        "test_ce": ce,
        "test_ppl": ppl,
        "candidate_hash": candidate_hash,
        "test_hash": test_hash,
    }


def cmd_promote(run_id: str, artifact_dir: Path) -> dict:
    """Mark a run as PROMOTED — emit promotion manifest."""
    _check_write(artifact_dir)
    run = get_loop_run(run_id)
    if run["status"] != "AWAITING_PROMOTION_DECISION":
        raise ValueError(
            f"Cannot promote: run {run_id} is {run['status']}, "
            f"expected AWAITING_PROMOTION_DECISION"
        )

    best = get_best_loop_checkpoint(run_id)
    ckpt_path = Path(best["file_path"])

    # Build promotion manifest
    manifest = {
        "run_id": run_id,
        "exp_id": run["exp_id"],
        "seed": run["seed"],
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(ckpt_path.resolve()),
            "sha256": _hash_file(ckpt_path),
            "step": best["step"],
            "val_ce": best["val_ce"],
        },
        "artifact_dir": str(Path(artifact_dir).resolve()),
    }

    manifest_path = Path(artifact_dir) / "PROMOTION.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    set_promotion(run_id, "promoted")
    update_loop_run_status(
        run_id,
        "AWAITING_PROMOTION_DECISION",
        "human_promoted",
        {"manifest": str(manifest_path)},
    )

    print(f"[{run_id}] PROMOTED — manifest written to {manifest_path}")
    return manifest


def cmd_reject(run_id: str, reason: str, artifact_dir: Path) -> dict:
    """Mark a run as REJECTED."""
    _check_write(artifact_dir)
    run = get_loop_run(run_id)
    if run["status"] != "AWAITING_PROMOTION_DECISION":
        raise ValueError(
            f"Cannot reject: run {run_id} is {run['status']}, "
            f"expected AWAITING_PROMOTION_DECISION"
        )

    manifest = {
        "run_id": run_id,
        "exp_id": run["exp_id"],
        "seed": run["seed"],
        "rejected_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }

    manifest_path = Path(artifact_dir) / "REJECTION.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    set_promotion(run_id, "rejected")
    update_loop_run_status(
        run_id,
        "AWAITING_PROMOTION_DECISION",
        "human_rejected",
        {"reason": reason, "manifest": str(manifest_path)},
    )

    print(f"[{run_id}] REJECTED — reason: {reason}")
    return manifest


def cmd_status(run_id: str) -> None:
    from db import get_evaluations
    run = get_loop_run(run_id)
    best = get_best_loop_checkpoint(run_id)

    print(f"Run: {run_id}")
    print(f"  Experiment: {run['exp_id']}")
    print(f"  Seed: {run['seed']}")
    print(f"  Status: {run['status']}")
    print(f"  Promotion: {run.get('promotion') or 'none'}")

    if best:
        print(f"  Best checkpoint: step={best['step']}  val_ce={best['val_ce']:.4f}")

    evals = get_evaluations(run_id)
    for e in sorted(evals, key=lambda x: x["step"]):
        marker = " ← final_test" if e["eval_set"] == "test_final" else ""
        print(f"  {e['eval_set']:12s}  step={e['step']:6d}  CE={e['ce']:.4f}  PPL={e['ppl']:.2f}{marker}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_approve = sub.add_parser("approve", help="Approve final test (transitions to FINAL_TEST_RUNNING)")
    p_approve.add_argument("--run-id", required=True)

    p_run = sub.add_parser("run", help="Run final test evaluation")
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument("--artifact-dir", required=True)
    p_run.add_argument("--corpus-dir",
                       default=str(_ROOT / "experiments/exp-003/corpus"))
    p_run.add_argument("--tokenizer-path", default=str(_ROOT / "vocab.json"))

    p_promote = sub.add_parser("promote", help="Promote run")
    p_promote.add_argument("--run-id", required=True)
    p_promote.add_argument("--artifact-dir", required=True)

    p_reject = sub.add_parser("reject", help="Reject run")
    p_reject.add_argument("--run-id", required=True)
    p_reject.add_argument("--reason", required=True)
    p_reject.add_argument("--artifact-dir", required=True)

    p_status = sub.add_parser("status", help="Show promotion status")
    p_status.add_argument("--run-id", required=True)

    args = parser.parse_args()

    if args.cmd == "approve":
        result = cmd_approve(args.run_id)
        print(json.dumps(result, indent=2))

    elif args.cmd == "run":
        result = cmd_run(
            run_id=args.run_id,
            artifact_dir=Path(args.artifact_dir),
            corpus_dir=Path(args.corpus_dir),
            tokenizer_path=args.tokenizer_path,
        )
        print(json.dumps(result, indent=2))

    elif args.cmd == "promote":
        result = cmd_promote(args.run_id, Path(args.artifact_dir))
        print(json.dumps(result, indent=2))

    elif args.cmd == "reject":
        result = cmd_reject(args.run_id, args.reason, Path(args.artifact_dir))
        print(json.dumps(result, indent=2))

    elif args.cmd == "status":
        cmd_status(args.run_id)

    else:
        parser.print_help()


if __name__ == "__main__":
    raise SystemExit(main())
