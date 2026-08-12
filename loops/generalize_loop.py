"""
loops/generalize_loop.py
========================
Evaluate on hard_dev + novel_eval; run gates; write gate_results.
Transitions run state: EVALUATING → GENERALIZING → AWAITING_FINAL_TEST_APPROVAL

Usage:
    python -m loops.generalize_loop run --run-id exp-004-train-seed42
    python -m loops.generalize_loop gates --run-id exp-004-train-seed42
    python -m loops.generalize_loop status --run-id exp-004-train-seed42
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from db import (
    get_loop_run,
    get_best_loop_checkpoint,
    save_gate_result,
    get_gate_results,
    update_loop_run_status,
    save_evaluation,
    get_evaluations,
    set_db_path,
)

_REGISTRY = str(Path(__file__).resolve().parents[1] / "state" / "registry.db")
set_db_path(_REGISTRY)


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


def _load_gate(gate_name: str) -> dict:
    gate_path = _ROOT / "state" / "gates" / f"{gate_name}.json"
    if not gate_path.exists():
        raise FileNotFoundError(f"Gate not found: {gate_path}")
    return json.loads(open(gate_path).read())


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


def _interpolate_argv(argv: list, run_id: str) -> list:
    return [arg.replace("{run_id}", run_id) for arg in argv]


def _eval_condition(metric_value: float, operator: str, threshold: float) -> bool:
    if operator == "<":
        return metric_value < threshold
    elif operator == "<=":
        return metric_value <= threshold
    elif operator == ">":
        return metric_value > threshold
    elif operator == ">=":
        return metric_value >= threshold
    elif operator == "==":
        return metric_value == threshold
    else:
        raise ValueError(f"Unknown operator: {operator}")


def run_gate(
    run_id: str,
    gate_name: str,
    checkpoint_path: str,
    corpus_dir: Path,
    tokenizer_path: str,
    artifact_dir: Path,
) -> dict:
    """
    Run a single gate: interpolate argv, run worker, evaluate pass_condition.
    Returns gate result dict.
    """
    gate = _load_gate(gate_name)
    started_at = datetime.now(timezone.utc).isoformat()

    argv = _interpolate_argv(gate["argv"], run_id)
    # Inject checkpoint-path and corpus-dir if not in argv
    argv = argv + [
        "--checkpoint-path", checkpoint_path,
        "--corpus-dir", str(corpus_dir),
        "--tokenizer-path", tokenizer_path,
    ]

    gate_artifact_dir = artifact_dir / "gates" / gate_name
    stdout_path = gate_artifact_dir / "stdout.txt"
    stderr_path = gate_artifact_dir / "stderr.txt"

    try:
        result = _run_worker(argv, gate_artifact_dir, gate.get("timeout_seconds", 120))

        metric = gate["pass_condition"]["metric"]
        operator = gate["pass_condition"]["operator"]
        threshold = gate["pass_condition"]["value"]
        actual_value = result["metrics"].get(metric)

        if actual_value is None:
            raise ValueError(
                f"Gate metric '{metric}' not found in worker result metrics: "
                f"{list(result['metrics'].keys())}"
            )

        passed = _eval_condition(actual_value, operator, threshold)

        gate_result = {
            "gate_name": gate_name,
            "passed": passed,
            "actual_value": actual_value,
            "threshold": threshold,
            "operator": operator,
            "duration_s": None,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "worker_result": result,
        }

        print(
            f"  {'PASS' if passed else 'FAIL'} {gate_name}: "
            f"{metric}={actual_value:.4f} ({operator} {threshold})"
        )

    except Exception as exc:
        gate_result = {
            "gate_name": gate_name,
            "passed": False,
            "actual_value": None,
            "threshold": gate["pass_condition"]["value"],
            "operator": gate["pass_condition"]["operator"],
            "duration_s": None,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "error": str(exc),
        }
        print(f"  ERROR {gate_name}: {exc}")

    ended_at = datetime.now(timezone.utc).isoformat()
    save_gate_result(
        run_id=run_id,
        gate_name=gate_name,
        passed=gate_result["passed"],
        duration_s=gate_result.get("duration_s"),
        stdout_path=gate_result["stdout_path"],
        stderr_path=gate_result["stderr_path"],
    )

    return gate_result


def run_generalization(config: dict) -> dict:
    """
    Run full generalization pipeline:
      1. Ensure best checkpoint has hard_dev + novel_eval evaluations
      2. Run all gates from state/gates/
      3. If all pass: transition to AWAITING_FINAL_TEST_APPROVAL
      4. If any fail: stop and report
    """
    run_id = config["run_id"]
    artifact_dir = Path(config["artifact_dir"])
    corpus_dir = Path(config.get("corpus_dir", str(_ROOT / "experiments/exp-003/corpus")))
    tokenizer_path = config.get("tokenizer_path", str(_ROOT / "vocab.json"))

    _check_write(artifact_dir)
    run_record = get_loop_run(run_id)
    best = get_best_loop_checkpoint(run_id)

    if not best:
        raise RuntimeError(f"No best checkpoint found for {run_id}")

    ckpt_path = best["file_path"]
    step = best["step"]

    # Transition to GENERALIZING
    if run_record["status"] == "EVALUATING":
        update_loop_run_status(run_id, "EVALUATING", "generalization_start",
                               {"checkpoint_step": step})

    gate_names = [p.stem for p in sorted((_ROOT / "state" / "gates").glob("*.json"))]

    print(f"[{run_id}] Running generalization gates ({len(gate_names)} gates)")

    gate_results = []
    all_passed = True

    for gate_name in gate_names:
        result = run_gate(
            run_id=run_id,
            gate_name=gate_name,
            checkpoint_path=ckpt_path,
            corpus_dir=corpus_dir,
            tokenizer_path=tokenizer_path,
            artifact_dir=artifact_dir,
        )
        gate_results.append(result)
        if not result["passed"]:
            all_passed = False

    # Summary
    passed_gates = [g for g in gate_results if g["passed"]]
    failed_gates = [g for g in gate_results if not g["passed"]]
    print(f"\n[{run_id}] Gate summary: {len(passed_gates)} passed, {len(failed_gates)} failed")

    if all_passed:
        print(f"[{run_id}] All gates passed — transitioning to AWAITING_FINAL_TEST_APPROVAL")
        update_loop_run_status(
            run_id, "GENERALIZING", "all_gates_passed",
            {"gates": [g["gate_name"] for g in gate_results]}
        )
        state = "AWAITING_FINAL_TEST_APPROVAL"
    else:
        print(f"[{run_id}] Some gates failed — run is stopped")
        update_loop_run_status(
            run_id, "GENERALIZING", "gate_failed",
            {"failed_gates": [g["gate_name"] for g in failed_gates]}
        )
        state = run_record["status"]  # don't advance

    return {
        "run_id": run_id,
        "state": state,
        "gate_results": gate_results,
        "all_passed": all_passed,
    }


def cmd_gates(run_id: str) -> None:
    results = get_gate_results(run_id)
    if not results:
        print(f"No gate results for {run_id}")
        return
    print(f"Gate results for {run_id}:")
    for r in sorted(results, key=lambda x: x["gate_name"]):
        status = "PASS" if r["passed"] else "FAIL"
        val_str = f"{r.get('actual_value'):.4f}" if r.get('actual_value') is not None else "N/A"
        print(f"  {status:4s}  {r['gate_name']}: {val_str}")


def cmd_status(run_id: str) -> None:
    from db import get_evaluations
    run = get_loop_run(run_id)
    best = get_best_loop_checkpoint(run_id)
    evals = get_evaluations(run_id)
    gates = get_gate_results(run_id)

    print(f"Run: {run_id}  status={run['status']}")
    if best:
        print(f"  Best: step={best['step']}  val_ce={best['val_ce']:.4f}")

    hard_dev = [e for e in evals if e["eval_set"] == "hard_dev"]
    novel = [e for e in evals if e["eval_set"] == "novel_eval"]
    if hard_dev:
        print(f"  hard_dev:  CE={hard_dev[0]['ce']:.4f}  PPL={hard_dev[0]['ppl']:.2f}")
    if novel:
        print(f"  novel_eval: CE={novel[0]['ce']:.4f}  PPL={novel[0]['ppl']:.2f}")

    if gates:
        print("  Gates:")
        for g in gates:
            status = "PASS" if g["passed"] else "FAIL"
            val = f"{g.get('actual_value'):.4f}" if g.get('actual_value') is not None else "N/A"
            print(f"    {status} {g['gate_name']}: {val}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run generalization + gates")
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument("--artifact-dir", required=True)
    p_run.add_argument("--corpus-dir",
                       default=str(_ROOT / "experiments/exp-003/corpus"))
    p_run.add_argument("--tokenizer-path", default=str(_ROOT / "vocab.json"))

    p_gates = sub.add_parser("gates", help="Show gate results")
    p_gates.add_argument("--run-id", required=True)

    p_status = sub.add_parser("status", help="Show generalization status")
    p_status.add_argument("--run-id", required=True)

    args = parser.parse_args()

    if args.cmd == "run":
        result = run_generalization(vars(args))
        print(json.dumps(result, indent=2))

    elif args.cmd == "gates":
        cmd_gates(args.run_id)

    elif args.cmd == "status":
        cmd_status(args.run_id)

    else:
        parser.print_help()


if __name__ == "__main__":
    raise SystemExit(main())
