"""
loops/feedback_loop.py
======================
Telemetry → corpus acquisition requests.
Reads evaluation metrics from registry, identifies gaps, emits acquisition requests.

Usage:
    python -m loops.feedback_loop analyze --exp-id exp-004
    python -m loops.feedback_loop request --exp-id exp-004 --gap hard_compounds
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from db import (
    get_loop_experiment,
    get_loop_runs_for_experiment,
    get_best_loop_checkpoint,
    get_evaluations,
    get_gate_results,
    set_db_path,
)

_REGISTRY = str(Path(__file__).resolve().parents[1] / "state" / "registry.db")
set_db_path(_REGISTRY)


# ── Gap analysis ─────────────────────────────────────────────────────────────

def analyze_gaps(exp_id: str) -> dict:
    """
    Analyze evaluation results to identify generalization gaps.
    Returns gap report with prioritized acquisition suggestions.
    """
    runs = get_loop_runs_for_experiment(exp_id)
    if not runs:
        return {"error": f"No runs found for experiment {exp_id}"}

    report = {
        "exp_id": exp_id,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "runs": [],
        "gaps": [],
        "acquisition_requests": [],
    }

    for run in runs:
        run_id = run["run_id"]
        best = get_best_loop_checkpoint(run_id)
        evals = get_evaluations(run_id)
        gates = get_gate_results(run_id)

        hard_dev = next((e for e in evals if e["eval_set"] == "hard_dev"), None)
        novel = next((e for e in evals if e["eval_set"] == "novel_eval"), None
)
        val = next((e for e in evals if e["eval_set"] == "val"), None)

        run_report = {
            "run_id": run_id,
            "seed": run["seed"],
            "best_step": best["step"] if best else None,
            "val_ce": best["val_ce"] if best else None,
            "hard_dev_ce": hard_dev["ce"] if hard_dev else None,
            "novel_eval_ce": novel["ce"] if novel else None,
            "gates": [g["gate_name"] for g in gates if g["passed"]],
        }
        report["runs"].append(run_report)

        # Identify gaps
        if hard_dev and val:
            gap = hard_dev["ce"] - val["ce"]
            if gap > 1.0:
                report["gaps"].append({
                    "run_id": run_id,
                    "type": "hard_dev_vs_val",
                    "gap": round(gap, 4),
                    "val_ce": val["ce"],
                    "hard_dev_ce": hard_dev["ce"],
                    "severity": "high" if gap > 2.0 else "medium",
                    "suggestion": "Acquire more hard_dev domain examples",
                })

        if novel and hard_dev:
            gap = novel["ce"] - hard_dev["ce"]
            if gap > 1.0:
                report["gaps"].append({
                    "run_id": run_id,
                    "type": "novel_vs_hard_dev",
                    "gap": round(gap, 4),
                    "hard_dev_ce": hard_dev["ce"],
                    "novel_ce": novel["ce"],
                    "severity": "high" if gap > 1.5 else "medium",
                    "suggestion": "Acquire novel phrase examples (unseen in training)",
                })

    # Generate acquisition requests
    severe_gaps = [g for g in report["gaps"] if g["severity"] == "high"]
    for gap in severe_gaps:
        report["acquisition_requests"].append({
            "priority": "high",
            "gap_type": gap["type"],
            "run_id": gap["run_id"],
            "suggestion": gap["suggestion"],
            "requested_at": datetime.now(timezone.utc).isoformat(),
        })

    return report


def emit_acquisition_request(exp_id: str, gap_type: str, priority: str = "medium",
                             notes: str = "") -> dict:
    """Record an explicit acquisition request."""
    request = {
        "exp_id": exp_id,
        "gap_type": gap_type,
        "priority": priority,
        "notes": notes,
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }

    artifact_dir = _ROOT / "state" / "artifacts" / "acquisition_requests"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # One file per gap type per experiment
    out_path = artifact_dir / f"{exp_id}_{gap_type}.json"
    with open(out_path, "w") as f:
        json.dump(request, f, indent=2)

    return request


def cmd_analyze(exp_id: str) -> None:
    report = analyze_gaps(exp_id)
    print(json.dumps(report, indent=2))


def cmd_request(exp_id: str, gap_type: str, priority: str = "medium",
                notes: str = "") -> None:
    result = emit_acquisition_request(exp_id, gap_type, priority, notes)
    print(f"Acquisition request saved:")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_analyze = sub.add_parser("analyze", help="Analyze generalization gaps")
    p_analyze.add_argument("--exp-id", required=True)

    p_request = sub.add_parser("request", help="Emit acquisition request")
    p_request.add_argument("--exp-id", required=True)
    p_request.add_argument("--gap-type", required=True,
                           choices=["hard_dev_vs_val", "novel_vs_hard_dev", "novel_vs_val"])
    p_request.add_argument("--priority", default="medium",
                           choices=["low", "medium", "high"])
    p_request.add_argument("--notes", default="")

    args = parser.parse_args()

    if args.cmd == "analyze":
        cmd_analyze(args.exp_id)
    elif args.cmd == "request":
        cmd_request(args.exp_id, args.gap_type, args.priority, args.notes)
    else:
        parser.print_help()


if __name__ == "__main__":
    raise SystemExit(main())
