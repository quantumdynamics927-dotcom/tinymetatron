"""
loops/corpus_loop.py
====================
Ingest → validate → dedupe → split → version → freeze manifest.
Owns all registry state; workers emit only result JSON.

Usage:
    python -m loops.corpus_loop run --exp-id exp-004 --corpus-dir data/raw
    python -m loops.corpus_loop status --exp-id exp-004
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
    create_loop_experiment,
    get_loop_experiment,
    init_db,
    update_loop_experiment_state,
    set_db_path,
)

_REGISTRY = str(Path(__file__).resolve().parents[1] / "state" / "registry.db")
# Respect TINYMETATRON_DB env if set; otherwise use default production path
import os as _os
if not _os.environ.get("TINYMETATRON_DB"):
    set_db_path(_REGISTRY)
init_db(None)  # Idempotent: initializes the active DB (respects TINYMETATRON_DB env)


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


def _run_worker(argv: list[str], artifact_dir: Path, timeout_seconds: int = 300) -> dict:
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
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        )
    result = json.loads(open(result_path).read())
    if result.get("status") == "error":
        raise RuntimeError(f"Worker error: {result.get('error')}")
    return result


# ── Corpus gate ──────────────────────────────────────────────────────────────

def _load_gate(gate_name: str) -> dict:
    gate_path = _ROOT / "state" / "gates" / f"{gate_name}.json"
    if not gate_path.exists():
        raise FileNotFoundError(f"Gate not found: {gate_path}")
    return json.loads(open(gate_path).read())


def _interpolate_argv(argv: list, manifest_path: str) -> list:
    return [arg.replace("{manifest_path}", manifest_path) for arg in argv]


def _eval_condition(metric_value, operator: str, threshold) -> bool:
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


def run_corpus_gate(exp_id: str, gate_name: str, manifest_path: Path,
                    artifact_dir: Path, threshold_override: float | None = None) -> dict:
    """
    Run a single corpus gate: interpolate {manifest_path}, run the gate worker,
    evaluate pass_condition. Mirrors generalize_loop.run_gate but is
    experiment-scoped (no run_id, no checkpoint).

    threshold_override: when set, replaces the gate definition's pass_condition
    value (used to make a gate's threshold configurable per run).

    On failure: records a CORPUS_GATE_FAILED loop_event with the gate result as
    payload, then raises RuntimeError so FROZEN_CORPUS is never reached.
    On success: returns the gate result dict.
    """
    gate = _load_gate(gate_name)
    started_at = datetime.now(timezone.utc).isoformat()

    argv = _interpolate_argv(gate["argv"], str(manifest_path))
    gate_artifact_dir = artifact_dir / "gates" / gate_name

    metric = gate["pass_condition"]["metric"]
    operator = gate["pass_condition"]["operator"]
    threshold = (threshold_override if threshold_override is not None
                 else gate["pass_condition"]["value"])

    try:
        result = _run_worker(argv, gate_artifact_dir,
                             gate.get("timeout_seconds", 60))

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
            "worker_result": result,
        }
        print(
            f"[{exp_id}] Corpus gate {'PASS' if passed else 'FAIL'} "
            f"{gate_name}: {metric}={actual_value} ({operator} {threshold})"
        )

    except Exception as exc:
        gate_result = {
            "gate_name": gate_name,
            "passed": False,
            "actual_value": None,
            "threshold": threshold,
            "operator": operator,
            "error": str(exc),
        }
        print(f"[{exp_id}] Corpus gate ERROR {gate_name}: {exc}")

    ended_at = datetime.now(timezone.utc).isoformat()

    if not gate_result["passed"]:
        update_loop_experiment_state(
            exp_id, "CORPUS_GATE_FAILED", "corpus_gate_failed",
            payload={
                "gate_name": gate_name,
                "gate_result": gate_result,
                "manifest_path": str(manifest_path),
                "started_at": started_at,
                "ended_at": ended_at,
            },
        )
        raise RuntimeError(
            f"Corpus gate '{gate_name}' FAILED for {exp_id}: "
            f"{gate['pass_condition']['metric']}={gate_result['actual_value']} "
            f"(expected {gate_result['operator']} {gate_result['threshold']}). "
            f"Experiment left in CORPUS_GATE_FAILED; not frozen."
        )

    return gate_result


def run_corpus_pipeline(config: dict) -> dict:
    """
    Full corpus pipeline:
      1. Validate corpus files
      2. Deduplicate
      3. Split into train/val/test
      4. Version and freeze manifest
    """
    exp_id = config["exp_id"]
    corpus_dir = Path(config["corpus_dir"])
    output_dir = Path(config["output_dir"])
    _check_write(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()

    # Ensure experiment exists
    exp = get_loop_experiment(exp_id)
    if exp is None:
        create_loop_experiment(
            exp_id=exp_id,
            hypothesis=f"exp-{exp_id} corpus pipeline",
        )

    pipeline_stages = []

    # Stage 1: Validate
    print(f"[{exp_id}] Stage 1: Validating corpus")
    val_artifact_dir = output_dir / "validate"
    val_result = _run_worker(
        argv=[
            "python", "-m", "workers.corpus.validate",
            "--corpus-dir", str(corpus_dir),
        ],
        artifact_dir=val_artifact_dir,
        timeout_seconds=120,
    )
    pipeline_stages.append({"stage": "validate", "result": val_result})

    # Stage 2: Dedupe — reads from validate stage output
    print(f"[{exp_id}] Stage 2: Deduplicating")
    dedupe_artifact_dir = output_dir / "dedupe"
    dedupe_result = _run_worker(
        argv=[
            "python", "-m", "workers.corpus.dedupe",
            "--corpus-dir", str(val_artifact_dir),
        ],
        artifact_dir=dedupe_artifact_dir,
        timeout_seconds=300,
    )
    pipeline_stages.append({"stage": "dedupe", "result": dedupe_result})

    # Stage 3: Split — reads from dedupe stage output
    print(f"[{exp_id}] Stage 3: Splitting corpus")
    split_artifact_dir = output_dir / "split"
    split_argv = [
        "python", "-m", "workers.corpus.split",
        "--corpus-dir", str(dedupe_artifact_dir),
        "--output-dir", str(output_dir / "corpus"),
        "--seed", str(config.get("seed", 42)),
        "--train-pct", str(config.get("train_pct", 0.80)),
        "--val-pct", str(config.get("val_pct", 0.10)),
    ]
    if config.get("max_rows_per_source"):
        split_argv += ["--max-rows-per-source", str(config["max_rows_per_source"])]
    split_result = _run_worker(
        argv=split_argv,
        artifact_dir=split_artifact_dir,
        timeout_seconds=300,
    )
    pipeline_stages.append({"stage": "split", "result": split_result})

    # Stage 4: Version
    print(f"[{exp_id}] Stage 4: Versioning and freezing manifest")
    version_artifact_dir = output_dir / "version"
    version_argv = [
        "python", "-m", "workers.corpus.version",
        "--corpus-dir", str(output_dir / "corpus"),
        "--output-dir", str(output_dir),
    ]
    if config.get("scope"):
        version_argv += ["--scope", str(config["scope"])]
    if config.get("revision") is not None:
        version_argv += ["--revision", str(config["revision"])]
    if config.get("max_source_share") is not None:
        version_argv += ["--max-source-share", str(config["max_source_share"])]
    version_result = _run_worker(
        argv=version_argv,
        artifact_dir=version_artifact_dir,
        timeout_seconds=60,
    )
    pipeline_stages.append({"stage": "version", "result": version_result})

    # Stage 5: Corpus gates — verify the frozen split is source-disjoint and
    # that no single source dominates a primary partition.
    print(f"[{exp_id}] Stage 5: Running corpus gates")
    manifest_path = output_dir / "MANIFEST.json"
    gate_result = run_corpus_gate(
        exp_id=exp_id,
        gate_name="corpus_source_disjoint_gate",
        manifest_path=manifest_path,
        artifact_dir=version_artifact_dir,
    )
    pipeline_stages.append({"stage": "gate", "result": gate_result})

    # The max-source-share gate is opt-in: it rejects a partition dominated by
    # one source (e.g. the 68% single-source val share that motivated it), but
    # a tiny synthetic corpus (smoke fixture) cannot satisfy it by construction.
    # Pass --max-source-share <threshold> to enforce it on a real corpus.
    if config.get("max_source_share"):
        share_gate_result = run_corpus_gate(
            exp_id=exp_id,
            gate_name="corpus_max_source_share_gate",
            manifest_path=manifest_path,
            artifact_dir=version_artifact_dir,
            threshold_override=float(config["max_source_share"]),
        )
        pipeline_stages.append({"stage": "gate_max_source_share", "result": share_gate_result})

    ended_at = datetime.now(timezone.utc).isoformat()

    update_loop_experiment_state(exp_id, "FROZEN_CORPUS", "corpus_pipeline_complete",
                               payload={
                                   "corpus_hash": version_result["metrics"]["corpus_hash"],
                               })

    summary = {
        "exp_id": exp_id,
        "corpus_hash": version_result["metrics"]["corpus_hash"],
        "total_rows": version_result["metrics"]["total_rows"],
        "unique_normalized": version_result["metrics"]["unique_normalized"],
        "splits": version_result["metrics"]["splits"],
        "stages": [s["stage"] for s in pipeline_stages],
        "started_at": started_at,
        "ended_at": ended_at,
    }

    print(f"\n[{exp_id}] Corpus pipeline complete:")
    print(f"  Corpus hash: {summary['corpus_hash']}")
    print(f"  Total rows:  {summary['total_rows']}")
    print(f"  Unique:      {summary['unique_normalized']}")
    print(f"  Splits:      {summary['splits']}")

    return summary


def cmd_status(exp_id: str) -> None:
    try:
        exp = get_loop_experiment(exp_id)
        print(f"Experiment: {exp_id}")
        print(f"  State: {exp['state']}")
        print(f"  Created: {exp['created_at']}")
        if exp.get("ended_at"):
            print(f"  Ended: {exp['ended_at']}")
    except KeyError:
        print(f"Experiment {exp_id} not found")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run full corpus pipeline")
    p_run.add_argument("--experiment", "--exp-id", dest="exp_id", required=True,
                       help="Experiment ID (e.g. exp-004)")
    p_run.add_argument("--input", "--corpus-dir", dest="corpus_dir", required=True,
                       help="Directory containing raw corpus JSONL files")
    p_run.add_argument("--output", "--output-dir", dest="output_dir", required=True,
                       help="Output directory for processed corpus and manifests")
    p_run.add_argument("--seed", type=int, default=42,
                       help="Deterministic split seed (default 42)")
    p_run.add_argument("--train-pct", type=float, default=0.80,
                       help="Target train fraction of rows (default 0.80)")
    p_run.add_argument("--val-pct", type=float, default=0.10,
                       help="Target val fraction of rows (default 0.10)")
    p_run.add_argument("--max-rows-per-source", type=int, default=500,
                       help="Cap rows per source group before splitting (default 500)")
    p_run.add_argument("--max-source-share", type=float, default=None,
                       help="Enforce the max-source-share gate with this threshold "
                            "(e.g. 0.25). Omit to skip the gate (smoke/synthetic corpora).")
    p_run.add_argument("--revision", type=int, default=None,
                       help="Corpus revision number written into MANIFEST.json "
                            "(re-freezing a corpus is a new revision, not a rewrite)")
    p_run.add_argument("--scope", default=None,
                       help="Experiment scope written into MANIFEST.json")

    p_status = sub.add_parser("status", help="Show corpus experiment status")
    p_status.add_argument("--exp-id", required=True)

    args = parser.parse_args()

    if args.cmd == "run":
        result = run_corpus_pipeline(vars(args))
        print(json.dumps(result, indent=2))

    elif args.cmd == "status":
        cmd_status(args.exp_id)

    else:
        parser.print_help()


if __name__ == "__main__":
    raise SystemExit(main())
