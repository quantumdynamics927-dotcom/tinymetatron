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
    split_result = _run_worker(
        argv=[
            "python", "-m", "workers.corpus.split",
            "--corpus-dir", str(dedupe_artifact_dir),
            "--output-dir", str(output_dir / "corpus"),
        ],
        artifact_dir=split_artifact_dir,
        timeout_seconds=300,
    )
    pipeline_stages.append({"stage": "split", "result": split_result})

    # Stage 4: Version
    print(f"[{exp_id}] Stage 4: Versioning and freezing manifest")
    version_artifact_dir = output_dir / "version"
    version_result = _run_worker(
        argv=[
            "python", "-m", "workers.corpus.version",
            "--corpus-dir", str(output_dir / "corpus"),
            "--output-dir", str(output_dir),
        ],
        artifact_dir=version_artifact_dir,
        timeout_seconds=60,
    )
    pipeline_stages.append({"stage": "version", "result": version_result})

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
