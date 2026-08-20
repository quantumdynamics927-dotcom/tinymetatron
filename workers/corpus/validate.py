"""
workers/corpus/validate.py
=========================
Quality-score, length, and syntax checks → result JSON.

Contract:
    Reads: config['corpus_dir'] (directory containing *.jsonl files)
    Writes: result.json to config['artifact_dir']

Output schema:
{
  "worker": "workers.corpus.validate",
  "version": 1,
  "status": "success",
  "metrics": {
    "total_files": int,
    "total_rows": int,
    "valid_rows": int,
    "rejected_empty": int,
    "rejected_too_short": int,
    "rejected_too_long": int,
    "rejected_no_text": int,
    "rejected_invalid_json": int
  },
  "artifacts": {"valid_rows": Path}
}

Usage:
    python -m workers.corpus.validate --corpus-dir data/raw --result result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

WORKER_VERSION = 1

MIN_TEXT_LEN = 20
MAX_TEXT_LEN = 100_000


def validate_row(row: dict) -> tuple[bool, str]:
    """Validate a single row. Returns (passes, reason)."""
    if not isinstance(row, dict):
        return False, "not_a_dict"

    text = row.get("text", "")
    if not isinstance(text, str):
        return False, "no_text"

    text = text.strip()
    if not text:
        return False, "empty"

    if len(text) < MIN_TEXT_LEN:
        return False, "too_short"

    if len(text) > MAX_TEXT_LEN:
        return False, "too_long"

    return True, ""


def run(config: dict) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    corpus_dir = Path(config["corpus_dir"])
    artifact_dir = Path(config.get("artifact_dir", "."))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(corpus_dir.glob("*.jsonl"))
    total_rows = 0
    valid_rows = 0
    rejected = {"empty": 0, "too_short": 0, "too_long": 0, "no_text": 0, "invalid_json": 0}

    valid_output_path = artifact_dir / "validated.jsonl"

    with open(valid_output_path, "w", encoding="utf-8") as out_f:
        for f in files:
            for line in open(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                total_rows += 1
                try:
                    row = json.loads(line)
                except Exception:
                    rejected["invalid_json"] += 1
                    continue

                passes, reason = validate_row(row)
                if passes:
                    valid_rows += 1
                    out_f.write(line + "\n")
                else:
                    key = reason.replace("_", "_")
                    if key in rejected:
                        rejected[key] += 1
                    else:
                        rejected["empty"] += 1

    ended_at = datetime.now(timezone.utc).isoformat()

    import hashlib
    h = hashlib.sha256()
    h.update(str(total_rows).encode())
    h.update(str(valid_rows).encode())
    input_hash = "sha256:" + h.hexdigest()[:16]

    h2 = hashlib.sha256()
    with open(valid_output_path, "rb") as vf:
        for chunk in iter(lambda: vf.read(65536), b""):
            h2.update(chunk)
    output_hash = "sha256:" + h2.hexdigest()[:16]

    result = {
        "worker": "workers.corpus.validate",
        "version": WORKER_VERSION,
        "status": "success",
        "input_hash": input_hash,
        "output_hash": output_hash,
        "artifact_paths": [str(valid_output_path)],
        "metrics": {
            "total_files": len(files),
            "total_rows": total_rows,
            "valid_rows": valid_rows,
            "rejected_empty": rejected["empty"],
            "rejected_too_short": rejected["too_short"],
            "rejected_too_long": rejected["too_long"],
            "rejected_no_text": rejected["no_text"],
            "rejected_invalid_json": rejected["invalid_json"],
        },
        "started_at": started_at,
        "ended_at": ended_at,
    }

    result_path = artifact_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--result", default=None)
    args = parser.parse_args()

    artifact_dir = Path(args.result).parent if args.result else Path(".")
    config = vars(args)
    config["artifact_dir"] = str(artifact_dir)
    result = run(config)

    if args.result:
        with open(args.result, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
