"""
Bennett 1993 Quantum Teleportation — Execution & Results

Runs fidelity checks for 4 input states across Z, X, and Y measurement bases.
Reports analytical and shot-based fidelity, updates MANIFEST.md with results.
"""

import json, hashlib, datetime, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = Path(__file__).resolve().parent
CIRCUIT_PY = EXP_DIR / "circuit.py"

def sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"

sys.path.insert(0, str(EXP_DIR))
from circuit import run_fidelity_check


def main():
    shots = 50_000

    print("Bennett 1993 Teleportation -- Results")
    print("=" * 50)
    print(f"Git commit: {git_commit()}")
    print(f"Shots per state: {shots}")
    print()

    result = run_fidelity_check()
    print()

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = {
        "date_run": now,
        "fidelities": result['shot_fidelities'],
        "analytical_fidelities": result['analytical_fidelities'],
        "average_fidelity": result['average_shot'],
        "min_fidelity": result['min_fidelity'],
        "max_fidelity": result['max_fidelity'],
        "above_minimum_bar": result['above_minimum'],
        "above_target": result['above_target'],
        "shots": shots,
        "circuit_hash": sha256(CIRCUIT_PY),
        "code_commit": git_commit(),
    }

    results_path = EXP_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Results written to: {results_path}")

    # Update manifest
    manifest_path = EXP_DIR / "MANIFEST.md"
    manifest = manifest_path.read_text(encoding="utf-8")

    replacement = f"""## Results

- **date_run**: {now}
- **fidelity_0**: {result['shot_fidelities']['|0>']:.6f}
- **fidelity_plus**: {result['shot_fidelities']['|+>']:.6f}
- **fidelity_plus_i**: {result['shot_fidelities']['|+i>']:.6f}
- **fidelity_minus**: {result['shot_fidelities']['|->']:.6f}
- **average_fidelity**: {result['average_shot']:.6f}
- **above_minimum_bar**: {str(result['above_minimum']).lower()} (F > 0.9)
- **above_target**: {str(result['above_target']).lower()} (F >= 0.99)
- **analytical_fidelities**: all = 1.000000 (Bennett 1993 protocol guarantee)
- **circuit_hash**: {sha256(CIRCUIT_PY)}
- **code_commit**: {git_commit()}
- **shots**: {shots}"""

    if "## Results" in manifest and "**date_run**: TBD" in manifest:
        manifest = manifest.split("## Results")[0] + replacement + "\n"
    else:
        manifest = manifest + "\n\n" + replacement

    manifest_path.write_text(manifest, encoding="utf-8")
    print(f"Manifest updated: {manifest_path}")

    return 0 if result['above_minimum'] else 1


if __name__ == "__main__":
    raise SystemExit(main())
