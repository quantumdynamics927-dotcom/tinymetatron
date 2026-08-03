"""
GHZ + Mermin Test — Execution & Results

Run the full Mermin experiment, record to results.json and update MANIFEST.md.
Shot-count sensitivity: run at 1024 AND 100k shots.
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
from circuit import run_mermin


def main():
    shots = 1024

    print(f"GHZ + Mermin Test -- Qiskit Aer (noiseless)")
    print(f"{'='*55}")

    result = run_mermin(shots=shots)
    M = result["M"]
    M_th = result["M_theory"]
    E_vals = result["E_vals"]

    print(f"Circuit hash (circuit.py): {sha256(CIRCUIT_PY)}")
    print(f"Git commit: {git_commit()}")
    print(f"Shots: {shots}")
    print()
    print(f"M = {M:.6f}")
    print(f"Classical bound: M <= 4.0  -> {'VIOLATED [+]' if M > 4 else 'NOT VIOLATED [-]'}")
    print(f"Quantum maximum: M <= {M_th:.4f}")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = {
        "date_run": now,
        "result_value": round(M, 6),
        "violated_bound": "classical" if M > 4 else "neither",
        "passes_minimum_bar": bool(M > 4),
        "passes_target": bool(M >= 5.5),
        "circuit_hash": sha256(CIRCUIT_PY),
        "code_commit": git_commit(),
        "shots": shots,
        "E_vals": {k: round(v, 6) for k, v in E_vals.items()},
        "classical_bound": 4.0,
        "quantum_maximum": round(M_th, 6),
        "shot_noise_verified": False,
    }

    results_path = EXP_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nResults written to: {results_path}")

    manifest_path = EXP_DIR / "MANIFEST.md"
    manifest = manifest_path.read_text(encoding="utf-8")

    replacement = f"""## Results

- **date_run**: {now}
- **result_value**: {M:.6f}
- **violated_bound**: {"classical" if M > 4 else "neither"}
- **circuit_hash**: {sha256(CIRCUIT_PY)}
- **code_commit**: {git_commit()}
- **passes_minimum_bar**: {str(M > 4).lower()} (M > 4 = quantum advantage)
- **passes_target**: {str(M >= 5.5).lower()} (M >= 5.5 = near quantum max)
- **shot_noise_verified**: false"""

    if "## Results" in manifest and "date_run**: TBD" in manifest:
        manifest = manifest.split("## Results")[0] + replacement + "\n"
    else:
        manifest = manifest + "\n\n" + replacement

    manifest_path.write_text(manifest, encoding="utf-8")
    print(f"Manifest updated: {manifest_path}")

    return 0 if M > 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
