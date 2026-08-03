"""
Bit-Flip QEC Execution Script — qrl-007
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
from circuit import run_shot_test


def main():
    shots = 20_000

    print("Bit-Flip QEC -- qrl-007")
    print("=" * 50)
    print(f"Git commit: {git_commit()}")
    print(f"Shots per state: {shots}")
    print()

    print("--- Noiseless ---")
    r_nl = run_shot_test(shots=shots, noisy=False)
    print(f"  QEC mean F={r_nl['qec_mean']:.6f}  min={r_nl['qec_min']:.6f}")
    print(f"  Baseline mean F={r_nl['baseline_mean']:.6f}  min={r_nl['baseline_min']:.6f}")
    print(f"  Mean QEC improvement: {r_nl['mean_improvement']:.6f}")
    print(f"  Syndrome accuracy: {r_nl['syndrome_accuracy']:.2%}")

    print("\n--- Noisy (p1q=0.002, p2q=0.015, ro=0.02) ---")
    r_no = run_shot_test(shots=shots, noisy=True)
    print(f"  QEC mean F={r_no['qec_mean']:.6f}  min={r_no['qec_min']:.6f}")
    print(f"  Baseline mean F={r_no['baseline_mean']:.6f}  min={r_no['baseline_min']:.6f}")
    print(f"  Mean QEC improvement: {r_no['mean_improvement']:.6f}")
    print(f"  Syndrome accuracy: {r_no['syndrome_accuracy']:.2%}")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = {
        "date_run": now,
        "noiseless": {
            "shots": shots,
            "qec_mean_fidelity": float(r_nl['qec_mean']),
            "qec_min_fidelity": float(r_nl['qec_min']),
            "baseline_mean_fidelity": float(r_nl['baseline_mean']),
            "mean_improvement": float(r_nl['mean_improvement']),
            "syndrome_accuracy": float(r_nl['syndrome_accuracy']),
        },
        "noisy": {
            "shots": shots,
            "qec_mean_fidelity": float(r_no['qec_mean']),
            "qec_min_fidelity": float(r_no['qec_min']),
            "baseline_mean_fidelity": float(r_no['baseline_mean']),
            "mean_improvement": float(r_no['mean_improvement']),
            "syndrome_accuracy": float(r_no['syndrome_accuracy']),
        },
        "circuit_hash": sha256(CIRCUIT_PY),
        "code_commit": git_commit(),
    }

    results_path = EXP_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nResults written to: {results_path}")

    # Update manifest
    manifest_path = EXP_DIR / "MANIFEST.md"
    manifest = manifest_path.read_text(encoding="utf-8")

    replacement = f"""## Results

- **date_run**: {now}
- **noiseless_qec_mean_fidelity**: {r_nl['qec_mean']:.6f}
- **noiseless_qec_min_fidelity**: {r_nl['qec_min']:.6f}
- **noiseless_baseline_mean_fidelity**: {r_nl['baseline_mean']:.6f}
- **noiseless_mean_improvement**: {r_nl['mean_improvement']:.6f}
- **noiseless_syndrome_accuracy**: {r_nl['syndrome_accuracy']:.2%}
- **noisy_qec_mean_fidelity**: {r_no['qec_mean']:.6f}
- **noisy_qec_min_fidelity**: {r_no['qec_min']:.6f}
- **noisy_baseline_mean_fidelity**: {r_no['baseline_mean']:.6f}
- **noisy_mean_improvement**: {r_no['mean_improvement']:.6f}
- **noisy_syndrome_accuracy**: {r_no['syndrome_accuracy']:.2%}
- **circuit_hash**: {sha256(CIRCUIT_PY)}
- **code_commit**: {git_commit()}
- **shots_per_test**: {shots}"""

    if "## Results" in manifest and "**date_run**: TBD" in manifest:
        manifest = manifest.split("## Results")[0] + replacement + "\n"
    else:
        manifest = manifest + "\n\n" + replacement

    manifest_path.write_text(manifest, encoding="utf-8")
    print(f"Manifest updated: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
