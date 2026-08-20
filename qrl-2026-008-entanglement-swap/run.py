"""
Entanglement Swapping Execution Script — qrl-008
"""

import json, hashlib, datetime, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = Path(__file__).resolve().parent
CIRCUIT_PY = EXP_DIR / "circuit.py"

# ── helpers ────────────────────────────────────────────────────────────────────

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


# ── import circuit logic ───────────────────────────────────────────────────────
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(1, str(EXP_DIR))
from circuit import swap_statevector_exact, swap_fidelity_dm


def main():
    shots_small = 1_024
    shots_large = 100_000

    print("Two-Hop Entanglement Swapping -- qrl-008")
    print("=" * 50)
    print(f"Git commit: {git_commit()}")
    print()

    # ── Statevector exact (noiseless) ─────────────────────────────────────────
    print("--- Noiseless Statevector (exact) ---")
    exact = swap_statevector_exact()
    print(f"  With corrections:    F = {exact['with_corrections']:.6f}")
    print(f"  Without corrections: F = {exact['without_corrections']:.6f}")
    print(f"  Improvement: +{exact['with_corrections'] - exact['without_corrections']:.6f}")

    # ── Noiseless (density matrix fidelity, 100k shots) ─────────────────────
    print(f"\n--- Noiseless ({shots_large:,} shots, density matrix) ---")
    r_nl = swap_fidelity_dm(shots=shots_large, noisy=False)
    wc_nl = r_nl['with_corrections']
    woc_nl = r_nl['without_corrections']
    print(f"  With corrections:    F = {wc_nl['fidelity']:.4f}  E = {wc_nl['correlator_E']:.4f}")
    print(f"  Without corrections: F = {woc_nl['fidelity']:.4f}  E = {woc_nl['correlator_E']:.4f}")
    print(f"  Fidelity improvement: +{wc_nl['fidelity'] - woc_nl['fidelity']:.4f}")

    # ── Noisy (density matrix fidelity, 100k shots) ─────────────────────────
    print(f"\n--- Noisy ({shots_large:,} shots, qrl-004 noise) ---")
    r_no = swap_fidelity_dm(shots=shots_large, noisy=True)
    wc_no = r_no['with_corrections']
    woc_no = r_no['without_corrections']
    print(f"  With corrections:    F = {wc_no['fidelity']:.4f}  E = {wc_no['correlator_E']:.4f}")
    print(f"  Without corrections: F = {woc_no['fidelity']:.4f}  E = {woc_no['correlator_E']:.4f}")
    print(f"  Fidelity improvement: +{wc_no['fidelity'] - woc_no['fidelity']:.4f}")

    # ── Record results ────────────────────────────────────────────────────────
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = {
        "date_run": now,
        "noiseless_statevector": {
            "with_corrections": float(exact['with_corrections']),
            "without_corrections": float(exact['without_corrections']),
        },
        "noiseless_dm": {
            "shots": shots_large,
            "with_corrections_fidelity": float(wc_nl['fidelity']),
            "without_corrections_fidelity": float(woc_nl['fidelity']),
            "with_corrections_E": float(wc_nl['correlator_E']),
            "without_corrections_E": float(woc_nl['correlator_E']),
        },
        "noisy_dm": {
            "shots": shots_large,
            "with_corrections_fidelity": float(wc_no['fidelity']),
            "without_corrections_fidelity": float(woc_no['fidelity']),
            "with_corrections_E": float(wc_no['correlator_E']),
            "without_corrections_E": float(woc_no['correlator_E']),
        },
        "circuit_hash": sha256(CIRCUIT_PY),
        "code_commit": git_commit(),
    }

    results_path = EXP_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nResults written to: {results_path}")

    # ── Update manifest ───────────────────────────────────────────────────────
    manifest_path = EXP_DIR / "MANIFEST.md"
    manifest = manifest_path.read_text(encoding="utf-8")

    replacement = f"""## Results

- **date_run**: {now}
- **noiseless_statevector_fidelity**: {exact['with_corrections']:.6f}
- **noiseless_baseline_fidelity**: {exact['without_corrections']:.6f}
- **noiseless_improvement**: {exact['with_corrections'] - exact['without_corrections']:.6f}
- **noisy_swap_fidelity**: {wc_no['fidelity']:.4f}
- **noisy_baseline_fidelity**: {woc_no['fidelity']:.4f}
- **noisy_improvement**: {wc_no['fidelity'] - woc_no['fidelity']:.4f}
- **circuit_hash**: {sha256(CIRCUIT_PY)}
- **code_commit**: {git_commit()}
- **shots**: {shots_large}"""

    if "## Results" in manifest and "**date_run**: TBD" in manifest:
        manifest = manifest.split("## Results")[0] + replacement + "\n"
    else:
        manifest = manifest + "\n\n" + replacement

    manifest_path.write_text(manifest, encoding="utf-8")
    print(f"Manifest updated: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
