"""
CHSH Bell Test — Execution & Results

Run the CHSH circuit, compute S, record to results.json and update MANIFEST.md.
"""

import json, hashlib, datetime, math, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = Path(__file__).resolve().parent
CIRCUIT_PY = __file__.replace("run.py", "circuit.py")

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
# Insert REPO_ROOT first so circuit.py (which adds its parent to sys.path)
# can find qrl_common from the repo root.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(1, str(EXP_DIR))
from circuit import make_chsh_circuit, compute_correlation
from qiskit_aer import AerSimulator


def run_chsh(shots: int = 1024) -> dict:
    """Run the full CHSH experiment and return results dict."""
    # Optimal CHSH angles for |Phi+> state
    a  = 0.0
    ap = math.pi / 2      # 90°
    b  = math.pi / 4     # 45°
    bp = 3 * math.pi / 4 # 135°

    angles = [(a, b), (a, bp), (ap, b), (ap, bp)]
    sim = AerSimulator()
    E_vals = {}
    raw_counts = {}

    for th0, th1 in angles:
        qc = make_chsh_circuit(th0, th1, shots=shots)
        job = sim.run(qc, shots=shots)
        counts = job.result().get_counts(qc)
        E = compute_correlation(counts)
        E_vals[(th0, th1)] = E
        raw_counts[f"{th0:.4f}_{th1:.4f}"] = counts

    S = abs(E_vals[(a, b)] - E_vals[(a, bp)] + E_vals[(ap, b)] + E_vals[(ap, bp)])

    return {
        "S": round(S, 6),
        "E_vals": {f"E({k[0]:.4f},{k[1]:.4f})": round(v, 6) for k, v in E_vals.items()},
        "raw_counts": raw_counts,
        "shots": shots,
        "angles": {"a": a, "a_prime": ap, "b": b, "b_prime": bp},
        "classical_bound": 2.0,
        "quantum_maximum": round(2 * math.sqrt(2), 6),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    shots = 1024

    print(f"CHSH Bell Test — Qiskit Aer (noiseless)")
    print(f"{'='*55}")

    result = run_chsh(shots=shots)
    S = result["S"]
    q_max = result["quantum_maximum"]

    print(f"Circuit hash (circuit.py): {sha256(Path(CIRCUIT_PY))}")
    print(f"Git commit: {git_commit()}")
    print(f"Shots: {shots}")
    print()
    print(f"S = {S:.6f}")
    print(f"Classical bound:  S <= 2.0  -> {'VIOLATED [OK]' if S > 2 else 'NOT VIOLATED [FAIL]'}")
    print(f"Quantum maximum: S <= {q_max}  -> {'AT QUANTUM LIMIT [OK]' if S > 2.7 else 'BELOW TARGET [FAIL]'}")

    # Build record
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = {
        "date_run": now,
        "result_value": S,
        "violated_bound": "classical" if S > 2 else "neither",
        "passes_minimum_bar": S > 2,
        "passes_target": S >= 2.7,
        "circuit_hash": sha256(Path(CIRCUIT_PY)),
        "code_commit": git_commit(),
        "shots": shots,
        "angles": result["angles"],
        "E_vals": result["E_vals"],
        "classical_bound": 2.0,
        "quantum_maximum": q_max,
    }

    # Write results.json
    results_path = EXP_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nResults written to: {results_path}")

    # Update MANIFEST.md results section
    manifest_path = EXP_DIR / "MANIFEST.md"
    manifest = manifest_path.read_text(encoding="utf-8")

    replacement = f"""## Results

- **date_run**: {now}
- **result_value**: {S:.6f}
- **violated_bound**: {"classical" if S > 2 else "neither"}
- **circuit_hash**: {sha256(Path(CIRCUIT_PY))}
- **code_commit**: {git_commit()}
- **passes_minimum_bar**: {str(S > 2).lower()} (S > 2 = any quantum advantage)
- **passes_target**: {str(S >= 2.7).lower()} (S >= 2.7 = near quantum maximum)"""

    if "## Results" in manifest and "date_run**: TBD" in manifest:
        manifest = manifest.split("## Results")[0] + replacement + "\n"
    else:
        manifest = manifest + "\n\n" + replacement

    manifest_path.write_text(manifest, encoding="utf-8")
    print(f"Manifest updated: {manifest_path}")

    return 0 if S > 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
