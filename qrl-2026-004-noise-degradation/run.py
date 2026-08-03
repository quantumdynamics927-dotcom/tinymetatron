"""
Noise Degradation Study — Execution & Results

Runs CHSH (qrl-002) and Mermin (qrl-003) under a realistic noise model.
Reports both noiseless and noisy results, with PASS/FAIL against classical bounds.
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
from circuit import run_chsh, run_mermin, make_noise_model


def main():
    shots = 100_000
    nm = make_noise_model()

    print("=== Noise Degradation Study ===")
    print(f"Git commit: {git_commit()}")
    print(f"Shots: {shots}")
    print()

    # Noiseless reference
    print("--- Noiseless baseline ---")
    chsh_clean = run_chsh(shots, noise_model=None)
    mermin_clean = run_mermin(shots, noise_model=None)
    print(f"CHSH  S = {chsh_clean['S']:.6f}  (theory = {chsh_clean['theoretical_S']:.6f})")
    print(f"Mermin M = {mermin_clean['M_exact']:.6f}  (theory = {mermin_clean['theoretical_M']:.6f})")
    print()

    # Noisy run
    print("--- With noise model (p1q=0.002, p2q=0.015, ro=0.02) ---")
    chsh_noisy = run_chsh(shots, noise_model=nm)
    mermin_noisy = run_mermin(shots, noise_model=nm)
    print(f"CHSH  S = {chsh_noisy['S']:.6f}  {'[ABOVE]' if chsh_noisy['violates'] else '[BELOW]'} classical bound (S>2)")
    print(f"Mermin M = {mermin_noisy['M_shot']:.6f}  {'[ABOVE]' if mermin_noisy['violates'] else '[BELOW]'} classical bound (M>2)")
    print()

    # Degradation summary
    chsh_deg = chsh_clean['S'] - chsh_noisy['S']
    mermin_deg = mermin_clean['M_shot'] - mermin_noisy['M_shot']
    print(f"CHSH degradation:  {chsh_deg:.6f}")
    print(f"Mermin degradation: {mermin_deg:.6f}")
    print()

    # Classification
    if chsh_noisy['violates'] and mermin_noisy['violates']:
        outcome = "BOTH_ABOVE — both remain above classical bound under noise"
    elif chsh_noisy['violates'] and not mermin_noisy['violates']:
        outcome = "CHSH_ONLY — CHSH is robust, Mermin degraded below bound (informative)"
    elif not chsh_noisy['violates'] and mermin_noisy['violates']:
        outcome = "MERMIN_ONLY — unexpected: Mermin survives but CHSH doesn't"
    else:
        outcome = "BOTH_BELOW — both failed under realistic noise"

    print(f"Outcome: {outcome}")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = {
        "date_run": now,
        "noiseless_CHSH_S": round(chsh_clean['S'], 6),
        "noisy_CHSH_S": round(chsh_noisy['S'], 6),
        "noiseless_Mermin_M": round(mermin_clean['M_shot'], 6),
        "noisy_Mermin_M": round(mermin_noisy['M_shot'], 6),
        "chsh_degradation": round(chsh_deg, 6),
        "mermin_degradation": round(mermin_deg, 6),
        "chsh_above_bound": bool(chsh_noisy['violates']),
        "mermin_above_bound": bool(mermin_noisy['violates']),
        "outcome": outcome,
        "noise_params": {"p1q": 0.002, "p2q": 0.015, "readout_error": 0.02},
        "circuit_hash_CHSH": sha256(CIRCUIT_PY),
        "circuit_hash_Mermin": sha256(CIRCUIT_PY),
        "code_commit": git_commit(),
        "shots": shots,
        "classical_bound": 2.0,
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
- **noiseless_CHSH_S**: {chsh_clean['S']:.6f}
- **noisy_CHSH_S**: {chsh_noisy['S']:.6f}
- **noiseless_Mermin_M**: {mermin_clean['M_shot']:.6f}
- **noisy_Mermin_M**: {mermin_noisy['M_shot']:.6f}
- **chsh_degradation**: {chsh_deg:.6f}
- **mermin_degradation**: {mermin_deg:.6f}
- **chsh_above_bound**: {str(chsh_noisy['violates']).lower()} (S > 2)
- **mermin_above_bound**: {str(mermin_noisy['violates']).lower()} (M > 2)
- **outcome**: {outcome}
- **noise_params**: p1q=0.002, p2q=0.015, readout=0.02
- **circuit_hash_CHSH**: {sha256(CIRCUIT_PY)}
- **circuit_hash_Mermin**: {sha256(CIRCUIT_PY)}
- **code_commit**: {git_commit()}
- **shots**: {shots}"""

    if "## Results" in manifest and "**date_run**: TBD" in manifest:
        manifest = manifest.split("## Results")[0] + replacement + "\n"
    else:
        manifest = manifest + "\n\n" + replacement

    manifest_path.write_text(manifest, encoding="utf-8")
    print(f"Manifest updated: {manifest_path}")

    # Return 0 if at least one experiment remains above bound
    return 0 if (chsh_noisy['violates'] or mermin_noisy['violates']) else 1


if __name__ == "__main__":
    raise SystemExit(main())
