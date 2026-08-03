"""
Noise Degradation Study — CHSH + Mermin under realistic noise model.

CHSH uses RY-based measurement (A = cos*Z + sin*X), validated in qrl-002.
Mermin uses RZ+H-based measurement (A = cos*X + sin*Y), validated in qrl-003.

Shared utilities: qrl-common (Bell/GHZ prep, correlators, noise model).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'qrl_common'))
from qrl_common import (
    make_bell_circuit,
    make_ghz_circuit,
    compute_correlator_2q,
    compute_correlator_3q,
    exact_correlator_3q,
    make_noise_model,
)

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ── CHSH (2-qubit Bell state) — RY-based measurement ───────────────────────

def make_chsh_measure(theta0: float, theta1: float) -> QuantumCircuit:
    """CHSH circuit: Bell state + RY-based measurement at (theta0, theta1)."""
    qc = make_bell_circuit()
    qc.ry(-theta0, 0)
    qc.ry(-theta1, 1)
    qc.measure_all()
    return qc


def run_chsh(shots: int, noise_model=None) -> dict:
    """Run full CHSH experiment. Angles: a=0, a'=pi/2, b=pi/4, b'=3pi/4."""
    a, ap, b, bp = 0.0, np.pi/2, np.pi/4, 3*np.pi/4

    sim = AerSimulator()
    settings = {
        'E(a,b)':    (a, b),
        'E(a,b\')':  (a, bp),
        'E(a\',b)':  (ap, b),
        'E(a\',b\')':(ap, bp),
    }

    E_vals = {}
    all_counts = {}
    for key, angles in settings.items():
        qc = make_chsh_measure(*angles)
        result = sim.run(qc, shots=shots, noise_model=noise_model).result()
        counts = result.get_counts(qc)
        all_counts[key] = counts
        E_vals[key] = compute_correlator_2q(counts)

    S = abs(E_vals['E(a,b)'] - E_vals['E(a,b\')']
               + E_vals['E(a\',b)'] + E_vals['E(a\',b\')'])

    return {
        'S': S,
        'E_vals': {k: round(v, 8) for k, v in E_vals.items()},
        'counts': all_counts,
        'shots': shots,
        'classical_bound': 2.0,
        'theoretical_S': 2*np.sqrt(2),
        'violates': bool(S > 2),
    }


# ── Mermin (3-qubit GHZ state) — RZ+H-based measurement ───────────────────

def make_mermin_measure(theta0: float, theta1: float, theta2: float) -> QuantumCircuit:
    """
    GHZ circuit with A(theta) = cos(theta)*X + sin(theta)*Y measurement.
    RZ(-theta) then H on each qubit.
    """
    qc = make_ghz_circuit()
    for qi, theta in enumerate([theta0, theta1, theta2]):
        qc.rz(-theta, qi)
        qc.h(qi)
    qc.measure_all()
    return qc


def run_mermin(shots: int, noise_model=None) -> dict:
    """Run full Mermin experiment on GHZ state."""
    sim = AerSimulator()
    th = np.pi / 2

    settings = {
        'E(0,0,0)':    (0.0, 0.0, 0.0),
        'E(0,th,th)':  (0.0, th, th),
        'E(th,0,th)':  (th, 0.0, th),
        'E(th,th,0)':  (th, th, 0.0),
    }

    E_vals_shot = {}
    E_vals_exact = {}
    all_counts = {}
    for key, angles in settings.items():
        qc = make_mermin_measure(*angles)
        result = sim.run(qc, shots=shots, noise_model=noise_model).result()
        counts = result.get_counts(qc)
        all_counts[key] = counts
        E_vals_shot[key] = compute_correlator_3q(counts, shots)
        E_vals_exact[key] = exact_correlator_3q(*angles)

    M = (E_vals_shot['E(0,0,0)']
         - E_vals_shot['E(0,th,th)']
         - E_vals_shot['E(th,0,th)']
         - E_vals_shot['E(th,th,0)'])

    return {
        'M_shot': M,
        'M_exact': (E_vals_exact['E(0,0,0)'] - E_vals_exact['E(0,th,th)']
                    - E_vals_exact['E(th,0,th)'] - E_vals_exact['E(th,th,0)']),
        'E_vals_shot': {k: round(v, 8) for k, v in E_vals_shot.items()},
        'E_vals_exact': {k: round(v, 8) for k, v in E_vals_exact.items()},
        'counts': all_counts,
        'shots': shots,
        'classical_bound': 2.0,
        'theoretical_M': 4.0,
        'violates': bool(M > 2),
    }


# ── Verification ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    nm = make_noise_model()

    print("=== Noise Degradation Study: verification ===\n")

    print("--- Noiseless (100k shots) ---")
    chsh_clean = run_chsh(100_000, noise_model=None)
    mermin_clean = run_mermin(100_000, noise_model=None)
    print(f"CHSH  S = {chsh_clean['S']:.6f}  (theory = {chsh_clean['theoretical_S']:.6f})")
    print(f"Mermin M = {mermin_clean['M_exact']:.6f}  (theory = {mermin_clean['theoretical_M']:.6f})")

    print("\n--- Noisy (100k shots, p1q=0.002, p2q=0.015, ro=0.02) ---")
    chsh_noisy = run_chsh(100_000, noise_model=nm)
    mermin_noisy = run_mermin(100_000, noise_model=nm)
    print(f"CHSH  S = {chsh_noisy['S']:.6f}  {'[ABOVE]' if chsh_noisy['violates'] else '[BELOW]'} classical bound (S>2)")
    print(f"Mermin M = {mermin_noisy['M_shot']:.6f}  {'[ABOVE]' if mermin_noisy['violates'] else '[BELOW]'} classical bound (M>2)")
