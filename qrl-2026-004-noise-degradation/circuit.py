"""
Noise Degradation Study — CHSH + Mermin under realistic noise model.

CHSH uses RY-based measurement (A = cos*Z + sin*X), validated in qrl-002.
Mermin uses RZ+H-based measurement (A = cos*X + sin*Y), validated in qrl-003.

These are DIFFERENT measurement operators because CHSH and Mermin are
different inequalities with different optimal strategies.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, PauliError, ReadoutError


# ── Noise Model ──────────────────────────────────────────────────────────────

def make_noise_model(p1q: float = 0.002, p2q: float = 0.015,
                     ro_err: float = 0.02) -> NoiseModel:
    """
    Build a realistic noise model based on typical IBM 7-qubit device
    calibration data. No credentials needed — pure qiskit_aer.

    Args:
        p1q: 1-qubit gate depolarizing probability (default 0.2% per H/RZ/SX/X)
        p2q: 2-qubit gate (CX) depolarizing probability (default 1.5%)
        ro_err: Per-qubit readout bit-flip probability (default 2%)
    """
    nm = NoiseModel()

    # 1-qubit gate errors (H, RZ, SX, X — applied to qubits 0,1,2)
    for qubit in range(3):
        for gate in ('h', 'rz', 'sx', 'x'):
            nm.add_quantum_error(
                PauliError(['X', 'I'], [p1q, 1 - p1q]), gate, [qubit])

    # 2-qubit gate errors (CX — both pairs used in our circuits)
    nm.add_quantum_error(
        PauliError(['XY', 'II'], [p2q, 1 - p2q]), 'cx', [0, 1])
    nm.add_quantum_error(
        PauliError(['XY', 'II'], [p2q, 1 - p2q]), 'cx', [0, 2])

    # Readout errors
    for qubit in range(3):
        nm.add_readout_error(
            ReadoutError([[1 - ro_err, ro_err], [ro_err, 1 - ro_err]]), [qubit])

    return nm


# ── CHSH (2-qubit Bell state) — RY-based measurement ───────────────────────
#
# Measurement: A(theta) = cos(theta)*Z + sin(theta)*X
# Implementation: RY(-theta) then measure Z
#   RY(-theta)|psi> then Z measurement = expectation of cos*Z + sin*X
# For |Phi+> with angles a=0, a'=90, b=45, b'=135: S = 2.828

def make_bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, name='Bell')
    qc.h(0)
    qc.cx(0, 1)
    return qc


def make_chsh_measure(theta0: float, theta1: float) -> QuantumCircuit:
    """CHSH circuit: Bell state + RY-based measurement at (theta0, theta1)."""
    qc = make_bell_circuit()
    qc.ry(-theta0, 0)
    qc.ry(-theta1, 1)
    qc.measure_all()
    return qc


def compute_correlator_2q(counts: dict) -> float:
    """
    Compute E = P(++) + P(--) - P(+-) - P(-+) for 2-qubit counts.
    Qiskit bit ordering: ['01'] = qubit0=0, qubit1=1.
    Convention: 0 -> +1, 1 -> -1.
    E = (N_00 + N_11 - N_01 - N_10) / N_total
    """
    n00 = counts.get('00', 0)
    n11 = counts.get('11', 0)
    n01 = counts.get('01', 0)
    n10 = counts.get('10', 0)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return (n00 + n11 - n01 - n10) / total


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

    # CHSH: S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|
    S = (abs(E_vals['E(a,b)'] - E_vals['E(a,b\')']
               + E_vals['E(a\',b)'] + E_vals['E(a\',b\')']))

    return {
        'S': S,
        'E_vals': {k: round(v, 8) for k, v in E_vals.items()},
        'counts': all_counts,
        'shots': shots,
        'classical_bound': 2.0,
        'theoretical_S': 2*np.sqrt(2),
        'violates': bool(S > 2),
    }


# ── Mermin (3-qubit GHZ state) — RZ+H-based measurement ────────────────────
#
# Measurement: A(theta) = cos(theta)*X + sin(theta)*Y
# Implementation: RZ(-theta) then H
# For |GHZ+> with (0, pi/2) settings per qubit: M = 4

def make_ghz_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3, name='GHZ')
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    return qc


def make_mermin_measure(theta0: float, theta1: float, theta2: float) -> QuantumCircuit:
    """
    GHZ circuit with A(theta) = cos(theta)*X + sin(theta)*Y measurement.
    Implemented as: RZ(-theta) then H on each qubit.
    Correlator: E = cos(theta_a + theta_b + theta_c)
    """
    qc = make_ghz_circuit()
    for qi, theta in enumerate([theta0, theta1, theta2]):
        qc.rz(-theta, qi)
        qc.h(qi)
    qc.measure_all()
    return qc


def compute_correlator_3q(counts: dict, shots: int) -> float:
    """
    Compute E = (N_+++ + N_--- - N_all_other) / N_total for 3-qubit GHZ.
    Qiskit bit ordering: '000' = qubit2=0, qubit1=0, qubit0=0 (leftmost = qubit2).
    Sign: '000' and '111' get +, all other 6 outcomes get -.
    """
    n000 = counts.get('000', 0)
    n111 = counts.get('111', 0)
    others = (counts.get('001', 0) + counts.get('010', 0) +
              counts.get('100', 0) + counts.get('011', 0) +
              counts.get('101', 0) + counts.get('110', 0))
    return (n000 + n111 - others) / shots


def run_mermin(shots: int, noise_model=None) -> dict:
    """
    Run full Mermin experiment on GHZ state.
    Uses RZ+H measurement (A = cos*X + sin*Y).
    Mermin operator: M = E(0,0,0) - E(0,pi/2,pi/2) - E(pi/2,0,pi/2) - E(pi/2,pi/2,0)
    """
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
        E_vals_exact[key] = np.cos(sum(angles))

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
    print("=== Noise Degradation Study: verification ===\n")

    nm = make_noise_model()

    # Noiseless reference
    print("--- Noiseless (100k shots) ---")
    chsh_clean = run_chsh(100_000, noise_model=None)
    mermin_clean = run_mermin(100_000, noise_model=None)
    print(f"CHSH  S = {chsh_clean['S']:.6f}  (theory = {chsh_clean['theoretical_S']:.6f})")
    print(f"Mermin M = {mermin_clean['M_exact']:.6f}  (theory = {mermin_clean['theoretical_M']:.6f})")

    # Noisy
    print("\n--- Noisy (100k shots, p1q=0.002, p2q=0.015, ro=0.02) ---")
    chsh_noisy = run_chsh(100_000, noise_model=nm)
    mermin_noisy = run_mermin(100_000, noise_model=nm)
    print(f"CHSH  S = {chsh_noisy['S']:.6f}  {'[ABOVE]' if chsh_noisy['violates'] else '[BELOW]'} classical bound (S>2)")
    print(f"Mermin M = {mermin_noisy['M_shot']:.6f}  {'[ABOVE]' if mermin_noisy['violates'] else '[BELOW]'} classical bound (M>2)")
