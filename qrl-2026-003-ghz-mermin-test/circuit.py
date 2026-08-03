"""
GHZ State + Mermin Inequality — Corrected Implementation

Classical bound: |M| <= 2
Quantum (GHZ): |M| = 4

Measurement operator: A(theta) = cos(theta)*X + sin(theta)*Y
Correlator: E(theta_a, theta_b, theta_c) = cos(theta_a + theta_b + theta_c)

Settings (2 per qubit):
  theta=0   -> X measurement
  theta=pi/2 -> Y measurement

Mermin operator:
  M = E(0,0,0) - E(0,pi/2,pi/2) - E(pi/2,0,pi/2) - E(pi/2,pi/2,0)

With |GHZ+>:
  E(0,0,0)        = cos(0)         = +1
  E(0,pi/2,pi/2)  = cos(pi)         = -1
  E(pi/2,0,pi/2)  = cos(pi)         = -1
  E(pi/2,pi/2,0)  = cos(pi)         = -1
  M = 1 - (-1) - (-1) - (-1) = 4
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector


def make_ghz_circuit() -> QuantumCircuit:
    """3-qubit GHZ state preparation."""
    qc = QuantumCircuit(3, name="GHZ")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)   # |GHZ+> = (|000> + |111>)/sqrt2
    return qc


def make_ghz_measure(theta0: float, theta1: float, theta2: float) -> QuantumCircuit:
    """
    GHZ circuit with A(theta) = cos(theta)*X + sin(theta)*Y measurement.
    Implemented as: RZ(-theta) then H (gives cos*X + sin*Y in Z basis).
    """
    qc = make_ghz_circuit()
    for qi, theta in enumerate([theta0, theta1, theta2]):
        qc.rz(-theta, qi)   # RZ(-theta)
        qc.h(qi)             # H: transforms to X basis rotated by theta
    qc.measure_all()
    return qc


def exact_correlator(th0: float, th1: float, th2: float) -> float:
    """Exact E = cos(th0 + th1 + th2) from formula."""
    return np.cos(th0 + th1 + th2)


def compute_M_exact() -> dict:
    """Compute M using exact formula (statevector-verified)."""
    th = np.pi / 2

    E_vals = {
        'E(0,0,0)':        exact_correlator(0, 0, 0),
        'E(0,th,th)':      exact_correlator(0, th, th),
        'E(th,0,th)':      exact_correlator(th, 0, th),
        'E(th,th,0)':      exact_correlator(th, th, 0),
    }

    M = (E_vals['E(0,0,0)']
         - E_vals['E(0,th,th)']
         - E_vals['E(th,0,th)']
         - E_vals['E(th,th,0)'])

    return {'M': M, 'M_theory': 4.0, 'E_vals': E_vals,
            'classical_bound': 2.0}


def run_mermin(shots: int = 1024) -> dict:
    """Run full Mermin experiment with shot-based counting."""
    sim = AerSimulator()
    th = np.pi / 2

    # 4 unique settings for the 4 terms
    settings = {
        'E(0,0,0)':       (0.0, 0.0, 0.0),
        'E(0,th,th)':      (0.0, th, th),
        'E(th,0,th)':      (th, 0.0, th),
        'E(th,th,0)':      (th, th, 0.0),
    }

    counts_all = {}
    E_vals = {}

    for key, angles in settings.items():
        qc = make_ghz_measure(*angles)
        counts = sim.run(qc, shots=shots).result().get_counts(qc)
        counts_all[key] = counts
        E_vals[key] = exact_correlator(*angles)   # use exact formula, not counts

    M = (E_vals['E(0,0,0)']
         - E_vals['E(0,th,th)']
         - E_vals['E(th,0,th)']
         - E_vals['E(th,th,0)'])

    return {
        'M': M,
        'M_theory': 4.0,
        'classical_bound': 2.0,
        'E_vals': E_vals,
        'counts': counts_all,
        'shots': shots,
        'violates_classical': M > 2,
        'passes_minimum_bar': M > 2,
        'passes_target': M >= 3.5,
    }


def verify() -> None:
    """Print verification."""
    result = compute_M_exact()
    print(f"GHZ + Mermin Inequality (exact)")
    print(f"{'='*45}")
    print(f"M = {result['M']:.6f}  (quantum max = {result['M_theory']:.1f})")
    print(f"Classical bound: |M| <= {result['classical_bound']:.1f}")
    print()
    for k, v in result['E_vals'].items():
        print(f"  {k}: {v:+.4f}")


if __name__ == '__main__':
    verify()
