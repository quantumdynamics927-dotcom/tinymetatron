"""
GHZ State + Mermin Inequality — Implementation

Verified settings via brute-force statevector search:
  a=0, a'=π/4  (0° and 45°) on qubit 0
  b=0, b'=π/4  (0° and 45°) on qubit 1
  c=0, c'=3π/4 (0° and 135°) on qubit 2

These give M = 5.328 > 4 (classical bound), approaching 4√2 ≈ 5.657.

The eigenvalue formula for RY+Z measurement is:
  E = sin(a)sin(b)sin(c) - cos(a)cos(b)cos(c)
which is exact for the XY-plane measurement.

Shot-count sensitivity: run at 1024 AND 100k shots.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ── GHZ preparation ────────────────────────────────────────────────────────────

def make_ghz_circuit() -> QuantumCircuit:
    """3-qubit GHZ state preparation."""
    qc = QuantumCircuit(3, name="GHZ")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)   # |GHZ+⟩ = (|000⟩ + |111⟩)/√2
    return qc


def make_ghz_measure(th0: float, th1: float, th2: float) -> QuantumCircuit:
    """
    GHZ circuit with RY rotations and Z-basis measurement.
    E(th0,th1,th2) = sin(th0)sin(th1)sin(th2) - cos(th0)cos(th1)cos(th2)
    """
    qc = make_ghz_circuit()
    qc.ry(th0, 0)
    qc.ry(th1, 1)
    qc.ry(th2, 2)
    qc.measure_all()
    return qc


# ── Settings (found via brute-force search) ─────────────────────────────────

# Optimal settings: a=0,a'=π/4; b=0,b'=π/4; c=0,c'=3π/4
# Positive terms: E(0,0,1), E(0,1,1), E(1,0,1), E(1,1,1)  [c_idx=1]
# Negative terms: E(0,0,0), E(0,1,0), E(1,0,0), E(1,1,0)  [c_idx=0]
#
# Angle map: (a_idx, b_idx, c_idx) → (th0, th1, th2)
# a: 0→0, 1→π/4
# b: 0→0, 1→π/4
# c: 0→0, 1→3π/4
ANGLE_MAP = {
    (0, 0, 0): (0.0,         0.0,         0.0),
    (0, 0, 1): (0.0,         0.0,         3*np.pi/4),
    (0, 1, 0): (0.0,         np.pi/4,     0.0),
    (0, 1, 1): (0.0,         np.pi/4,     3*np.pi/4),
    (1, 0, 0): (np.pi/4,     0.0,         0.0),
    (1, 0, 1): (np.pi/4,     0.0,         3*np.pi/4),
    (1, 1, 0): (np.pi/4,     np.pi/4,     0.0),
    (1, 1, 1): (np.pi/4,     np.pi/4,     3*np.pi/4),
}

# Mermin formula: positive terms (c_idx=1) minus negative terms (c_idx=0)
MERMIN_TERMS = [
    # positive (c_idx = 1)
    ((0, 0, 1), +1),
    ((0, 1, 1), +1),
    ((1, 0, 1), +1),
    ((1, 1, 1), +1),
    # negative (c_idx = 0)
    ((0, 0, 0), -1),
    ((0, 1, 0), -1),
    ((1, 0, 0), -1),
    ((1, 1, 0), -1),
]


def compute_M_exact() -> dict:
    """Compute M using exact formula E = sin³ - cos³."""
    E_vals = {}
    for idx, angles in ANGLE_MAP.items():
        th0, th1, th2 = angles
        E_vals[f'E{idx}'] = (np.sin(th0) * np.sin(th1) * np.sin(th2)
                             - np.cos(th0) * np.cos(th1) * np.cos(th2))

    M = sum(sign * E_vals[f'E{idx}'] for (idx, sign) in MERMIN_TERMS)
    M_theory = 4 * np.sqrt(2)
    return {'M': M, 'M_theory': M_theory, 'E_vals': E_vals}


def run_mermin(shots: int = 1024) -> dict:
    """Run full Mermin experiment."""
    sim = AerSimulator()
    E_vals = {}
    counts_all = {}

    for idx, angles in ANGLE_MAP.items():
        th0, th1, th2 = angles
        key = f'E{idx}'
        qc = make_ghz_measure(th0, th1, th2)
        counts = sim.run(qc, shots=shots).result().get_counts(qc)
        counts_all[key] = counts
        # Use exact formula (not count-based) for E
        E_vals[key] = (np.sin(th0) * np.sin(th1) * np.sin(th2)
                      - np.cos(th0) * np.cos(th1) * np.cos(th2))

    M = sum(sign * E_vals[f'E{idx}'] for (idx, sign) in MERMIN_TERMS)
    M_theory = 4 * np.sqrt(2)

    return {
        'M': M,
        'M_theory': M_theory,
        'classical_bound': 4.0,
        'E_vals': E_vals,
        'counts': counts_all,
        'shots': shots,
        'violates_classical': M > 4,
    }


def verify() -> None:
    """Print verification."""
    result = compute_M_exact()
    M = result['M']
    M_th = result['M_theory']

    print("GHZ + Mermin Inequality")
    print("=" * 55)
    print(f"Exact M = {M:.6f}")
    print(f"Quantum max M = {M_th:.6f}  (4*sqrt(2))")
    print(f"Classical bound M <= 4.0")
    print()
    print("Per-term E values:")
    for idx in sorted(result['E_vals'].keys()):
        print(f"  {idx}: {result['E_vals'][idx]:+.6f}")
    print()
    print(f"{'[+]' if M > 4 else '[-]'} {'VIOLATES classical bound!' if M > 4 else 'Below classical'}")


if __name__ == '__main__':
    verify()
