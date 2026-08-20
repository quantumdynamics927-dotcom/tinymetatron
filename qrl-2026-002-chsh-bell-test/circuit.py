"""
CHSH Bell Inequality Violation — Quantum Circuit

Prepares a |Phi+> Bell state and measures in the optimal CHSH
bases to compute S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|.

Expected result (noiseless): S = 2*sqrt(2) ~= 2.828
Classical bound: S <= 2

Shared utilities (Bell prep, correlator): qrl-common.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'qrl_common'))
from qrl_common import make_bell_circuit, compute_correlator_2q

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import math


def make_chsh_circuit(angle0: float, angle1: float, shots: int = 1024) -> QuantumCircuit:
    """
    CHSH circuit: Bell state + RY-based measurement at (angle0, angle1).

    Measurement: A(theta) = cos(theta)*Z + sin(theta)*X
    Implemented as RY(-theta) then measure Z.
    Optimal CHSH angles for |Phi+>: a=0, a'=90, b=45, b'=135.
    """
    qc = QuantumCircuit(2, 2, name=f"chsh_{angle0:.2f}_{angle1:.2f}")

    # Bell state |Phi+> = (|00> + |11>)/sqrt(2)
    qc.h(0)
    qc.cx(0, 1)

    # Measurement rotations
    qc.ry(-angle0, 0)
    qc.ry(-angle1, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def compute_correlation(counts: dict) -> float:
    """Alias for compute_correlator_2q for backward compatibility."""
    return compute_correlator_2q(counts)


if __name__ == "__main__":
    a  = 0.0
    ap = math.pi / 2
    b  = math.pi / 4
    bp = 3 * math.pi / 4

    angles = [(a, b), (a, bp), (ap, b), (ap, bp)]

    print("CHSH Bell Test -- Qiskit Aer (noiseless)")
    print(f"{'='*50}")

    sim = AerSimulator()
    E_vals = {}

    for (th0, th1) in angles:
        qc = make_chsh_circuit(th0, th1, shots=1024)
        counts = sim.run(qc, shots=1024).result().get_counts(qc)
        E = compute_correlator_2q(counts)
        E_vals[(th0, th1)] = E
        print(f"  {qc.name}: E = {E:+.5f}  counts={counts}")

    S = abs(E_vals[(a, b)] - E_vals[(a, bp)] + E_vals[(ap, b)] + E_vals[(ap, bp)])

    print()
    print(f"{'='*50}")
    print(f"S = {S:.6f}")
    print(f"Classical bound: S <= 2  -> {'VIOLATED [+]' if S > 2 else 'NOT VIOLATED [-]'}")
    print(f"Quantum maximum: S = 2*sqrt(2) ~= 2.828")
    print(f"{'='*50}")
