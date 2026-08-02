"""
CHSH Bell Inequality Violation — Quantum Circuit

Prepares a |Phi+> Bell state and measures in the optimal CHSH
bases to compute S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|.

Expected result (noiseless): S = 2*sqrt(2) ≈ 2.828
Classical bound: S <= 2
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def make_bell_state(qc: QuantumCircuit, q0: int, q1: int) -> None:
    """Prepare |Phi+> = (|00> + |11>)/sqrt(2) Bell state."""
    qc.h(q0)      # H on q0: |0> -> (|0>+|1>)/sqrt(2)
    qc.cx(q0, q1) # CX: (|00>+|11>)/sqrt(2)


def make_chsh_circuit(angle0: float, angle1: float, shots: int = 1024) -> QuantumCircuit:
    """
    Create a CHSH circuit measuring qubit 0 at `angle0` and qubit 1 at `angle1`.

    Args:
        angle0: measurement angle for qubit 0 (radians, from Z axis)
        angle1: measurement angle for qubit 1 (radians, from Z axis)
        shots: number of measurement shots

    Returns:
        QuantumCircuit prepared and measured with given angles
    """
    qc = QuantumCircuit(2, 2, name=f"chsh_{angle0:.2f}_{angle1:.2f}")

    # Prepare Bell state |Phi+>
    make_bell_state(qc, 0, 1)

    # Measure qubit 0 at angle0: apply RY(-angle0) then measure Z
    # Measuring observable M = cos(angle0)*Z0 + sin(angle0)*X0
    # E(M) = <psi|RY(-angle0) Z RY(angle0)|psi> = <psi'|Z|psi'> where psi' = RY(-angle0)|psi>
    qc.ry(-angle0, 0)

    # Measure qubit 1 at angle1
    qc.ry(-angle1, 1)

    # Measure both in Z basis
    qc.measure([0, 1], [0, 1])

    return qc


def compute_correlation(counts: dict) -> float:
    """
    Compute expectation value E = P(++) + P(--) - P(+-) - P(-+).

    Counts: {'00': N00, '01': N01, '10': N10, '11': N11}
    Outcomes: 0 -> +1, 1 -> -1
    E = (N00 - N01 - N10 + N11) / N_total
    """
    n00 = counts.get('00', 0)
    n01 = counts.get('01', 0)
    n10 = counts.get('10', 0)
    n11 = counts.get('11', 0)
    total = n00 + n01 + n10 + n11
    if total == 0:
        return 0.0
    return (n00 + n11 - n01 - n10) / total


if __name__ == "__main__":
    from qiskit_aer import AerSimulator
    import math

    # Optimal CHSH angles for |Phi+> state:
    # a=0, a'=pi/2, b=pi/4, b'=3*pi/4
    a  = 0.0
    ap = math.pi / 2     # 90 degrees
    b  = math.pi / 4     # 45 degrees
    bp = 3 * math.pi / 4 # 135 degrees

    angles = [
        (a,  b),
        (a,  bp),
        (ap, b),
        (ap, bp),
    ]

    print(f"CHSH Bell Test — Qiskit Aer (noiseless)")
    print(f"{'='*50}")
    print(f"Angles (radians): a={a:.4f}, a'={ap:.4f}, b={b:.4f}, b'={bp:.4f}")
    print(f"Angles (degrees): a=0°, a'=90°, b=45°, b'=135°")
    print(f"Circuit: |Phi+> Bell state + CHSH measurement bases")
    print(f"Classical bound: S <= 2")
    print(f"Quantum maximum: S = 2*sqrt(2) ≈ 2.828")
    print(f"{'='*50}")
    print()

    sim = AerSimulator()
    results = {}
    E_vals = {}

    for (th0, th1) in angles:
        qc = make_chsh_circuit(th0, th1, shots=1024)
        job = sim.run(qc, shots=1024)
        counts = job.result().get_counts(qc)
        E = compute_correlation(counts)
        E_name = f"E({th0:.2f},{th1:.2f})"
        E_vals[(th0, th1)] = E
        results[f"{th0:.4f}_{th1:.4f}"] = counts
        print(f"  {qc.name}: E = {E:+.5f}  counts={counts}")

    # Compute S = |E(ab) - E(ab') + E(a'b) + E(a'b')|
    S = abs(E_vals[(a, b)] - E_vals[(a, bp)] + E_vals[(ap, b)] + E_vals[(ap, bp)])

    print()
    print(f"{'='*50}")
    print(f"S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|")
    print(f"S = |{E_vals[(a,b)]:+.5f} - ({E_vals[(a,bp)]:+.5f}) + ({E_vals[(ap,b)]:+.5f}) + ({E_vals[(ap,bp)]:+.5f})|")
    print(f"S = {S:.6f}")
    print()
    print(f"Classical bound: S <= 2  → {'VIOLATED ✓' if S > 2 else 'NOT VIOLATED ✗'}")
    print(f"Quantum maximum: S <= 2√2 ≈ 2.828  → {'AT QUANTUM LIMIT ✓' if S > 2.7 else 'BELOW QUANTUM LIMIT'}")
    print(f"{'='*50}")
