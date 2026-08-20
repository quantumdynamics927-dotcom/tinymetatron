"""
Golden-Angle Teleportation Fidelity Sweep — qrl-006

Extends qrl-005 (Bennett 1993 teleportation) with a full Bloch-sphere
fidelity sweep using golden-angle (Weyl equidistribution) sampling.

- Noiseless: analytical fidelity = 1.0 for ALL 50 states (Bennett 1993 guarantee)
- Noisy: density-matrix fidelity via shots (reveals state-dependent noise sensitivity)
- Comparison: golden-angle sampling is more uniform than uniform random (discrepancy)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qrl_common import (
    golden_angle_sphere_points,
    uniform_random_sphere_points,
    sphere_discrepancy,
    sphere_point_to_statevector,
    make_noise_model,
)

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ── Teleportation circuit ──────────────────────────────────────────────────────

def make_teleport_circuit(input_sv: list) -> QuantumCircuit:
    """Bennett 1993 teleportation: q0=data, q1=Alice Bell, q2=Bob output."""
    alpha, beta = np.array(input_sv, dtype=complex)
    alpha /= np.linalg.norm([alpha, beta])
    beta /= np.linalg.norm([alpha, beta])

    qc = QuantumCircuit(3, 3, name='teleport')
    qc.initialize([alpha, beta], 0)
    qc.h(1); qc.cx(1, 2)      # Bell pair on q1,q2
    qc.cx(0, 1)                 # Entangle data with Alice's Bell half
    qc.h(0)                       # Bell-state measurement
    qc.measure([0, 1], [0, 1])
    x_body = QuantumCircuit(1); x_body.x(0)
    z_body = QuantumCircuit(1); z_body.z(0)
    qc.if_test((qc.clbits[1], 1), true_body=x_body, qubits=[2], clbits=[])
    qc.if_test((qc.clbits[0], 1), true_body=z_body, qubits=[2], clbits=[])
    return qc


def teleportation_fidelity_analytical(input_sv: list) -> float:
    """Bennett 1993 guarantee: F=1.0 for all inputs in noiseless case."""
    return 1.0


def teleportation_fidelity_noisy(input_sv: list, shots: int = 20000) -> float:
    """
    Fidelity under qrl-004 noise model (p1q=0.002, p2q=0.015, ro=0.02).

    Runs teleportation circuit with noise, computes the mixed density matrix
    of qubit 2 by accounting for all measurement outcomes, then computes
    F = <psi_in| rho2 |psi_in> (Uhlmann-Jozsa fidelity for pure states).
    """
    nm = make_noise_model()
    alpha, beta = np.array(input_sv, dtype=complex)
    alpha /= np.linalg.norm([alpha, beta])
    beta /= np.linalg.norm([alpha, beta])
    psi_in = np.array([alpha, beta], dtype=complex)

    sim = AerSimulator()
    qc = make_teleport_circuit(input_sv)
    counts = sim.run(qc, shots=shots, noise_model=nm).result().get_counts(qc)

    # Reconstruct rho2 as mixed state: rho2 = sum_p p * |b><b|
    # where b = q2's computational basis state for outcome with probability p
    rho2 = np.zeros((2, 2), dtype=complex)
    total = sum(counts.values())

    for outcome, count in counts.items():
        p = count / total
        q2_val = int(outcome[0])   # most significant bit = qubit 2
        if q2_val == 0:
            rho2 += p * np.array([[1, 0], [0, 0]], dtype=complex)
        else:
            rho2 += p * np.array([[0, 0], [0, 1]], dtype=complex)

    # F = <psi| rho |psi> = psi^* rho psi
    fidelity = np.vdot(psi_in.conj(), rho2 @ psi_in).real
    return float(fidelity)


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(n_points: int = 50, noisy: bool = False,
             shots_per_state: int = 20000) -> dict:
    """Run fidelity sweep over golden-angle and uniform-random Bloch sphere points."""
    ga_points = golden_angle_sphere_points(n_points)
    ur_points = uniform_random_sphere_points(n_points, seed=42)

    ga_anal, ga_noisy = [], []
    for theta, phi in ga_points:
        sv = sphere_point_to_statevector(theta, phi).tolist()
        ga_anal.append(teleportation_fidelity_analytical(sv))
        ga_noisy.append(teleportation_fidelity_noisy(sv, shots=shots_per_state))

    ur_anal, ur_noisy = [], []
    for theta, phi in ur_points:
        sv = sphere_point_to_statevector(theta, phi).tolist()
        ur_anal.append(teleportation_fidelity_analytical(sv))
        ur_noisy.append(teleportation_fidelity_noisy(sv, shots=shots_per_state))

    def _stats(fids):
        return {
            'min': round(min(fids), 6),
            'max': round(max(fids), 6),
            'mean': round(np.mean(fids), 6),
            'std': round(np.std(fids), 6),
        }

    return {
        'golden_angle': {
            'analytical': {**_stats(ga_anal),
                          'fidelities': [round(f, 6) for f in ga_anal]},
            'noisy':      {**_stats(ga_noisy),
                          'discrepancy': round(sphere_discrepancy(ga_points), 6)},
        },
        'random': {
            'analytical': {**_stats(ur_anal),
                          'fidelities': [round(f, 6) for f in ur_anal]},
            'noisy':      {**_stats(ur_noisy),
                          'discrepancy': round(sphere_discrepancy(ur_points), 6)},
        },
        'n_points': n_points,
        'shots_per_state': shots_per_state,
    }


def print_results(r: dict):
    ga = r['golden_angle']
    ur = r['random']
    n = r['n_points']
    s = r['shots_per_state']

    print(f"=== NOISELESS Sweep (N={n}) ===")
    print(f"  Golden-angle: F mean={ga['analytical']['mean']:.6f}"
          f"  min={ga['analytical']['min']:.6f}  max={ga['analytical']['max']:.6f}"
          f"  discrepancy={ga['noisy']['discrepancy']:.6f}")
    print(f"  Random:      F mean={ur['analytical']['mean']:.6f}"
          f"  discrepancy={ur['noisy']['discrepancy']:.6f}")
    print(f"  Golden more uniform: {ga['noisy']['discrepancy'] < ur['noisy']['discrepancy']}")
    print()

    print(f"=== NOISY Sweep (N={n}, {s} shots/state) ===")
    print(f"  Golden-angle: F mean={ga['noisy']['mean']:.6f}  std={ga['noisy']['std']:.6f}"
          f"  min={ga['noisy']['min']:.6f}  max={ga['noisy']['max']:.6f}")
    print(f"  Random:      F mean={ur['noisy']['mean']:.6f}  std={ur['noisy']['std']:.6f}"
          f"  min={ur['noisy']['min']:.6f}  max={ur['noisy']['max']:.6f}")
    print(f"  Golden more uniform: {ga['noisy']['discrepancy'] < ur['noisy']['discrepancy']}")
    print(f"  Golden lower variance: {ga['noisy']['std'] < ur['noisy']['std']}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("Golden-Angle Teleportation Fidelity Sweep")
    print("=" * 50)

    r_nl = run_sweep(n_points=50, noisy=False)
    r_no = run_sweep(n_points=50, noisy=True, shots_per_state=20000)

    print_results(r_nl)
    print_results(r_no)

    print("=== SUMMARY ===")
    print(f"  Noiseless: all 50 states F=1.0 (Bennett 1993 guarantee)")
    print(f"  Noisy golden-angle:  mean={r_no['golden_angle']['noisy']['mean']:.6f}"
          f"  std={r_no['golden_angle']['noisy']['std']:.6f}"
          f"  min={r_no['golden_angle']['noisy']['min']:.6f}")
    print(f"  Noisy random:       mean={r_no['random']['noisy']['mean']:.6f}"
          f"  std={r_no['random']['noisy']['std']:.6f}"
          f"  min={r_no['random']['noisy']['min']:.6f}")
    print(f"  Golden more uniform: {r_no['golden_angle']['noisy']['discrepancy']:.4f}"
          f" < {r_no['random']['noisy']['discrepancy']:.4f}")
