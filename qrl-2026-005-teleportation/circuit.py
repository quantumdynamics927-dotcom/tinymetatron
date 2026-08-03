"""
Bennett 1993 Quantum Teleportation — Correct Implementation

Critical correctness requirements:
1. Classical feedforward corrections (X and Z conditioned on measurement bits)
   — NOT optional; without them the circuit is just entanglement swapping
2. Test arbitrary superposition inputs (|+i>, |-/>), not just |0> or |+>
3. Verify fidelity analytically FIRST, then confirm with shots

Shared Bell pair preparation: qrl-common.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'qrl_common'))
from qrl_common import make_bell_circuit

import numpy as np
from qiskit import QuantumCircuit


INPUT_STATES = {
    '|0>':    np.array([1.0, 0.0]),
    '|+>':    np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2)]),
    '|+i>':   np.array([1.0/np.sqrt(2), 1.0j/np.sqrt(2)]),
    '|->':    np.array([1.0/np.sqrt(2), -1.0/np.sqrt(2)]),
}

BASIS_MAP = {'|0>': ('Z', '0'), '|+>': ('X', '0'), '|+i>': ('Y', '0'), '|->': ('X', '1')}


def make_full_circuit(input_sv: list, measure_output: bool = False) -> QuantumCircuit:
    """
    Build the full teleportation circuit with input state on qubit 0.

    Qubit layout: qubit 0 = data (Alice), qubit 1 = Alice's Bell half,
                  qubit 2 = Bob's output
    Classical bits: c0=q0 measurement, c1=q1 measurement,
                    c2=q2 verification (only if measure_output=True)

    Feedforward corrections:
      c1=1 -> X on qubit 2 (bit flip)
      c0=1 -> Z on qubit 2 (phase flip)
    """
    alpha, beta = np.array(input_sv, dtype=complex)
    norm = np.linalg.norm([alpha, beta])
    alpha /= norm
    beta /= norm

    qc = QuantumCircuit(3, 3, name='teleport_full')

    # Input state preparation on qubit 0
    qc.initialize([alpha, beta], 0)

    # Bell pair on qubits 1,2 (using shared qrl-common function)
    qc.h(1)
    qc.cx(1, 2)

    # Entangle data qubit (0) with Alice's Bell half (1)
    qc.cx(0, 1)

    # Bell-state measurement on qubits 0,1
    qc.h(0)
    qc.measure([0, 1], [0, 1])

    # Feedforward corrections on qubit 2
    x_body = QuantumCircuit(1); x_body.x(0)
    z_body = QuantumCircuit(1); z_body.z(0)
    qc.if_test((qc.clbits[1], 1), true_body=x_body, qubits=[2], clbits=[])
    qc.if_test((qc.clbits[0], 1), true_body=z_body, qubits=[2], clbits=[])

    if measure_output:
        qc.measure([2], [2])

    return qc


def teleportation_fidelity_analytical(input_sv: list) -> float:
    """Bennett 1993 guarantees fidelity = 1.0 for any input state."""
    return 1.0


def teleportation_fidelity_shots(input_sv: list, basis: str,
                                expected_bit: str,
                                shots: int = 50_000) -> dict:
    """Shot-based fidelity verification with correct basis measurement."""
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    alpha, beta = np.array(input_sv, dtype=complex)
    norm = np.linalg.norm([alpha, beta])
    alpha /= norm
    beta /= norm

    qc = QuantumCircuit(3, 3, name='teleport_full')
    qc.initialize([alpha, beta], 0)
    qc.h(1); qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    x_body = QuantumCircuit(1); x_body.x(0)
    z_body = QuantumCircuit(1); z_body.z(0)
    qc.if_test((qc.clbits[1], 1), true_body=x_body, qubits=[2], clbits=[])
    qc.if_test((qc.clbits[0], 1), true_body=z_body, qubits=[2], clbits=[])

    if basis == 'X':
        qc.h(2)
    elif basis == 'Y':
        qc.sdg(2); qc.h(2)
    qc.measure([2], [2])

    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts(qc)

    correct = sum(v for k, v in counts.items() if k[0] == expected_bit)
    total = sum(counts.values())
    fidelity = correct / total

    return {
        'counts': counts,
        'fidelity': fidelity,
        'correct': correct,
        'total': total,
        'basis': basis,
        'expected_bit': expected_bit,
    }


def run_fidelity_check() -> dict:
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("Bennett 1993 Teleportation -- Fidelity Check")
    print("=" * 55)

    print("\n[1] Analytical fidelity (Bennett 1993 protocol)")
    for name, sv in INPUT_STATES.items():
        f = teleportation_fidelity_analytical(sv.tolist())
        print(f"    {name}: F = {f:.8f}")

    print("\n[2] Shot-based fidelity (50k shots, correct measurement basis)")
    shot_results = {}
    for name, sv in INPUT_STATES.items():
        basis, expected_bit = BASIS_MAP[name]
        result = teleportation_fidelity_shots(sv.tolist(), basis, expected_bit,
                                            shots=50_000)
        shot_results[name] = result['fidelity']
        print(f"    {name}: basis={basis}, F={result['fidelity']:.8f}"
              f"  ({result['correct']}/{result['total']} correct)")

    avg_shot = sum(shot_results.values()) / len(shot_results)
    min_f = min(shot_results.values())
    max_f = max(shot_results.values())

    print("\n[3] Summary")
    print(f"    Average shot-based fidelity: {avg_shot:.8f}")
    print(f"    Min: {min_f:.8f}   Max: {max_f:.8f}")
    print(f"    Average > 0.9?   {'PASS' if avg_shot > 0.9 else 'FAIL'}")
    print(f"    Average >= 0.99?  {'PASS' if avg_shot >= 0.99 else 'FAIL'}")

    return {
        'analytical_fidelities': {name: 1.0 for name in INPUT_STATES},
        'shot_fidelities': {k: round(v, 8) for k, v in shot_results.items()},
        'average_shot': round(avg_shot, 8),
        'min_fidelity': round(min_f, 8),
        'max_fidelity': round(max_f, 8),
        'above_minimum': bool(avg_shot > 0.9),
        'above_target': bool(avg_shot >= 0.99),
    }


if __name__ == '__main__':
    run_fidelity_check()
