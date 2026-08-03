"""
qrl-common — Shared utilities for QuantumResearchLab experiments.

Provides:
- Bell state preparation (2-qubit |Phi+>)
- GHZ state preparation (3-qubit |GHZ+>)
- Correlator computation functions (2-qubit and 3-qubit)
- Realistic noise model builder (qiskit_aer.noise)
- Shot-count constants

Each experiment imports what it needs from here rather than duplicating code.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, PauliError, ReadoutError


# ── Constants ─────────────────────────────────────────────────────────────────

SHOT_SENSITIVITY_SMALL = 1_024
SHOT_SENSITIVITY_LARGE = 100_000
SHOT_TELEPORTATION = 50_000


# ── State Preparation ─────────────────────────────────────────────────────────

def make_bell_circuit() -> QuantumCircuit:
    """
    Prepare the |Phi+> = (|00> + |11>)/sqrt(2) Bell state on qubits 0,1.
    """
    qc = QuantumCircuit(2, name='Bell')
    qc.h(0)
    qc.cx(0, 1)
    return qc


def make_ghz_circuit(num_qubits: int = 3) -> QuantumCircuit:
    """
    Prepare the |GHZ+> = (|0...0> + |1...1>)/sqrt(2) state.

    Args:
        num_qubits: number of qubits (default 3)
    """
    qc = QuantumCircuit(num_qubits, name='GHZ')
    qc.h(0)
    for i in range(1, num_qubits):
        qc.cx(0, i)
    return qc


# ── Correlator Computations ───────────────────────────────────────────────────

def compute_correlator_2q(counts: dict) -> float:
    """
    Compute E = P(++) + P(--) - P(+-) - P(-+) for 2-qubit measurement.

    Qiskit bit ordering: '00' = qubit0=0, qubit1=0.
    Convention: qubit value 0 -> +1, 1 -> -1.
    E = (N_00 + N_11 - N_01 - N_10) / N_total

    Args:
        counts: Qiskit counts dict from AerSimulator

    Returns:
        Expectation value E in [-1, 1]
    """
    n00 = counts.get('00', 0)
    n11 = counts.get('11', 0)
    n01 = counts.get('01', 0)
    n10 = counts.get('10', 0)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return (n00 + n11 - n01 - n10) / total


def compute_correlator_3q(counts: dict, shots: int) -> float:
    """
    Compute E = (N_+++ + N_--- - sum of all other 6 outcomes) / N_total
    for 3-qubit GHZ measurement.

    Qiskit bit ordering: '000' = qubit2=0, qubit1=0, qubit0=0 (leftmost = highest-index qubit).
    Outcomes '000' (+++) and '111' (---) get sign +1; all others get -1.

    Args:
        counts: Qiskit counts dict
        shots: total number of shots (for normalization)

    Returns:
        Expectation value E in [-1, 1]
    """
    n000 = counts.get('000', 0)
    n111 = counts.get('111', 0)
    others = sum(counts.get(k, 0) for k in ['001', '010', '011', '100', '101', '110'])
    return (n000 + n111 - others) / shots


def exact_correlator_3q(th0: float, th1: float, th2: float) -> float:
    """
    Exact E = cos(th0 + th1 + th2) for |GHZ+> with RZ+H measurement
    (A(theta) = cos*X + sin*Y).

    Verified by statevector in qrl-003.
    """
    return np.cos(th0 + th1 + th2)


# ── Noise Model ─────────────────────────────────────────────────────────────

def make_noise_model(p1q: float = 0.002, p2q: float = 0.015,
                    ro_err: float = 0.02, num_qubits: int = 3) -> NoiseModel:
    """
    Build a realistic noise model based on typical IBM 7-qubit device calibration data.
    Pure qiskit_aer — no credentials or live backend required.

    Args:
        p1q: 1-qubit gate depolarizing probability (default 0.2% per H/RZ/SX/X)
        p2q: 2-qubit gate (CX) depolarizing probability (default 1.5%)
        ro_err: Per-qubit readout bit-flip probability (default 2%)
        num_qubits: Number of qubits to apply errors to (default 3)

    Returns:
        NoiseModel ready for AerSimulator(noise_model=...)
    """
    nm = NoiseModel()

    for qubit in range(num_qubits):
        for gate in ('h', 'rz', 'sx', 'x'):
            nm.add_quantum_error(
                PauliError(['X', 'I'], [p1q, 1 - p1q]), gate, [qubit])

    # CX gate errors (qubit pairs used in Bell and GHZ circuits)
    for q0, q1 in [(0, 1), (0, 2)]:
        nm.add_quantum_error(
            PauliError(['XY', 'II'], [p2q, 1 - p2q]), 'cx', [q0, q1])

    # Readout errors
    for qubit in range(num_qubits):
        nm.add_readout_error(
            ReadoutError([[1 - ro_err, ro_err], [ro_err, 1 - ro_err]]), [qubit])

    return nm
