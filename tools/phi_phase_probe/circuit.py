"""
tools/phi_phase_probe/circuit.py
===============================
φ (golden ratio) phase-encoding circuits and controls.

φ = (1 + √5) / 2 ≈ 1.6180339887...
Continued fraction convergents of φ:
  [1;1,1,1,...] → 3/2, 5/3, 8/5, 13/8, 21/13, 34/21, 55/34, 89/55, 144/89, ...

We use the first few convergents to define phase angles:
  θ_n = 2π × (p_n / q_n)  where p_n/q_n → φ

The probe encodes these angles on qubits via RZ(θ) gates and measures
whether the resulting interference patterns differ systematically from a
uniform-phase null (RZ(π/4) as a control that has no special relationship to φ).
"""

from __future__ import annotations

import numpy as np

# Golden ratio
PHI: float = (1 + np.sqrt(5)) / 2

# Continued-fraction convergents of φ (stopping before the large denominator)
# Computed from the regular continued fraction [1; 1,1,1,...]
CONVERGENTS: list[tuple[int, int]] = [
    (3, 2),    # 3/2  = 1.5
    (5, 3),    # 5/3  ≈ 1.6667
    (8, 5),    # 8/5  = 1.6
    (13, 8),   # 13/8 = 1.625
    (21, 13),  # 21/13 ≈ 1.6154
    (34, 21),  # 34/21 ≈ 1.6190
    (55, 34),  # 55/34 ≈ 1.6176
    (89, 55),  # 89/55 ≈ 1.6182
    (144, 89), # 144/89 ≈ 1.6179
]


def phi_angle(n: int = 0) -> float:
    """Return 2π × (p_n / q_n) where p_n/q_n is the n-th convergent of φ."""
    p, q = CONVERGENTS[min(n, len(CONVERGENTS) - 1)]
    return 2 * np.pi * p / q


def uniform_angle() -> float:
    """Control angle with no special relationship to φ: π/4 = 45°."""
    return np.pi / 4


def build_phi_circuit(num_qubits: int = 4, convergent_idx: int = 0,
                      include_hadamard: bool = True) -> dict:
    """
    Build a circuit that encodes a φ convergent's phase on each qubit.

    Structure:
      1. Hadamard on all qubits → equal superposition
      2. RZ(θ) on each qubit, where θ = 2π × (p/q) from convergent
      3. CNOT ladder to create entanglement (parity-sensitivity)
      4. Measure all qubits

    Args:
        num_qubits: number of qubits (3–8 recommended for real hardware)
        convergent_idx: which convergent to use (0 = 3/2, 1 = 5/3, ...)
        include_hadamard: whether to start in |+⟩ superposition

    Returns:
        dict with circuit metadata and qiskit.QuantumCircuit
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: pip install qiskit. {exc}")

    θ = phi_angle(convergent_idx)
    p, q = CONVERGENTS[convergent_idx]

    qc = QuantumCircuit(num_qubits, num_qubits, name=f"phi_c{p}_{q}")

    if include_hadamard:
        for i in range(num_qubits):
            qc.h(i)

    # Phase encoding
    for i in range(num_qubits):
        qc.rz(θ, i)

    # Entangling layer: CNOT ladder for parity sensitivity
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    qc.measure(range(num_qubits), range(num_qubits))

    return {
        "circuit": qc,
        "type": "phi",
        "num_qubits": num_qubits,
        "convergent": (p, q),
        "angle": θ,
        "angle_str": f"2π×{p}/{q}≈{θ:.6f}",
        "description": f"φ convergent {p}/{q} ≈ {p/q:.6f}, angle={θ:.6f} rad",
    }


def build_control_circuit(num_qubits: int = 4,
                          include_hadamard: bool = True) -> dict:
    """
    Build a control circuit: same structure as phi circuit but with uniform
    phase angle π/4 (no special relationship to φ).

    Null hypothesis: if φ circuits differ from this, something interesting is
    happening beyond generic phase encoding.

    Args:
        num_qubits: number of qubits
        include_hadamard: whether to start in |+⟩ superposition

    Returns:
        dict with circuit metadata and qiskit.QuantumCircuit
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: pip install qiskit. {exc}")

    θ = uniform_angle()
    qc = QuantumCircuit(num_qubits, num_qubits, name="control_uniform")

    if include_hadamard:
        for i in range(num_qubits):
            qc.h(i)

    # Uniform phase encoding
    for i in range(num_qubits):
        qc.rz(θ, i)

    # Same entangling layer
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    qc.measure(range(num_qubits), range(num_qubits))

    return {
        "circuit": qc,
        "type": "control",
        "num_qubits": num_qubits,
        "angle": θ,
        "angle_str": f"π/4={θ:.6f}",
        "description": f"Control: uniform phase angle π/4 ≈ {θ:.6f} rad",
    }


def build_all_probe_circuits(num_qubits: int = 4) -> list[dict]:
    """Build the full probe set: all φ convergents + control."""
    circuits = []

    for idx in range(len(CONVERGENTS)):
        circuits.append(build_phi_circuit(num_qubits=num_qubits, convergent_idx=idx))

    circuits.append(build_control_circuit(num_qubits=num_qubits))

    return circuits
