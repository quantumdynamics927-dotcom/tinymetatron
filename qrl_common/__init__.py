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


# ── Golden-Angle Bloch Sphere Sampling ─────────────────────────────────────────
#
# Reference: Weyl equidistribution theorem — for irrational α, the sequence
# {nα mod 1} is uniformly distributed on [0, 1]. The golden ratio φ = (1+√5)/2
# is particularly well-suited because its reciprocal 1/φ = φ - 1 ≈ 0.618...
# is also irrational, giving the golden angle:
#   θ_g = 2π * (1 - 1/φ) = 2π/φ² ≈ 2.4021 rad ≈ 137.508°
#
# For N points on the Bloch sphere (Weyl/golden-angle sampling):
#   θ_n = arccos(1 - 2n/(N-1))   n = 0, ..., N-1   (elevation, uniform in cos θ)
#   φ_n = n * θ_g  (azimuthal, irrational rotation per step)
#
# The resulting set {φ_n mod 2π} is equidistributed, giving more uniform
# coverage than uniform random (which clusters at poles) or naive grid.
#
# This is standard in quantum tomography and sphere sampling literature.

GOLDEN_RATIO = (1 + 5 ** 0.5) / 2  # φ ≈ 1.6180339887
GOLDEN_ANGLE = 2 * np.pi * (1 - 1 / GOLDEN_RATIO)  # ≈ 2.4021 rad ≈ 137.508°


def golden_angle_sphere_points(n_points: int) -> list:
    """
    Generate N points uniformly distributed on the Bloch sphere
    using the golden-angle (Weyl equidistribution) sequence.

    Args:
        n_points: number of points to generate (N)

    Returns:
        List of (theta, phi) tuples in radians.
        theta ∈ [0, π] (polar angle from +Z axis)
        phi ∈ [0, 2π) (azimuthal angle)

    Reference: Weyl (1916), golden-angle spiral on sphere.
    """
    points = []
    for n in range(n_points):
        # Uniform in cos(θ) from -1 to 1: z = cos(θ) = 1 - 2n/(N-1)
        cos_theta = 1.0 - (2.0 * n) / (n_points - 1) if n_points > 1 else 0.0
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        # Azimuthal: golden-angle increment per point
        phi = (n * GOLDEN_ANGLE) % (2 * np.pi)
        points.append((theta, phi))
    return points


def sphere_point_to_statevector(theta: float, phi: float) -> np.ndarray:
    """
    Convert a Bloch sphere point to a normalized 2-qubit statevector.

    |ψ(θ,φ)⟩ = cos(θ/2) |0⟩ + e^{iφ} sin(θ/2) |1⟩

    Args:
        theta: polar angle in radians [0, π]
        phi: azimuthal angle in radians [0, 2π)

    Returns:
        Normalized complex statevector [α, β] where |α|²+|β|² = 1
    """
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    sv = np.array([alpha, beta], dtype=complex)
    return sv / np.linalg.norm(sv)


def uniform_random_sphere_points(n_points: int, seed: int = 42) -> list:
    """
    Generate N points on the Bloch sphere via naive uniform random sampling.
    Uses the standard spherical coordinate method: cos(θ) ~ U(-1,1), φ ~ U(0, 2π).

    This is a comparison baseline for the golden-angle method.
    The random method clusters near the poles; golden-angle avoids this.

    Args:
        n_points: number of points
        seed: random seed for reproducibility

    Returns:
        List of (theta, phi) tuples in radians.
    """
    rng = np.random.default_rng(seed)
    cos_thetas = rng.uniform(-1, 1, size=n_points)
    thetas = np.arccos(cos_thetas)
    phis = rng.uniform(0, 2 * np.pi, size=n_points)
    return list(zip(thetas, phis))


def sphere_discrepancy(points: list) -> float:
    """
    Compute a simple discrepancy statistic for a set of sphere points.

    Uses the centered Euclidean nearest-neighbor distance variance:
    - For each point, compute Euclidean distance to its nearest neighbor
    - Compute variance of these distances
    Lower variance = more uniform spacing.

    For N evenly spaced points, variance → 0.
    For clustered points, variance is high.

    Args:
        points: list of (theta, phi) tuples (radians)

    Returns:
        Float — coefficient of variation of nearest-neighbor distances
        (std / mean). Lower is more uniform.
    """
    if len(points) < 2:
        return 0.0

    # Convert to Cartesian unit vectors
    vecs = []
    for theta, phi in points:
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        vecs.append(np.array([x, y, z]))
    vecs = np.array(vecs)

    # Nearest-neighbor distances
    nn_dists = []
    for i, v in enumerate(vecs):
        dists = np.linalg.norm(vecs - v, axis=1)
        dists[i] = np.inf  # exclude self
        nn_dists.append(np.min(dists))

    nn_dists = np.array(nn_dists)
    mean = np.mean(nn_dists)
    std = np.std(nn_dists)
    if mean == 0:
        return 0.0
    return std / mean  # coefficient of variation

