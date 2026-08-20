"""
Compute the 5th configuration properly: Static circuit + post-hoc corrections.
The QSG methodology uses Bhattacharyya coefficient, which gives ~0.9995.
We verify this and also check whether it's measuring the right thing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve()))
sys.path.insert(0, str(Path(__file__).resolve() / 'qrl_common'))

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix
from qrl_common import make_noise_model

THETA_Y = np.pi / 3
THETA_Z = np.pi / 7
EXPECTED_P0 = np.cos(THETA_Y / 2) ** 2  # 0.75

alpha = np.cos(THETA_Y / 2)
beta = np.exp(1j * THETA_Z) * np.sin(THETA_Y / 2)
PSI = np.array([alpha, beta], dtype=complex)
psi = PSI.reshape(2, 1)
RHO_TARGET = psi @ psi.conj().T


def bhattacharyya(p0_measured):
    """QSG fidelity metric: Bhattacharyya coefficient."""
    p1 = 1 - p0_measured
    return np.sqrt(EXPECTED_P0 * p0_measured) + np.sqrt((1 - EXPECTED_P0) * p1)


def fidelity_quantum(rho):
    """True quantum fidelity: <psi|rho|psi>."""
    return float(np.real(np.trace(RHO_TARGET @ rho)))


def trace_out(dm_data, keep_qubit):
    """Partial trace: keep keep_qubit, trace out all others (qubits 0,1)."""
    dm = np.asarray(dm_data, dtype=complex)
    dim = 8
    rho = np.zeros((2, 2), dtype=complex)
    for i in range(dim):
        b0_i = i & 1; b1_i = (i >> 1) & 1; b2_i = (i >> 2) & 1
        for j in range(dim):
            b0_j = j & 1; b1_j = (j >> 1) & 1; b2_j = (j >> 2) & 1
            if keep_qubit == 2:
                if b0_i == b0_j and b1_i == b1_j:
                    rho[b2_i, b2_j] += dm[i, j]
            elif keep_qubit == 1:
                if b0_i == b0_j and b2_i == b2_j:
                    rho[b1_i, b1_j] += dm[i, j]
            elif keep_qubit == 0:
                if b1_i == b1_j and b2_i == b2_j:
                    rho[b0_i, b0_j] += dm[i, j]
    return rho


def make_static_full():
    """Full static circuit: Bell pair + entanglement + Bell measurement + Bob measurement."""
    qc = QuantumCircuit(3, 3, name='static_full')
    qc.ry(THETA_Y, 0)
    qc.rz(THETA_Z, 0)
    qc.h(1); qc.cx(1, 2)
    qc.cx(0, 1); qc.h(0)
    qc.measure([0, 1], [0, 1])  # Alice: c[0]=q0, c[1]=q1
    qc.measure([2], [2])          # Bob: c[2]=q2
    return qc


def make_static_dm():
    """For density matrix: no Bob measurement."""
    qc = QuantumCircuit(3, 2, name='static_dm')
    qc.ry(THETA_Y, 0)
    qc.rz(THETA_Z, 0)
    qc.h(1); qc.cx(1, 2)
    qc.cx(0, 1); qc.h(0)
    qc.measure([0, 1], [0, 1])
    qc.save_density_matrix()
    return qc


sim = AerSimulator()
nm = make_noise_model(num_qubits=3)
shots = 100_000

print("=" * 60)
print("CONFIG: Static circuit + post-hoc corrections")
print("=" * 60)
print()

# ── NOISELESS ────────────────────────────────────────────────────────────────
print("--- NOISELESS ---")

res_nl = sim.run(make_static_full(), shots=shots).result()
counts_nl = res_nl.get_counts()
total_nl = sum(counts_nl.values())

# Raw P(0)
raw0_nl = sum(cnt for bits, cnt in counts_nl.items() if bits[0] == '0')
p0_raw_nl = raw0_nl / total_nl
bhatt_raw_nl = bhattacharyya(p0_raw_nl)
print(f"Raw: P(0)={p0_raw_nl:.4f}, Bhatt={bhatt_raw_nl:.4f}")

# Corrected P(0) — apply X if c1=1
corr0_nl = 0
for bits, cnt in counts_nl.items():
    bob = int(bits[0])
    c1 = int(bits[1])
    corrected = bob ^ c1  # X correction
    if corrected == 0:
        corr0_nl += cnt
p0_corr_nl = corr0_nl / total_nl
bhatt_corr_nl = bhattacharyya(p0_corr_nl)
print(f"Corrected: P(0)={p0_corr_nl:.4f}, Bhatt={bhatt_corr_nl:.4f}")

# Quantum fidelity from density matrix (no output measurement)
dm_nl = sim.run(make_static_dm(), shots=shots).result().data(0)['density_matrix']
rho_nl = trace_out(dm_nl, keep_qubit=2)
F_nl = fidelity_quantum(rho_nl)
p0_rho_nl = float(np.real(rho_nl[0, 0]))
print(f"Quantum fidelity from DM: F={F_nl:.4f}, P(0)={p0_rho_nl:.4f}")

# ── NOISY ──────────────────────────────────────────────────────────────────
print()
print("--- NOISY qrl-004 ---")

res_no = sim.run(make_static_full(), noise_model=nm, shots=shots).result()
counts_no = res_no.get_counts()
total_no = sum(counts_no.values())

raw0_no = sum(cnt for bits, cnt in counts_no.items() if bits[0] == '0')
p0_raw_no = raw0_no / total_no
bhatt_raw_no = bhattacharyya(p0_raw_no)
print(f"Raw: P(0)={p0_raw_no:.4f}, Bhatt={bhatt_raw_no:.4f}")

corr0_no = 0
for bits, cnt in counts_no.items():
    bob = int(bits[0])
    c1 = int(bits[1])
    corrected = bob ^ c1
    if corrected == 0:
        corr0_no += cnt
p0_corr_no = corr0_no / total_no
bhatt_corr_no = bhattacharyya(p0_corr_no)
print(f"Corrected: P(0)={p0_corr_no:.4f}, Bhatt={bhatt_corr_no:.4f}")

# Quantum fidelity from DM
dm_no = sim.run(make_static_dm(), noise_model=nm, shots=shots).result().data(0)['density_matrix']
rho_no = trace_out(dm_no, keep_qubit=2)
F_no = fidelity_quantum(rho_no)
p0_rho_no = float(np.real(rho_no[0, 0]))
print(f"Quantum fidelity from DM: F={F_no:.4f}, P(0)={p0_rho_no:.4f}")

# ── COMPLETE TABLE ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("COMPLETE CORRECTED TABLE (5 configurations)")
print("=" * 60)
print()
print("Configuration                          | Metric           | Value")
print("-" * 60)
print(f"[A] Static noiseless (no corrections)  | Quantum F        | {F_nl:.4f}")
print(f"[A] Static noisy (no corrections)    | Quantum F        | {F_no:.4f}")
print(f"[A] Static noiseless + post-hoc corr | Bhatt (QSG)     | {bhatt_corr_nl:.4f}")
print(f"[A] Static noisy + post-hoc corr     | Bhatt (QSG)     | {bhatt_corr_no:.4f}")
print(f"[A] Static noisy + post-hoc corr     | Quantum F       | (need different method)")
print(f"[B] Dynamic noiseless (analytical)   | Quantum F        | 1.000")
print(f"[B] Dynamic noisy DM                 | Quantum F        | 0.962")
print()
print("KEY INSIGHT:")
print(f"  Bhattacharyya for static+post-hoc noiseless: {bhatt_corr_nl:.4f}")
print(f"  This is what QSG reports as ~0.9995 locally, ~0.9848 on hardware.")
print(f"  It is a distribution overlap metric, not quantum fidelity.")
print()
print(f"  Quantum fidelity for static+post-hoc noisy: CANNOT be computed from shots alone.")
print(f"  The corrected P(0) tells us about classical distribution,")
print(f"  not about quantum state fidelity after corrections.")
print()
print(f"  The DM gives the TRUE quantum fidelity: static=noiseless F={F_nl:.4f}")
print(f"  But the DM was computed WITHOUT applying corrections.")
print()
print("CONSISTENCY CHECK:")
print(f"  Staticnoiseless F (no corrections): {F_nl:.4f} ~ 0.50: {'PASS' if abs(F_nl-0.5)<0.15 else 'FAIL'}")
print(f"  Static noisy F (no corrections): {F_no:.4f} ~ 0.50: {'PASS' if abs(F_no-0.5)<0.15 else 'FAIL'}")
print(f"  Dynamic noiseless F: 1.000: PASS (analytical)")
