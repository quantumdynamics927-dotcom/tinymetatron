# Bit-Flip Quantum Error Correction — qrl-007

## Metadata
- **exp_id**: `007`
- **title**: Three-Qubit Bit-Flip Code — Quantum Error Correction
- **date_proposed**: 2026-08-03
- **circuit_family**: Three-qubit bit-flip QEC (Shor code variant, simplest QEC)
- **backend**: simulator (Qiskit Aer, noiseless + noisy)

## Physics Background

The three-qubit bit-flip code protects a logical qubit against single bit-flip errors using entanglement and parity measurements:

**Encoding:**
- |0⟩_L = |000⟩, |1⟩_L = |111⟩ (three physical qubits)
- Input state |ψ⟩ = α|0⟩ + β|1⟩ → α|000⟩ + β|111⟩

**Error model:** Single-qubit bit-flip (X) on any of the three qubits.

**Syndrome measurement (parity checks):**
- Measure parity of qubits 0 and 1 (Z⊗Z); outcome 1 → X error on q0 or q1
- Measure parity of qubits 1 and 2 (Z⊗Z); outcome 1 → X error on q1 or q2

**Syndrome table:**
| Z₀Z₁ | Z₁Z₂ | Error | Correction |
|------|------|-------|------------|
| 0    | 0    | none  | identity   |
| 1    | 0    | X₀    | apply X to q0 |
| 1    | 1    | X₁    | apply X to q1 |
| 0    | 1    | X₂    | apply X to q2 |

**Recovery:** Apply X to the flagged qubit, restoring the correct state.

**Why it works:**
- The code space is the +1 eigenspace of stabilizer {Z₀Z₁, Z₁Z₂}
- Errors map the codeword to a different eigenstate; syndrome measurement identifies which
- No collapse of logical state (error detection is non-destructive when no correction applied)

## Success Criteria
| Threshold | Value | Meaning |
|-----------|-------|---------|
| Noiseless fidelity | F = 1.0 | Perfect correction — code is exact |
| Noisy: without QEC | F < 1.0 | Single X error degrades fidelity |
| Noisy: with QEC | F > without QEC | QEC provides measurable protection |
| Syndrome accuracy | 100% | Correctable errors always detected |

## Implementation Notes
- 5-qubit circuit: 3 code qubits (q0,q1,q2) + 2 ancilla (q3,q4) for syndrome
- Syndrome extraction: CNOT cascade (XOR ladder) on ancilla, then measurement
- Syndrome table (for |000> codeword): 00=none, 10=q0, 11=q1, 01=q2
- Test: encode |ψ⟩ = α|0⟩ + β|1⟩, inject X error, run recovery, measure fidelity
- Test both with and without recovery to isolate QEC benefit
- Statevector verification first (noiseless F=1.0), then shots, then noisy

## Results

- **date_run**: 2026-08-03T20:21:53.740702+00:00
- **noiseless_qec_mean_fidelity**: 1.000000
- **noiseless_qec_min_fidelity**: 1.000000
- **noiseless_baseline_mean_fidelity**: 0.250000
- **noiseless_mean_improvement**: 0.750000
- **noiseless_syndrome_accuracy**: 100.00%
- **noisy_qec_mean_fidelity**: 0.493837
- **noisy_qec_min_fidelity**: 0.000000
- **noisy_baseline_mean_fidelity**: 0.442187
- **noisy_mean_improvement**: 0.051650
- **noisy_syndrome_accuracy**: 49.90%
- **circuit_hash**: b0bfb0042a1912736b2630c5044809d481b3ed219f0e2230f42830ea9d50fcbf
- **code_commit**: 6c26eaf
- **shots_per_test**: 20000
