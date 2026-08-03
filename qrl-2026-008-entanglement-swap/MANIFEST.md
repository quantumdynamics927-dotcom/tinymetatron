# Entanglement Swapping — qrl-008

## Metadata
- **exp_id**: `008`
- **title**: Two-Hop Quantum Repeater via Entanglement Swapping
- **date_proposed**: 2026-08-03
- **circuit_family**: 4-qubit entanglement swapping (Duan et al., Nature 2003 quantum repeater)
- **backend**: simulator (Qiskit Aer, noiseless + noisy)

## Physics Background

Entanglement swapping extends the Bennett 1993 teleportation concept to distribute entanglement between two non-interacting nodes using an intermediate "repeater" node:

**Qubit layout:**
- Node A = q0 (left end node)
- Repeater-left = q1 (left half of first Bell pair)
- Repeater-right = q2 (right half of second Bell pair)
- Node B = q3 (right end node)

**Two independent Bell pairs:**
1. Pair 1: (q0, q1) = |Phi+> = (|00> + |11>)/sqrt(2)
2. Pair 2: (q2, q3) = |Phi+> = (|00> + |11>)/sqrt(2)

**Entanglement swap on (q1, q2):**
- Bell-state measurement on (q1, q2): H(q1) + CX(q1,q2) + measure
- This projects (q0, q3) onto a Bell state, entangling nodes that never interacted
- Feedforward corrections on q0 and/or q3 based on (q1, q2) measurement outcome

**Why this is different from teleportation:**
- No quantum state is "teleported" — instead, two independent entanglements are "fused" into one
- The intermediate qubits (q1, q2) are measured and discarded; only q0 and q3 remain entangled
- This is the fundamental primitive of a quantum repeater network

## Success Criteria
| Threshold | Value | Meaning |
|-----------|-------|---------|
| Noiseless fidelity | F = 1.0 | Perfect swap — q0, q3 become a Bell pair |
| Without corrections | F < 1.0 | Corrections are load-bearing (not vestigial) |
| Noisy fidelity | F > baseline | Noise degrades but QEC helps |

## Feedforward Correction Logic

After measuring (q1, q2), the joint state of (q0, q3) depends on the measurement outcome:

| q1_meas | q2_meas | Outcome state (q0,q3) | Correction |
|---------|---------|----------------------|------------|
| 0 | 0 | |Phi+> = (|00>+|11>)/sqrt(2) | none |
| 0 | 1 | |Psi+> = (|01>+|10>)/sqrt(2) | X on q3 |
| 1 | 0 | |Phi-> = (|00>-|11>)/sqrt(2) | Z on q0 |
| 1 | 1 | |Psi-> = (|01>-|10>)/sqrt(2) | Z on q0, X on q3 |

After correction, (q0, q3) = |Phi+> always.

## Bug Classes Encountered

1. **Bell measurement mapping**: The circuit used CX then H, but the correct order for Bell measurement is H then CX. This maps all four Bell states to distinguishable computational basis states.

2. **Shot-based fidelity for entangled states**: Computational-basis measurement (P(00)+P(11)) gives 0.5 for ALL Bell states — it cannot distinguish them. The correct fidelity metric requires density matrix projection.

3. **Feedforward with `if_test` in shot-based simulation**: Measurement collapses the state before corrections can create entanglement. The density matrix approach (averaging over measurement outcomes in quantum superposition) is the correct method.

## Results

- **date_run**: 2026-08-03T22:14:09.353114+00:00
- **noiseless_qec_fidelity**: 1.000000 (density matrix, 100k shots)
- **noiseless_baseline_fidelity**: 0.260742 (without corrections)
- **noiseless_improvement**: 0.739258
- **noisy_qec_fidelity**: 0.947266 (qrl-004 noise, 100k shots)
- **noisy_baseline_fidelity**: 0.259766 (without corrections)
- **noisy_improvement**: 0.687500
- **key_finding**: Corrections are load-bearing; without them fidelity is ~0.26 (mixture of all four Bell states). With corrections, F = 1.0 noiseless, 0.95 noisy. The extra entanglement operations (vs single-hop teleportation) make swap more sensitive to noise.
- **circuit_hash**: f080ad5ebc58dba2c26b6fd99cacec258e5126ee0002ca3c22405856f89c5f42
- **code_commit**: fb45ec0
- **shots**: 100000


## Results

- **date_run**: 2026-08-03T22:15:47.653630+00:00
- **noiseless_statevector_fidelity**: 1.000000
- **noiseless_baseline_fidelity**: 0.262695
- **noiseless_improvement**: 0.737305
- **noisy_swap_fidelity**: 0.9521
- **noisy_baseline_fidelity**: 0.2695
- **noisy_improvement**: 0.6826
- **circuit_hash**: 089d7b3bc6f675feb0b27e943af6c40dc47c15339ae66c612bd2fb4523b781ee
- **code_commit**: fb45ec0
- **shots**: 100000