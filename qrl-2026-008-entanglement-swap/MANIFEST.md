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

## Bug Classes Found During Development

The following bugs were discovered and fixed through systematic debugging:

1. **Bell measurement gate order wrong**: Initial circuit used `CX then H`, but the correct order for Bell measurement is `H then CX`. With the wrong order, all four Bell states map to overlapping superpositions and cannot be distinguished. Fix: `H(bsm_control) then CX(bsm_control, bsm_target)` maps each Bell state to a unique computational basis outcome.

2. **Shot-based fidelity metric for entangled states**: Using computational-basis measurement P(00)+P(11) as a fidelity proxy gives 0.5 for ALL four Bell states — it cannot distinguish |Phi+>, |Phi->, |Psi+>, or |Psi->. Fix: project the full density matrix onto the |Phi+> subspace and compute fidelity there. The correlator E = P(00)+P(11)-P(01)-P(10) also fails here since |Phi+> and |Phi-> have the same E=1.

3. **`if_test` feedforward collapsing state in shot-based simulation**: When measuring (q1,q2) first then applying corrections, the state collapses before corrections can act. More critically, the `if_test` conditional gate was somehow affecting the syndrome distribution, causing only 2 of 4 possible syndromes to appear. Fix: use `AerSimulator` with `save_density_matrix()` on a circuit that measures syndrome and applies corrections but does NOT measure the output (q0,q3). This correctly represents quantum feedforward as a coherent conditional operation.

## Noisy Fidelity Computation Note

The noisy fidelity (F=0.944) uses `AerSimulator(method='density_matrix')` with `save_density_matrix()` on the full circuit including syndrome measurement and feedforward corrections. This mirrors real hardware feedforward: measurement → classical outcome → conditional unitary applied. The density matrix captures noise effects (depolarizing CX errors, readout errors) averaged over 100k shots. This is the same method used by qrl-005 teleportation, making the F=0.944 vs F=1.0 comparison between swap and teleportation apples-to-apples.

## Results

- **date_run**: 2026-08-03T22:18:36.460693+00:00
- **noiseless_qec_fidelity**: 1.000000 (density matrix, 100k shots)
- **noiseless_baseline_fidelity**: 0.261719 (without corrections, uniform mixture of 4 Bell states = 1/4)
- **noiseless_improvement**: 0.738281
- **noisy_qec_fidelity**: 0.944300 (qrl-004 noise, 100k shots)
- **noisy_baseline_fidelity**: 0.263700 (without corrections)
- **noisy_improvement**: 0.680700
- **key_finding**: Corrections are load-bearing; without them F ≈ 0.26 (maximally mixed, 1/4 per Bell state). With corrections, F = 1.0 noiseless, 0.944 noisy. The extra entanglement operations (two Bell pairs + extra BSM) make swap more noise-sensitive than single-hop teleportation (qrl-005: F=1.0 noisy).
- **circuit_hash**: 089d7b3bc6f675feb0b27e943af6c40dc47c15339ae66c612bd2fb4523b781ee
- **code_commit**: dad8e05
- **shots**: 100000
