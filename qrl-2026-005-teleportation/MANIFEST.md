# Bennett 1993 Quantum Teleportation — Simulator Experiment

## Metadata
- **exp_id**: `005`
- **title**: Bennett 1993 Quantum Teleportation with Classical Feedforward Corrections
- **date_proposed**: 2026-08-03
- **circuit_family**: 3-qubit teleportation (Bennett et al., PRL 1993)
- **backend**: simulator (Qiskit Aer noise-free)
- **hypothesis**: The Bennett 1993 teleportation protocol will transfer an arbitrary unknown quantum state from Alice to Bob with fidelity ≈ 1.0, using a shared Bell pair and classical feedforward of measurement outcomes.

## Physics Background

The Bennett 1993 teleportation protocol uses a shared Bell pair and classical communication to teleport an unknown quantum state:

**Circuit (3 qubits):**
- Qubit 0: Alice's data qubit (unknown state |ψ⟩ to teleport)
- Qubit 1: Alice's half of Bell pair
- Qubit 2: Bob's half of Bell pair

**Steps:**
1. Prepare Bell pair: H(1) → CX(1,2) → |Φ+⟩ = (|00⟩+|11⟩)/√2
2. Entangle data qubit: CX(0,1)
3. Alice's measurement: H(0) → measure qubits 0,1 → classical bits (c0, c1)
4. Bob's feedforward corrections: X(2) if c1=1, Z(2) if c0=1
5. Qubit 2 now holds |ψ⟩ (teleported)

**Feedforward corrections (critical):**
- If measurement outcome is |00⟩: c0=0, c1=0 → no correction needed
- If |01⟩: c0=0, c1=1 → apply X (bit flip)
- If |10⟩: c0=1, c1=0 → apply Z (phase flip)
- If |11⟩: c0=1, c1=1 → apply X then Z

**Why arbitrary superposition inputs matter:**
- |0⟩ and |+⟩ can trivially "teleport" even with broken circuits (no phase/amplitude sensitivity)
- |+i⟩ = (|0⟩+i|1⟩)/√2 tests phase sensitivity (catches Z-axis bugs)
- |-⟩ = (|0⟩-|1⟩)/√2 tests sign sensitivity (catches global phase bugs)

## Success Criteria
| Threshold | Value | Meaning |
|-----------|-------|---------|
| Classical bound | N/A | No classical analogue of teleportation |
| Minimum bar | F > 0.9 | Any state teleported with >90% fidelity |
| Target | F ≥ 0.99 | Near-perfect fidelity for all test states |

## Circuit Description

**Bell pair preparation:**
- H gate on qubit 1 → superposition
- CX(1,2) → entangled Bell pair |Φ+⟩ on qubits 1,2

**Bell-state measurement (Alice):**
- CX(0,1) → entangle data qubit with Alice's Bell half
- H(0) → Hadamard before measurement
- MEASURE qubits 0,1 → classical bits (c0, c1)

**Feedforward correction (Bob):**
- X gate on qubit 2 conditioned on c1=1
- Z gate on qubit 2 conditioned on c0=1
- Uses Qiskit's `.c_if()` for classical-feedforward simulation

## Implementation Notes

- Use `AerSimulator` with no noise (this is a fidelity test, not a bound comparison)
- Use Qiskit `Statevector` to compute exact output state vs. input state
- Compute fidelity F = |⟨ψ_in|ψ_out⟩|² for each input
- Test 4 input states: |0⟩, |+⟩, |+i⟩, |-⟩ — average fidelity must exceed 0.99
- Use `qc.initialize()` to prepare arbitrary input states

## Results

- **date_run**: 2026-08-03T18:15:20.123382+00:00
- **fidelity_0**: 1.000000
- **fidelity_plus**: 1.000000
- **fidelity_plus_i**: 1.000000
- **fidelity_minus**: 1.000000
- **average_fidelity**: 1.000000
- **above_minimum_bar**: true (F > 0.9)
- **above_target**: true (F >= 0.99)
- **analytical_fidelities**: all = 1.000000 (Bennett 1993 protocol guarantee)
- **circuit_hash**: 0638fb077f374fd3344165314a5ac4247b301f4a95d3ae6d8e61a3bf3d7fb9a8
- **code_commit**: 97d4d85
- **shots**: 50000


## Results

- **date_run**: 2026-08-03T21:13:59.598143+00:00
- **fidelity_0**: 1.000000
- **fidelity_plus**: 1.000000
- **fidelity_plus_i**: 1.000000
- **fidelity_minus**: 1.000000
- **average_fidelity**: 1.000000
- **above_minimum_bar**: true (F > 0.9)
- **above_target**: true (F >= 0.99)
- **analytical_fidelities**: all = 1.000000 (Bennett 1993 protocol guarantee)
- **circuit_hash**: 4498ca8dcef531c490f62fa16f8c399537b739f6a30e7bc6d490d351bf1c7685
- **code_commit**: 152b474
- **shots**: 50000

## Results

- **date_run**: 2026-08-03T21:44:31.903593+00:00
- **fidelity_0**: 1.000000
- **fidelity_plus**: 1.000000
- **fidelity_plus_i**: 1.000000
- **fidelity_minus**: 1.000000
- **average_fidelity**: 1.000000
- **above_minimum_bar**: true (F > 0.9)
- **above_target**: true (F >= 0.99)
- **analytical_fidelities**: all = 1.000000 (Bennett 1993 protocol guarantee)
- **circuit_hash**: 4498ca8dcef531c490f62fa16f8c399537b739f6a30e7bc6d490d351bf1c7685
- **code_commit**: fb45ec0
- **shots**: 50000

## Results

- **date_run**: 2026-08-03T22:17:08.245121+00:00
- **fidelity_0**: 1.000000
- **fidelity_plus**: 1.000000
- **fidelity_plus_i**: 1.000000
- **fidelity_minus**: 1.000000
- **average_fidelity**: 1.000000
- **above_minimum_bar**: true (F > 0.9)
- **above_target**: true (F >= 0.99)
- **analytical_fidelities**: all = 1.000000 (Bennett 1993 protocol guarantee)
- **circuit_hash**: 4498ca8dcef531c490f62fa16f8c399537b739f6a30e7bc6d490d351bf1c7685
- **code_commit**: dad8e05
- **shots**: 50000