# GHZ State + Mermin Inequality — Simulator Experiment

## Metadata
- **exp_id**: `003`
- **title**: Mermin Inequality Violation on a Simulated 3-Qubit GHZ State
- **date_proposed**: 2026-08-02
- **circuit_family**: 3-qubit GHZ entangled state (|GHZ⟩ = (|000⟩ + |111⟩)/√2)
- **backend**: simulator (Qiskit Aer noise-free)
- **hypothesis**: A Mermin measurement on an ideal 3-qubit GHZ state, simulated noiselessly, will yield M > 4, violating the classical bound of 4 and approaching the theoretical quantum maximum of 4√2 ≈ 5.657.

## Physics Background

The Mermin inequality for 3 qubits is a generalization of the CHSH (2-qubit) Bell inequality:

For settings a, a' on qubit 1; b, b' on qubit 2; c, c' on qubit 3:

```
M = E(a,b,c) + E(a,b',c) + E(a',b,c) + E(a',b',c')
    - E(a,b,c') - E(a,b',c') - E(a',b,c') - E(a',b',c')
```

where E(a,b,c) is the 3-qubit correlator: expectation value of the product of the three measurement outcomes (±1).

**Bounds:**
- **Classical**: |M| ≤ 4
- **Quantum (GHZ state, optimal settings)**: |M| ≤ 4√2 ≈ 5.657

The |GHZ⟩ = (|000⟩ + |111⟩)/√2 state achieves the quantum bound when all measurement angles are 0° or 90° (in the X basis).

## Success Criteria
| Threshold | Value | Meaning |
|-----------|-------|---------|
| Classical bound | M ≤ 4 | Local realism limit |
| Minimum bar (PASS/FAIL) | M > 4 | Any violation confirms quantum advantage |
| Target | M ≥ 5.5 | Near-quantum-maximum with noiseless simulator |

## Circuit Description

**GHZ state preparation:**
- H gate on qubit 0 → superposition
- CX(0, 1) → entangle qubits 0,1
- CX(0, 2) → entangle qubits 0,2 → |GHZ⟩ = (|000⟩ + |111⟩)/√2

**Mermin measurement bases:**
- All three qubits measured at either 0° (Z basis) or 90° (X basis)
- 8 combinations of (a,b,c) settings, each computed as a 3-qubit correlator
- M = sum of 4 positive terms minus sum of 4 negative terms

**3-qubit correlator E(a,b,c):**
Each outcome is ±1. For an n-qubit correlator, the expectation is:
E = (N_even_parity - N_odd_parity) / N_total
where even parity = even number of |1⟩ outcomes.

## Implementation Notes
- Use Qiskit `AerSimulator` with no noise model
- Measure all 3 qubits simultaneously with RY rotations
- Counts dict has 8 keys ('000' through '111')
- Shot-count sensitivity: run at 1024 AND 100k shots

## Results

- **date_run**: 2026-08-03T15:33:44.947429+00:00
- **result_value**: 5.328427
- **violated_bound**: classical
- **circuit_hash**: 80b8b36b561576d49139e9f987f2dd1d8f9005fa80302bc347f96caf5fe19366
- **code_commit**: f2ba1eb
- **passes_minimum_bar**: true (M > 4 = quantum advantage)
- **passes_target**: false (M >= 5.5 = near quantum max)
- **shot_noise_verified**: false
