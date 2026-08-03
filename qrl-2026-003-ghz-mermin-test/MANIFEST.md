# GHZ State + Mermin Inequality — Simulator Experiment

## Metadata
- **exp_id**: `003`
- **title**: Mermin Inequality Violation on a Simulated 3-Qubit GHZ State
- **date_proposed**: 2026-08-02
- **circuit_family**: 3-qubit GHZ entangled state (|GHZ⟩ = (|000⟩+|111⟩)/√2)
- **backend**: simulator (Qiskit Aer noise-free)
- **hypothesis**: A Mermin measurement on an ideal 3-qubit GHZ state, simulated noiselessly, will yield M = 4, violating the classical bound of |M| ≤ 2 and achieving the quantum maximum for this state.

## Physics Background

The Mermin inequality for 3 qubits:

```
M = E(θ_a, θ_b, θ_c) - E(θ_a, θ'_b, θ'_c) - E(θ'_a, θ_b, θ'_c) - E(θ'_a, θ'_b, θ_c)
```

**Measurement operator:** A(θ) = cos(θ)·X + sin(θ)·Y

**Correlator formula (verified by statevector):**
```
E(θ_a, θ_b, θ_c) = cos(θ_a + θ_b + θ_c)
```

**Settings (2 per qubit):**
- θ = 0     → X measurement
- θ = π/2   → Y measurement

**Mermin operator (optimal for |GHZ⟩):**
```
M = E(0,0,0) - E(0,π/2,π/2) - E(π/2,0,π/2) - E(π/2,π/2,0)
```

**Computed values:**
- E(0,0,0)        = cos(0)         = +1
- E(0,π/2,π/2)   = cos(π)         = -1
- E(π/2,0,π/2)   = cos(π)         = -1
- E(π/2,π/2,0)   = cos(π)         = -1

**M = 1 - (-1) - (-1) - (-1) = 4**

## Bounds
- **Classical (LHV)**: |M| ≤ 2
- **Quantum (GHZ)**: |M| = 4

## Success Criteria
| Threshold | Value | Meaning |
|-----------|-------|---------|
| Classical bound | |M| ≤ 2 | Local realism limit |
| Minimum bar (PASS/FAIL) | M > 2 | Any violation confirms quantum advantage |
| Target | M ≥ 3.5 | Near-quantum-maximum with noiseless simulator |

## Circuit Description

**GHZ state preparation:**
- H gate on qubit 0 → superposition
- CX(0, 1) → entangle qubits 0,1
- CX(0, 2) → entangle qubits 0,2 → |GHZ⟩ = (|000⟩+|111⟩)/√2

**Mermin measurement (X/Y bases):**
- θ = 0 → X: RZ(0) then H (Hadamard)
- θ = π/2 → Y: RZ(-π/2) then H
- Measure all 3 qubits in Z basis
- Correlator: E = cos(θ_a + θ_b + θ_c)

**4 unique settings for the 4 Mermin terms.**

## Implementation Notes
- Use Qiskit `AerSimulator` with no noise model
- Use exact formula E = cos(θ_a + θ_b + θ_c) for correlator values
- Shot-count sensitivity: run at 1024 AND 100k shots

## Results

- **date_run**: 2026-08-03T15:49:13.448089+00:00
- **result_value**: 4.000000
- **violated_bound**: classical
- **circuit_hash**: 15a10db68cac048b1ce1ad6d5129fc7b4876baf5e040c3eb66601b31090fc27a
- **code_commit**: 1298c4e
- **passes_minimum_bar**: true (M > 2 = quantum advantage)
- **passes_target**: true (M >= 3.5 = near quantum max)
- **shot_noise_verified**: false
