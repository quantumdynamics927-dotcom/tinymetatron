# CHSH Bell Inequality Violation — Simulator Experiment

## Metadata
- **exp_id**: `002`
- **title**: CHSH Bell Inequality Violation on a Simulated 2-Qubit Bell State
- **date_proposed**: 2026-08-02
- **circuit_family**: 2-qubit entangled Bell state (|Φ+⟩ = (|00⟩ + |11⟩)/√2)
- **backend**: simulator (Qiskit Aer noise-free)
- **hypothesis**: A CHSH measurement on an ideal 2-qubit Bell state, simulated noiselessly, will yield S ≥ 2.7, violating the classical bound of 2 and approaching the theoretical quantum maximum of 2√2 ≈ 2.828.

## Physics Background

The CHSH inequality states that for any local hidden variable theory:
```
S = |E(a,b) - E(a, b') + E(a', b) + E(a', b')| ≤ 2
```
where E(a, b) is the expectation value of the product of measurements along angles a and b.

For the optimal quantum strategy (measure each qubit at 45°-separated bases):
- **|Φ+⟩ state** measured at angles (0°, 45°, 90°, 135°) gives S = 2√2 ≈ 2.828

## Success Criteria
| Threshold | Value | Meaning |
|-----------|-------|---------|
| Classical bound | S ≤ 2 | Local realism limit |
| Minimum bar (PASS/FAIL) | S > 2 | Any violation confirms quantum advantage |
| Target | S ≥ 2.7 | Near-quantum-maximum with noiseless simulator |

## Circuit Description

**Bell state preparation:**
- Apply H gate to qubit 0 → superposition (|0⟩ + |1⟩)/√2
- Apply CX gate with qubit 0 as control, qubit 1 as target → |Φ+⟩ = (|00⟩ + |11⟩)/√2

**CHSH measurement bases:**
- Qubit 0: angles a=0°, a'=90°
- Qubit 1: angles b=45°, b'=135°

This configuration is the standard optimal CHSH strategy for the |Φ+⟩ state.

## Measurement & Statistics

- **Shots**: 10,000 (sufficient for S ≈ 2.828 ± 0.02 on noiseless simulator)
- **Pairs of settings**: (a,b), (a,b'), (a',b), (a',b')
- **For each pair**: compute E = (N_++ + N_-- - N_+- - N_-+) / N_total
  - N_++ = count both qubits measured +1
  - N_-- = count both qubits measured -1
  - etc.
- **S = |E(ab) - E(ab') + E(a'b) + E(a'b')|**

## Implementation Notes
- Use Qiskit `AerSimulator` with `noise_model=None` for ideal simulation
- Use `BasicAer` or `AerSimulator` with no noise
- Measurement must be in Z basis; rotation applied before measurement via `ry`/`rz` gates
- Use ` shots=1024` (minimum) to `shots=100000` — 10,000 is the standard choice for Bell tests

## Results

- **date_run**: 2026-08-02T20:09:10.859441+00:00
- **result_value**: 2.828120 (corrected — 100k shots; 1024-shot result 2.830078 was shot noise)
- **violated_bound**: classical
- **circuit_hash**: 158766364b2d2c9f3ed75a346e3b2c6d47e285c2bc6edaea8df8641d96e5512b
- **code_commit**: edefbb9
- **passes_minimum_bar**: true (S > 2 = any quantum advantage)
- **passes_target**: true (S >= 2.7 = near quantum maximum)
- **shot_noise_verified**: true (S converged to 2.828120 at 100k shots, within 0.0003 of theoretical 2.828427)
- **note**: Original 1024-shot result (S=2.830078) exceeded Tsirelson bound due to shot noise. Re-run at 100k shots confirmed S=2.828120.

