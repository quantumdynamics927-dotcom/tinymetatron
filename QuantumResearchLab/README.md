# QuantumResearchLab (QRL) — Experiment Library

A validated simulator experiment library demonstrating core quantum computing phenomena using Qiskit Aer.

## Experiments

| exp_id | Name | Core Result | Status |
|--------|------|-------------|--------|
| `qrl-2026-002` | CHSH Bell Inequality | S = 2.828 > 2 (violates classical bound) | ✅ Validated |
| `qrl-2026-003` | GHZ + Mermin Inequality | M = 4.0 > 2 (achieves quantum maximum) | ✅ Validated |
| `qrl-2026-004` | Noise Degradation Study | CHSH robust; Mermin collapses to ~1.0 under noise | ✅ Validated |
| `qrl-2026-005` | Bennett 1993 Teleportation | F = 1.0 for all states (|+i>, |-}}, etc.) | ✅ Validated |
| `qrl-2026-006` | Golden-Angle Sphere Sweep | F=1.0 noiseless; noisy min=0.000 (golden) vs 0.044 (random) | ✅ Validated |
| `qrl-2026-007` | Bit-Flip QEC | QEC F=1.0 noiseless; noisy improvement +0.055, syndrome accuracy 98.4% | ✅ Validated |
| `qrl-2026-008` | Entanglement Swapping | F=1.0 noiseless; noisy +0.68 improvement over baseline | ✅ Validated |

## Quick Start

```bash
# Run any experiment
python qrl-2026-002-chsh-bell-test/run.py
python qrl-2026-003-ghz-mermin-test/run.py
python qrl-2026-004-noise-degradation/run.py
python qrl-2026-005-teleportation/run.py
python qrl-2026-006-teleport-sphere-sweep/run.py
python qrl-2026-007-bitflip-qec/run.py
python qrl-2026-008-entanglement-swap/run.py

# Verification (statevector dry-run)
python qrl-2026-003-ghz-mermin-test/circuit.py
python qrl-2026-005-teleportation/circuit.py
```

## Shared Library

`qrl_common/` — shared utilities used by all experiments:
- `make_bell_circuit()` — |Phi+> Bell state preparation
- `make_ghz_circuit(num_qubits)` — |GHZ+> preparation
- `compute_correlator_2q(counts)` — 2-qubit correlator E
- `compute_correlator_3q(counts, shots)` — 3-qubit correlator E
- `exact_correlator_3q(th0, th1, th2)` — exact E = cos(th0+th1+th2) for GHZ
- `make_noise_model(p1q, p2q, ro_err)` — realistic IBM-calibrated noise model
- `golden_angle_sphere_points(n)` — Weyl equidistribution sampling on Bloch sphere
- `sphere_point_to_statevector(theta, phi)` — Bloch point → normalized statevector
- `uniform_random_sphere_points(n)` — baseline random sampling for comparison
- `sphere_discrepancy(points)` — nearest-neighbor CV uniformity measure
- `SHOT_SENSITIVITY_SMALL`, `SHOT_SENSITIVITY_LARGE`, `SHOT_TELEPORTATION` — shot constants

## Protocol

Every experiment follows: **propose → implement → validate → report**

1. **Propose**: fill `MANIFEST.md` with hypothesis and success criteria BEFORE writing circuit code
2. **Implement**: write `circuit.py` + `run.py`
3. **Validate**: statevector-exact first, then shots, then noise (where relevant)
4. **Report**: fill `results.json`, update `MANIFEST.md`, commit

## Key Findings

- **CHSH + Mermin require different measurement operators** — unifying them with one basis choice fails; CHSH uses RY (A=cos·Z+sin·X), Mermin uses RZ+H (A=cos·X+sin·Y)
- **Feedforward corrections are load-bearing** in teleportation — removing them drops fidelity from 1.0 to ~0.50
- **Three-qubit bit-flip QEC** corrects all single-X errors perfectly (noiseless); under qrl-004 noise, syndrome accuracy ~98% and QEC provides +0.055 mean fidelity improvement over no-QEC baseline
- **Two-hop entanglement swap** creates Bell pair between non-interacting nodes via intermediate repeater; noiseless F=1.0, degrades to ~0.95 under qrl-004 noise (+0.68 over uncorrected baseline); corrections are load-bearing (F drops to ~0.26 without them)
- **Multi-qubit entanglement degrades exponentially** under realistic noise — Mermin/GHZ collapses to classical (M~1.0) at p1q=0.2%, p2q=1.5% while CHSH remains above bound

## Experiment Structure

```
qrl-2026-XXX-name/
  MANIFEST.md     — hypothesis, criteria, results, physics background
  circuit.py      — quantum circuit implementation
  run.py         — execution + measurement + results.json
  results.json    — generated after run
```

## Noise Model Parameters (qrl-004 and qrl_common)

Based on typical IBM 7-qubit device calibration data:
- **p1q = 0.002** (0.2%): 1-qubit gate depolarizing error
- **p2q = 0.015** (1.5%): 2-qubit gate (CX) depolarizing error
- **ro_err = 0.02** (2%): Per-qubit readout bit-flip probability

No credentials or live backend required — pure `qiskit_aer.noise`.
