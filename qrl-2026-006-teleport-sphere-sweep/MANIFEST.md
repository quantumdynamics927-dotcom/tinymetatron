# Golden-Angle Teleportation Fidelity Sweep — qrl-006

## Metadata
- **exp_id**: `006`
- **title**: Bloch-Sphere Fidelity Sweep: Golden-Angle Sampling vs. Uniform Random
- **date_proposed**: 2026-08-03
- **circuit_family**: Bennett 1993 teleportation (3-qubit, from qrl-005)
- **backend**: simulator (Qiskit Aer, noiseless + noisy)
- **hypothesis**: Teleportation fidelity is uniform across the Bloch sphere under noiseless conditions; under realistic noise, fidelity may show state-dependent variation near the poles or equator due to differential gate errors.

## Physics Background

The golden-angle sequence (Weyl equidistribution) gives uniformly distributed points on the sphere:
- Golden ratio φ = (1+√5)/2
- Golden angle = 2π(1 - 1/φ) ≈ 137.508°
- θ_n = arccos(1 - 2n/(N-1)), φ_n = n × golden_angle

Uniform random sampling clusters near poles; golden-angle avoids this.

## Success Criteria
| Threshold | Value | Meaning |
|-----------|-------|---------|
| Noiseless fidelity | F ≥ 0.99 for all 50 states | Protocol uniform across Bloch sphere |
| Noisy fidelity | Any state F < 0.5 | Reveals noise-sensitive states |
| Sampling comparison | Golden-angle discrepancy < random | Validates sampling technique |

## Implementation Notes
- Uses qrl-005 teleportation circuit (feedforward corrections validated)
- 50 golden-angle states via `qrl_common.golden_angle_sphere_points()`
- 50 uniform-random states via `qrl_common.uniform_random_sphere_points()`
- Noiseless run first, then noisy (qrl-004 noise model)
- Statevector-exact verification on 5 representative states before full sweep

## Results

- **date_run**: 2026-08-03T19:18:28.328387+00:00
- **noiseless_min_fidelity**: 1.000000
- **noiseless_avg_fidelity**: 1.000000
- **noisy_min_fidelity**: 0.000000
- **noisy_avg_fidelity**: 0.500000
- **noisy_golden_std**: 0.294508
- **noisy_random_mean**: 0.535233
- **noisy_random_std**: 0.277838
- **golden_angle_discrepancy**: 0.114576
- **random_discrepancy**: 0.503156
- **sampling_uniformity**: Golden-angle more uniform (0.1146 < 0.5032)
- **key_finding**: Golden-angle found states with F near 0.000 under noise (min=0.000); random sampling's worst case was F=0.044. Both methods show ~0.50 mean noisy fidelity.
- **circuit_hash**: 1afdf0cb852b8efde6d3287f2903758a17e1f99dbe2c00917845e1675823de11
- **code_commit**: 42c0315
- **shots_per_state**: 20000


## Results

- **date_run**: 2026-08-03T21:06:54.314387+00:00
- **noiseless_min_fidelity**: 1.000000
- **noiseless_avg_fidelity**: 1.000000
- **noisy_min_fidelity**: 0.000000
- **noisy_avg_fidelity**: 0.500000
- **noisy_golden_std**: 0.294508
- **noisy_random_mean**: 0.535233
- **noisy_random_std**: 0.277838
- **golden_angle_discrepancy**: 0.114576
- **random_discrepancy**: 0.503156
- **sampling_uniformity**: Golden-angle more uniform (0.1146 < 0.5032)
- **key_finding**: Golden-angle found states with F near 0.000 under noise (min=0.000); random sampling's worst case was F=0.044. Both methods show ~0.50 mean noisy fidelity.
- **circuit_hash**: 1afdf0cb852b8efde6d3287f2903758a17e1f99dbe2c00917845e1675823de11
- **code_commit**: 152b474
- **shots_per_state**: 20000

## Results

- **date_run**: 2026-08-03T21:15:12.075687+00:00
- **noiseless_min_fidelity**: 1.000000
- **noiseless_avg_fidelity**: 1.000000
- **noisy_min_fidelity**: 0.000000
- **noisy_avg_fidelity**: 0.500000
- **noisy_golden_std**: 0.294508
- **noisy_random_mean**: 0.535233
- **noisy_random_std**: 0.277838
- **golden_angle_discrepancy**: 0.114576
- **random_discrepancy**: 0.503156
- **sampling_uniformity**: Golden-angle more uniform (0.1146 < 0.5032)
- **key_finding**: Golden-angle found states with F near 0.000 under noise (min=0.000); random sampling's worst case was F=0.044. Both methods show ~0.50 mean noisy fidelity.
- **circuit_hash**: 1afdf0cb852b8efde6d3287f2903758a17e1f99dbe2c00917845e1675823de11
- **code_commit**: 152b474
- **shots_per_state**: 20000

## Results

- **date_run**: 2026-08-03T21:16:55.912884+00:00
- **noiseless_min_fidelity**: 1.000000
- **noiseless_avg_fidelity**: 1.000000
- **noisy_min_fidelity**: 0.000000
- **noisy_avg_fidelity**: 0.500000
- **noisy_golden_std**: 0.294508
- **noisy_random_mean**: 0.535233
- **noisy_random_std**: 0.277838
- **golden_angle_discrepancy**: 0.114576
- **random_discrepancy**: 0.503156
- **sampling_uniformity**: Golden-angle more uniform (0.1146 < 0.5032)
- **key_finding**: Golden-angle found states with F near 0.000 under noise (min=0.000); random sampling's worst case was F=0.044. Both methods show ~0.50 mean noisy fidelity.
- **circuit_hash**: 1afdf0cb852b8efde6d3287f2903758a17e1f99dbe2c00917845e1675823de11
- **code_commit**: 152b474
- **shots_per_state**: 20000