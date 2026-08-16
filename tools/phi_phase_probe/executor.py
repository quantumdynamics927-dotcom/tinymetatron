"""
tools/phi_phase_probe/executor.py
================================
Execute φ-phase probe circuits against real hardware or AerSimulator.

Execution path:
  1. Check quantum_jobs.db for available backends (ibm_fez, ibm_marrakesh, ibm_torino)
  2. If credentials available → submit to real IBM Quantum backend
  3. Otherwise → fall back to AerSimulator (local, no credentials needed)

Results are stored in quantum_jobs.db and returned as structured dicts.

Statistical test:
  - Total Variation Distance (TVD) between observed distribution and uniform
  - Permutation test: pool φ and control shots, resample, compute TVD difference
  - Report: TVD per circuit, p-value, conclusion
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np

# ── statistical helpers ────────────────────────────────────────────────────────

def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """TVD = 0.5 × Σ |p_i - q_i|. p, q must be same length and sum to 1."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: {p.shape} vs {q.shape}")
    if abs(p.sum() - 1.0) > 1e-9 or abs(q.sum() - 1.0) > 1e-9:
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
    return 0.5 * np.sum(np.abs(p - q))


def uniform_distribution(num_qubits: int) -> np.ndarray:
    """Uniform distribution over all 2^n bitstrings."""
    n = 2 ** num_qubits
    return np.ones(n) / n


def observed_distribution(counts: dict, num_qubits: int) -> np.ndarray:
    """Convert shot-counts dict to probability distribution."""
    n = 2 ** num_qubits
    probs = np.zeros(n)
    for key, val in counts.items():
        idx = int(key, 2) if isinstance(key, str) else key
        if 0 <= idx < n:
            probs[idx] = val
    total = probs.sum()
    return probs / total if total > 0 else probs


def permutation_test(phi_counts: dict, control_counts: dict,
                    num_qubits: int, n_permutations: int = 5000,
                    seed: int = 42) -> dict:
    """
    Permutation test: does phi produce measurably different TVD from uniform
    than the uniform-phase control?

    H0: phi and control produce the same TVD from uniform.
    H1: phi TVD differs from control TVD.

    Uses multinomial resampling: pools the observed counts across all
    bitstrings and resamples two groups of the same sizes as phi/control.
    """
    rng = np.random.default_rng(seed)

    phi_dist = observed_distribution(phi_counts, num_qubits)
    ctrl_dist = observed_distribution(control_counts, num_qubits)
    unif = uniform_distribution(num_qubits)

    obs_tvd_phi = total_variation_distance(phi_dist, unif)
    obs_tvd_ctrl = total_variation_distance(ctrl_dist, unif)
    obs_diff = abs(obs_tvd_phi - obs_tvd_ctrl)

    total_shots = sum(phi_counts.values()) + sum(control_counts.values())
    n = 2 ** num_qubits

    if n == 0 or total_shots == 0:
        return {
            "observed_tvd_phi": float(obs_tvd_phi),
            "observed_tvd_control": float(obs_tvd_ctrl),
            "observed_diff": float(obs_diff),
            "p_value": 1.0,
            "n_permutations": n_permutations,
            "conclusion": "insufficient_data",
        }

    phi_shots = sum(phi_counts.values())
    ctrl_shots = sum(control_counts.values())

    # Pool into a multinomial over the full 2^n outcome space
    pool = np.zeros(n)
    for k, v in phi_counts.items():
        idx = int(k, 2) if isinstance(k, str) else k
        if 0 <= idx < n:
            pool[idx] += v
    for k, v in control_counts.items():
        idx = int(k, 2) if isinstance(k, str) else k
        if 0 <= idx < n:
            pool[idx] += v

    # Resample n_permutations times
    count_as_or_more_extreme = 0
    for _ in range(n_permutations):
        # Multinomial resample of the pooled distribution
        # into two groups with the same sizes as phi/ctrl
        perm = rng.multinomial(int(pool.sum()), pool / pool.sum())
        perm_phi = rng.multinomial(phi_shots, perm / perm.sum() if perm.sum() > 0 else np.ones(n) / n)
        perm_ctrl = pool.astype(int) - perm_phi

        eps = 1e-10
        perm_phi_p = np.maximum(perm_phi, 1) / max(perm_phi.sum(), 1)
        perm_ctrl_p = np.maximum(perm_ctrl, 1) / max(perm_ctrl.sum(), 1)

        perm_tvd_phi = total_variation_distance(perm_phi_p, unif)
        perm_tvd_ctrl = total_variation_distance(perm_ctrl_p, unif)

        if abs(perm_tvd_phi - perm_tvd_ctrl) >= obs_diff:
            count_as_or_more_extreme += 1

    p_value = count_as_or_more_extreme / n_permutations

    return {
        "observed_tvd_phi": float(obs_tvd_phi),
        "observed_tvd_control": float(obs_tvd_ctrl),
        "observed_diff": float(obs_diff),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "conclusion": (
            "significant" if p_value < 0.05
            else " suggestive" if p_value < 0.10
            else "null"
        ),
    }


# ── circuit execution ──────────────────────────────────────────────────────────

def execute_with_qiskit(circuit_dict: dict, backend_name: str,
                        shots: int = 4096) -> dict:
    """
    Execute a single circuit on IBM Quantum or AerSimulator.

    Args:
        circuit_dict: output of build_phi_circuit / build_control_circuit
        backend_name: 'aer_simulator', 'ibm_fez', 'ibm_marrakesh', 'ibm_torino'
        shots: number of measurement shots

    Returns:
        dict with counts, distribution, backend, shots, timestamp
    """
    qc = circuit_dict["circuit"]

    try:
        from qiskit import QuantumCircuit
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency. Install with: pip install qiskit qiskit-ibm-runtime qiskit-aer. {exc}"
        )

    counts: dict[str, int] = {}
    service_backend: Optional[str] = None

    if backend_name == "aer_simulator":
        simulator = AerSimulator()
        job = simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        service_backend = "aer_simulator"
    else:
        # Real hardware via IBM Quantum
        try:
            service = QiskitRuntimeService(channel="ibm_quantum")
            backend = service.backend(backend_name)
            sampler = Sampler(backend=backend)
            job = sampler.run(qc, shots=shots)
            result = job.result()
            counts = result[0].data.c.get_counts()
            service_backend = backend_name
        except Exception as exc:
            print(f"WARN: backend {backend_name} failed: {exc}. Falling back to AerSimulator.")
            simulator = AerSimulator()
            job = simulator.run(qc, shots=shots)
            result = job.result()
            counts = result.get_counts(qc)
            service_backend = "aer_simulator (fallback)"

    # Normalize counts keys (ensure consistent format)
    normalized_counts = {}
    for key, val in counts.items():
        if isinstance(key, int):
            key = format(key, f"0{circuit_dict['num_qubits']}b")
        normalized_counts[str(key)] = val

    return {
        "circuit_type": circuit_dict["type"],
        "convergent": circuit_dict.get("convergent"),
        "angle_str": circuit_dict.get("angle_str"),
        "description": circuit_dict.get("description"),
        "num_qubits": circuit_dict["num_qubits"],
        "backend": service_backend,
        "shots": sum(normalized_counts.values()),
        "counts": normalized_counts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_probe(num_qubits: int = 4, shots: int = 4096,
              backend: str = "aer_simulator",
              convergent_indices: Optional[list[int]] = None,
              seed: int = 42) -> dict:
    """
    Run the full phi-phase probe: all convergents + control, TVD analysis.

    Args:
        num_qubits: number of qubits (3-8 recommended)
        shots: shots per circuit
        backend: 'aer_simulator' or 'ibm_fez'/'ibm_marrakesh'/'ibm_torino'
        convergent_indices: which convergents to test (default: all 0-8)
        seed: random seed for permutation test

    Returns:
        dict with per-circuit results, TVD stats, permutation test results
    """
    from tools.phi_phase_probe.circuit import build_all_probe_circuits, CONVERGENTS

    if convergent_indices is None:
        convergent_indices = list(range(9))  # all 9 convergents

    # Build all 10 circuits (9 phi + 1 control)
    all_circuits = build_all_probe_circuits(num_qubits=num_qubits)

    # Select: requested phi convergents + always include control
    selected = []
    for c in all_circuits:
        if c["type"] == "control":
            selected.append(c)
        elif c.get("convergent") and c["convergent"] in CONVERGENTS:
            idx = CONVERGENTS.index(c["convergent"])
            if idx in convergent_indices:
                selected.append(c)

    # Execute selected circuits
    results = []
    for circuit_dict in selected:
        result = execute_with_qiskit(circuit_dict, backend_name=backend, shots=shots)
        results.append(result)

    # Per-circuit TVD vs uniform
    unif = uniform_distribution(num_qubits)
    for r in results:
        dist = observed_distribution(r["counts"], num_qubits)
        r["tvd_vs_uniform"] = float(total_variation_distance(dist, unif))

    # Permutation test: compare phi (avg over all convergents) vs control
    phi_results = [r for r in results if r["circuit_type"] == "phi"]
    control_results = [r for r in results if r["circuit_type"] == "control"]

    perm_test: dict = {
        "observed_tvd_phi": None,
        "observed_tvd_control": None,
        "p_value": None,
        "conclusion": "no_data",
    }

    if phi_results and control_results:
        # Pool counts across all phi convergents for permutation test
        avg_phi_counts: dict[str, int] = {}
        for r in phi_results:
            for k, v in r["counts"].items():
                avg_phi_counts[k] = avg_phi_counts.get(k, 0) + v
        ctrl_result = control_results[0]
        perm_test = permutation_test(
            avg_phi_counts, ctrl_result["counts"],
            num_qubits=num_qubits, seed=seed
        )

    return {
        "probe": "phi_phase",
        "version": 1,
        "num_qubits": num_qubits,
        "shots": shots,
        "backend": backend,
        "convergents_tested": convergent_indices,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "circuits": results,
        "summary": {
            "n_phi_circuits": len(phi_results),
            "n_control_circuits": len(control_results),
            "phi_tvd_mean": float(np.mean([r["tvd_vs_uniform"] for r in phi_results])) if phi_results else None,
            "phi_tvd_std": float(np.std([r["tvd_vs_uniform"] for r in phi_results])) if phi_results else None,
            "control_tvd": control_results[0]["tvd_vs_uniform"] if control_results else None,
        },
        "permutation_test": perm_test,
    }
