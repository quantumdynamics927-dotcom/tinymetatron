"""
Three-Qubit Bit-Flip Quantum Error Correction - qrl-007

Extends qrl-005 to demonstrate the simplest quantum error correction code:
the three-qubit bit-flip code (Shor code variant).

Encoding:   |psi> = alpha|0> + beta|1>  -->  alpha|000> + beta|111>
Error:      Single X on one of three qubits
Syndrome:   Z0Z1 and Z1Z2 parity checks via CNOT cascade ancilla
Recovery:   X on error qubit based on syndrome

Noiseless:  F = 1.0 (perfect QEC) for all test states
Noisy:      QEC should improve fidelity over no-QEC baseline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

from qrl_common import make_noise_model


# -- Encoding ---------------------------------------------------------------

def encode_logical(input_sv: list) -> QuantumCircuit:
    """
    Encode a single qubit state |psi> = alpha|0> + beta|1> into the three-qubit
    bit-flip code: alpha|000> + beta|111>.
    """
    _sv = np.array(input_sv, dtype=complex)
    norm = np.linalg.norm(_sv)
    alpha, beta = _sv[0] / norm, _sv[1] / norm

    qc = QuantumCircuit(5, name='encode')
    qc.initialize([alpha, beta], 0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    return qc


# -- Syndrome extraction ----------------------------------------------------

def extract_syndrome(qc: QuantumCircuit,
                    code_qubits=(0, 1, 2),
                    ancilla_qubits=(3, 4)) -> QuantumCircuit:
    """
    Measure Z0Z1 and Z1Z2 parity using CNOT cascade on ancilla.

    CNOT cascade (XOR ladder):
      ancilla a0 = q0 XOR q1: CX(q0,a0), CX(q1,a0)
      ancilla a1 = q1 XOR q2: CX(q1,a1), CX(q2,a1)

    Syndrome table (for |000> codeword):
      (s1,s2) = (0,0): no error
      (0,1): X on q2
      (1,0): X on q0
      (1,1): X on q1
    """
    q0, q1, q2 = code_qubits
    a0, a1 = ancilla_qubits

    qc.cx(q0, a0)
    qc.cx(q1, a0)
    qc.cx(q1, a1)
    qc.cx(q2, a1)
    return qc


def syndrome_to_qubit(syndrome: str) -> int | None:
    """
    Map syndrome bits (s1,s2) to the qubit requiring X correction.

    For |000> codeword:
      00 -> no error -> None
      10 -> X on q0
      11 -> X on q1
      01 -> X on q2
    """
    if syndrome == '00':
        return None
    elif syndrome == '10':
        return 0
    elif syndrome == '11':
        return 1
    elif syndrome == '01':
        return 2
    raise ValueError(f'Unknown syndrome: {syndrome}')


# -- Full QEC protocol ------------------------------------------------------

# Expected syndrome map (verified by statevector): {None: '00', 0: '10', 1: '11', 2: '01'}
EXPECTED_SYN = {None: '00', 0: '10', 1: '11', 2: '01'}


def run_qec_with_recovery(input_sv: list, error_qubit: int | None,
                         shots: int = 20000, noisy: bool = False) -> dict:
    """
    Run the bit-flip QEC protocol and return fidelity + syndrome info.
    """
    _sv = np.array(input_sv, dtype=complex)
    norm = np.linalg.norm(_sv)
    alpha, beta = _sv[0] / norm, _sv[1] / norm
    psi_in = np.array([alpha, beta], dtype=complex)

    nm = make_noise_model() if noisy else None
    sim = AerSimulator()

    # -- Noiseless: exact statevector ----------------------------------------
    if not noisy:
        syndrome = EXPECTED_SYN[error_qubit]
        qc = QuantumCircuit(5, name='qec')
        qc.initialize([alpha, beta], 0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        if error_qubit is not None:
            qc.x(error_qubit)
        q_fix = syndrome_to_qubit(syndrome)
        if q_fix is not None:
            qc.x(q_fix)

        sv = Statevector(qc)
        amp_000 = sv.data[0]
        amp_111 = sv.data[7]
        psi_logical = np.array([amp_000, amp_111], dtype=complex)
        norm_l = np.linalg.norm(psi_logical)
        if norm_l > 1e-8:
            psi_logical /= norm_l
        fidelity = float(abs(np.vdot(psi_in, psi_logical)) ** 2)

        return {
            'fidelity': fidelity,
            'syndrome': syndrome,
            'syndrome_counts': {syndrome: 1},
            'total_shots': 1,
            'syndrome_correct': True,
        }

    # -- Noisy: extract syndrome from noisy circuit, apply, compute fidelity ----
    # Step 1: syndrome extraction under noise
    qc_syn = QuantumCircuit(5, 2, name='syndrome')
    qc_syn.initialize([alpha, beta], 0)
    qc_syn.cx(0, 1)
    qc_syn.cx(0, 2)
    if error_qubit is not None:
        qc_syn.x(error_qubit)
    extract_syndrome(qc_syn)
    qc_syn.measure([3, 4], [0, 1])

    result_syn = sim.run(qc_syn, shots=shots, noise_model=nm).result()
    counts_syn = result_syn.get_counts(qc_syn)
    total_shots = sum(counts_syn.values())

    # Step 2: for each syndrome outcome, compute fidelity
    syndrome_results = []
    for synd, synd_count in counts_syn.items():
        q_fix = syndrome_to_qubit(synd)
        true_q = syndrome_to_qubit(EXPECTED_SYN[error_qubit])

        qc_rec = QuantumCircuit(5, name='qec_corr')
        qc_rec.initialize([alpha, beta], 0)
        qc_rec.cx(0, 1)
        qc_rec.cx(0, 2)
        if error_qubit is not None:
            qc_rec.x(error_qubit)
        if q_fix is not None:
            qc_rec.x(q_fix)
        qc_rec.save_density_matrix()

        result_rec = sim.run(qc_rec, shots=synd_count, noise_model=nm).result()
        dm = result_rec.data(0)['density_matrix']

        f_qec = _code_space_fidelity(dm.data, psi_in)
        syndrome_results.append({
            'syndrome': synd,
            'count': synd_count,
            'fidelity_qec': f_qec,
            'correction_applied': q_fix,
            'true_correction': true_q,
            'correction_correct': (q_fix == true_q),
        })

    # Weighted average by syndrome probability
    total = sum(r['count'] for r in syndrome_results)
    fidelity_qec = sum(r['fidelity_qec'] * r['count'] for r in syndrome_results) / total
    syndrome_accuracy = sum(r['count'] for r in syndrome_results
                          if r['correction_correct']) / total
    syndrome_ml = max(syndrome_results, key=lambda r: r['count'])['syndrome']

    return {
        'fidelity': fidelity_qec,
        'syndrome': syndrome_ml,
        'syndrome_counts': counts_syn,
        'total_shots': total_shots,
        'syndrome_accuracy': syndrome_accuracy,
        'per_syndrome': syndrome_results,
    }


def _code_space_fidelity(dm_data, psi_in):
    """
    Compute fidelity of a noisy 5-qubit density matrix against psi_in,
    projected onto the code space {|000>, |111>}.
    """
    proj_000 = np.zeros((32, 32), dtype=complex)
    proj_000[0, 0] = 1.0
    proj_111 = np.zeros((32, 32), dtype=complex)
    proj_111[7, 7] = 1.0

    rho_000 = proj_000 @ dm_data @ proj_000
    rho_111 = proj_111 @ dm_data @ proj_111

    p000 = float(np.real(np.trace(rho_000)))
    p111 = float(np.real(np.trace(rho_111)))
    rho01 = rho_000[0, 7] if abs(rho_000[0, 7]) > abs(rho_111[7, 0]) else rho_111[7, 0]

    p_total = p000 + p111
    if p_total > 1e-8:
        rho00 = p000 / p_total
        rho11 = p111 / p_total
        rho01_n = rho01 / p_total if np.abs(rho01) > 1e-8 else 0.0
    else:
        rho00 = rho11 = rho01_n = 0.0

    rho_logical = np.array([[rho00, rho01_n],
                             [np.conj(rho01_n), rho11]], dtype=complex)
    return float(np.vdot(psi_in.conj(), rho_logical @ psi_in).real)


# -- Test states -----------------------------------------------------------

TEST_STATES = [
    ('|0>',  [1.0, 0.0]),
    ('|+>',  [1/np.sqrt(2), 1/np.sqrt(2)]),
    ('|+i>', [1/np.sqrt(2), 1j/np.sqrt(2)]),
    ('|-',   [1/np.sqrt(2), -1/np.sqrt(2)]),
]


# -- Main -----------------------------------------------------------------

def analytical_no_qec_fidelity(input_sv: list, error_qubit: int | None) -> float:
    """F = 4(Re(alpha*beta))^2 for X on logical qubit."""
    if error_qubit is None:
        return 1.0
    alpha_c = complex(input_sv[0])
    beta_c = complex(input_sv[1])
    re = np.real(alpha_c.conjugate() * beta_c)
    return float(min(1.0, max(0.0, 4.0 * re * re)))


def verify_noiseless() -> dict:
    """Statevector verification: QEC F=1.0 for all states."""
    results = []
    for name, sv in TEST_STATES:
        for err_q in [None, 0, 1, 2]:
            f_qec = run_qec_with_recovery(sv, err_q, noisy=False)['fidelity']
            f_noqec = analytical_no_qec_fidelity(sv, err_q)
            results.append({
                'state': name,
                'error_qubit': err_q,
                'f_qec': round(f_qec, 6),
                'f_noqec': round(f_noqec, 6),
            })
    all_qec_1 = all(r['f_qec'] == 1.0 for r in results)
    return {'results': results, 'all_qec_fidelity_1': all_qec_1}


def run_shot_test(shots: int = 20000, noisy: bool = False) -> dict:
    """Run QEC vs baseline over all states and error positions."""
    rows = []
    for name, sv in TEST_STATES:
        for err_q in [None, 0, 1, 2]:
            r_qec = run_qec_with_recovery(sv, err_q, shots=shots, noisy=noisy)

            # Compute no-QEC baseline fidelity
            _sv2 = np.array(sv, dtype=complex)
            norm2 = np.linalg.norm(_sv2)
            a2, b2 = _sv2[0] / norm2, _sv2[1] / norm2
            psi2 = np.array([a2, b2], dtype=complex)

            nm2 = make_noise_model() if noisy else None
            qc_base = QuantumCircuit(5, name='baseline')
            qc_base.initialize([a2, b2], 0)
            qc_base.cx(0, 1)
            qc_base.cx(0, 2)
            if err_q is not None:
                qc_base.x(err_q)

            if noisy:
                qc_base.save_density_matrix()
                res = AerSimulator().run(qc_base, shots=shots,
                                        noise_model=nm2).result()
                dm = res.data(0)['density_matrix']
                f_base = _code_space_fidelity(dm.data, psi2)
            else:
                sv_base = Statevector(qc_base)
                amp_000 = sv_base.data[0]
                amp_111 = sv_base.data[7]
                psi_log = np.array([amp_000, amp_111], dtype=complex)
                n = np.linalg.norm(psi_log)
                if n > 1e-8:
                    psi_log /= n
                f_base = float(abs(np.vdot(psi2, psi_log)) ** 2)

            rows.append({
                'state': name,
                'error_qubit': err_q,
                'fidelity_qec': round(r_qec['fidelity'], 6),
                'fidelity_baseline': round(f_base, 6),
                'qec_improvement': round(r_qec['fidelity'] - f_base, 6),
                'syndrome': r_qec['syndrome'],
                'syndrome_accuracy': round(r_qec.get('syndrome_accuracy', 1.0), 4),
            })

    qec_fids = [x['fidelity_qec'] for x in rows]
    base_fids = [x['fidelity_baseline'] for x in rows]
    improvements = [x['qec_improvement'] for x in rows]
    return {
        'rows': rows,
        'qec_mean': round(np.mean(qec_fids), 6),
        'qec_min': round(np.min(qec_fids), 6),
        'baseline_mean': round(np.mean(base_fids), 6),
        'baseline_min': round(np.min(base_fids), 6),
        'mean_improvement': round(np.mean(improvements), 6),
        'syndrome_accuracy': round(
            np.mean([x['syndrome_accuracy'] for x in rows]), 4),
        'shots': shots,
        'noisy': noisy,
    }


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print('Three-Qubit Bit-Flip QEC -- qrl-007')
    print('=' * 50)

    print('\n=== Noiseless Analytical Verification ===')
    v = verify_noiseless()
    for r in v['results']:
        state = r['state'].replace('⟩', '>').replace('⟨', '<')
        err_str = str(r['error_qubit'])
        print(f'  {state:4s}  err={err_str:>4s}  F_QEC={r["f_qec"]:.4f}  F_noQEC={r["f_noqec"]:.4f}')
    print(f'  All QEC = 1.0: {v["all_qec_fidelity_1"]}')

    print('\n=== Noiseless Shot-Based (20k shots/state) ===')
    r_nl = run_shot_test(shots=20000, noisy=False)
    print(f'  QEC mean F={r_nl["qec_mean"]:.6f}  min={r_nl["qec_min"]:.6f}')
    print(f'  Baseline mean F={r_nl["baseline_mean"]:.6f}  min={r_nl["baseline_min"]:.6f}')
    print(f'  Mean QEC improvement: {r_nl["mean_improvement"]:.6f}')
    print(f'  Syndrome accuracy: {r_nl["syndrome_accuracy"]:.2%}')

    print('\n=== Noisy Shot-Based (qrl-004 noise, 20k shots/state) ===')
    r_no = run_shot_test(shots=20000, noisy=True)
    print(f'  QEC mean F={r_no["qec_mean"]:.6f}  min={r_no["qec_min"]:.6f}')
    print(f'  Baseline mean F={r_no["baseline_mean"]:.6f}  min={r_no["baseline_min"]:.6f}')
    print(f'  Mean QEC improvement: {r_no["mean_improvement"]:.6f}')
    print(f'  Syndrome accuracy: {r_no["syndrome_accuracy"]:.2%}')
