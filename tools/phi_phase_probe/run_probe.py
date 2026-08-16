"""
tools/phi_phase_probe/run_probe.py
=================================
CLI entry point for the phi-phase probe.

Usage:
  python tools/phi_phase_probe/run_probe.py --num-qubits 4 --shots 4096 --backend aer_simulator
  python tools/phi_phase_probe/run_probe.py --num-qubits 4 --shots 8192 --backend ibm_fez
  python tools/phi_phase_probe/run_probe.py --num-qubits 5 --shots 4096 --backend aer_simulator --output phi_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


def main() -> int:
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Run phi-phase probe: phi-encoding circuits vs uniform-phase control"
    )
    parser.add_argument(
        "--num-qubits", "-n", type=int, default=4,
        help="Number of qubits (3–8 recommended; default: 4)"
    )
    parser.add_argument(
        "--shots", "-s", type=int, default=4096,
        help="Shots per circuit (default: 4096)"
    )
    parser.add_argument(
        "--backend", "-b",
        default="aer_simulator",
        choices=["aer_simulator", "ibm_fez", "ibm_marrakesh", "ibm_torino"],
        help="Backend: aer_simulator (local, no credentials) or IBM Quantum hardware"
    )
    parser.add_argument(
        "--convergents", "-c", type=str, default="0-8",
        help="Which convergents to test (comma-separated or range, e.g. '0,1,2' or '0-4'; default: all 0-8)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON path (default: prints to stdout)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for permutation test (default: 42)"
    )
    args = parser.parse_args()

    # Parse convergents
    if args.convergents == "all":
        convergent_indices = list(range(9))
    elif "-" in args.convergents:
        parts = args.convergents.split("-")
        start = int(parts[0])
        end = int(parts[-1])
        convergent_indices = list(range(start, end + 1))
    else:
        convergent_indices = [int(x) for x in args.convergents.split(",")]

    print(f"phi-phase probe: {args.num_qubits} qubits, {args.shots} shots, backend={args.backend}")
    print(f"Testing convergents: {convergent_indices}")
    print()

    try:
        from tools.phi_phase_probe.executor import run_probe
    except ImportError:
        # Fallback: adjust path for script-in-directory invocation
        import os
        _sdk = Path(__file__).resolve().parents[1]
        if str(_sdk) not in sys.path:
            sys.path.insert(0, str(_sdk))
        from tools.phi_phase_probe.executor import run_probe

    result = run_probe(
        num_qubits=args.num_qubits,
        shots=args.shots,
        backend=args.backend,
        convergent_indices=convergent_indices,
        seed=args.seed,
    )

    # Print summary
    print(f"\n=== phi-phase probe results ===")
    print(f"Backend: {result['backend']}")
    print(f"Circuits run: {result['summary']['n_phi_circuits']} phi + {result['summary']['n_control_circuits']} control")
    print(f"phi TVD mean vs uniform: {result['summary']['phi_tvd_mean']:.4f} +/- {result['summary']['phi_tvd_std']:.4f}")
    print(f"Control TVD vs uniform:  {result['summary']['control_tvd']:.4f}")
    pt = result["permutation_test"]
    print(f"Permutation test p-value: {pt.get('p_value', 'N/A')}")
    print(f"Conclusion: {pt.get('conclusion', 'N/A')}")

    # Per-circuit table
    print(f"\n--- Per-circuit TVD vs uniform ---")
    for c in result["circuits"]:
        cv = c.get("convergent")
        label = f"phi {cv[0]}/{cv[1]}" if cv else "control"
        print(f"  {label:12s}  TVD={c['tvd_vs_uniform']:.4f}  shots={c['shots']}")

    result["finished_at"] = datetime.now(timezone.utc).isoformat()

    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to: {output_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
