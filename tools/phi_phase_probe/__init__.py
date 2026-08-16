"""
tools/phi_phase_probe
=====================
Empirical probe: do φ-phase-encoded quantum circuits produce measurably
different output distributions than uniform-phase controls?

Falsifiable claim: if φ-phase circuits produce measurably different output
distributions from uniform-phase controls in phase-sensitive tasks, that's a
real signal worth characterizing. If they don't, the null holds.

Design:
  - Encode φ's continued-fraction convergents as RZ gate sequences
  - Compare output TVD vs uniform-phase null across real hardware or AerSimulator
  - Uses quantum_jobs.db for backend selection and job storage
  - No consciousness / biomimetic narrative — pure measurement + statistics
"""

from .circuit import PHI, CONVERGENTS, build_phi_circuit, build_control_circuit

__all__ = ["PHI", "CONVERGENTS", "build_phi_circuit", "build_control_circuit"]
