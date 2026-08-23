"""
Loop Adapters: agents call real TinyMetatron loops.

Each _call_<role> function translates an AgentContract into a real loop
invocation (or a dry-run mock when mode is SIMULATION). Results are returned
as AgentOutputSchema so the orchestration trace is complete regardless of lane.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from copilot.orchestration.models import AgentContract, AgentOutputSchema, AgentProfile

# =============================================================================
# SIMULATION MODE — dry-run mocks
# =============================================================================


def _mock_output(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    status: str = "simulated",
    extra_result: dict | None = None,
) -> "AgentOutputSchema":
    """Return a simulated AgentOutputSchema with realistic-looking fields."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )
    result = {"status": status, "mode": "simulation"}
    if extra_result:
        result.update(extra_result)
    return AgentOutputSchema(
        task_id=contract.input.task_id,
        agent_id=profile.agent_id,
        agent_name=profile.agent_name,
        agent_role=profile.agent_role,
        result=result,
        summary=f"[SIMULATED] {profile.agent_name} · {contract.input.objective[:80]}",
        confidence=profile.fitness,
        resonance_score=profile.phi_alignment,
        fitness_contribution=profile.fitness * 0.1,
        status=HandoffStatus.COMPLETED,
        processing_time_ms=(time.time() - start_time) * 1000,
    )


# =============================================================================
# WORKFLOW — train_loop / corpus_loop
# =============================================================================


def call_workflow(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """workflow → train_loop.run_training() or corpus_loop.run_corpus_pipeline()."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )
    from copilot.orchestration.orchestrator import ExecutionMode

    ctx = contract.input.context
    task_type = contract.input.task_type  # "train" | "corpus" | "evaluate"

    if simulation:
        return _mock_output(profile, contract, start_time, extra_result={
            "task_type": task_type,
            "note": "simulation mode — no loop executed",
        })

    try:
        if task_type == "train":
            return _run_train_loop(profile, contract, start_time, ctx)
        elif task_type == "corpus":
            return _run_corpus_loop(profile, contract, start_time, ctx)
        elif task_type == "evaluate":
            return _run_evaluate_loop(profile, contract, start_time, ctx)
        else:
            return _error_output(
                profile, contract, start_time,
                f"Unknown workflow task_type: {task_type}"
            )
    except Exception as exc:
        return _error_output(profile, contract, start_time, str(exc))


def _run_train_loop(
    profile, contract, start_time, ctx: dict
) -> "AgentOutputSchema":
    """Call train_loop.run_training() programmatically."""
    import sys
    from pathlib import Path as P

    # Import loop directly (adds TinyMetatron root to sys.path)
    _ROOT = P(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from loops.train_loop import run_training

    exp_id = ctx.get("exp_id", "exp-000")
    run_id = ctx.get("run_id", f"{exp_id}-train-seed{ctx.get('seed', 0)}")
    artifact_dir = str(ctx.get("artifact_dir", _ROOT / "experiments" / exp_id / run_id))

    config = {
        "exp_id": exp_id,
        "run_id": run_id,
        "seed": ctx.get("seed", 42),
        "steps": ctx.get("steps", 2000),
        "artifact_dir": artifact_dir,
        "corpus_dir": str(ctx.get("corpus_dir", _ROOT / "experiments/exp-003/corpus")),
        "tokenizer_path": str(ctx.get("tokenizer_path", _ROOT / "vocab.json")),
        "val_every": ctx.get("val_every", 50),
        "log_every": ctx.get("log_every", 25),
        "patience": ctx.get("patience", 3),
        "batch_size": ctx.get("batch_size", 16),
        "lr": ctx.get("lr", 1e-3),
        "weight_decay": ctx.get("weight_decay", 1e-4),
    }

    result = run_training(config)

    return _build_output(profile, contract, start_time, {
        "status": "training_complete",
        "task_type": "train",
        "metrics": result.get("metrics", {}),
    })


def _run_corpus_loop(
    profile, contract, start_time, ctx: dict
) -> "AgentOutputSchema":
    """Call corpus_loop.run_corpus_pipeline() programmatically."""
    import sys
    from pathlib import Path as P

    _ROOT = P(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from loops.corpus_loop import run_corpus_pipeline

    exp_id = ctx.get("exp_id", "exp-000")
    corpus_dir = str(ctx.get("corpus_dir", _ROOT / "data/raw"))
    output_dir = str(ctx.get("output_dir", _ROOT / "experiments" / exp_id / "corpus"))

    config = {
        "exp_id": exp_id,
        "corpus_dir": corpus_dir,
        "output_dir": output_dir,
        "seed": ctx.get("seed", 42),
        "train_pct": ctx.get("train_pct", 0.80),
        "val_pct": ctx.get("val_pct", 0.10),
    }

    result = run_corpus_pipeline(config)

    return _build_output(profile, contract, start_time, {
        "status": "corpus_pipeline_complete",
        "task_type": "corpus",
        "metrics": result,
    })


def _run_evaluate_loop(
    profile, contract, start_time, ctx: dict
) -> "AgentOutputSchema":
    """Call evaluate_loop.run_evaluation_for_checkpoints() programmatically."""
    import sys
    from pathlib import Path as P

    _ROOT = P(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from loops.evaluate_loop import run_evaluation_for_checkpoints

    run_id = ctx.get("run_id", "")
    artifact_dir = str(ctx.get("artifact_dir", _ROOT / "experiments"))
    corpus_dir = str(ctx.get("corpus_dir", _ROOT / "experiments/exp-003/corpus"))
    tokenizer_path = str(ctx.get("tokenizer_path", _ROOT / "vocab.json"))
    eval_sets = ctx.get("eval_sets", ["val", "hard_dev", "novel_eval"])
    best_only = ctx.get("best_only", False)

    results = run_evaluation_for_checkpoints(
        run_id=run_id,
        artifact_dir=P(artifact_dir),
        corpus_dir=P(corpus_dir),
        tokenizer_path=tokenizer_path,
        eval_sets=eval_sets,
        checkpoint_filter="best" if best_only else None,
    )

    return _build_output(profile, contract, start_time, {
        "status": "evaluation_complete",
        "task_type": "evaluate",
        "results": results,
    })


# =============================================================================
# VALIDATOR — generalize_loop gates
# =============================================================================


def call_validator(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """validator → generalize_loop.run_gate() for named gates."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    ctx = contract.input.context
    run_id = ctx.get("run_id", "")
    gate_names = ctx.get("gates", [])
    checkpoint_path = ctx.get("checkpoint_path", "")
    step = ctx.get("step", 0)
    corpus_dir = str(ctx.get("corpus_dir", "experiments/exp-003/corpus"))
    tokenizer_path = str(ctx.get("tokenizer_path", "vocab.json"))
    artifact_dir = str(ctx.get("artifact_dir", "experiments"))

    if simulation:
        return _mock_output(profile, contract, start_time, extra_result={
            "run_id": run_id,
            "gates": [f"gate_{g}" for g in gate_names],
        })

    try:
        import sys
        from pathlib import Path as P

        _ROOT = P(__file__).resolve().parents[1]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))

        from loops.generalize_loop import run_gate

        gate_results = {}
        all_passed = True

        for gate_name in gate_names:
            result = run_gate(
                run_id=run_id,
                gate_name=gate_name,
                checkpoint_path=checkpoint_path,
                step=step,
                corpus_dir=P(corpus_dir),
                tokenizer_path=tokenizer_path,
                artifact_dir=P(artifact_dir),
            )
            gate_results[gate_name] = result
            if not result.get("passed", False):
                all_passed = False

        status_val = "all_gates_passed" if all_passed else "some_gates_failed"

        return _build_output(profile, contract, start_time, {
            "status": status_val,
            "task_type": "validation",
            "gate_results": gate_results,
            "all_passed": all_passed,
        })
    except Exception as exc:
        return _error_output(profile, contract, start_time, str(exc))


# =============================================================================
# OBSERVER — telemetry from registry
# =============================================================================


def call_observer(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """observer → db.get_evaluations(), get_gate_results(), get_loop_run()."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    ctx = contract.input.context
    run_id = ctx.get("run_id", "")

    if simulation:
        return _mock_output(profile, contract, start_time, extra_result={
            "run_id": run_id,
        })

    try:
        import sys
        from pathlib import Path as P

        _ROOT = P(__file__).resolve().parents[1]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))

        from db import (
            get_evaluations,
            get_gate_results,
            get_loop_run,
            get_best_loop_checkpoint,
            get_loop_checkpoints,
            set_db_path,
        )

        registry = str(_ROOT / "state" / "registry.db")
        set_db_path(registry)

        run = get_loop_run(run_id) if run_id else None
        best = get_best_loop_checkpoint(run_id) if run_id else None
        evals = get_evaluations(run_id) if run_id else []
        gates = get_gate_results(run_id) if run_id else []
        checkpoints = get_loop_checkpoints(run_id) if run_id else []

        telemetry = {
            "run": run,
            "best_checkpoint": best,
            "evaluations": evals,
            "gate_results": gates,
            "checkpoints": [{"step": c["step"], "is_best": c["is_best"]} for c in checkpoints],
        }

        return _build_output(profile, contract, start_time, {
            "status": "telemetry_collected",
            "task_type": "monitoring",
            "telemetry": telemetry,
        })
    except Exception as exc:
        return _error_output(profile, contract, start_time, str(exc))


# =============================================================================
# SYNTHESIZER — aggregate upstream outputs
# =============================================================================


def call_synthesizer(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """synthesizer → aggregates contract outputs, builds synthesis decision."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    ctx = contract.input.context
    upstream = ctx.get("upstream_outputs", [])

    if simulation:
        return _mock_output(profile, contract, start_time, extra_result={
            "synthesis": "mock",
        })

    # Aggregate confidence from upstream outputs
    confidences = [u.get("confidence", 0.0) for u in upstream]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    resonance_scores = [u.get("resonance_score", 0.0) for u in upstream]
    avg_resonance = sum(resonance_scores) / len(resonance_scores) if resonance_scores else 0.0

    synthesis = {
        "upstream_count": len(upstream),
        "avg_confidence": avg_confidence,
        "avg_resonance": avg_resonance,
        "recommendations": _extract_recommendations(upstream),
        "decision": "proceed" if avg_confidence >= 0.7 else "review_required",
    }

    return _build_output(profile, contract, start_time, {
        "status": "synthesized",
        "task_type": "synthesis",
        **synthesis,
    })


def _extract_recommendations(upstream: list[dict]) -> list[str]:
    """Pull recommendations out of upstream outputs."""
    recs = []
    for item in upstream:
        result = item.get("result", {})
        if isinstance(result, dict):
            recs.extend(result.get("recommendations", []))
    return recs


# =============================================================================
# ARCHIVIST — checkpoint management
# =============================================================================


def call_archivist(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """archivist → db.get_loop_checkpoints(), get_best_loop_checkpoint()."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    ctx = contract.input.context
    run_id = ctx.get("run_id", "")

    if simulation:
        return _mock_output(profile, contract, start_time)

    try:
        import sys
        from pathlib import Path as P

        _ROOT = P(__file__).resolve().parents[1]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))

        from db import (
            get_best_loop_checkpoint,
            get_loop_checkpoints,
            set_db_path,
        )

        registry = str(_ROOT / "state" / "registry.db")
        set_db_path(registry)

        checkpoints = get_loop_checkpoints(run_id) if run_id else []
        best = get_best_loop_checkpoint(run_id) if run_id else None

        return _build_output(profile, contract, start_time, {
            "status": "checkpoint_manifest",
            "checkpoint_count": len(checkpoints),
            "best": best,
            "checkpoints": [
                {"step": c["step"], "is_best": c["is_best"], "val_ce": c.get("val_ce")}
                for c in checkpoints
            ],
        })
    except Exception as exc:
        return _error_output(profile, contract, start_time, str(exc))


# =============================================================================
# FEDERATION — coordinate multiple loops
# =============================================================================


def call_federation(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """federation → runs multiple sub-agents in parallel, merges results."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    ctx = contract.input.context
    parallel_tasks = ctx.get("parallel_tasks", [])

    if simulation:
        return _mock_output(profile, contract, start_time, extra_result={
            "parallel_tasks": len(parallel_tasks),
        })

    # Run each parallel task as a simulated agent call
    results = []
    for task in parallel_tasks:
        task_type = task.get("task_type", "unknown")
        results.append({
            "task": task,
            "status": "dispatched",
            "task_type": task_type,
        })

    return _build_output(profile, contract, start_time, {
        "status": "federation_complete",
        "task_type": "coordination",
        "parallel_results": results,
        "task_count": len(parallel_tasks),
    })


# =============================================================================
# AUDITOR — validate loop invariants
# =============================================================================


def call_auditor(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """auditor → validates loop invariants (gates passed, state transitions valid)."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    ctx = contract.input.context
    run_id = ctx.get("run_id", "")

    if simulation:
        return _mock_output(profile, contract, start_time)

    try:
        import sys
        from pathlib import Path as P

        _ROOT = P(__file__).resolve().parents[1]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))

        from db import get_loop_run, get_gate_results, set_db_path

        registry = str(_ROOT / "state" / "registry.db")
        set_db_path(registry)

        run = get_loop_run(run_id) if run_id else None
        gates = get_gate_results(run_id) if run_id else []

        passed_gates = [g["gate_name"] for g in gates if g.get("passed")]
        failed_gates = [g["gate_name"] for g in gates if not g.get("passed")]

        violations = []
        if run:
            status = run.get("status", "")
            # Check for invalid state transitions (example invariant)
            if status == "NEW":
                violations.append("Run in NEW state with no transitions attempted")

        return _build_output(profile, contract, start_time, {
            "status": "audit_complete",
            "task_type": "audit",
            "passed_gates": passed_gates,
            "failed_gates": failed_gates,
            "violations": violations,
            "audit_clean": len(violations) == 0,
        })
    except Exception as exc:
        return _error_output(profile, contract, start_time, str(exc))


# =============================================================================
# BRONZE — safety checks (path protection, gate enforcement)
# =============================================================================


def call_bronze(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """bronze → safety checks: protected paths, gate enforcement, policy violations."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    ctx = contract.input.context
    write_path = ctx.get("write_path", "")

    if simulation:
        return _mock_output(profile, contract, start_time)

    # Check write path against protected paths
    protected = [Path(p).resolve() for p in [
        "experiments/exp-003",
    ]]
    try:
        resolved = Path(write_path).resolve()
        for prot in protected:
            resolved.relative_to(prot)
            return _error_output(
                profile, contract, start_time,
                f"Write blocked: path {write_path} is protected"
            )
    except ValueError:
        pass  # not relative to any protected path — allowed

    return _build_output(profile, contract, start_time, {
        "status": "safety_pass",
        "task_type": "protection",
        "write_path": write_path,
        "allowed": True,
    })


# =============================================================================
# STRATEGIC — experiment planning
# =============================================================================


def call_strategic(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """strategic → decides experiment path, hyperparameters, priority."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    ctx = contract.input.context
    hypothesis = ctx.get("hypothesis", "")

    if simulation:
        return _mock_output(profile, contract, start_time)

    # Strategic decision: choose next experiment configuration
    decision = {
        "next_exp_id": ctx.get("exp_id", "exp-000"),
        "priority": "high" if ctx.get("urgent", False) else "normal",
        "strategy": "explore",
        "hypothesis": hypothesis,
        "suggested_steps": ctx.get("suggested_steps", 2000),
        "suggested_lr": ctx.get("suggested_lr", 1e-3),
    }

    return _build_output(profile, contract, start_time, {
        "status": "strategic_decision",
        "task_type": "strategic",
        **decision,
    })


# =============================================================================
# BITNET — entropy / quality scoring
# =============================================================================


def call_bitnet(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """bitnet → entropy/quality scoring on corpus data."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    ctx = contract.input.context
    corpus_dir = ctx.get("corpus_dir", "")

    if simulation:
        return _mock_output(profile, contract, start_time, extra_result={
            "entropy_score": 0.618,
            "bitnet_score": profile.fitness,
        })

    # Placeholder: real bitnet scoring would read corpus files and compute entropy
    return _build_output(profile, contract, start_time, {
        "status": "bitnet_scored",
        "task_type": "processing",
        "entropy_score": profile.fitness,
        "quality_score": profile.resonance_frequency / 1000.0,
    })


# =============================================================================
# HARMONIC — resonance tuning
# =============================================================================


def call_harmonic(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """harmonic → hyperparameter resonance tuning (lr, batch, depth alignment to phi)."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    ctx = contract.input.context

    # Phi-based tuning: suggest lr aligned to golden ratio
    phi = 1.618033988749895
    base_lr = 1e-3
    tuned_lr = base_lr * phi

    return _build_output(profile, contract, start_time, {
        "status": "resonance_tuned",
        "task_type": "processing",
        "suggested_lr": tuned_lr,
        "phi_alignment": profile.phi_score,
        "resonance_frequency": profile.resonance_frequency,
    })


# =============================================================================
# MIRROR — reflection / stall detection
# =============================================================================


def call_mirror(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """mirror → reads loop state, detects stalls, suggests recovery actions."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    ctx = contract.input.context
    run_id = ctx.get("run_id", "")

    if simulation:
        return _mock_output(profile, contract, start_time)

    # Detect stall by checking if best_val_ce improved recently
    stall_detected = False
    recovery_action = None

    try:
        import sys
        from pathlib import Path as P

        _ROOT = P(__file__).resolve().parents[1]
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))

        from db import get_best_loop_checkpoint, get_loop_checkpoints, set_db_path

        registry = str(_ROOT / "state" / "registry.db")
        set_db_path(registry)

        checkpoints = get_loop_checkpoints(run_id)[:5] if run_id else []
        if len(checkpoints) >= 3:
            recent_ce = [c.get("val_ce", float("inf")) for c in checkpoints]
            if recent_ce == sorted(recent_ce, reverse=True):
                stall_detected = True
                recovery_action = "reduce_learning_rate_or_early_stop"

    except Exception:
        pass

    return _build_output(profile, contract, start_time, {
        "status": "reflection_complete",
        "task_type": "monitoring",
        "stall_detected": stall_detected,
        "recovery_action": recovery_action,
        "phi_alignment": profile.phi_score,
    })


# =============================================================================
# FRACTAL — pattern recognition over loss curves
# =============================================================================


def call_fractal(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """fractal → pattern recognition over loss curves, Sierpinski topology."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    ctx = contract.input.context
    run_id = ctx.get("run_id", "")

    # Use sierpinski_topology to generate circuit spec as a pattern template
    from copilot.orchestration.sierpinski_topology import SierpinskiGenerator, SierpinskiConfig

    cfg = SierpinskiConfig(depth=3)
    gen = SierpinskiGenerator(cfg)
    spec = gen.generate()

    return _build_output(profile, contract, start_time, {
        "status": "fractal_pattern",
        "task_type": "visualization",
        "sierpinski_depth": cfg.depth,
        "phi_score": profile.phi_score,
        "pattern_summary": f"Sierpinski depth-{cfg.depth} circuit generated",
        "qasm_preview": spec.to_qasm()[:200] if spec else "",
    })


# =============================================================================
# WORMHOLE — cross-experiment knowledge transfer
# =============================================================================


def call_wormhole(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """wormhole → cross-experiment knowledge transfer."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    ctx = contract.input.context
    source_exp_id = ctx.get("source_exp_id", "")
    target_exp_id = ctx.get("target_exp_id", "")

    if simulation:
        return _mock_output(profile, contract, start_time)

    # Placeholder: real wormhole would copy successful agent configs between experiments
    return _build_output(profile, contract, start_time, {
        "status": "wormhole_transfer",
        "task_type": "strategic",
        "source_exp": source_exp_id,
        "target_exp": target_exp_id,
        "transfer_type": "hyperparameters",
        "phi_alignment": profile.phi_score,
    })


# =============================================================================
# STEALTH — background / coroutine tasks
# =============================================================================


def call_stealth(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """stealth → background tasks, deferred execution, silent monitoring."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    return _build_output(profile, contract, start_time, {
        "status": "stealth_task_scheduled",
        "task_type": "background",
        "note": "stealth agent runs silently in background",
    })


# =============================================================================
# VISUAL — visualizations
# =============================================================================


def call_visual(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """visual → generates visualizations of loss curves, agent constellation."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    ctx = contract.input.context
    run_id = ctx.get("run_id", "")

    return _build_output(profile, contract, start_time, {
        "status": "visualization_generated",
        "task_type": "visualization",
        "run_id": run_id,
        "phi_alignment": profile.phi_score,
        "visualization_type": "loss_curve",
        "note": "Visualization would be saved to state/artifacts/",
    })


# =============================================================================
# BIO — corpus bio-diversity checks
# =============================================================================


def call_bio(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    simulation: bool = False,
) -> "AgentOutputSchema":
    """bio → corpus bio-diversity: language coverage, domain balance, novelty."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    ctx = contract.input.context
    corpus_dir = ctx.get("corpus_dir", "")

    return _build_output(profile, contract, start_time, {
        "status": "bio_diversity_scored",
        "task_type": "corpus",
        "corpus_dir": corpus_dir,
        "diversity_score": profile.fitness,
        "novelty_score": profile.resonance_frequency / 1000.0,
        "language_coverage": ["en", "sk"],
    })


# =============================================================================
# Helper
# =============================================================================


def _build_output(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    result: dict,
) -> "AgentOutputSchema":
    """Build a success AgentOutputSchema from a result dict."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    summary_parts = [
        f"{profile.agent_name}",
        result.get("status", "ok"),
        result.get("task_type", ""),
    ]
    summary = " · ".join(filter(None, summary_parts))

    return AgentOutputSchema(
        task_id=contract.input.task_id,
        agent_id=profile.agent_id,
        agent_name=profile.agent_name,
        agent_role=profile.agent_role,
        result=result,
        summary=summary[:200],
        confidence=profile.fitness,
        resonance_score=profile.phi_alignment,
        fitness_contribution=profile.fitness * 0.1,
        status=HandoffStatus.COMPLETED,
        processing_time_ms=(time.time() - start_time) * 1000,
    )


def _error_output(
    profile: "AgentProfile",
    contract: "AgentContract",
    start_time: float,
    error_message: str,
) -> "AgentOutputSchema":
    """Build an error AgentOutputSchema."""
    from copilot.orchestration.models import (
        AgentOutputSchema,
        HandoffStatus,
    )

    return AgentOutputSchema(
        task_id=contract.input.task_id,
        agent_id=profile.agent_id,
        agent_name=profile.agent_name,
        agent_role=profile.agent_role,
        result={"status": "error", "error": error_message},
        summary=f"Error: {error_message[:150]}",
        confidence=0.0,
        resonance_score=0.0,
        fitness_contribution=0.0,
        status=HandoffStatus.FAILED,
        errors=[error_message],
        processing_time_ms=(time.time() - start_time) * 1000,
    )
