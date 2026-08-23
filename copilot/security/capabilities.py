"""
Capability map — least-privilege per agent role.

Each AgentRole maps to a bounded set of loop/function capabilities.
check_capability() validates a role against a requested action.

This enforces that no agent can call loops outside its designated scope,
limiting blast radius if an agent is compromised or misbehaves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from copilot.types import AgentRole

# ── Capability definitions ─────────────────────────────────────────────────────────

class Capability:
    """A callable capability with optional max values."""

    def __init__(
        self,
        actions: set[str],
        *,
        max_steps: int | None = None,
        max_output_tokens: int | None = None,
        max_time_seconds: float | None = None,
    ) -> None:
        self.actions = actions
        self.max_steps = max_steps
        self.max_output_tokens = max_output_tokens
        self.max_time_seconds = max_time_seconds

    def allows(self, action: str) -> bool:
        return action in self.actions

    def __repr__(self) -> str:
        return (
            f"Capability(actions={sorted(self.actions)!r}, "
            f"max_steps={self.max_steps}, "
            f"max_output_tokens={self.max_output_tokens}, "
            f"max_time_seconds={self.max_time_seconds})"
        )


# Maximum constraints — overrides are capped at these values
MAX_AGENT_STEPS = 100
MAX_OUTPUT_TOKENS = 4096
MAX_AGENT_TIME_SECONDS = 30.0


# ── Per-role capability map ───────────────────────────────────────────────────────

# Format: AgentRole → Capability(actions=set, max_steps, max_output_tokens, max_time_seconds)
#
# Actions are string identifiers matching the loop/function names each role may call.
# Unknown roles default to an empty (no-op) capability.

CAPABILITIES: dict[str, Capability] = {

    # ── Core orchestration ─────────────────────────────────────────────────────

    "validator": Capability(
        actions={
            "generalize_gate",
            "generalize_loop.run_gate",
            "db.get_gate_results",
            "db.get_evaluations",
        },
        max_steps=50,
        max_output_tokens=1024,
        max_time_seconds=15.0,
    ),

    "synthesizer": Capability(
        actions={
            "aggregate",
            "synthesize_decision",
            "db.get_loop_run",
            "db.get_evaluations",
        },
        max_steps=80,
        max_output_tokens=2048,
        max_time_seconds=20.0,
    ),

    "workflow": Capability(
        actions={
            "train_loop.run_training",
            "corpus_loop.run_corpus_pipeline",
            "generalize_gate",
            "db.get_loop_checkpoints",
            "db.get_best_loop_checkpoint",
        },
        max_steps=MAX_AGENT_STEPS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_time_seconds=MAX_AGENT_TIME_SECONDS,
    ),

    "observer": Capability(
        actions={
            "db.get_evaluations",
            "db.get_gate_results",
            "db.get_loop_run",
            "db.get_loop_checkpoints",
            "telemetry_stream",
        },
        max_steps=30,
        max_output_tokens=512,
        max_time_seconds=10.0,
    ),

    # ── Persistence ──────────────────────────────────────────────────────────

    "archivist": Capability(
        actions={
            "db.get_loop_checkpoints",
            "db.get_best_loop_checkpoint",
            "db.get_loop_runs",
            "checkpoint.write",
            "checkpoint.read",
        },
        max_steps=40,
        max_output_tokens=1024,
        max_time_seconds=15.0,
    ),

    "auditor": Capability(
        actions={
            "db.get_loop_runs",
            "db.get_loop_run",
            "db.get_evaluations",
            "audit_invariants",
        },
        max_steps=60,
        max_output_tokens=1536,
        max_time_seconds=20.0,
    ),

    # ── Safety ────────────────────────────────────────────────────────────────

    "bronze": Capability(
        actions={
            "path_protection",
            "gate_enforcement",
            "safety_check",
            "db.get_loop_runs",
        },
        max_steps=20,
        max_output_tokens=512,
        max_time_seconds=8.0,
    ),

    "stealth": Capability(
        actions={
            "background_task",
            "coroutine_schedule",
            "checkpoint.read",
        },
        max_steps=15,
        max_output_tokens=256,
        max_time_seconds=10.0,
    ),

    # ── Intelligence ─────────────────────────────────────────────────────────

    "strategic": Capability(
        actions={
            "plan_experiment",
            "db.get_loop_runs",
            "db.get_best_loop_checkpoint",
            "synthesize_decision",
        },
        max_steps=70,
        max_output_tokens=2048,
        max_time_seconds=25.0,
    ),

    "federation": Capability(
        actions={
            "coordinate_multi_loop",
            "train_loop.run_training",
            "corpus_loop.run_corpus_pipeline",
            "db.get_evaluations",
            "aggregate",
        },
        max_steps=MAX_AGENT_STEPS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_time_seconds=MAX_AGENT_TIME_SECONDS,
    ),

    "bitnet": Capability(
        actions={
            "entropy_score",
            "quality_score",
            "db.get_loop_runs",
            "db.get_evaluations",
        },
        max_steps=40,
        max_output_tokens=1024,
        max_time_seconds=12.0,
    ),

    "harmonic": Capability(
        actions={
            "resonance_tune",
            "hyperparameter_search",
            "db.get_loop_run",
            "db.get_best_loop_checkpoint",
        },
        max_steps=60,
        max_output_tokens=1024,
        max_time_seconds=18.0,
    ),

    "mirror": Capability(
        actions={
            "reflect_state",
            "stall_detection",
            "db.get_loop_run",
            "db.get_loop_checkpoints",
        },
        max_steps=30,
        max_output_tokens=512,
        max_time_seconds=10.0,
    ),

    "fractal": Capability(
        actions={
            "pattern_recognition",
            "loss_curve_analysis",
            "db.get_loop_runs",
            "db.get_evaluations",
        },
        max_steps=50,
        max_output_tokens=1536,
        max_time_seconds=15.0,
    ),

    "wormhole": Capability(
        actions={
            "cross_experiment_transfer",
            "knowledge_retrieval",
            "db.get_loop_runs",
            "db.get_best_loop_checkpoint",
        },
        max_steps=60,
        max_output_tokens=2048,
        max_time_seconds=20.0,
    ),

    # ── Output ───────────────────────────────────────────────────────────────

    "visual": Capability(
        actions={
            "visualize",
            "plot_loss_curve",
            "checkpoint.read",
            "db.get_loop_run",
        },
        max_steps=50,
        max_output_tokens=2048,
        max_time_seconds=15.0,
    ),

    # ── Corpus / bio ─────────────────────────────────────────────────────────

    "bio": Capability(
        actions={
            "bio_diversity_check",
            "corpus_quality_check",
            "db.get_loop_runs",
        },
        max_steps=40,
        max_output_tokens=1024,
        max_time_seconds=12.0,
    ),
}


# ── Validation ─────────────────────────────────────────────────────────────────

class CapabilityError(ValueError):
    """Raised when an agent attempts an action outside its capability."""

    def __init__(self, role: str, action: str, capability: Capability) -> None:
        self.role = role
        self.action = action
        self.capability = capability
        super().__init__(
            f"Role '{role}' attempted unauthorized action '{action}'. "
            f"Allowed actions: {sorted(capability.actions)}"
        )


def check_capability(role: str, action: str) -> None:
    """
    Validate that `role` is permitted to perform `action`.

    Raises CapabilityError if:
      - `role` is unknown (not in CAPABILITIES)
      - `action` is not in the role's allowed set

    Usage at API entry points (invoke agent):
        check_capability(agent_role, "train_loop.run_training")

    Usage inside orchestrator before each loop call:
        check_capability(agent_role, "generalize_gate")
    """
    capability = CAPABILITIES.get(role)
    if capability is None:
        raise CapabilityError(role, action, Capability(actions=set()))

    if not capability.allows(action):
        raise CapabilityError(role, action, capability)

    # Validate numeric constraints
    # (These can be enforced by the orchestrator before calling a loop)
    # The check here is informational — the orchestrator must respect the caps.
    return


def get_role_capability(role: str) -> Capability:
    """Return the Capability for `role`, or an empty Capability if unknown."""
    return CAPABILITIES.get(role, Capability(actions=set()))
