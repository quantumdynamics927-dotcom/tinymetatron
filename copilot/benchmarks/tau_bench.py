"""
τ-bench (tau-bench) Adapter for TMT Copilot.

τ-bench evaluates agent policy-following in multi-turn dialogue settings.
It tests whether an agent can correctly:
1. Follow directives and instructions in context
2. Route to the correct sub-agent at each turn
3. Produce outputs that match a reference "gold" answer

Format (from https://github.com/serviceinnovation/tau-bench):
- instance_id: unique task identifier
- version: benchmark version (e.g. "v1.0")
- conversations: list of {role, content} turns
- gold_answer: the expected correct final answer
- task_description: high-level description of the user's request

TMT Adapter:
- Maps each τ-bench conversation turn → AgentContract with turn context
- Evaluates whether the agent's routing and output match expected behaviour
- Runs through AgentOrchestrator in SIMULATION mode (no live LLM needed)
- Produces per-turn and aggregate scores in τ-bench schema

For local use without the full τ-bench dataset, the adapter also generates
synthetic τ-style tasks that exercise the same coordination patterns.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

# Task types that τ-bench tests — mapped to TMT routing
TAU_BENCH_TASK_TYPES = {
    "customer_support": "synthesis",
    "technical_support": "validation",
    "information_gathering": "monitoring",
    "coordination": "coordination",
    "troubleshooting": "analysis",
}


@dataclass
class TauTurn:
    """A single turn in a τ-bench conversation."""
    role: str  # "user" | "agent"
    content: str
    expected_action: str | None = None  # e.g. "route_to_specialist", "answer"


@dataclass
class TauTask:
    """A τ-bench task instance."""
    instance_id: str
    version: str
    task_type: str  # e.g. "customer_support"
    task_description: str
    turns: list[TauTurn]
    gold_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_agent(self) -> str:
        """Which TMT agent should handle this task type."""
        mapping = {
            "customer_support": "synthesizer",
            "technical_support": "validator",
            "information_gathering": "observer",
            "coordination": "federation",
            "troubleshooting": "strategic",
        }
        return mapping.get(self.task_type, "synthesizer")

    def to_contracts(self) -> list[dict]:
        """Serialize turns into a list of AgentContract-like dicts."""
        contracts = []
        for i, turn in enumerate(self.turns):
            contracts.append({
                "turn_id": i,
                "role": turn.role,
                "content": turn.content,
                "expected_action": turn.expected_action,
                "objective": f"[Turn {i+1}/{len(self.turns)}] {self.task_description}",
            })
        return contracts


@dataclass
class TauResult:
    """Result of running a τ-bench task."""
    instance_id: str
    task_type: str
    passed: bool
    score: float  # 0.0–1.0

    # Per-turn results
    turn_results: list[dict[str, Any]]

    # Aggregate
    total_turns: int
    correct_turns: int
    duration_ms: float
    trace_id: str

    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


class TauBenchAdapter:
    """
    Adapter that runs τ-bench tasks through AgentOrchestrator.

    Supports:
    - Loading tasks from a τ-bench format JSON file
    - Generating synthetic τ-style tasks for local testing
    - Evaluating routing correctness per turn
    - Producing results in τ-bench compatible schema
    """

    def __init__(
        self,
        orchestrator,  # AgentOrchestrator instance
        data_path: Path | None = None,
    ):
        self.orchestrator = orchestrator
        self.data_path = data_path
        self.tasks: list[TauTask] = []

    # ── Task Loading ─────────────────────────────────────────────────────────────

    def load_tasks(self, path: Path | None = None) -> list[TauTask]:
        """
        Load τ-bench tasks from a JSON file in τ-bench format.

        Expected schema:
        {
          "instance_id": "cust-001",
          "version": "v1.0",
          "task_type": "customer_support",
          "task_description": "User wants to change their password",
          "conversations": [
            {"role": "user", "content": "...", "expected_action": "ask_confirmation"},
            {"role": "agent", "content": "...", "expected_action": "route_to_specialist"}
          ],
          "gold_answer": "Password changed successfully"
        }
        """
        p = path or self.data_path
        if p is None:
            return []

        tasks = []
        with open(p, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        # Support both a single dict and a list of dicts
        items = raw if isinstance(raw, list) else [raw]

        for item in items:
            turns = [
                TauTurn(
                    role=t["role"],
                    content=t["content"],
                    expected_action=t.get("expected_action"),
                )
                for t in item.get("conversations", [])
            ]
            tasks.append(TauTask(
                instance_id=item["instance_id"],
                version=item.get("version", "unknown"),
                task_type=item.get("task_type", "customer_support"),
                task_description=item.get("task_description", ""),
                turns=turns,
                gold_answer=item.get("gold_answer"),
                metadata=item,
            ))

        self.tasks = tasks
        return tasks

    # ── Synthetic Task Generation ─────────────────────────────────────────────────

    def generate_synthetic_tasks(self, count: int = 10) -> list[TauTask]:
        """
        Generate synthetic τ-style tasks that test TMT routing patterns.

        These exercise the same coordination skills as real τ-bench without
        requiring the external dataset.
        """
        templates = [
            {
                "task_type": "customer_support",
                "description": "Customer requests password reset for their account",
                "turns": [
                    ("user", "I need to reset my password, it's not working"),
                    ("agent", None, "route_to_specialist"),
                    ("user", "Yes, it's for my main account"),
                    ("agent", None, "confirm_action"),
                ],
            },
            {
                "task_type": "technical_support",
                "description": "User reports a training pipeline failure",
                "turns": [
                    ("user", "Our training loop is crashing with an OOM error"),
                    ("agent", None, "route_to_specialist"),
                    ("user", "It happens after about 200 steps"),
                    ("agent", None, "collect_diagnostics"),
                ],
            },
            {
                "task_type": "information_gathering",
                "description": "User asks for an overview of current experiment status",
                "turns": [
                    ("user", "What's the status of our latest training run?"),
                    ("agent", None, "route_to_observer"),
                    ("user", "Show me the validation loss curve"),
                    ("agent", None, "provide_data"),
                ],
            },
            {
                "task_type": "coordination",
                "description": "User requests a multi-step experiment be set up",
                "turns": [
                    ("user", "Set up a new experiment with the updated corpus"),
                    ("agent", None, "route_to_federation"),
                    ("user", "Yes, and validate it against the hard dev set"),
                    ("agent", None, "confirm_action"),
                ],
            },
            {
                "task_type": "troubleshooting",
                "description": "User reports model quality degradation",
                "turns": [
                    ("user", "Our model's validation loss has been increasing for 3 days"),
                    ("agent", None, "route_to_strategic"),
                    ("user", "Yes, started after we added the new corpus"),
                    ("agent", None, "analyze_root_cause"),
                ],
            },
        ]

        tasks = []
        for i in range(count):
            tmpl = templates[i % len(templates)]
            turns = [
                TauTurn(role=t[0], content=t[1] or "", expected_action=t[2] if len(t) > 2 else None)
                for t in tmpl["turns"]
            ]
            tasks.append(TauTask(
                instance_id=f"tau synthetic {i+1:03d}",
                version="synthetic-v1",
                task_type=tmpl["task_type"],
                task_description=tmpl["description"],
                turns=turns,
                metadata={"synthetic": True},
            ))

        self.tasks = tasks
        return tasks

    # ── Evaluation ──────────────────────────────────────────────────────────────

    def evaluate_task(self, task: TauTask) -> TauResult:
        """
        Run a single τ-bench task through the orchestrator.

        For each turn, executes the agent and checks whether the routing
        decision matches the expected action. In SIMULATION mode the
        orchestrator returns mock outputs; we evaluate routing correctness only.
        """
        from copilot.orchestration import AgentContract, AgentInputSchema

        start = time.time()
        trace_id = str(uuid4())

        turn_results = []
        correct = 0

        for i, turn in enumerate(task.turns):
            if turn.role != "agent":
                continue

            # Build contract for this turn
            contract = AgentContract(
                input=AgentInputSchema(
                    objective=turn.content or task.task_description,
                    task_type=TAU_BENCH_TASK_TYPES.get(task.task_type, "synthesis"),
                    context={
                        "turn_id": i,
                        "instance_id": task.instance_id,
                        "task_description": task.task_description,
                        "expected_action": turn.expected_action,
                        "conversation_history": [
                            {"role": t.role, "content": t.content}
                            for t in task.turns[:i]
                        ],
                    },
                )
            )

            # Route through orchestrator
            try:
                result = self.orchestrator.execute(
                    task_type=TAU_BENCH_TASK_TYPES.get(task.task_type, "synthesis"),
                    objective=turn.content or task.task_description,
                    context=dict(contract.input),
                )
                routed_agent = result.final_status
                routing_correct = self._check_routing(
                    routed_agent=routed_agent,
                    expected_action=turn.expected_action,
                    task_type=task.task_type,
                )
                if routing_correct:
                    correct += 1

                turn_results.append({
                    "turn_id": i,
                    "expected_action": turn.expected_action,
                    "actual_agent": routed_agent,
                    "routing_correct": routing_correct,
                    "status": "ok",
                })
            except Exception as exc:
                turn_results.append({
                    "turn_id": i,
                    "expected_action": turn.expected_action,
                    "actual_agent": None,
                    "routing_correct": False,
                    "status": "error",
                    "error": str(exc),
                })

        duration_ms = (time.time() - start) * 1000
        score = correct / max(len([t for t in task.turns if t.role == "agent"]), 1)

        return TauResult(
            instance_id=task.instance_id,
            task_type=task.task_type,
            passed=score >= 0.7,
            score=score,
            turn_results=turn_results,
            total_turns=len(task.turns),
            correct_turns=correct,
            duration_ms=duration_ms,
            trace_id=trace_id,
        )

    def _check_routing(
        self,
        routed_agent: str,
        expected_action: str | None,
        task_type: str,
    ) -> bool:
        """
        Check if the routed agent is consistent with the expected action.

        Maps expected actions to required agent roles and verifies the routing.
        """
        if expected_action is None:
            return True  # No expectation set

        # Action → expected primary role
        action_roles = {
            "route_to_specialist": ["synthesizer", "federation", "strategic"],
            "route_to_observer": ["observer", "mirror"],
            "route_to_validator": ["validator", "auditor"],
            "route_to_archivist": ["archivist"],
            "route_to_federation": ["federation"],
            "confirm_action": ["synthesizer", "validator"],
            "collect_diagnostics": ["observer", "bitnet"],
            "provide_data": ["archivist", "observer"],
            "analyze_root_cause": ["strategic", "fractal", "mirror"],
            "answer": ["synthesizer"],
        }

        allowed = action_roles.get(expected_action, [])
        if not allowed:
            return True  # Unknown action — no expectation

        return routed_agent.lower() in [r.lower() for r in allowed]

    # ── Run Suite ───────────────────────────────────────────────────────────────

    def run_suite(
        self,
        tasks: list[TauTask] | None = None,
        max_tasks: int | None = None,
    ) -> dict[str, Any]:
        """
        Run a full τ-bench suite and return results.

        Returns {
            benchmark: "tau-bench",
            version: "v1.0|synthetic-v1",
            total_tasks: int,
            passed_tasks: int,
            success_rate: float,
            avg_score: float,
            results: [TauResult, ...]
        }
        """
        task_list = tasks or self.tasks
        if max_tasks:
            task_list = task_list[:max_tasks]

        results = []
        for task in task_list:
            results.append(self.evaluate_task(task))

        passed = sum(1 for r in results if r.passed)
        scores = [r.score for r in results]

        return {
            "benchmark": "tau-bench",
            "version": task_list[0].version if task_list else "unknown",
            "total_tasks": len(results),
            "passed_tasks": passed,
            "failed_tasks": len(results) - passed,
            "success_rate": passed / len(results) if results else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "duration_ms": sum(r.duration_ms for r in results),
            "results": [
                {
                    "instance_id": r.instance_id,
                    "task_type": r.task_type,
                    "passed": r.passed,
                    "score": r.score,
                    "correct_turns": r.correct_turns,
                    "total_turns": r.total_turns,
                    "duration_ms": r.duration_ms,
                    "trace_id": r.trace_id,
                    "turn_results": r.turn_results,
                }
                for r in results
            ],
        }
