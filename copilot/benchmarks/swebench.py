"""
SWE-bench Adapter for TMT Copilot.

SWE-bench (Software Engineering Benchmark) evaluates LLMs on real GitHub issues
from popular Python repositories. Each instance contains:
- repo: the repository identifier
- version: version or commit
- problem_statement: the GitHub issue description
- failing_tests: test cases that should pass after a fix
- patch: the reference gold patch (not visible to the agent)

Format (from https://github/princeton-nlp/SWE-bench):
{
  "instance_id": "django__django-11099",
  "repo": "django/django",
  "version": "main",
  "problem_statement": "...",
  "hints_text": "...",
  "file_output": [...],
  "image_name": "...",
  "FAIL_TO_PASS": ["tests/..."],
  "PASS_TO_PASS": ["tests/..."],
}

TMT Adapter:
- Maps each SWE-bench instance → AgentContract with task_type="analysis"
- The Synthesizer/Strategic agents handle the planning step
- The Validator agent evaluates whether the approach is sound
- Produces a "swe_result" with: instance_id, problem_summary, approach,
  agents_involved, success_likelihood (0-1), and reasoning

For local use without the full SWE-bench dataset, the adapter generates
synthetic SWE-style tasks that exercise the same planning/diagnostic patterns.
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

# Task type for SWE-bench tasks — maps to analysis/strategic routing
SWE_BENCH_TASK_TYPES = {
    "debugging": "analysis",
    "feature_implementation": "analysis",
    "refactoring": "analysis",
    "test_generation": "analysis",
    "performance": "analysis",
}


@dataclass
class SWETask:
    """A SWE-bench task instance."""
    instance_id: str
    repo: str
    version: str
    problem_statement: str
    failing_tests: list[str] = field(default_factory=list)
    pass_tests: list[str] = field(default_factory=list)
    hints_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task_type(self) -> str:
        """Infer task type from problem statement content."""
        text = self.problem_statement.lower()
        if any(k in text for k in ["bug", "crash", "error", "fail", "wrong"]):
            return "debugging"
        if any(k in text for k in ["implement", "add", "new feature", "support"]):
            return "feature_implementation"
        if any(k in text for k in ["refactor", "restructure", "clean up"]):
            return "refactoring"
        if any(k in text for k in ["performance", "slow", "optimize", "bottleneck"]):
            return "performance"
        return "debugging"

    def to_contract(self) -> dict:
        """Serialize into an AgentContract-like dict."""
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "version": self.version,
            "objective": self.problem_statement,
            "task_type": self.task_type,
            "failing_tests": self.failing_tests,
            "context": {
                "pass_tests": self.pass_tests,
                "hints": self.hints_text,
            },
        }


@dataclass
class SWEResult:
    """Result of evaluating a SWE-bench task."""
    instance_id: str
    repo: str
    task_type: str
    problem_summary: str  # TMT's summary of the problem
    approach: str  # TMT's proposed approach
    agents_involved: list[str]
    success_likelihood: float  # 0.0–1.0
    reasoning: str

    duration_ms: float
    trace_id: str

    # SWE-bench specific
    predicted_patch: str | None = None  # TMT's predicted fix (if any)
    evaluation_notes: str = ""

    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


class SWEBenchAdapter:
    """
    Adapter that runs SWE-bench tasks through AgentOrchestrator.

    Supports:
    - Loading tasks from a SWE-bench format JSON file
    - Generating synthetic SWE-style tasks for local testing
    - Producing per-task results with problem analysis and approach
    - Returning aggregate scores in SWE-bench compatible schema
    """

    def __init__(
        self,
        orchestrator,  # AgentOrchestrator instance
        data_path: Path | None = None,
    ):
        self.orchestrator = orchestrator
        self.data_path = data_path
        self.tasks: list[SWETask] = []

    # ── Task Loading ──────────────────────────────────────────────────────────

    def load_tasks(self, path: Path | None = None) -> list[SWETask]:
        """
        Load SWE-bench tasks from a JSON file in SWE-bench format.

        Supports both the full SWE-bench format (with FAIL_TO_PASS etc.)
        and the Lite format.
        """
        p = path or self.data_path
        if p is None:
            return []

        with open(p, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        items = raw if isinstance(raw, list) else [raw]
        tasks = []
        for item in items:
            tasks.append(SWETask(
                instance_id=item["instance_id"],
                repo=item.get("repo", "unknown"),
                version=item.get("version", "main"),
                problem_statement=item.get("problem_statement", item.get("description", "")),
                failing_tests=item.get("FAIL_TO_PASS", item.get("fail_to_pass", [])),
                pass_tests=item.get("PASS_TO_PASS", item.get("pass_to_pass", [])),
                hints_text=item.get("hints_text", ""),
                metadata=dict(item),
            ))

        self.tasks = tasks
        return tasks

    # ── Synthetic Task Generation ───────────────────────────────────────────

    def generate_synthetic_tasks(self, count: int = 10) -> list[SWETask]:
        """
        Generate synthetic SWE-style tasks for local testing.

        These exercise the same analysis/planning patterns as real SWE-bench
        without requiring the external dataset.
        """
        templates = [
            {
                "instance_id": "swe-synthetic-{:03d}",
                "repo": "example/repo",
                "task_type": "debugging",
                "problem": (
                    "Training loop crashes with IndexError after epoch 5. "
                    "Stack trace shows the error originates in the attention mask computation. "
                    "The issue only appears when batch_size > 16 and seq_len > 128."
                ),
                "failing": ["tests/test_attention.py::test_mask_forward"],
            },
            {
                "instance_id": "swe-synthetic-{:03d}",
                "repo": "example/repo",
                "task_type": "feature_implementation",
                "problem": (
                    "Need to add gradient checkpointing to reduce memory usage "
                    "during training. The model currently OOMs on 24GB GPUs "
                    "when batch_size >= 32 and seq_len >= 512."
                ),
                "failing": ["tests/test_checkpoint.py::test_gradient_checkpointing"],
            },
            {
                "instance_id": "swe-synthetic-{:03d}",
                "repo": "example/repo",
                "task_type": "performance",
                "problem": (
                    "Loss curve shows oscillating validation loss after step 1000. "
                    "Learning rate schedule appears correct. Suspect gradient "
                    "accumulation is interacting badly with the warmup schedule."
                ),
                "failing": ["tests/test_scheduler.py::test_warmup_stability"],
            },
            {
                "instance_id": "swe-synthetic-{:03d}",
                "repo": "example/repo",
                "task_type": "refactoring",
                "problem": (
                    "The router in multi_head_attention.py has 847 lines and handles "
                    "3 distinct concerns: routing, masking, and output projection. "
                    "Need to extract the router logic into a dedicated module."
                ),
                "failing": ["tests/test_router.py::test_router_interface"],
            },
        ]

        tasks = []
        for i in range(count):
            tmpl = templates[i % len(templates)]
            tasks.append(SWETask(
                instance_id=tmpl["instance_id"].format(i + 1),
                repo=tmpl["repo"],
                version="main",
                problem_statement=tmpl["problem"],
                failing_tests=tmpl["failing"],
                pass_tests=["tests/test_basic.py::test_model_init"],
                metadata={"synthetic": True},
            ))

        self.tasks = tasks
        return tasks

    # ── Evaluation ──────────────────────────────────────────────────────────

    def evaluate_task(self, task: SWETask) -> SWEResult:
        """
        Run a single SWE-bench task through the orchestrator.

        The task is dispatched as "analysis" routing to the Strategic/Synthesizer
        agents, which plan an approach. The Validator then evaluates soundness.
        Returns a structured SWEResult with problem summary, approach, and success likelihood.
        """
        from copilot.orchestration import AgentContract, AgentInputSchema

        start = time.time()
        trace_id = str(uuid4())

        # Build contract — use analysis routing
        contract = AgentContract(
            input=AgentInputSchema(
                objective=task.problem_statement,
                task_type="analysis",
                context={
                    "instance_id": task.instance_id,
                    "repo": task.repo,
                    "task_type": task.task_type,
                    "failing_tests": task.failing_tests,
                    "pass_tests": task.pass_tests,
                    "hints": task.hints_text,
                },
            )
        )

        try:
            result = self.orchestrator.execute(
                task_type="analysis",
                objective=task.problem_statement,
                context=dict(contract.input),
            )

            # Extract agents involved from the trace
            agents_involved = list({
                getattr(d, "primary_agent", None)
                for d in getattr(result, "decisions", [])
            })
            agents_involved = [a for a in agents_involved if a]

            # Build problem summary (from orchestrator output)
            problem_summary = self._summarize_problem(task.problem_statement)

            # Build approach from contract
            approach = self._extract_approach(result)

            # Estimate success likelihood based on routing quality
            success_likelihood = self._estimate_success(
                task=task,
                agents_involved=agents_involved,
                result=result,
            )

            reasoning = (
                f"TMT routed to {agents_involved or 'synthesizer'} for this "
                f"{task.task_type} task. "
                f"Confidence: {result.final_confidence:.2f}. "
                f"Problem involves {len(task.failing_tests)} failing test(s). "
                f"Task complexity: {'high' if len(task.failing_tests) > 2 else 'medium'}."
            )

            return SWEResult(
                instance_id=task.instance_id,
                repo=task.repo,
                task_type=task.task_type,
                problem_summary=problem_summary,
                approach=approach,
                agents_involved=agents_involved,
                success_likelihood=success_likelihood,
                reasoning=reasoning,
                duration_ms=(time.time() - start) * 1000,
                trace_id=trace_id,
            )

        except Exception as exc:
            return SWEResult(
                instance_id=task.instance_id,
                repo=task.repo,
                task_type=task.task_type,
                problem_summary=f"ERROR: {str(exc)[:200]}",
                approach="",
                agents_involved=[],
                success_likelihood=0.0,
                reasoning=f"Orchestrator error: {str(exc)[:300]}",
                duration_ms=(time.time() - start) * 1000,
                trace_id=trace_id,
                evaluation_notes="Task failed during execution",
            )

    def _summarize_problem(self, statement: str) -> str:
        """Extract a brief problem summary from the statement."""
        # Strip markdown/code block markers
        text = re.sub(r"```[\s\S]*?```", "", statement)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"#+\s*", "", text)
        text = re.sub(r"\n+", " ", text).strip()
        return text[:300] + ("…" if len(text) > 300 else "")

    def _extract_approach(self, result) -> str:
        """Extract the proposed approach from the orchestrator result."""
        try:
            decisions = getattr(result, "decisions", [])
            if decisions:
                last = decisions[-1]
                directive = getattr(last, "directive", None) or getattr(last, "summary", "")
                return str(directive)[:500]
        except Exception:
            pass
        return "Approach determined by orchestrator routing"

    def _estimate_success(
        self,
        task: SWETask,
        agents_involved: list[str],
        result,
    ) -> float:
        """
        Estimate how likely TMT is to succeed on this SWE-bench instance.

        Based on:
        - Number of failing tests (fewer = easier)
        - Whether the right agents were involved
        - Confidence score from orchestrator
        """
        base = 0.5

        # Fewer failing tests = easier
        if len(task.failing_tests) == 1:
            base += 0.15
        elif len(task.failing_tests) > 3:
            base -= 0.15

        # Strategic agent involved = better planning
        if any("strategic" in a.lower() for a in agents_involved):
            base += 0.10
        if any("synthesizer" in a.lower() for a in agents_involved):
            base += 0.05

        # Confidence from orchestrator
        conf = getattr(result, "final_confidence", 0.5)
        base = base * 0.5 + conf * 0.5

        return min(max(base, 0.0), 1.0)

    # ── Run Suite ──────────────────────────────────────────────────────────

    def run_suite(
        self,
        tasks: list[SWETask] | None = None,
        max_tasks: int | None = None,
    ) -> dict[str, Any]:
        """
        Run a full SWE-bench suite and return results.

        Returns {
            benchmark: "swebench",
            total_tasks: int,
            avg_success_likelihood: float,
            results: [SWEResult, ...]
        }
        """
        task_list = tasks or self.tasks
        if max_tasks:
            task_list = task_list[:max_tasks]

        results = []
        for task in task_list:
            results.append(self.evaluate_task(task))

        avg_likelihood = (
            sum(r.success_likelihood for r in results) / len(results)
            if results else 0.0
        )

        return {
            "benchmark": "swebench",
            "total_tasks": len(results),
            "avg_success_likelihood": round(avg_likelihood, 4),
            "total_duration_ms": sum(r.duration_ms for r in results),
            "results": [
                {
                    "instance_id": r.instance_id,
                    "repo": r.repo,
                    "task_type": r.task_type,
                    "problem_summary": r.problem_summary,
                    "approach": r.approach,
                    "agents_involved": r.agents_involved,
                    "success_likelihood": round(r.success_likelihood, 4),
                    "reasoning": r.reasoning,
                    "duration_ms": round(r.duration_ms, 1),
                    "trace_id": r.trace_id,
                }
                for r in results
            ],
        }
