"""
TMT Benchmark Adapters — Phase 4.

Provides adapters for external benchmark suites and exposes them via API:
- tau_bench: τ-bench adapter (agent policy-following, multi-turn dialogue)
- swebench: SWE-bench adapter (software engineering task resolution)

Each adapter:
1. Fetches tasks from the benchmark format
2. Translates them into TMT AgentContract form
3. Runs them through AgentOrchestrator
4. Returns results in the benchmark's expected schema
"""

from .tau_bench import TauBenchAdapter, TAU_BENCH_TASK_TYPES
from .swebench import SWEBenchAdapter, SWE_BENCH_TASK_TYPES

__all__ = [
    "TauBenchAdapter",
    "TAU_BENCH_TASK_TYPES",
    "SWEBenchAdapter",
    "SWE_BENCH_TASK_TYPES",
]
