"""
Copilot Orchestration Layer for TinyMetatron.

Multi-agent coordination system with 17 specialized agents, routing engine,
execution planner, conflict resolution, and coordination metrics.

Usage:
    from copilot import AgentOrchestrator

    orchestrator = AgentOrchestrator(vault_path=Path("copilot/agents"))
    trace = orchestrator.execute(
        task_type="validation",
        objective="Validate system integrity",
    )
"""

from copilot.orchestration import (
    AgentOrchestrator,
    AgentProfile,
    ExecutionPlanner,
    ExecutionStage,
    ExecutionPlan,
    HandoffManager,
    RoutingEngine,
    AgentBus,
    AgentChannel,
    ChannelRegistry,
    CoordinationMetrics,
    CoordinationTrace,
    RoutingPolicy,
    RoutingDecision,
    AgentLayer,
    AgentRole,
    AgentContract,
    AgentInputSchema,
    AgentOutputSchema,
    AgentMessage,
)

__all__ = [
    "AgentOrchestrator",
    "AgentProfile",
    "ExecutionPlanner",
    "ExecutionStage",
    "ExecutionPlan",
    "HandoffManager",
    "RoutingEngine",
    "AgentBus",
    "AgentChannel",
    "ChannelRegistry",
    "CoordinationMetrics",
    "CoordinationTrace",
    "RoutingPolicy",
    "RoutingDecision",
    "AgentLayer",
    "AgentRole",
    "AgentContract",
    "AgentInputSchema",
    "AgentOutputSchema",
    "AgentMessage",
]
