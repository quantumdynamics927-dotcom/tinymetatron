"""
Human-in-the-Loop Approval Gate — P2.

Defines HIGH_RISK_ACTIONS that require explicit human approval before execution.
Manages a pending-approvals store and provides the /copilot/agents/{id}/approve
endpoint for callers to grant or deny approval.

Usage in orchestrator or api.py before dispatching a high-risk action:
    from copilot.orchestration.approval import HIGH_RISK_ACTIONS, ApprovalManager

    if action in HIGH_RISK_ACTIONS:
        approval = ApprovalManager.request(profile.agent_id, trace_id, action)
        if approval.status == "pending":
            return {"status": "pending_approval", "approval_id": approval.approval_id}
        if approval.status == "denied":
            raise PermissionError("Approval denied for high-risk action")
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC
from datetime import datetime as dt
from enum import Enum


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


# ── High-risk action registry ──────────────────────────────────────────────────

HIGH_RISK_ACTIONS: set[str] = {
    "train_loop.run_training",
    "corpus_loop.run_corpus_pipeline",
    "checkpoint_promote",
    "cross_experiment_transfer",
    "checkpoint.delete",
    "experiment.delete",
    "corpus.delete",
}


# ── Approval record ─────────────────────────────────────────────────────────────

@dataclass
class ApprovalRecord:
    approval_id: str
    agent_id: int
    agent_role: str
    action: str
    trace_id: str
    status: ApprovalStatus
    requested_at: float
    resolved_at: float | None = None
    expires_at: float = field(default_factory=lambda: time.time() + 300.0)
    reason: str = ""

    def is_expired(self) -> bool:
        return self.status == ApprovalStatus.PENDING and time.time() > self.expires_at


# ── Approval manager ──────────────────────────────────────────────────────────


class ApprovalManager:
    """
    Thread-safe in-memory store of pending approvals.

    In a production deployment this would back a DB table or Redis.
    Expiry is checked on every access (no background thread needed).
    """

    _instance: "ApprovalManager | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._approvals: dict[str, ApprovalRecord] = {}
        self._by_agent: dict[int, list[str]] = {}  # agent_id → [approval_ids]

    @classmethod
    def get_instance(cls) -> "ApprovalManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def request(
        self,
        agent_id: int,
        agent_role: str,
        action: str,
        trace_id: str,
        reason: str = "",
        *,
        ttl_seconds: float = 300.0,
    ) -> ApprovalRecord:
        """
        File a new approval request. Returns the ApprovalRecord.

        In demo/read-only mode this auto-approves; set DEMO_MODE=True to bypass.
        """
        # Demo mode: auto-approve everything (no human-in-the-loop blocking demos)
        demo_mode = _is_demo_mode()

        approval_id = uuid.uuid4().hex[:12]
        record = ApprovalRecord(
            approval_id=approval_id,
            agent_id=agent_id,
            agent_role=agent_role,
            action=action,
            trace_id=trace_id,
            status=ApprovalStatus.APPROVED if demo_mode else ApprovalStatus.PENDING,
            requested_at=time.time(),
            expires_at=time.time() + ttl_seconds,
            reason=reason,
        )

        with self._lock:
            self._approvals[approval_id] = record
            self._by_agent.setdefault(agent_id, []).append(approval_id)

        return record

    def approve(self, approval_id: str) -> ApprovalRecord:
        """Resolve a pending approval as APPROVED. Raises KeyError if not found."""
        with self._lock:
            record = self._approvals[approval_id]
            if record.is_expired():
                record.status = ApprovalStatus.EXPIRED
                raise ValueError(f"Approval {approval_id} has expired")
            record.status = ApprovalStatus.APPROVED
            record.resolved_at = time.time()
            return record

    def deny(self, approval_id: str, reason: str = "") -> ApprovalRecord:
        """Resolve a pending approval as DENIED. Raises KeyError if not found."""
        with self._lock:
            record = self._approvals[approval_id]
            if record.is_expired():
                record.status = ApprovalStatus.EXPIRED
                raise ValueError(f"Approval {approval_id} has expired")
            record.status = ApprovalStatus.DENIED
            record.resolved_at = time.time()
            if reason:
                record.reason = reason
            return record

    def get(self, approval_id: str) -> ApprovalRecord | None:
        """Get an approval record by ID. Returns None if not found."""
        with self._lock:
            record = self._approvals.get(approval_id)
            if record and record.is_expired():
                record.status = ApprovalStatus.EXPIRED
            return record

    def list_pending(self, agent_id: int | None = None) -> list[ApprovalRecord]:
        """List all pending (or recently expired) approvals, optionally filtered by agent."""
        with self._lock:
            now = time.time()
            results = []
            for record in self._approvals.values():
                if record.is_expired() and record.status == ApprovalStatus.PENDING:
                    record.status = ApprovalStatus.EXPIRED
                if agent_id is not None and record.agent_id != agent_id:
                    continue
                if record.status in (ApprovalStatus.PENDING, ApprovalStatus.EXPIRED):
                    results.append(record)
            results.sort(key=lambda r: r.requested_at, reverse=True)
            return results


def _is_demo_mode() -> bool:
    import os
    return os.environ.get("TMT_DEPLOY_MODE", "demo") == "demo"
