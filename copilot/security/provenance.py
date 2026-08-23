"""
Provenance Tracking — P2 data lineage audit log.

Records every data input/output event to state/provenance.jsonl
(append-only, one JSON per line). Downstream auditors can replay
the log to reconstruct the full data lineage of any agent execution.

Schema per line:
{
  "timestamp": <unix timestamp>,
  "event": "input" | "output" | "checkpoint_read" | "checkpoint_write" | "loop_call",
  "agent_id": <int>,
  "agent_role": <str>,
  "trace_id": <str>,
  "data_hash": <sha256 of the data content>,
  "source": <path or "loop:<loop_name>">,
  "destination": <path or "agent:<role>">,
  "checksum_ok": <bool>,
}

Usage in loop_adapters:
    from copilot.security.provenance import record_provenance
    record_provenance("input", agent_id=profile.agent_id,
                      agent_role=profile.agent_role,
                      trace_id=trace_id,
                      data=data_bytes,
                      source=corpus_path,
                      destination=f"agent:{profile.agent_role}")
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import UTC
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Optional


_PROV_LOCK = threading.Lock()


def _prov_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "state" / "provenance.jsonl"


def _ensure_prov_dir() -> None:
    _prov_path().parent.mkdir(parents=True, exist_ok=True)


def _hash_data(data: Any) -> str:
    """Return SHA-256 hex of a JSON-serializable payload."""
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    if isinstance(data, str):
        return hashlib.sha256(data.encode()).hexdigest()
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def record_provenance(
    event: str,
    *,
    agent_id: int,
    agent_role: str,
    trace_id: str,
    data: Any,
    source: str,
    destination: str,
    checksum_ok: bool = True,
) -> None:
    """
    Append a provenance record to state/provenance.jsonl.

    This function is designed to be called in a finally: block inside
    every loop adapter, so that even on error the partial record is kept.

    Args:
        event: one of "input", "output", "checkpoint_read", "checkpoint_write", "loop_call"
        agent_id: ID of the agent performing the operation
        agent_role: role string of the agent
        trace_id: execution trace UUID (string)
        data: the actual data being operated on (used only for checksum hashing)
        source: where the data came from (path or "loop:<name>")
        destination: where the data went (path or "agent:<role>")
        checksum_ok: whether integrity check passed
    """
    record = {
        "timestamp": dt.now(UTC).isoformat(),
        "event": event,
        "agent_id": agent_id,
        "agent_role": agent_role,
        "trace_id": trace_id,
        "data_hash": _hash_data(data),
        "source": source,
        "destination": destination,
        "checksum_ok": checksum_ok,
    }

    _ensure_prov_dir()
    with _PROV_LOCK:
        with open(_prov_path(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")


def query_provenance(
    trace_id: str | None = None,
    agent_id: int | None = None,
    event: str | None = None,
    limit: int = 1000,
) -> list[dict]:
    """
    Read back provenance records, optionally filtered.

    Returns up to `limit` matching records as dicts, newest first.
    """
    prov_file = _prov_path()
    if not prov_file.is_file():
        return []

    results: list[tuple[float, dict]] = []
    with _PROV_LOCK:
        try:
            with open(prov_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    # filter
                    if trace_id and rec.get("trace_id") != trace_id:
                        continue
                    if agent_id is not None and rec.get("agent_id") != agent_id:
                        continue
                    if event and rec.get("event") != event:
                        continue
                    ts = rec.get("timestamp", "")
                    results.append((ts, rec))
        except Exception:
            return []

    # sort newest first
    results.sort(key=lambda x: x[0], reverse=True)
    return [rec for _, rec in results[:limit]]
