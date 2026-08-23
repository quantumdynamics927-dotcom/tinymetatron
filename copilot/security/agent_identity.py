"""
Agent Identity & Message Signing — P2 inter-agent security.

Provides:
- agent_fingerprint(agent_id): stable HMAC-SHA256 identity for an agent
- sign_agent_message(agent_id, payload, timestamp): HMAC-SHA256 signature
- verify_agent_message(agent_id, payload, timestamp, signature): verify + TTL check
- Two-step WS auth: issue short-lived JWT-like tokens for WS connections

Signing spec:
- Algorithm: HMAC-SHA256
- TTL: 30 seconds for inter-agent messages
- Payload for signing: f"{agent_id}:{timestamp}:{json_payload}"
- Signature format: hex-encoded HMAC-SHA256
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any

# ── Global secret (set once at startup) ──────────────────────────────────────────

_AGENT_SIGNING_KEY: str | None = None


def _get_signing_key() -> bytes:
    global _AGENT_SIGNING_KEY
    if _AGENT_SIGNING_KEY is None:
        raw = os.environ.get("TMT_AGENT_SIGNING_KEY")
        if raw:
            _AGENT_SIGNING_KEY = raw
        else:
            # Generate a random key on first use — fine for single-instance demo
            _AGENT_SIGNING_KEY = secrets.token_hex(32)
    return _AGENT_SIGNING_KEY.encode()


# ── Agent fingerprint ─────────────────────────────────────────────────────────────


def agent_fingerprint(agent_id: int) -> str:
    """
    Return a stable HMAC-SHA256 fingerprint for an agent_id.

    The fingerprint is derived as HMAC-SHA256(signing_key, f"agent:{agent_id}").
    It is deterministic across restarts (stable key) and unique per agent.
    """
    return hmac.new(
        _get_signing_key(),
        f"agent:{agent_id}".encode(),
        hashlib.sha256,
    ).hexdigest()[:16]


# ── Message signing ───────────────────────────────────────────────────────────────


def sign_agent_message(
    agent_id: int,
    payload: dict[str, Any],
    timestamp: float | None = None,
) -> tuple[str, float]:
    """
    Sign a message payload for a given agent.

    Returns (signature_hex, timestamp). Timestamp is returned so verifiers
    can enforce TTL without clock skew.

    Signing input: f"{agent_id}:{timestamp}:{json.dumps(payload, sort_keys=True)}"
    """
    if timestamp is None:
        timestamp = time.time()

    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    signing_input = f"{agent_id}:{timestamp}:{payload_json}"

    signature = hmac.new(
        _get_signing_key(),
        signing_input.encode(),
        hashlib.sha256,
    ).hexdigest()

    return signature, timestamp


# ── Message verification ──────────────────────────────────────────────────────────


class SignatureError(ValueError):
    """Raised when a signature check fails."""
    pass


def verify_agent_message(
    agent_id: int,
    payload: dict[str, Any],
    timestamp: float,
    signature: str,
    *,
    ttl_seconds: float = 30.0,
) -> None:
    """
    Verify a signed message from an agent.

    Raises SignatureError if:
      - Signature does not match (tampering)
      - Timestamp is outside the TTL window (replay attack)
      - Agent ID in signature does not match
    """
    # Check TTL
    now = time.time()
    if abs(now - timestamp) > ttl_seconds:
        raise SignatureError(
            f"Message from agent {agent_id} is stale "
            f"({now - timestamp:.1f}s old, max {ttl_seconds}s)"
        )

    # Recompute expected signature
    expected_sig, _ = sign_agent_message(agent_id, payload, timestamp)
    if not hmac.compare_digest(signature, expected_sig):
        raise SignatureError(
            f"Signature mismatch for agent {agent_id} — possible tampering"
        )


# ── WS two-step auth tokens ──────────────────────────────────────────────────────


_WS_TOKEN_SECRET: str | None = None


def _get_ws_secret() -> bytes:
    global _WS_TOKEN_SECRET
    if _WS_TOKEN_SECRET is None:
        raw = os.environ.get("TMT_WS_TOKEN_SECRET")
        if raw:
            _WS_TOKEN_SECRET = raw
        else:
            _WS_TOKEN_SECRET = secrets.token_hex(48)
    return _WS_TOKEN_SECRET.encode()


def issue_ws_token(agent_id: int, *, ttl_seconds: float = 300.0) -> tuple[str, float]:
    """
    Issue a short-lived WS auth token for an agent.

    Returns (token_hex, expires_at).  Caller presents the token to the WS
    endpoint within the TTL window.  Token = HMAC-SHA256(secret, f"ws:{agent_id}:exp:{expires_at}")

    Demo mode (TMT_DEPLOY_MODE=demo): returns ("demo-token", time.time() + 300)
    """
    expires_at = time.time() + ttl_seconds
    token_input = f"ws:{agent_id}:exp:{int(expires_at)}"
    token = hmac.new(_get_ws_secret(), token_input.encode(), hashlib.sha256).hexdigest()
    return token, expires_at


def verify_ws_token(
    token: str,
    agent_id: int,
    expires_at: float,
) -> bool:
    """
    Verify a WS auth token presented by a connecting client.

    Returns True if valid and not expired.  Uses constant-time comparison.
    """
    if time.time() > expires_at:
        return False
    expected_input = f"ws:{agent_id}:exp:{int(expires_at)}"
    expected_token = hmac.new(_get_ws_secret(), expected_input.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(token, expected_token)
