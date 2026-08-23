"""
Copilot Security Module — P1-P2 hardening for TinyMetatron Copilot v2.

Provides:
- P1 Prompt injection detection and blocking
- P1 Output sanitization (API keys, tokens, private keys)
- P1 Least-privilege capability per agent role
- P1 RAG content isolation wrappers
- P2 Inter-agent message signing (HMAC-SHA256)
- P2 Corpus integrity (SHA-256 verification)
- P2 Provenance tracking (append-only audit log)
- P2 Human-in-the-loop approval gate

Usage:
    from copilot.security import PromptGuard, OutputGuard, CAPABILITIES
    guard = PromptGuard()
    is_safe, reason = guard.is_safe(user_input)
    if not is_safe:
        raise ValueError(f"Blocked prompt injection: {reason}")

    cleaner = OutputGuard()
    clean = cleaner.sanitize(agent_output)
"""

from .prompt_guard import PromptGuard
from .output_guard import OutputGuard
from .capabilities import CAPABILITIES, check_capability
from .agent_identity import (
    agent_fingerprint,
    sign_agent_message,
    verify_agent_message,
    issue_ws_token,
    verify_ws_token,
    SignatureError,
)
from .corpus_guard import (
    register_corpus_file,
    verify_corpus_integrity,
    CorpusIntegrityError,
)
from .provenance import record_provenance, query_provenance

__all__ = [
    # P1
    "PromptGuard",
    "OutputGuard",
    "CAPABILITIES",
    "check_capability",
    # P2
    "agent_fingerprint",
    "sign_agent_message",
    "verify_agent_message",
    "issue_ws_token",
    "verify_ws_token",
    "SignatureError",
    "register_corpus_file",
    "verify_corpus_integrity",
    "CorpusIntegrityError",
    "record_provenance",
    "query_provenance",
]
