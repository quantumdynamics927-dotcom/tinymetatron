"""
Copilot Security Module — P1 hardening for TinyMetatron Copilot v2.

Provides:
- Prompt injection detection and blocking
- Output sanitization (API keys, tokens, private keys)
- Least-privilege capability per agent role
- RAG content isolation wrappers

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

__all__ = [
    "PromptGuard",
    "OutputGuard",
    "CAPABILITIES",
    "check_capability",
]
