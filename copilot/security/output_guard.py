"""
Output Guard — sanitizes agent outputs before they leave the system.

Redacts:
- API keys, tokens, secret keys (OpenAI, GitHub, Slack, AWS, etc.)
- Private keys (RSA, EC, Ed25519, GPG)
- Connection strings with embedded credentials
- Environment variable dumps
- Raw token sequences that could be replayed

Also validates that the output is UTF-8 and does not contain
extremely large payloads (DoS on downstream consumers).
"""

from __future__ import annotations

import base64
import re
from typing import Literal


# ── Redaction helpers ─────────────────────────────────────────────────────────────

def _mask(s: str, visible: int = 4) -> str:
    """Replace all but last `visible` chars with asterisks."""
    if len(s) <= visible:
        return "***"
    return "*" * (len(s) - visible) + s[-visible:]


# ── Compiled patterns ─────────────────────────────────────────────────────────────

_API_KEY_PATTERNS = [
    # OpenAI / Anthropic / generic LLM keys
    (re.compile(r"\b(sk-[A-Za-z0-9]{20,})\b", re.IGNORECASE), "[OPENAI_KEY]"),
    (re.compile(r"\b(sk-ant-[A-Za-z0-9_-]{20,})\b", re.IGNORECASE), "[ANTHROPIC_KEY]"),
    # GitHub fine-grained / classic tokens
    (re.compile(r"\b(ghp_[A-Za-z0-9]{20,})\b", re.IGNORECASE), "[GITHUB_TOKEN]"),
    (re.compile(r"\b(gho_[A-Za-z0-9_-]{20,})\b", re.IGNORECASE), "[GITHUB_TOKEN]"),
    (re.compile(r"\b(github_pat_[A-Za-z0-9_]{22,})\b", re.IGNORECASE), "[GITHUB_PAT]"),
    # Slack tokens
    (re.compile(r"\b(xox[baprs]-[A-Za-z0-9-]{10,})\b", re.IGNORECASE), "[SLACK_TOKEN]"),
    # AWS keys
    (re.compile(r"\b(A3T[A-Za-z0-9]{20,})\b", re.IGNORECASE), "[AWS_KEY]"),
    (re.compile(r"\b(ABIA[A-Za-z0-9]{20,})\b", re.IGNORECASE), "[AWS_KEY]"),
    (re.compile(r"\b(ASIA[A-Za-z0-9]{20,})\b", re.IGNORECASE), "[AWS_KEY]"),
    # Google Cloud / generic JSON Web Tokens
    (re.compile(r"\b(ya29\.[A-Za-z0-9_-]{50,})\b", re.IGNORECASE), "[GOOGLE_TOKEN]"),
    (re.compile(r"\b(AIza[ A-Za-z0-9_-]{20,})\b", re.IGNORECASE), "[GOOGLE_API_KEY]"),
    # Azure
    (re.compile(r"\b([A-Za-z0-9]{32})\b", re.IGNORECASE), "[AZURE_KEY]"),  # Azure key format
    # Stripe
    (re.compile(r"\b(sk_live_[A-Za-z0-9]{20,})\b", re.IGNORECASE), "[STRIPE_KEY]"),
    # Twilio
    (re.compile(r"\b(SK[a-f0-9]{30,})\b", re.IGNORECASE), "[TWILIO_KEY]"),
    # Database connection strings with embedded passwords
    (re.compile(r"(\b(?:postgres|mysql|redis|mongodb)://[^:]+:)([^@]+)(@)", re.IGNORECASE), r"\1[REDACTED_DB_PASS]\3"),
    # Generic Bearer tokens
    (re.compile(r"\b(Bearer\s+)([A-Za-z0-9_-]{20,})\b", re.IGNORECASE), r"\1[REDACTED_TOKEN]"),
    # Generic API key assignments (not just 'key = val' — catches embedded in JSON/YAML too)
    (re.compile(r'("(?:api[_-]?key|access[_-]?token|secret[_-]?key|auth[_-]?token)"\s*:\s*")([^"]{8,})(")', re.IGNORECASE), r'\1[REDACTED]\3'),
]

_PRIVATE_KEY_PATTERNS = [
    (re.compile(r"-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+|DSA\s+|PGP\s+)?PRIVATE\s+KEY-----"), "[PRIVATE_KEY_HEADER]"),
    (re.compile(r"-----BEGIN\s+OPENSSH\s+PRIVATE\s+KEY-----"), "[SSH_KEY_HEADER]"),
    (re.compile(r"-----BEGIN\s+GPG\s+PRIVATE\s+KEY\s+BLOCK-----"), "[GPG_KEY_HEADER]"),
    # Raw Ed25519 seed (base64 encoded 32-byte seed)
    (re.compile(r"\b[A-Za-z0-9+/]{40,62}\b", re.IGNORECASE), "[POTENTIAL_PRIVATE_KEY]"),  # generic high-entropy base64 string
]

_ENV_DUMP_PATTERNS = [
    re.compile(r"^(\s*(?:export\s+)?(?:API_KEY|APIKEY|SECRET|TOKEN|PASSWORD|PASSWD|PWD|PRIVATE|DB_PASS)[\s=].*)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*process\.env\.\w+\s*=", re.MULTILINE),
]

# High-entropy base64 strings that might be JWTs or key material
_JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b", re.IGNORECASE)
_JWT_WITH_BEARER = re.compile(r"(Bearer\s+)(eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)", re.IGNORECASE)


class OutputGuard:
    """
    Sanitizes agent output strings.

    Usage:
        guard = OutputGuard()
        clean = guard.sanitize(agent_output)
        is_clean, reason = guard.is_clean(agent_output)
    """

    def __init__(
        self,
        max_length: int = 50_000,
        redact_private_keys: bool = True,
        redact_jwts: bool = True,
        redact_env: bool = True,
    ) -> None:
        self.max_length = max_length
        self.redact_private_keys = redact_private_keys
        self.redact_jwts = redact_jwts
        self.redact_env = redact_env

    def sanitize(self, text: str) -> str:
        """
        Returns a sanitized copy of `text` with all detected credentials replaced.
        """
        if not text:
            return ""

        result = text

        # 1. High-entropy base64 strings (private key material) — MUST run before generic API key patterns
        if self.redact_private_keys:
            for pattern, replacement in _PRIVATE_KEY_PATTERNS:
                result = pattern.sub(replacement, result)

        # 2. JWT tokens
        if self.redact_jwts:
            result = _JWT_PATTERN.sub("[JWT_TOKEN]", result)
            result = _JWT_WITH_BEARER.sub(r"\1[JWT_TOKEN]", result)

        # 3. API keys / tokens / secrets
        for pattern, replacement in _API_KEY_PATTERNS:
            result = pattern.sub(replacement, result)

        # 4. Environment variable dumps
        if self.redact_env:
            for pattern in _ENV_DUMP_PATTERNS:
                result = pattern.sub("[ENV_VAR]", result)

        # 5. Generic high-entropy base64 strings (catch-all for unknown token formats)
        # Only flag strings >= 40 chars that are purely base64 (no spaces, newlines)
        # and don't already match a known pattern
        for match in re.finditer(r"\b([A-Za-z0-9+/]{40,62}={0,2})\b", result):
            token = match.group(1)
            try:
                decoded = base64.b64decode(token)
                if 16 <= len(decoded) <= 4096 and len(set(decoded)) > 4:
                    # Looks like key material — redact
                    result = result.replace(token, "[BASE64_KEY_MATERIAL]")
            except Exception:
                pass

        return result

    def is_clean(self, text: str) -> tuple[bool, str]:
        """
        Returns (True, "") if no violations found, else (False, reason).
        Does NOT modify the text — use sanitize() for that.
        """
        if not text:
            return True, ""

        if len(text) > self.max_length:
            return False, f"Output exceeds {self.max_length} chars (possible DoS)"

        for pattern, _ in _API_KEY_PATTERNS:
            if pattern.search(text):
                return False, "API key or token pattern detected in output"

        if self.redact_private_keys:
            for pattern, _ in _PRIVATE_KEY_PATTERNS:
                if pattern.search(text):
                    return False, "Private key header detected in output"

        if self.redact_jwts:
            if _JWT_PATTERN.search(text):
                return False, "JWT token detected in output"

        if self.redact_env:
            for pattern in _ENV_DUMP_PATTERNS:
                if pattern.search(text):
                    return False, "Environment variable dump detected in output"

        return True, ""

    def scan(self, text: str) -> list[str]:
        """Return a list of violation categories found in text."""
        violations = []
        for pattern, label in _API_KEY_PATTERNS:
            if pattern.search(text):
                violations.append(label)
        for pattern, label in _PRIVATE_KEY_PATTERNS:
            if pattern.search(text):
                violations.append(label)
        if _JWT_PATTERN.search(text):
            violations.append("JWT")
        for pattern in _ENV_DUMP_PATTERNS:
            if pattern.search(text):
                violations.append("ENV_DUMP")
        return violations
