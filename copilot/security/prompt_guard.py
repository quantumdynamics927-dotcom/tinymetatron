"""
Prompt Guard — blocks prompt injection attempts before they reach agents.

Uses regex patterns to detect:
- Role assertion / jailbreak patterns ("ignore previous instructions", etc.)
- System prompt override attempts ("you are now", "act as", etc.)
- Delimiter injection (```system, <?xml, etc.)
- Encoded evasion (base64, hex, rot13)

Also checks for unusually long inputs (DoS vector) and
repeated padding attempts.
"""

from __future__ import annotations

import base64
import re
import html


# ── Detection patterns ─────────────────────────────────────────────────────────────

# Direct instruction override / role play
_INJECTION_PATTERNS = [
    # Classic prompt injection
    re.compile(r"ignore\s+(all\s+)?previous\s+(instructions?|commands?|directives?)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(your\s+)?(instructions?|rules?|constraints?)", re.IGNORECASE),
    re.compile(r"(forget|reset)\s+(all\s+)?(instructions?|rules?|context)", re.IGNORECASE),
    re.compile(r"disobey\s+\w+", re.IGNORECASE),
    re.compile(r"(you\s+are\s+now|you\s+must\s+now|pretend\s+you\s+are|act\s+as)\s+(a\s+)?", re.IGNORECASE),
    re.compile(r"you\s+(are\s+)?(now\s+)?(a|an|restricted?\s+)", re.IGNORECASE),
    re.compile(r"new\s+(system\s+)?(prompt|instructions?|role|persona)", re.IGNORECASE),

    # Override prefixes
    re.compile(r"^\s*(system|instruction|priming)[\s:]*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"<\s*/?(system|instructions?|prompt)\s*>", re.IGNORECASE),
    re.compile(r"<\?xml\s", re.IGNORECASE),

    # Delimiter-based injection
    re.compile(r"```\s*system", re.IGNORECASE),
    re.compile(r"```\s*json\s*\{", re.IGNORECASE),
    re.compile(r"<!--\s*prompt:\s*", re.IGNORECASE),
    re.compile(r"^\s*–\s*prompt\s*:\s*", re.IGNORECASE | re.MULTILINE),

    # Markdown meta-injection
    re.compile(r"^#\s*system\s*prompt", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^#\s*hidden\s*prompt", re.IGNORECASE | re.MULTILINE),

    # Base64/hex obfuscation attempts
    re.compile(r'(?:base64|b64)["\'\s:]*([A-Za-z0-9+/]{20,})', re.IGNORECASE),
    re.compile(r"(?:\\x[0-9a-f]{2}){4,}", re.IGNORECASE),  # hex-encoded strings

    # ROT13 / cipher evasion
    re.compile(r"rot13\s*[:=]?\s*(?:[A-Za-z]{20,})", re.IGNORECASE),

    # Unicode spoofing (homoglyphs)
    re.compile(r"[​-‏﻿]"),  # zero-width / BOM chars

    # Padding / whitespace abuse (repeated invisible chars)
    re.compile(r"[  -‏]{10,}"),

    # Credential scraping patterns
    re.compile(r"(?:api[_-]?key|access[_-]?token|secret[_-]?key|auth[_-]?token)\s*[:=]\s*", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9]{20,}", re.IGNORECASE),  # OpenAI-style keys
    re.compile(r"ghp_[A-Za-z0-9]{20,}", re.IGNORECASE),  # GitHub tokens
    re.compile(r"xox[baprs]-[A-Za-z0-9]{10,}", re.IGNORECASE),  # Slack tokens
]

# Context length / DoS limits
_MAX_INPUT_LENGTH = 8000  # chars — well below 32-token context window
_MAX_TOKEN_ESTIMATE = 2000  # ~8000 chars at 4 chars/token


class PromptGuard:
    """
    Scans user-provided text for prompt injection patterns.

    Usage:
        guard = PromptGuard()
        is_safe, reason = guard.is_safe(user_input)
        if not is_safe:
            raise ValueError(f"Blocked: {reason}")
    """

    def __init__(
        self,
        max_length: int = _MAX_INPUT_LENGTH,
        strict: bool = False,
    ) -> None:
        self.max_length = max_length
        self.strict = strict
        self._encoded_check_cache: dict[str, bool] = {}

    def is_safe(self, text: str) -> tuple[bool, str]:
        """
        Returns (True, "") if clean, or (False, reason) if blocked.
        """
        if not text:
            return True, ""

        # 1. Length check
        if len(text) > self.max_length:
            return False, f"Input too long ({len(text)} > {self.max_length} chars)"

        # 2. Whitespace padding / steganography check
        if re.search(r"[  -‏　]{3,}", text):
            return False, "Unicode whitespace padding detected"

        # 3. Zero-width char check
        if "​" in text or "﻿" in text:
            return False, "Zero-width characters detected (steganography)"

        # 4. HTML/entity injection check
        if html.escape(text) != text and "<" in text:
            # Contains unescaped HTML or entities that aren't just escaped version
            if re.search(r"<[^>]*on\w+\s*=", text, re.IGNORECASE):
                return False, "HTML event handler injection (onclick, onerror, etc.)"

        # 5. Pattern-based detection
        for i, pattern in enumerate(_INJECTION_PATTERNS):
            try:
                match = pattern.search(text)
            except re.error:
                continue
            if match:
                return False, f"Pattern {i} matched: {match.group()!r}"

        # 6. Base64 content check (decoded content also scanned)
        b64_hits = re.findall(r"(?:base64|b64)['\"\s:]*([A-Za-z0-9+/=\s]{40,})", text, re.IGNORECASE)
        for hit in b64_hits:
            hit = hit.replace(" ", "").replace("\n", "")
            try:
                decoded = base64.b64decode(hit).decode("utf-8", errors="ignore")
                if not self.is_safe(decoded)[0]:
                    return False, f"Base64 payload contains blocked content"
            except Exception:
                pass

        # 7. Repetitive padding check (DoS / confusion)
        if len(set(text)) < max(5, len(text) * 0.05):
            return False, "Insufficient entropy (possible obfuscation or junk injection)"

        return True, ""

    def scan(self, text: str) -> list[str]:
        """Return a list of all matched patterns (for logging/audit)."""
        violations = []
        for i, pattern in enumerate(_INJECTION_PATTERNS):
            try:
                if pattern.search(text):
                    violations.append(f"pattern_{i}")
            except re.error:
                pass
        return violations

    def block(self, text: str) -> None:
        """Raise ValueError if text is not safe. Use in API entry points."""
        safe, reason = self.is_safe(text)
        if not safe:
            raise ValueError(f"PromptGuard blocked input: {reason}")
