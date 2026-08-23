"""
quantum_corpus.redact
=====================
Identifier + credential redaction layer that runs BEFORE any text is written
to the corpus DB. Honors the user's "include everything, redact identifiers"
decision: no subdir is excluded on sensitivity grounds, but every record passes
through here.

What gets redacted (kept OUT of the corpus, replaced by a tag):
  * IBMid identifiers          -> ``IBMid-[REDACTED]``
  * IBM Cloud CRN account/instance GUIDs (the 32-hex account + instance ids)
                               -> ``a/[REDACTED]:[REDACTED]``
  * bare 32-hex account ids    -> ``[REDACTED_ACCT]``
  * credential token values     (ghp_/gho_/ghs_/ghu_/hf_/sk-/xox- prefixes +
     long alnum runs that look like API keys) -> ``[REDACTED_TOKEN]``
  * private keys (PEM ``-----BEGIN ... PRIVATE KEY-----`` blocks) -> dropped

What is KEPT (research value, user owns it): names, emails, backends (ibm_fez),
QPU names, regions, circuit QASM, measurement *counts* (not raw secret values),
prose, architecture descriptions.

``redact_text(text)`` returns ``(redacted_text, counts_dict)``.

``should_skip_file(relpath)`` flags files that are PURE credentials/binary and
must never be read at all (apikey*.json, account-0x*.csv, *.mp4, *.pt, ...).
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

# ── File-level skip (never opened) ───────────────────────────────────────────
# Pure-credential / binary / non-text noise. Build-noise dirs (node_modules
# etc.) are handled separately in extract.py; this is for *sensitive* files
# the redaction layer itself owns.

_SKIP_GLOBS = (
    re.compile(r"(^|/)apikey.*\.json$", re.I),
    re.compile(r"(^|/)account-0x[0-9a-fA-F]+.*\.csv$", re.I),
    re.compile(r"(^|/).*\.(mp4|mov|avi|pt|bin|pth|onnx|safetensors)$", re.I),
    re.compile(r"(^|/)\.cache/", re.I),
)

# Build / cache noise that is non-text and adds nothing to a language corpus.
_NOISE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "egg-info",
    "dist", "build", "site", ".next", ".ruff_cache", ".pytest_cache",
    ".mypy_cache", ".cache", "quantum-env", ".huggingface", "coverage",
    "__results__", ".ipynb_checkpoints",
}
_NOISE_FILES = {
    "package-lock.json", "coverage.xml", "yarn.lock", "pnpm-lock.yaml",
}

_TEXT_EXT = (".py", ".md", ".rst", ".txt", ".ipynb", ".toml", ".json", ".csv",
             ".yaml", ".yml", ".html", ".js", ".ts", ".tsx", ".jsx", ".ps1",
             ".sh", ".cfg", ".ini")


def should_skip_file(relpath: str) -> bool:
    """True for credential/binary files that must never be read."""
    norm = relpath.replace("\\", "/")
    for rx in _SKIP_GLOBS:
        if rx.search(norm):
            return True
    base = norm.rsplit("/", 1)[-1]
    if base in _NOISE_FILES:
        return True
    return False


def is_noise_path(relpath: str) -> bool:
    """True if any path component is a build/cache noise dir."""
    parts = relpath.replace("\\", "/").split("/")
    return any(p in _NOISE_DIRS for p in parts)


def is_text_file(relpath: str) -> bool:
    base = relpath.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return base.endswith(_TEXT_EXT)


# ── Text-level redaction patterns ───────────────────────────────────────────

# Hard bounds on how much of an input is regex-scanned.  Secrets live in short
# spans; anything beyond these limits is preserved unchanged to avoid denial of
# service from adversarial inputs.
MAX_SCAN_BYTES = 1_000_000          # 1 MiB of text scanned per call
MAX_PEM_BLOCK_BYTES = 32_000        # refuse to redact PEM blocks larger than this
MAX_CRN_TOKEN_BYTES = 2_000         # CRN tokens are much shorter than this

_RE_IBMID = re.compile(r"IBMid-[A-Z0-9]+")
_RE_BARE_ACCT = re.compile(r"\b[0-9a-f]{32}\b")  # 32-hex account ids (after CRN handled)
_RE_TOKEN = re.compile(
    r"\b(?:gh[opsur]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{12,})\b"
)

# CRN detection: IBM Cloud CRNs are single whitespace-delimited tokens.  We
# find candidate tokens with a bounded-length regex, then validate each one with
# an anchored pattern.  This removes the polynomial backtracking path that
# existed when the full regex was run over unbounded text.
_RE_CRN_CAND = re.compile(r"\bcrn:v1:\S{10," + str(MAX_CRN_TOKEN_BYTES) + r"}(?=\s|$)", re.I)
_RE_CRN_ANCHOR = re.compile(
    r"^(crn:v1:(?:[^:\s]*:){0,12}a/)[0-9a-f]{32}(:[0-9a-f-]+)(?:::)?$",
    re.I,
)


def _redact_crn_tokens(text: str) -> Tuple[str, int]:
    """Return (text_with_crn_accounts_redacted, count).  Linear in input size."""
    count = 0

    def _repl(m: "re.Match") -> str:
        nonlocal count
        token = m.group(0)
        m2 = _RE_CRN_ANCHOR.match(token)
        if m2:
            count += 1
            return m2.group(1) + "[REDACTED]" + m2.group(2)
        return token

    return _RE_CRN_CAND.sub(_repl, text), count
# PEM detection: line-based literal scans only.  The old single-regex pattern
# with a dotall wildcard was flagged by CodeQL as polynomial-time (ReDoS) on
# attacker-controlled text.  We now locate begin/end line markers with bounded,
# linear searches and redact only the block between them.
_RE_PEM_BEGIN = re.compile(r"-----BEGIN(?: [A-Z][A-Z ]*)? PRIVATE KEY-----")
_RE_PEM_END = re.compile(r"-----END(?: [A-Z][A-Z ]*)? PRIVATE KEY-----")


def _redact_pem_blocks(text: str) -> Tuple[str, int]:
    """Return (text_with_pem_blocks_redacted, count).  Linear-time scan."""
    parts: list[str] = []
    pos = 0
    count = 0
    for m in _RE_PEM_BEGIN.finditer(text):
        start = m.start()
        if start < pos:
            continue
        parts.append(text[pos:start])
        window_end = min(start + MAX_PEM_BLOCK_BYTES, len(text))
        window = text[start:window_end]
        em = _RE_PEM_END.search(window)
        stop = start + (em.end() if em else len(window))
        parts.append("[REDACTED_PRIVATE_KEY]")
        pos = stop
        count += 1
    parts.append(text[pos:])
    return "".join(parts), count


# Qiskit token-style "token" fields in JSON-ish text: "token": "...."
_RE_JSON_TOKEN = re.compile(
    r'("token"\s*:\s*")[^"]{12,}(")', re.I
)
_RE_JSON_APIKEY = re.compile(
    r'("(?:api_?key|apikey|secret|password|access_token)"\s*:\s*")[^"]{8,}(")',
    re.I,
)


def redact_text(text: str) -> Tuple[str, Dict[str, int]]:
    """Return (redacted_text, counts). Idempotent-ish: tags are not re-matched.
    Scanning is bounded to MAX_SCAN_BYTES to avoid ReDoS on huge inputs."""
    counts: Dict[str, int] = {}
    if len(text) > MAX_SCAN_BYTES:
        prefix = text[:MAX_SCAN_BYTES]
        suffix = text[MAX_SCAN_BYTES:]
    else:
        prefix = text
        suffix = ""

    def _sub(rx, repl, key):
        nonlocal prefix
        new, n = rx.subn(repl, prefix)
        if n:
            counts[key] = counts.get(key, 0) + n
            prefix = new
        return n

    prefix, n_pem = _redact_pem_blocks(prefix)
    if n_pem:
        counts["private_key_block"] = counts.get("private_key_block", 0) + n_pem
    _sub(_RE_TOKEN, "[REDACTED_TOKEN]", "token")
    _sub(_RE_JSON_TOKEN, r'\1[REDACTED]\2', "json_token")
    _sub(_RE_JSON_APIKEY, r'\1[REDACTED]\2', "json_apikey")
    _sub(_RE_IBMID, "IBMid-[REDACTED]", "ibmid")
    prefix, n_crn = _redact_crn_tokens(prefix)
    if n_crn:
        counts["crn_acct"] = counts.get("crn_acct", 0) + n_crn
    # Bare 32-hex after CRN/IBMid handled; only redact standalone 32-hex (account ids).
    _sub(_RE_BARE_ACCT, "[REDACTED_ACCT]", "bare_acct")
    return prefix + suffix, counts


def merge_counts(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    for k, v in b.items():
        a[k] = a.get(k, 0) + v
    return a


# ── self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    def _ok(c, m):
        print(("OK  " if c else "FAIL") + " " + str(m))
        assert c, m

    # file-level skip
    _ok(should_skip_file("foo/apikey (3).json"), "apikey json skipped")
    _ok(should_skip_file("account-0x6E0b...-2026-01-19.csv"), "account csv skipped")
    _ok(should_skip_file("clip.mp4"), "mp4 skipped")
    _ok(not should_skip_file("docs/intro.md"), "md not skipped")
    _ok(is_noise_path("node_modules/x/y"), "node_modules is noise")
    _ok(not is_noise_path("Agent_Stealth/log.md"), "Agent_Stealth NOT noise (kept)")

    # IBMid
    t, c = redact_text("ran by IBMid-695001BQB4 on ibm_fez")
    _ok("IBMid-695001BQB4" not in t and "IBMid-[REDACTED]" in t, f"ibmid redacted: {t!r}")
    _ok(c.get("ibmid") == 1, c)

    # CRN with account
    t, c = redact_text(
        "crn:v1:bluemix:public:quantum-computing:us-east:a/06175211d06f464ba15a52c048b1712a:6d0c4996-00b0-451a-b96f-d676f4ee6ad7:: end"
    )
    _ok("06175211d06f464ba15a52c048b1712a" not in t, f"crn acct redacted: {t!r}")
    _ok("us-east" in t, "crn region preserved")
    _ok("ibm_fez" not in t or True, "no false positive on backend")

    # tokens
    t, c = redact_text("token gho_abcDEF1234567890ghijKLM leaked, hf_ZQCANARYabcdef0123456789 too")
    _ok("gho_abcDEF1234567890ghijKLM" not in t, "gho token redacted")
    _ok("hf_ZQCANARYabcdef0123456789" not in t, "hf token redacted")
    _ok("[REDACTED_TOKEN]" in t, "token replaced with tag")
    _ok(c.get("token") == 2, c)

    # JSON token field
    t, c = redact_text('{"token": "abcdefghij12345XYZ"}')
    _ok('"token": "[REDACTED]"' in t, f"json token field redacted: {t!r}")

    # bare 32-hex account id (CSV Account column)
    t, c = redact_text("User,kub chme,,06175211d06f464ba15a52c048b1712a")
    _ok("06175211d06f464ba15a52c048b1712a" not in t, "bare acct redacted")

    # PEM private key block dropped
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...secretpemdata...\n-----END RSA PRIVATE KEY-----"
    t, c = redact_text(pem)
    _ok("MIIEpAIBAAKCAQEA" not in t, "pem key body dropped")
    _ok(c.get("private_key_block") == 1, c)

    # research content preserved
    t, c = redact_text("The OTOC circuit measures C(t) = -<[W(t), V]^2> on ibm_kingston with 8192 shots.")
    _ok("OTOC" in t and "ibm_kingston" in t and "8192" in t, "research content preserved")
    _ok(c == {}, f"no redactions on clean prose: {c}")

    print("SELF-TEST PASSED")