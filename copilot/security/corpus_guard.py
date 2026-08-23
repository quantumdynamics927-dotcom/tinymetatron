"""
Corpus Integrity Guard — P2 corpus hash verification.

Provides:
- register_corpus_file(path, sha256): register a corpus file with its hash
- verify_corpus_integrity(path): verify a file against its registered hash
- CorpusIntegrityError: raised on mismatch

All corpus files used in loop_adapters must be verified before use.
Hashes are stored in state/corpus_hashes.jsonl (one JSON per line, append-only).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Optional

# ── Storage ────────────────────────────────────────────────────────────────────

_HASH_DB: dict[str, str] = {}  # path → sha256
_HASH_DB_LOCK = threading.Lock()


def _hash_db_path() -> Path:
    root = Path(__file__).resolve().parents[2]  # copilot/ → repo root
    return root / "state" / "corpus_hashes.jsonl"


def _load_hash_db() -> None:
    """Load the hash DB from disk on first use."""
    global _HASH_DB
    db_path = _hash_db_path()
    if not db_path.is_file():
        return
    with _HASH_DB_LOCK:
        if _HASH_DB:
            return  # already loaded
        try:
            with open(db_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("type") == "corpus_hash":
                            _HASH_DB[record["path"]] = record["sha256"]
                    except Exception:
                        pass
        except Exception:
            pass


def _append_record(record: dict) -> None:
    """Append a JSON record to the hash log (append-only, no truncation)."""
    db_path = _hash_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with open(db_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")


# ── SHA-256 computation ───────────────────────────────────────────────────────


def _sha256_of_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Public API ────────────────────────────────────────────────────────────────


class CorpusIntegrityError(ValueError):
    """Raised when a corpus file fails integrity check."""
    pass


def register_corpus_file(path: str | Path, sha256: str | None = None) -> str:
    """
    Register a corpus file with its SHA-256 hash.

    If sha256 is not provided, it is computed from the file contents.

    Returns the sha256 hash that was registered.
    """
    _load_hash_db()
    p = str(Path(path).resolve())
    if sha256 is None:
        sha256 = _sha256_of_file(Path(p))

    with _HASH_DB_LOCK:
        _HASH_DB[p] = sha256

    _append_record({
        "type": "corpus_hash",
        "path": p,
        "sha256": sha256,
    })
    return sha256


def verify_corpus_integrity(path: str | Path) -> bool:
    """
    Verify a corpus file against its registered SHA-256 hash.

    Returns True if the file's current SHA-256 matches the registered hash.

    Raises FileNotFoundError if the file does not exist.
    Raises CorpusIntegrityError if the hash does not match (corruption or tampering).
    """
    _load_hash_db()
    p = str(Path(path).resolve())

    if not os.path.isfile(p):
        raise FileNotFoundError(f"Corpus file not found: {p}")

    current_hash = _sha256_of_file(Path(p))

    with _HASH_DB_LOCK:
        registered_hash = _HASH_DB.get(p)

    if registered_hash is None:
        # Not registered — register it now (first-seen trust)
        register_corpus_file(p, current_hash)
        return True

    if current_hash != registered_hash:
        raise CorpusIntegrityError(
            f"Corpus integrity failure: {p}\n"
            f"  expected: {registered_hash}\n"
            f"  got:      {current_hash}\n"
            f"File may be corrupted or tampered with."
        )

    return True


def get_registered_hash(path: str | Path) -> Optional[str]:
    """Return the registered SHA-256 for a path, or None if not registered."""
    _load_hash_db()
    return _HASH_DB.get(str(Path(path).resolve()))
