"""
quantum_corpus.split
====================
Train/val/test partitioning BY DOCUMENT, not by chunk.

All records sharing a ``doc_id`` land in the same split, so chunks of one
document never leak across train and val/test (the leakage risk flagged in the
roadmap). Assignment is deterministic (sha256 of doc_id) so re-running the build
is stable; no Math.random (would break resume/ reproducibility).

Default ratios: 80 / 10 / 10. Configurable via ``ratios=(train,val,test)``.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Sequence

TRAIN, VAL, TEST = "train", "val", "test"


def _bucket(doc_id: str) -> int:
    """Stable 0..99 bucket from doc_id (sha256, not Python's randomized hash)."""
    h = hashlib.sha256(doc_id.encode("utf-8", "replace")).hexdigest()
    return int(h[:8], 16) % 100


def assign_splits(records: Sequence[dict],
                  ratios=(0.8, 0.1, 0.1)) -> List[Dict]:
    """
    Given records (dicts with ``id`` + ``doc_id``), return a list of
    ``{id, split}`` with a consistent split per doc_id.

    records may be a heterogeneous list; dedup by doc_id is implicit because we
    cache the first decision per doc_id.
    """
    tr, va, _te = ratios
    assert 0 < tr + va + _te <= 1.0 + 1e-9, "ratios must sum to <= 1"
    train_cut = int(tr * 100)
    val_cut = train_cut + int(va * 100)
    seen: Dict[str, str] = {}
    out: List[Dict] = []
    for r in records:
        doc = r["doc_id"]
        if doc not in seen:
            b = _bucket(doc)
            seen[doc] = TRAIN if b < train_cut else (VAL if b < val_cut else TEST)
        out.append({"id": r["id"], "split": seen[doc]})
    return out


def split_summary(assignments: Iterable[dict]) -> Dict[str, int]:
    s = {TRAIN: 0, VAL: 0, TEST: 0}
    for a in assignments:
        s[a["split"]] = s.get(a["split"], 0) + 1
    return s


# ── self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    def _ok(c, m):
        print(("OK  " if c else "FAIL") + " " + m)
        assert c, m

    recs = [{"id": i, "doc_id": f"proj:doc{i}"} for i in range(2000)]
    a = assign_splits(recs)
    # all chunks of one doc share a split
    by_doc = {}
    for r, asg in zip(recs, a):
        by_doc.setdefault(r["doc_id"], set()).add(asg["split"])
    _ok(all(len(v) == 1 for v in by_doc.values()), "one split per doc_id")
    s = split_summary(a)
    print("  split sizes:", s)
    # roughly 80/10/10
    _ok(1500 <= s[TRAIN] <= 1700, f"train ~80%: {s[TRAIN]}")
    _ok(50 <= s[VAL] <= 300, f"val ~10%: {s[VAL]}")
    _ok(50 <= s[TEST] <= 300, f"test ~10%: {s[TEST]}")

    # determinism: re-run identical
    a2 = assign_splits(recs)
    _ok(a == a2, "deterministic")

    # chunked records (same doc_id repeated) -> same split
    chunked = [{"id": 1, "doc_id": "p:x"}, {"id": 2, "doc_id": "p:x"},
               {"id": 3, "doc_id": "p:y"}]
    ac = assign_splits(chunked)
    _ok(ac[0]["split"] == ac[1]["split"], "chunks of same doc share split")
    print("SELF-TEST PASSED")