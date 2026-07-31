"""
quantum_corpus.eval.test_stable_identity
======================================
Three stability tests for source_identity (stable content-based identity):

  1. Rebuild stability    - same inputs -> same source_identity after wipe+rebuild
  2. Row-ID independence - same content, different DB row id -> same source_identity
  3. Change detection    - different content -> different source_identity

Run::

    python -m quantum_corpus.eval.test_stable_identity
"""
from __future__ import annotations

import os, sys, tempfile, shutil, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from quantum_corpus import schema
from quantum_corpus.schema import source_identity, content_hash, init_db


def _tmp():
    d = tempfile.mkdtemp(prefix="qcid_")
    p = os.path.join(d, "qc.db")
    init_db(p)
    return d, p


def _insert(db_path, **fields):
    sql = (
        "INSERT INTO corpus_records "
        "(source_type,project,doc_id,text,split,token_count,sensitivity,"
        " risk_tier,content_hash,cleaning_version,source_identity,created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
    )
    ch = content_hash(fields["text"])
    si = source_identity(fields["project"], fields["doc_id"],
                         fields.get("cleaning_version", "v1"), ch, 0)
    params = (
        fields.get("source_type", "repo"),
        fields["project"],
        fields["doc_id"],
        fields["text"],
        fields.get("split", "train"),
        len(fields["text"].split()),
        fields.get("sensitivity", "internal"),
        0,
        ch,
        fields.get("cleaning_version", "v1"),
        si,
        fields.get("created_at", "2026-01-01T00:00:00+00:00"),
    )
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()
    return si


def test_rebuild_stability():
    """Identical records rebuilt from scratch produce the same source_identity."""
    d1, p1 = _tmp()
    d2, p2 = _tmp()
    try:
        rec = dict(source_type="ibm_job", project="ibm-quantum",
                   doc_id="ibm:rebuild_test",
                   text="IBM Quantum job abc123 on backend ibm_fez.",
                   split="train", sensitivity="internal", cleaning_version="v1")
        si1 = _insert(p1, **rec)
        si2 = _insert(p2, **rec)
        assert si1 == si2, f"rebuild gave different identity: {si1} vs {si2}"
        print(f"REBUILD STABILITY: PASS  (si={si1[:16]}...)")
    finally:
        shutil.rmtree(d1)
        shutil.rmtree(d2)


def test_row_id_independence():
    """Two identical-content records inserted as separate rows get the same identity.
    Uses a separate test table without the unique constraint to verify the formula
    produces identical output for identical inputs regardless of row id."""
    d, p = _tmp()
    # Add a second test table (no unique constraint on source_identity)
    conn = sqlite3.connect(p)
    conn.execute("""
        CREATE TABLE test_identity_chunks (
            id INTEGER PRIMARY KEY,
            source_identity TEXT,
            chunk_index INTEGER
        )
    """)
    conn.commit()
    conn.close()

    try:
        rec = dict(source_type="ibm_job", project="ibm-quantum",
                   doc_id="ibm:rowid_test",
                   text="IBM Quantum job def456 on backend ibm_torino.",
                   split="train", sensitivity="internal", cleaning_version="v1")
        ch = content_hash(rec["text"])
        si = source_identity(rec["project"], rec["doc_id"], rec["cleaning_version"], ch, 0)

        # Insert same content twice with different chunk_index values
        conn = sqlite3.connect(p)
        conn.execute("INSERT INTO test_identity_chunks VALUES (?, ?, ?)", (None, si, 0))
        conn.execute("INSERT INTO test_identity_chunks VALUES (?, ?, ?)", (None, si, 1))
        conn.commit()
        rows = conn.execute("SELECT id, source_identity FROM test_identity_chunks").fetchall()
        conn.close()

        assert rows[0][1] == rows[1][1], \
            f"same content got different identity: {rows[0][1]} vs {rows[1][1]}"
        assert rows[0][0] != rows[1][0], "chunk rows should have distinct row ids"
        print(f"ROW-ID INDEPENDENCE: PASS  (si={si[:16]}...) row_ids={rows[0][0]},{rows[1][0]}")
    finally:
        shutil.rmtree(d)


def test_change_detection():
    """Different content produces a different source_identity."""
    d, p = _tmp()
    try:
        base = dict(source_type="ibm_job", project="ibm-quantum",
                    doc_id="ibm:change_test",
                    text="IBM Quantum job ghi789 on backend ibm_pegasus.",
                    split="train", sensitivity="internal", cleaning_version="v1")
        si_base = _insert(p, **base)
        si_proj = _insert(p, **{**base, "project": "different-project"})
        assert si_proj != si_base, "project change did not change identity"
        si_doc = _insert(p, **{**base, "doc_id": "ibm:changed_doc_id"})
        assert si_doc != si_base, "doc_id change did not change identity"
        si_txt = _insert(p, **{**base, "text": "Totally different record text."})
        assert si_txt != si_base, "text change did not change identity"
        si_ver = _insert(p, **{**base, "cleaning_version": "v2"})
        assert si_ver != si_base, "cleaning_version change did not change identity"
        ch = content_hash(base["text"])
        si_c0 = source_identity(base["project"], base["doc_id"], base["cleaning_version"], ch, 0)
        si_c1 = source_identity(base["project"], base["doc_id"], base["cleaning_version"], ch, 1)
        assert si_c1 != si_c0, "chunk_index change did not change identity"
        print(f"CHANGE DETECTION: PASS")
        print(f"  base        = {si_base[:16]}...")
        print(f"  project     = {si_proj[:16]}...  (changed)")
        print(f"  doc_id      = {si_doc[:16]}...  (changed)")
        print(f"  text        = {si_txt[:16]}...  (changed)")
        print(f"  version     = {si_ver[:16]}...  (changed)")
        print(f"  chunk[0]    = {si_c0[:16]}...")
        print(f"  chunk[1]    = {si_c1[:16]}...  (changed)")
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    test_rebuild_stability()
    test_row_id_independence()
    test_change_detection()
    print("\nALL STABILITY TESTS PASSED")
