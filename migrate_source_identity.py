"""Migrate existing DB: add source_identity column and backfill."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_corpus import schema
import sqlite3

path = schema.default_db_path()
print(f"DB path: {path}")

conn = sqlite3.connect(path)
try:
    conn.execute("SELECT source_identity FROM corpus_records LIMIT 1")
    print("source_identity column already exists")
except sqlite3.OperationalError:
    print("Adding source_identity column...")
    conn.execute("ALTER TABLE corpus_records ADD COLUMN source_identity TEXT NOT NULL DEFAULT ''")
    conn.commit()
    print("Column added")
conn.close()

count = schema.backfill_source_identities(path)
print(f"Backfilled {count} records")
print("Migration complete")
