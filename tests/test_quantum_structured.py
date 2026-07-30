"""
tests/test_quantum_structured.py
================================
Allowlisted structured-query route: named SELECT templates, read-only
connection, forbidden-SQL rejection, and the rule-based intent classifier.

Mirrors the ``quantum_corpus.structured.self_test`` in-memory sidecar DB but
as pytest cases so it runs under ``python -m pytest tests/ -q``.
"""

from __future__ import annotations

import os
import sys
import sqlite3
import tempfile
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from quantum_corpus.structured import (
    StructuredQuery, classify_intent, parse_job_text, parse_workload_text,
    _assert_select, TEMPLATES,
)


# ── text-template parsers ────────────────────────────────────────────────────

def test_parse_job_text_extracts_all_fields():
    jt = ("IBM Quantum job d5an4o7p3tbc73atau4g on backend ibm_fez, status Completed, "
          "program sampler, tags ['Composer'], cost 600, created 2025-12-31T18:58:40.651125Z. "
          "Measurement samples: 10000. Circuit (OPENQASM 3.0):\nOPENQASM 3.0;")
    p = parse_job_text(jt)
    assert p is not None
    assert p["jid"] == "d5an4o7p3tbc73atau4g"
    assert p["backend"] == "ibm_fez"
    assert p["status"] == "Completed"
    assert p["samples"] == 10000
    assert p["tags"] == ["Composer"]
    assert p["cost"] == 600


def test_parse_job_text_rejects_messy_and_nonjob():
    assert parse_job_text("IBM Quantum job abc on backend <bound method X>, status Completed, "
                          "program sampler") is None
    assert parse_job_text("not a job record") is None


def test_parse_workload_text():
    wt = ("IBM Quantum workload summary from all_time-workloads (3).csv: 35 jobs. "
          "By backend: {'ibm_fez': 34, 'ibm_torino': 1}. By status: {'completed': 34, 'failed': 1}. "
          "Total usage (seconds): 132. Accounts/user identifiers redacted.")
    w = parse_workload_text(wt)
    assert w is not None
    assert w["jobs_count"] == 35
    assert w["total_usage"] == 132
    assert w["by_backend"] == {"ibm_fez": 34, "ibm_torino": 1}


# ── named templates over an in-memory sidecar DB ─────────────────────────────

@pytest.fixture
def sq(tmp_path):
    sdb = str(tmp_path / "struct_test.db")
    o = sqlite3.connect(sdb)
    o.executescript("""
        CREATE TABLE jobs (record_id INTEGER PRIMARY KEY, jid TEXT, backend TEXT,
            status TEXT, program TEXT, cost INTEGER, samples INTEGER, created TEXT,
            project TEXT, tags TEXT);
        CREATE TABLE workload_summaries (record_id INTEGER PRIMARY KEY, csv_name TEXT,
            jobs_count INTEGER, by_backend TEXT, by_status TEXT, total_usage INTEGER);
    """)
    o.executemany("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)", [
        (1, "j1", "ibm_fez", "Completed", "sampler", 600, 0, "2025-12-31T03:00:00Z", "x", "[]"),
        (2, "j2", "ibm_fez", "Completed", "sampler", 600, 10000, "2025-12-31T18:58:00Z", "x", "['Composer']"),
        (3, "j3", "ibm_torino", "Completed", "sampler", 600, 0, "2025-12-21T15:35:00Z", "x", "[]"),
        (4, "j4", "ibm_fez", "Failed", "sampler", 600, 4096, "2026-01-03T15:12:00Z", "x", "[]"),
    ])
    o.execute("INSERT INTO workload_summaries VALUES (1,'all_time-workloads (3).csv',35,'{}','{}',132)")
    o.commit(); o.close()
    return StructuredQuery(sdb)


def test_jobs_with_samples_above(sq):
    r = sq.run("jobs_with_samples_above", {"min_samples": 1, "n": 200})
    assert r["row_count"] == 2
    assert set(r["record_ids"]) == {2, 4}
    assert r["template_name"] == "jobs_with_samples_above"


def test_jobs_on_backend(sq):
    r = sq.run("jobs_on_backend", {"backend": "ibm_torino", "n": 200})
    assert r["record_ids"] == [3]


def test_jobs_on_backend_with_status(sq):
    r = sq.run("jobs_on_backend_with_status", {"backend": "ibm_fez", "status": "Failed", "n": 200})
    assert r["record_ids"] == [4]


def test_jobs_created_on(sq):
    r = sq.run("jobs_created_on", {"day": "2025-12-31", "n": 200})
    assert set(r["record_ids"]) == {1, 2}


def test_count_by_backend(sq):
    r = sq.run("count_by_backend", {})
    assert any(row["backend"] == "ibm_fez" and row["n"] == 3 for row in r["rows"])


def test_count_by_status(sq):
    r = sq.run("count_by_status", {})
    assert any(row["status"] == "Completed" and row["n"] == 3 for row in r["rows"])


def test_total_samples(sq):
    r = sq.run("total_samples", {})
    assert r["rows"][0]["total_samples"] == 14096


def test_workload_summary_by_name(sq):
    r = sq.run("workload_summary_by_name", {"name": "%all_time-workloads (3).csv%"})
    assert r["row_count"] == 1
    assert r["rows"][0]["jobs_count"] == 35


def test_classify_intent_workload_uses_name_param():
    ci = classify_intent("What is the workload CSV all_time-workloads (3).csv summary?")
    assert ci is not None and ci[0] == "workload_summary_by_name"
    assert "name" in ci[1] and "%" in ci[1]["name"]


def test_run_provenance_fields(sq):
    r = sq.run("jobs_on_backend", {"backend": "ibm_fez", "n": 200})
    assert {"template_name", "params", "rows", "row_count", "record_ids"} <= set(r)
    assert r["params"]["backend"] == "ibm_fez"


def test_unknown_template_rejected(sq):
    with pytest.raises(ValueError):
        sq.run("not_a_template", {})


def test_all_templates_are_select_only():
    """Every named template must be a single SELECT (no writes/pragma/attach)."""
    for name, sql in TEMPLATES.items():
        # should not raise
        _assert_select(sql)


# ── read-only connection ─────────────────────────────────────────────────────

def test_readonly_rejects_writes(sq, tmp_path):
    sdb = str(tmp_path / "struct_test.db").replace("\\", "/")
    uri = f"file:{sdb}?mode=ro"
    c = sqlite3.connect(uri, uri=True)
    with pytest.raises(sqlite3.OperationalError):
        c.execute("INSERT INTO jobs VALUES (99,'z','x','y','s',0,0,'t','p','[]')")
    c.close()


@pytest.mark.parametrize("bad", [
    "DROP TABLE jobs;",
    "PRAGMA database_list",
    "INSERT INTO jobs VALUES(1)",
    "ATTACH 'x' AS y",
    "SELECT 1; DROP TABLE jobs",
])
def test_assert_select_rejects_forbidden_sql(bad):
    with pytest.raises(ValueError):
        _assert_select(bad)


# ── rule-based intent classifier ─────────────────────────────────────────────

@pytest.mark.parametrize("question, want", [
    ("Which jobs have nonzero samples?", "jobs_with_samples_above"),
    ("how many jobs ran on ibm_torino", "jobs_on_backend"),
    ("List the IBM Quantum jobs created on 2025-12-31", "jobs_created_on"),
    ("how many jobs are completed", "count_by_status"),
    ("show jobs on ibm_fez that failed", "jobs_on_backend_with_status"),
    ("What is the workload CSV all_time-workloads (3).csv summary?", "workload_summary_by_name"),
    ("how many jobs per backend", "count_by_backend"),
])
def test_classify_intent_maps(question, want):
    ci = classify_intent(question)
    got = ci[0] if ci else None
    assert got == want, f"{question!r} -> {got} (want {want})"


def test_classify_intent_none_for_conceptual():
    assert classify_intent("What is the spectral gap in QSG run 042?") is None
    assert classify_intent("Explain the OTOC Lyapunov exponent measurement.") is None


def test_classify_intent_named_backend_filters_not_aggregates():
    """Regression: 'how many jobs ran on ibm_torino' must filter, not aggregate."""
    ci = classify_intent("how many jobs ran on ibm_torino")
    assert ci is not None
    assert ci[0] == "jobs_on_backend"
    assert ci[1].get("backend") == "ibm_torino"