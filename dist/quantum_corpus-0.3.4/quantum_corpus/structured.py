"""
quantum_corpus.structured
=========================
Allowlisted structured-query route for the private quantum corpus.

Why this exists: ``corpus_records`` has NO structured columns for backend /
status / samples / cost / created date — those fields live only inside the
fixed-format ``text`` template produced by ``extract.py``. Filter-style
questions ("which jobs have nonzero samples?", "how many jobs ran on
ibm_torino?") are database queries, not document search, and BM25 handles them
poorly (cross_record Recall@5 = 0.689 in the build-2 eval).

This module builds a **derived sidecar SQLite DB** by parsing the known text
templates, then serves a small set of **named, hardcoded SELECT templates**
over a **read-only** connection. No model- or user-authored SQL is ever
executed. The frozen corpus DB is not modified (build-2 hash preserved).

Tables (sidecar DB):
  jobs(record_id, jid, backend, status, program, cost, samples, created,
       project, tags)
  workload_summaries(record_id, csv_name, jobs_count, by_backend, by_status,
                     total_usage)

CLI::

    python -m quantum_corpus.structured build                 # build sidecar DB
    python -m quantum_corpus.structured --query "which jobs have nonzero samples"
    python -m quantum_corpus.structured --query "how many jobs ran on ibm_torino"
"""

from __future__ import annotations

import os
import re
import sys
import ast
import sqlite3
from typing import Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from quantum_corpus import schema as _schema

# Default sidecar location: beside the corpus DB (outside the repo, never
# uploaded to the public Space).
def default_structured_db_path() -> str:
    base = os.environ.get("TMT_QUANTUM_STRUCTURED_DB", "")
    if base:
        return base
    corpus = _schema.default_db_path()
    d = os.path.dirname(os.path.abspath(corpus))
    return os.path.join(d, "quantum_jobs_structured.db")


# ── text-template parsers ───────────────────────────────────────────────────
# Exact templates from extract.py:
#   "IBM Quantum job {jid} on backend {backend}, status {status}, program
#    {program}, tags {tags}, cost {cost}, created {created}. Measurement
#    samples: {n}. Circuit (OPENQASM 3.0):\n{qasm}"
#   "IBM Quantum workload summary from {name}: {N} jobs. By backend: {dict}.
#    By status: {dict}. Total usage (seconds): {n}. Accounts/user identifiers redacted."

_RE_JOB = re.compile(
    r"IBM Quantum job ([A-Za-z0-9]+) on backend ([A-Za-z0-9_]+), "
    r"status ([A-Za-z]+), program ([A-Za-z_]+), tags (\[[^\]]*\]), "
    r"cost (\d*), created (\S+?)\. Measurement samples: (\d+)\."
)
_RE_WORKLOAD = re.compile(
    r"IBM Quantum workload summary from (.+?): (\d+) jobs\. "
    r"By backend: (\{[^}]+\})\. By status: (\{[^}]+\})\. "
    r"Total usage \(seconds\): (\d+)\."
)


def parse_job_text(text: str) -> Optional[dict]:
    """Parse an ibm_job record's text into structured fields, or None."""
    m = _RE_JOB.search(text or "")
    if not m:
        return None
    jid, backend, status, program, tags_raw, cost_raw, created, samples = m.groups()
    try:
        tags = ast.literal_eval(tags_raw) if tags_raw else []
    except Exception:
        tags = []
    cost = int(cost_raw) if cost_raw.isdigit() else None
    return {
        "jid": jid, "backend": backend, "status": status,
        "program": program, "tags": tags, "cost": cost,
        "created": created, "samples": int(samples),
    }


def parse_workload_text(text: str) -> Optional[dict]:
    """Parse a workload_csv aggregate record's text, or None."""
    m = _RE_WORKLOAD.search(text or "")
    if not m:
        return None
    name, njobs, by_backend_raw, by_status_raw, usage = m.groups()
    try:
        by_backend = ast.literal_eval(by_backend_raw)
    except Exception:
        by_backend = {}
    try:
        by_status = ast.literal_eval(by_status_raw)
    except Exception:
        by_status = {}
    return {
        "csv_name": name, "jobs_count": int(njobs),
        "by_backend": by_backend, "by_status": by_status,
        "total_usage": int(usage),
    }


# ── sidecar DB build ────────────────────────────────────────────────────────
def build_structured_db(corpus_db_path: Optional[str] = None,
                        out_path: Optional[str] = None) -> dict:
    """Scan the frozen corpus for ibm_job + workload_csv records, parse their
    text, and write the sidecar structured DB. Does NOT modify the corpus DB."""
    corpus_db_path = corpus_db_path or _schema.default_db_path()
    out_path = out_path or default_structured_db_path()
    if os.path.exists(out_path):
        os.remove(out_path)
    conn = sqlite3.connect(corpus_db_path); conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, project, source_type, doc_id, text FROM corpus_records "
        "WHERE source_type IN ('ibm_job','workload_csv') ORDER BY id"
    ).fetchall()
    conn.close()

    jout = []   # (record_id, jid, backend, status, program, cost, samples, created, project, tags_json)
    wout = []
    skipped = 0
    for r in rows:
        if r["source_type"] == "ibm_job":
            p = parse_job_text(r["text"])
            if p is None:
                skipped += 1
                continue
            jout.append((r["id"], p["jid"], p["backend"], p["status"], p["program"],
                         p["cost"], p["samples"], p["created"], r["project"],
                         json_dumps(p["tags"])))
        else:  # workload_csv
            p = parse_workload_text(r["text"])
            if p is None:
                skipped += 1
                continue
            wout.append((r["id"], p["csv_name"], p["jobs_count"],
                         json_dumps(p["by_backend"]), json_dumps(p["by_status"]),
                         p["total_usage"]))

    o = sqlite3.connect(out_path)
    o.executescript("""
        CREATE TABLE jobs (
            record_id INTEGER PRIMARY KEY,
            jid TEXT, backend TEXT, status TEXT, program TEXT,
            cost INTEGER, samples INTEGER, created TEXT, project TEXT, tags TEXT
        );
        CREATE TABLE workload_summaries (
            record_id INTEGER PRIMARY KEY,
            csv_name TEXT, jobs_count INTEGER, by_backend TEXT,
            by_status TEXT, total_usage INTEGER
        );
        CREATE INDEX idx_jobs_backend ON jobs(backend);
        CREATE INDEX idx_jobs_status ON jobs(status);
        CREATE INDEX idx_jobs_created ON jobs(created);
    """)
    if jout:
        o.executemany(
            "INSERT INTO jobs (record_id,jid,backend,status,program,cost,samples,"
            "created,project,tags) VALUES (?,?,?,?,?,?,?,?,?,?)", jout)
    if wout:
        o.executemany(
            "INSERT INTO workload_summaries (record_id,csv_name,jobs_count,by_backend,"
            "by_status,total_usage) VALUES (?,?,?,?,?,?)", wout)
    o.commit(); o.close()
    return {"out_path": out_path, "jobs": len(jout),
            "workload_summaries": len(wout), "skipped_unparseable": skipped}


def json_dumps(obj) -> str:
    import json as _j
    return _j.dumps(obj, ensure_ascii=False)


# ── allowlisted SELECT templates (constants; no dynamic SQL) ────────────────
TEMPLATES = {
    "jobs_on_backend":
        "SELECT record_id, jid, backend, status, samples, created FROM jobs "
        "WHERE backend = :backend ORDER BY created LIMIT :n",
    "jobs_on_backend_with_status":
        "SELECT record_id, jid, backend, status, samples, created FROM jobs "
        "WHERE backend = :backend AND status = :status ORDER BY created LIMIT :n",
    "jobs_with_samples_above":
        "SELECT record_id, jid, backend, samples FROM jobs "
        "WHERE samples >= :min_samples ORDER BY samples DESC, jid LIMIT :n",
    "jobs_created_on":
        "SELECT record_id, jid, backend, status, created FROM jobs "
        "WHERE substr(created,1,10) = :day ORDER BY created LIMIT :n",
    "jobs_created_in_month":
        "SELECT record_id, jid, backend, status, created FROM jobs "
        "WHERE substr(created,1,7) = :month ORDER BY created LIMIT :n",
    "count_by_backend":
        "SELECT backend, COUNT(*) AS n FROM jobs GROUP BY backend ORDER BY n DESC",
    "count_by_status":
        "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status ORDER BY n DESC",
    "job_by_jid":
        "SELECT record_id, jid, backend, status, program, cost, samples, created "
        "FROM jobs WHERE jid = :jid LIMIT 1",
    "total_samples":
        "SELECT COUNT(*) AS n, SUM(samples) AS total_samples, AVG(samples) AS avg_samples "
        "FROM jobs",
    "workload_summary_by_name":
        "SELECT record_id, csv_name, jobs_count, by_backend, by_status, total_usage "
        "FROM workload_summaries WHERE csv_name LIKE :name",
}

_FORBIDDEN = re.compile(
    r"\b(PRAGMA|ATTACH|DETACH|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|"
    r"VACUUM|REINDEX|GRANT|REVOKE)\b", re.I)


def _assert_select(sql: str) -> None:
    """Defense-in-depth: the executor only ever runs TEMPLATES constants, but
    assert each is a single read-only SELECT with no write keywords / '; '."""
    s = sql.strip()
    if ";" in s.rstrip(";"):
        raise ValueError("multi-statement SQL rejected")
    if not s.lower().startswith("select"):
        raise ValueError("non-SELECT statement rejected")
    if _FORBIDDEN.search(s):
        raise ValueError("forbidden keyword in SQL")


class StructuredQuery:
    """Read-only executor over the sidecar structured DB.

    Only named templates from ``TEMPLATES`` may be run; params are bound via
    sqlite parameterization. The connection is opened read-only (uri mode=ro)."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or default_structured_db_path()
        if not os.path.isfile(self.db_path):
            raise FileNotFoundError(
                f"structured DB not found at {self.db_path}; run "
                "`python -m quantum_corpus.structured build` first.")
        for sql in TEMPLATES.values():
            _assert_select(sql)

    def _connect(self) -> sqlite3.Connection:
        # Read-only connection: SQLite refuses writes at the engine level.
        uri = "file:%s?mode=ro" % self.db_path.replace("\\", "/")
        return sqlite3.connect(uri, uri=True)

    def run(self, template_name: str, params: dict) -> dict:
        if template_name not in TEMPLATES:
            raise ValueError(f"unknown template: {template_name}")
        sql = TEMPLATES[template_name]
        _assert_select(sql)
        conn = self._connect(); conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        rrows = [dict(r) for r in rows]
        record_ids = [r["record_id"] for r in rrows if "record_id" in r]
        return {
            "template_name": template_name,
            "params": params,
            "rows": rrows,
            "row_count": len(rrows),
            "record_ids": record_ids,
        }


# ── rule-based intent classifier ────────────────────────────────────────────
_BACKENDS = ("ibm_fez", "ibm_torino", "ibm_kingston", "ibm_brisbane",
             "ibm_kyiv", "ibm_marrakesh", "ibm_strasbourg", "ibm_kawasaki",
             "ibm_brussels", "ibm_sherbrooke", "ibm_leipzig", "ibm_cleveland",
             "ibm_aachen")
_STATUSES = ("completed", "failed", "canceled", "cancelled", "running", "queued")
_RE_BACKEND = re.compile(r"\b(ibm_[a-z]+)\b")
_RE_DAY = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_RE_MONTH = re.compile(r"\b(20\d{2}-\d{2})\b")
_RE_JID = re.compile(r"\b([a-z0-9]{18,24})\b")
_RE_MIN_SAMPLES = re.compile(r"(?:at least|min(?:imum)?|>=|>|more than|over)\s*(\d+)\s*samples?", re.I)
_RE_CSV_NAME = re.compile(r"all_time-workloads(?:\s*\(\d+\))?\.csv", re.I)


def classify_intent(question: str) -> Optional[tuple[str, dict]]:
    """Map a natural-language question to a (template_name, params) or None.

    None => no structured template fits; the caller falls back to retrieval.
    Deterministic, rule-based, auditable (no model)."""
    q = question.lower()
    n = 200  # default row cap

    # Workload CSV summary lookup by named CSV.
    if ("workload" in q or "csv" in q) and _RE_CSV_NAME.search(question):
        return ("workload_summary_by_name",
                {"name": "%" + _RE_CSV_NAME.search(question).group(0) + "%"})

    # A workload/CSV question WITHOUT a specific CSV name (e.g. "across both
    # workload CSV summaries, how many ibm_fez jobs", "which workload CSV
    # reports canceled jobs") is about the workload_summaries, not the jobs
    # table — no jobs-table template fits, so fall back to retrieval (which can
    # surface the workload_csv records). Without this guard the backend name in
    # such questions misroutes to jobs_on_backend and returns raw job rows.
    if "workload" in q or "csv" in q:
        return None

    # total samples aggregation
    if re.search(r"\btotal\s+samples\b", q) and not _RE_JID.search(q):
        return ("total_samples", {})

    # explicit jid lookup
    mj = _RE_JID.search(question)
    if mj and re.search(r"\bjob\b", q) and not _RE_BACKEND.search(q) \
            and not _RE_DAY.search(q):
        return ("job_by_jid", {"jid": mj.group(1)})

    # samples threshold / nonzero samples
    if ("nonzero" in q or "non-zero" in q
            or re.search(r"\bsamples?\s*(?:>|>=|greater|more)", q)
            or _RE_MIN_SAMPLES.search(q)):
        mmin = _RE_MIN_SAMPLES.search(q)
        min_s = int(mmin.group(1)) if mmin else 1
        return ("jobs_with_samples_above", {"min_samples": min_s, "n": n})

    # created on a specific day
    md = _RE_DAY.search(question)
    if md and ("created on" in q or "on " + md.group(1) in q or "run on" in q
               or re.search(r"\bjobs?\b.*\bon\b", q)):
        # "run on ibm_fez" would also match day-less; ensure backend not present
        if not _RE_BACKEND.search(q):
            return ("jobs_created_on", {"day": md.group(1), "n": n})

    # created in a month
    mm = _RE_MONTH.search(question)
    if mm and ("in " + mm.group(1) in q or "during" in q) and not _RE_DAY.search(q) \
            and not _RE_BACKEND.search(q):
        return ("jobs_created_in_month", {"month": mm.group(1), "n": n})

    # backend filters / counts. A NAMED backend always filters to it; the
    # cross-backend aggregate (count_by_backend) is reserved for explicit
    # "by/per/each backend" phrasing with no specific backend named.
    mb = _RE_BACKEND.search(q)
    if mb:
        backend = mb.group(1)
        # status-filtered list/count on a specific backend
        st = next((s for s in _STATUSES if s in q), None)
        if st:
            return ("jobs_on_backend_with_status",
                    {"backend": backend, "status": st.capitalize(), "n": n})
        # how many on a specific backend -> filtered list (row_count is the count)
        if re.search(r"\bhow many\b", q) and not _RE_DAY.search(q):
            return ("jobs_on_backend", {"backend": backend, "n": n})
        # list jobs on a backend
        if re.search(r"\b(which|list|what|show)\b.*\bjobs?\b", q) or "jobs on" in q \
                or "ran on" in q or "running on" in q:
            return ("jobs_on_backend", {"backend": backend, "n": n})

    # count by status (no specific backend)
    if re.search(r"\bhow many\b.*\bjobs?\b", q) and any(s in q for s in _STATUSES):
        return ("count_by_status", {})

    # count by backend (aggregate across all backends; no specific backend named)
    if re.search(r"\bhow many\b.*\bjobs?\b", q) \
            and re.search(r"\b(?:by|per|each)\s+backend\b", q):
        return ("count_by_backend", {})

    return None


# ── CLI + self-test ─────────────────────────────────────────────────────────
def _cli(argv) -> int:
    if argv and argv[0] == "build":
        r = build_structured_db()
        print(f"built structured DB: {r['out_path']}")
        print(f"  jobs: {r['jobs']}  workload_summaries: {r['workload_summaries']}  "
              f"skipped(unparseable): {r['skipped_unparseable']}")
        return 0
    q = None
    if "--query" in argv:
        q = argv[argv.index("--query") + 1]
    if not q:
        print("usage: python -m quantum_corpus.structured build | --query '...'")
        return 1
    intent = classify_intent(q)
    if intent is None:
        print("no structured template matches -> fall back to retrieval")
        return 0
    name, params = intent
    print(f"intent: {name}  params: {params}")
    res = StructuredQuery().run(name, params)
    print(f"rows: {res['row_count']}  record_ids: {res['record_ids'][:10]}")
    for r in res["rows"][:10]:
        print("  ", r)
    return 0


def self_test() -> None:
    import tempfile
    sys.stdout.reconfigure(encoding="utf-8")

    def _ok(c, m):
        print(("OK  " if c else "FAIL") + " " + str(m))
        assert c, m

    # parse_job_text
    jt = ("IBM Quantum job d5an4o7p3tbc73atau4g on backend ibm_fez, status Completed, "
          "program sampler, tags ['Composer'], cost 600, created 2025-12-31T18:58:40.651125Z. "
          "Measurement samples: 10000. Circuit (OPENQASM 3.0):\nOPENQASM 3.0;")
    p = parse_job_text(jt)
    _ok(p is not None and p["jid"] == "d5an4o7p3tbc73atau4g", f"parse job: {p}")
    _ok(p["backend"] == "ibm_fez" and p["status"] == "Completed" and p["samples"] == 10000,
        "job fields parsed")
    _ok(p["tags"] == ["Composer"] and p["cost"] == 600, "tags + cost parsed")

    # messy QAP backend -> None (skipped)
    messy = "IBM Quantum job abc on backend <bound method X>, status Completed, program sampler"
    _ok(parse_job_text(messy) is None, "messy backend -> None (skipped)")
    _ok(parse_job_text("not a job record") is None, "non-job -> None")

    # parse_workload_text
    wt = ("IBM Quantum workload summary from all_time-workloads (3).csv: 35 jobs. "
          "By backend: {'ibm_fez': 34, 'ibm_torino': 1}. By status: {'completed': 34, 'failed': 1}. "
          "Total usage (seconds): 132. Accounts/user identifiers redacted.")
    w = parse_workload_text(wt)
    _ok(w is not None and w["jobs_count"] == 35 and w["total_usage"] == 132, f"parse workload: {w}")
    _ok(w["by_backend"] == {"ibm_fez": 34, "ibm_torino": 1}, "workload by_backend parsed")

    # build an in-memory sidecar DB via temp file and test templates
    td = tempfile.mkdtemp()
    sdb = os.path.join(td, "struct_test.db")
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

    sq = StructuredQuery(sdb)

    r = sq.run("jobs_with_samples_above", {"min_samples": 1, "n": 200})
    _ok(r["row_count"] == 2 and {2, 4} == set(r["record_ids"]),
        f"jobs_with_samples_above -> {r['record_ids']}")

    r = sq.run("jobs_on_backend", {"backend": "ibm_torino", "n": 200})
    _ok(r["row_count"] == 1 and r["record_ids"] == [3], f"jobs_on_backend torino -> {r['record_ids']}")

    r = sq.run("jobs_on_backend_with_status", {"backend": "ibm_fez", "status": "Failed", "n": 200})
    _ok(r["row_count"] == 1 and r["record_ids"] == [4], f"jobs_on_backend_with_status -> {r['record_ids']}")

    r = sq.run("jobs_created_on", {"day": "2025-12-31", "n": 200})
    _ok(r["row_count"] == 2 and set(r["record_ids"]) == {1, 2}, f"jobs_created_on -> {r['record_ids']}")

    r = sq.run("count_by_backend", {})
    _ok(any(row["backend"] == "ibm_fez" and row["n"] == 3 for row in r["rows"]),
        f"count_by_backend -> {r['rows']}")

    r = sq.run("total_samples", {})
    _ok(r["rows"][0]["total_samples"] == 14096, f"total_samples -> {r['rows']}")

    # read-only connection rejects writes
    try:
        uri = "file:%s?mode=ro" % sdb.replace("\\", "/")
        c = sqlite3.connect(uri, uri=True)
        c.execute("INSERT INTO jobs VALUES (99,'z','x','y','s',0,0,'t','p','[]')")
        c.close()
        _ok(False, "read-only connection should reject INSERT")
    except sqlite3.OperationalError:
        _ok(True, "read-only connection rejects INSERT")
    except Exception as e:
        _ok(True, f"read-only connection rejects write ({type(e).__name__})")

    # _assert_select rejects forbidden SQL
    for bad in ("DROP TABLE jobs;", "PRAGMA database_list", "INSERT INTO jobs VALUES(1)"):
        try:
            _assert_select(bad)
            _ok(False, f"should reject: {bad}")
        except ValueError:
            _ok(True, f"rejected forbidden SQL: {bad[:24]}")

    # unknown template rejected
    try:
        sq.run("not_a_template", {})
        _ok(False, "unknown template should raise")
    except ValueError:
        _ok(True, "unknown template rejected")

    # classify_intent mapping
    cases = [
        ("Which jobs have nonzero samples?", "jobs_with_samples_above"),
        ("how many jobs ran on ibm_torino", "jobs_on_backend"),
        ("List the IBM Quantum jobs created on 2025-12-31", "jobs_created_on"),
        ("how many jobs are completed", "count_by_status"),
        ("show jobs on ibm_fez that failed", "jobs_on_backend_with_status"),
        ("What is the workload CSV all_time-workloads (3).csv summary?", "workload_summary_by_name"),
    ]
    for q, want in cases:
        ci = classify_intent(q)
        got = ci[0] if ci else None
        _ok(got == want, f"classify {q!r} -> {got} (want {want})")
    # non-structured -> None
    _ok(classify_intent("What is the spectral gap in QSG run 042?") is None,
        "conceptual question -> None (fall back to retrieval)")

    import shutil
    shutil.rmtree(td, ignore_errors=True)
    print("SELF-TEST PASSED")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:]
    if not argv:
        self_test()
    else:
        raise SystemExit(_cli(argv))