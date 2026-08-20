"""
quantum_jobs/ingest.py
=====================
Ingest all IBM Quantum job data from E:\\Descargas into quantum_jobs.db.

Sources processed:
  1. workloads (N).zip    - bulk zip files (44 files, ~569 jobs)
  2. workloads.zip         - aggregated zip
  3. job-*.zip           - individual job zips (120 unique)
  4. workloads (N)/        - loose JSON dirs (17 jobs)
  5. all_time-workloads*.csv - IBM usage CSVs (8 files, ~515 rows)

The SQLite DB lives in quantum_jobs.db (beside this module) and is created
if absent. JIDs are unique-keyed — duplicate jobs across zips are skipped.

Usage:
  python -m quantum_jobs.ingest
  python -m quantum_jobs.ingest --db-path /custom/path/quantum_jobs.db
  python -m quantum_jobs.ingest --skip-csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from quantum_jobs import schema as _schema


# ── Source roots ─────────────────────────────────────────────────────────────

DESCARGAS = Path("E:/Descargas")
DB_PATH = ""          # filled in main()
SKIP_CSV = False


# ── Job record builders ─────────────────────────────────────────────────────

def _parse_samples(result: dict) -> Tuple[List[str], int]:
    """Return (hex_samples_list, count) from a result dict."""
    try:
        res = result.get("results") or []
        if res and isinstance(res, list):
            data = (res[0].get("data") or {}) if isinstance(res[0], dict) else {}
            c = data.get("c") or data.get("register") or data.get("meas") or {}
            samples = c.get("samples") if isinstance(c, dict) else None
            if isinstance(samples, list):
                return samples, len(samples)
    except Exception:
        pass
    return [], 0


def _job_from_info_result(
    info: dict,
    result: dict,
    provenance: str,
    workload_name: str = "",
    source_file: str = "",
) -> Optional[dict]:
    """Build a job dict from a paired info + result, or None if invalid."""
    if not isinstance(info, dict):
        return None
    jid = info.get("id") or info.get("job_id")
    if not jid:
        return None

    # Backend
    backend = info.get("backend", "?")
    if not isinstance(backend, str):
        return None

    # Status
    state = info.get("state") or info.get("status", "?")
    status = state.get("status") if isinstance(state, dict) else str(state)

    # Program
    prog = info.get("program") or info.get("program_id") or ""
    program = ""
    if isinstance(prog, dict):
        program = prog.get("id", "")
    elif isinstance(prog, str):
        program = prog

    # Tags, cost, created
    tags_raw = info.get("tags") or []
    tags = json.dumps(tags_raw) if isinstance(tags_raw, list) else str(tags_raw)
    cost = info.get("cost")
    if cost is not None:
        try:
            cost = int(cost)
        except (ValueError, TypeError):
            cost = None
    created = info.get("created", "")

    # QASM
    qasm = ""
    params = info.get("params")
    if isinstance(params, dict):
        pubs = params.get("pubs") or []
        if pubs and isinstance(pubs[0], list) and pubs[0] and isinstance(pubs[0][0], str):
            qasm = pubs[0][0]
        elif isinstance(params.get("circuit"), str):
            qasm = params["circuit"]
    elif isinstance(params, str):
        qasm = params

    # Raw samples
    raw_samples, samples_count = _parse_samples(result)
    raw_samples_json = json.dumps(raw_samples) if raw_samples else ""

    return {
        "jid": str(jid),
        "backend": backend,
        "status": str(status),
        "program": str(program),
        "cost": cost,
        "shots": samples_count,
        "created": str(created),
        "tags": tags,
        "qasm": qasm,
        "raw_samples": raw_samples_json,
        "samples_count": samples_count,
        "provenance": provenance,
        "workload_name": workload_name,
        "source_file": source_file,
    }


# ── Zip processors ───────────────────────────────────────────────────────────

def _ingest_zip(
    zip_path: Path,
    project: str = "ibm-quantum",
    workload_name: str = "",
    source_file: str = "",
) -> Tuple[List[dict], dict]:
    """Ingest all job pairs from a zip file. Returns (jobs, stats)."""
    infos: Dict[str, dict] = {}
    results: Dict[str, dict] = {}
    skipped = 0
    provenance = str(zip_path.resolve())

    try:
        zf = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile):
        return [], {"bad_zip": 1}

    try:
        for name in zf.namelist():
            if name.endswith("-info.json"):
                key = name[: -len("-info.json")]
                try:
                    infos[key] = json.loads(zf.read(name))
                except Exception:
                    skipped += 1
            elif name.endswith("-result.json"):
                key = name[: -len("-result.json")]
                try:
                    results[key] = json.loads(zf.read(name))
                except Exception:
                    skipped += 1
    finally:
        zf.close()

    jobs = []
    for key, info in infos.items():
        job = _job_from_info_result(
            info, results.get(key, {}),
            provenance=provenance,
            workload_name=workload_name or zip_path.stem,
            source_file=key,
        )
        if job:
            jobs.append(job)

    return jobs, {"jobs": len(jobs), "bad_info_result": skipped}


def ingest_workloads_zips() -> List[dict]:
    """Ingest all workloads (N).zip and workloads.zip files."""
    all_jobs = []
    patterns = ["workloads.zip", "workloads (*).zip"]
    for pattern in patterns:
        for zp in sorted(DESCARGAS.glob(pattern)):
            name = zp.stem if "(" in zp.stem else "workloads"
            jobs, stats = _ingest_zip(zp, workload_name=name)
            print(f"  {zp.name}: {stats}")
            all_jobs.extend(jobs)
    return all_jobs


def ingest_job_zips() -> List[dict]:
    """Ingest all job-*.zip files (skipping apparent (1) duplicates)."""
    seen = set()
    all_jobs = []
    for zp in sorted(DESCARGAS.glob("job-*.zip")):
        base = zp.stem.replace(" (1)", "")
        if base in seen:
            continue
        seen.add(base)
        jobs, stats = _ingest_zip(zp, workload_name="job-zip", source_file=base)
        print(f"  {zp.name}: {stats}")
        all_jobs.extend(jobs)
    return all_jobs


# ── Loose JSON dirs ─────────────────────────────────────────────────────────

def ingest_loose_dirs() -> List[dict]:
    """Ingest workloads (N)/ directories with loose JSON files."""
    all_jobs = []
    for d in sorted(DESCARGAS.glob("workloads (*)")):
        if not d.is_dir():
            continue
        infos: Dict[str, dict] = {}
        results: Dict[str, dict] = {}
        for fn in d.glob("*.json"):
            stem = fn.stem
            try:
                with open(fn, "rb") as f:
                    data = json.loads(f.read())
            except Exception:
                continue
            if fn.name.endswith("-info.json"):
                key = stem[: -len("-info")]
                infos[key] = data
            elif fn.name.endswith("-result.json"):
                key = stem[: -len("-result")]
                results[key] = data

        jobs = []
        for key, info in infos.items():
            job = _job_from_info_result(
                info, results.get(key, {}),
                provenance=str(d.resolve()),
                workload_name=d.name,
                source_file=key,
            )
            if job:
                jobs.append(job)
        print(f"  {d.name}/: {len(jobs)} jobs")
        all_jobs.extend(jobs)
    return all_jobs


# ── CSV ingestors ────────────────────────────────────────────────────────────

def ingest_csvs() -> Tuple[List[dict], List[dict]]:
    """Ingest all_time-workloads*.csv files.

    Returns (csv_job_rows, workload_summary_rows) for the workloads table.
    """
    import csv as _csv

    csv_rows = []       # for csv_jobs table
    wl_rows = []        # for workloads table

    csv_files = sorted(DESCARGAS.glob("all_time-workloads*.csv"))
    for csvp in csv_files:
        try:
            with open(csvp, "r", encoding="utf-8", errors="replace", newline="") as f:
                reader = _csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            print(f"  WARN: {csvp.name} read failed: {e}")
            continue

        if not rows:
            continue

        # Aggregate stats for workloads table
        by_backend: Dict[str, int] = defaultdict(int)
        by_status: Dict[str, int] = defaultdict(int)
        total_usage = 0.0
        job_count = len(rows)

        for r in rows:
            jid = r.get("WorkloadId", "")
            backend = r.get("QPU") or r.get("backend") or "?"
            status = r.get("Status", "?")
            qpu = r.get("QPU", "")
            usage = r.get("Usage (seconds)", "0")
            user = r.get("User", "")
            created = r.get("Created", "")
            completed = r.get("Completed", "")
            tags = r.get("Tags", "")
            account = r.get("Account", "")

            by_backend[backend] += 1
            by_status[status] += 1
            try:
                total_usage += float(usage)
            except (ValueError, TypeError):
                pass

            csv_rows.append({
                "csv_name": csvp.name,
                "csv_path": str(csvp.resolve()),
                "workload_id": None,
                "jid": jid,
                "backend": backend,
                "status": status,
                "qpu": qpu,
                "usage_seconds": float(usage) if usage else 0,
                "user_name": user,
                "created_date": created[:10] if created else "",
                "completed_date": completed[:10] if completed else "",
                "tags": tags,
                "account": account,
            })

        wl_rows.append({
            "name": csvp.name,
            "kind": "csv",
            "source_path": str(csvp.resolve()),
            "jobs_count": job_count,
            "total_shots": 0,
            "by_backend": json.dumps(dict(by_backend)),
            "by_status": json.dumps(dict(by_status)),
        })
        print(f"  {csvp.name}: {job_count} job rows, backends={dict(by_backend)}")

    return csv_rows, wl_rows


# ── Main ─────────────────────────────────────────────────────────────────────

def run(db_path: str = "") -> dict:
    global DB_PATH
    DB_PATH = db_path or _schema.default_db_path()

    actual_path = _schema.init_db(DB_PATH)
    print(f"\nDB: {actual_path}\n")

    # Collect all jobs from all zip sources
    print("=== Processing workloads zips ===")
    zip_jobs = ingest_workloads_zips()

    print("\n=== Processing job-*.zip files ===")
    jobzip_jobs = ingest_job_zips()

    print("\n=== Processing loose workloads dirs ===")
    loose_jobs = ingest_loose_dirs()

    all_jobs = zip_jobs + jobzip_jobs + loose_jobs
    print(f"\nTotal raw job records: {len(all_jobs)}")

    # Deduplicate by jid
    seen, unique = set(), []
    for j in all_jobs:
        if j["jid"] not in seen:
            seen.add(j["jid"])
            unique.append(j)
    print(f"After jid dedup: {len(unique)}")

    # Write jobs
    inserted, skipped = _schema.write_jobs(DB_PATH, unique)
    print(f"Jobs written: {inserted} inserted, {skipped} skipped (dup jid)")

    # Build workload summaries for zip files
    workloads = []
    for name, jobs_group in [
        ("workloads", zip_jobs),
        ("job-zips", jobzip_jobs),
    ]:
        by_backend: Dict[str, int] = defaultdict(int)
        by_status: Dict[str, int] = defaultdict(int)
        total_shots = 0
        for j in jobs_group:
            by_backend[j["backend"]] += 1
            by_status[j["status"]] += 1
            total_shots += j.get("shots", 0) or 0
        if jobs_group:
            workloads.append({
                "name": name,
                "kind": "zip",
                "source_path": "E:/Descargas",
                "jobs_count": len(jobs_group),
                "total_shots": total_shots,
                "by_backend": json.dumps(dict(by_backend)),
                "by_status": json.dumps(dict(by_status)),
            })

    # Write workload summaries
    _schema.write_workloads(DB_PATH, workloads)
    print(f"Workload summaries written: {len(workloads)}")

    # CSV ingest
    if not SKIP_CSV:
        print("\n=== Processing CSV files ===")
        csv_rows, wl_rows = ingest_csvs()
        _schema.write_csv_jobs(DB_PATH, csv_rows)
        _schema.write_workloads(DB_PATH, wl_rows)
        print(f"CSV job rows written: {len(csv_rows)}, workload summaries: {len(wl_rows)}")
    else:
        print("\n[CSV ingest skipped]")

    # Final stats
    s = _schema.stats(DB_PATH)
    print(f"\n=== Final DB stats ===")
    print(f"  Total jobs in DB:     {s['jobs']}")
    print(f"  Total shots:          {s['shots_total']:,}")
    print(f"  By backend:          {s['by_backend']}")
    print(f"  By status:           {s['by_status']}")
    print(f"  Workload summaries:   {s['workloads']}")
    print(f"  CSV job rows:       {s['csv_jobs']}")

    return s


def main() -> int:
    global SKIP_CSV
    import argparse

    parser = argparse.ArgumentParser(description="Ingest IBM Quantum jobs from E:\\Descargas")
    parser.add_argument("--db-path", default="", help="Path for quantum_jobs.db")
    parser.add_argument("--skip-csv", action="store_true", help="Skip CSV ingestion")
    args = parser.parse_args()
    SKIP_CSV = args.skip_csv

    run(args.db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
