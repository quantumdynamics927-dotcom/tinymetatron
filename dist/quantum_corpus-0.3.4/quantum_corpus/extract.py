"""
quantum_corpus.extract
======================
Per-source-type extractors. Each returns ``(records, redaction_counts)`` where
``records`` are ``schema.Record`` objects whose ``text`` is ALREADY redacted
and chunked, and ``redaction_counts`` is a dict of what the redaction layer
stripped (for the build report).

Robustness contract: one malformed/giant/missing file is skipped + logged, never
fatal. Every file passes through ``redact.redact_text`` before it becomes a
Record (identifiers + credential tokens removed, research content kept).

Sources:
  * extract_repo(root, project, license, sensitivity, provenance_url)
  * extract_ibm_jobs_zip(zip_path, project)        # E:\\Descargas\\workloads*.zip
  * extract_ibm_jobs_dir(dir_path, project)        # QAP data/quantum_jobs (loose)
  * extract_manifest(path, project)                # wormhole_experiment_manifest.json
  * extract_workload_csv(path, project)            # all_time-workloads*.csv (summary)
  * extract_pdf(path, project)                     # optional (PyMuPDF)

Repo files matching ``job-*-info.json`` / ``job-*-result.json`` are skipped by
the repo walker (handled by the dedicated, structured IBM-job extractor).
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from typing import Dict, List, Tuple

from .redact import redact_text, merge_counts, should_skip_file, is_noise_path, is_text_file
from .schema import Record

MAX_CHUNK = 4000          # chars per record (retrieval granularity)
MAX_FILE_CHARS = 200_000  # cap a single file's contribution; log if exceeded

_JOB_INFO_RE = None
try:
    import re as _re
    _JOB_INFO_RE = _re.compile(r"job-[^/]+-(info|result)\.json$")
except Exception:
    pass


# ── helpers ──────────────────────────────────────────────────────────────────

def _chunk(text: str, max_chars: int = MAX_CHUNK) -> List[str]:
    """Split ``text`` into <= max_chars chunks on paragraph/line boundaries."""
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    buf = ""
    for para in text.split("\n\n"):
        piece = para if not buf else buf + "\n\n" + para
        if len(piece) <= max_chars:
            buf = piece
            continue
        # piece too big: flush buf, then split piece by lines / hard cut
        if buf:
            chunks.append(buf)
            buf = ""
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        for line in para.split("\n"):
            if len(line) <= max_chars:
                chunks.append(line)
            else:
                # hard cut
                for i in range(0, len(line), max_chars):
                    chunks.append(line[i:i + max_chars])
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


def _make_records(project: str, doc_id: str, text: str, source_type: str,
                   subdomain: str, license: str, sensitivity: str,
                   provenance: str, risk_tier: int = 0) -> Tuple[List[Record], Dict[str, int]]:
    """Redact + chunk a text blob into one or more Records."""
    red, counts = redact_text(text or "")
    chunks = _chunk(red)
    recs = [
        Record(
            source_type=source_type, project=project, text=ch, doc_id=doc_id,
            subdomain=subdomain, source_license=license, provenance_url=provenance,
            sensitivity=sensitivity, risk_tier=risk_tier,
        )
        for ch in chunks
    ]
    return recs, counts


def _is_job_artifact(relpath: str) -> bool:
    if _JOB_INFO_RE is None:
        base = os.path.basename(relpath)
        return base.startswith("job-") and (base.endswith("-info.json") or
                                            base.endswith("-result.json"))
    return bool(_JOB_INFO_RE.search(relpath.replace("\\", "/")))


# ── repo walker ──────────────────────────────────────────────────────────────

def extract_repo(root: str, project: str, license: str, sensitivity: str,
                  provenance_url: str) -> Tuple[List[Record], Dict[str, int]]:
    records: List[Record] = []
    counts: Dict[str, int] = {}
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # prune noise dirs in-place
        dirnames[:] = [d for d in dirnames if d not in {
            ".git", "node_modules", "__pycache__", ".venv", "venv", "egg-info",
            "dist", "build", "site", ".next", ".ruff_cache", ".pytest_cache",
            ".mypy_cache", ".cache", "quantum-env", ".huggingface", "coverage",
            "__results__", ".ipynb_checkpoints",
        }]
        for fn in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            if should_skip_file(rel) or not is_text_file(rel):
                skipped += 1
                continue
            if _is_job_artifact(rel):
                # handled by the structured IBM-job extractor
                skipped += 1
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    raw = f.read(MAX_FILE_CHARS + 1)
            except OSError:
                skipped += 1
                continue
            if len(raw) > MAX_FILE_CHARS:
                raw = raw[:MAX_FILE_CHARS]
                counts["_files_truncated"] = counts.get("_files_truncated", 0) + 1
            text, ext, subdomain = _extract_file_text(path, fn, raw)
            if not text.strip():
                skipped += 1
                continue
            doc_id = f"{project}:{rel.replace(chr(92), '/')}"
            recs, c = _make_records(
                project, doc_id, text, "repo", subdomain or ext,
                license, sensitivity, provenance_url,
                risk_tier=2 if sensitivity == "sensitive" else 1 if sensitivity == "internal" else 0,
            )
            records.extend(recs)
            merge_counts(counts, c)
    counts["_repo_files_skipped"] = counts.get("_repo_files_skipped", 0) + skipped
    return records, counts


def _extract_file_text(path: str, fn: str, raw: str) -> Tuple[str, str, str]:
    """Return (text, ext_label, subdomain). Special handling for ipynb/json."""
    ext = os.path.splitext(fn)[1].lower()
    if ext == ".ipynb":
        try:
            nb = json.loads(raw)
            parts: List[str] = []
            for cell in nb.get("cells", []):
                ct = cell.get("cell_type", "")
                src = cell.get("source", "")
                if isinstance(src, list):
                    src = "".join(src)
                parts.append(f"[{ct}]\n{src}")
            return "\n\n".join(parts), "ipynb", "notebooks"
        except Exception:
            return raw, "ipynb", "notebooks"
    if ext == ".json":
        try:
            obj = json.loads(raw)
            return _json_to_text(obj), "json", "data"
        except Exception:
            return raw, "json", "data"
    # first path component as subdomain hint
    return raw, ext.lstrip(".") or "txt", ""


def _json_to_text(obj, depth: int = 0) -> str:
    """Flatten a JSON object into readable key:value text (string values kept)."""
    if depth > 6:
        return ""
    parts: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                inner = _json_to_text(v, depth + 1)
                if inner:
                    parts.append(f"{k}: {inner}")
            elif v is None:
                continue
            elif isinstance(v, bool):
                parts.append(f"{k}: {str(v).lower()}")
            else:
                parts.append(f"{k}: {v}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            inner = _json_to_text(v, depth + 1)
            if inner:
                parts.append(f"[{i}] {inner}")
    else:
        parts.append(str(obj))
    return ", ".join(parts)


# ── IBM job records (zips + loose dirs) ──────────────────────────────────────

def _job_record_from_info(info: dict, project: str, num_samples: int,
                           provenance: str) -> Tuple[List[Record], Dict[str, int]]:
    """Build a job record from an info dict. Schema-adaptive: handles both the
    IBM Cloud zip format (id/state/program/params.pubs) and the QAP format
    (job_id/status/program_id/params)."""
    if not isinstance(info, dict):
        return [], {"_non_dict_info": 1}
    jid = info.get("id") or info.get("job_id") or "?"
    backend = info.get("backend", "?")
    state = info.get("state")
    status = state.get("status") if isinstance(state, dict) else info.get("status", "?")
    tags = info.get("tags", []) or []
    cost = info.get("cost", "")
    created = info.get("created", "")
    prog = info.get("program")
    program = prog.get("id") if isinstance(prog, dict) else (
        info.get("program_id") or (prog if isinstance(prog, str) else "") or "")
    # QASM lives in params.pubs[0][0] (IBM Cloud) or params.circuit (QAP) or
    # params itself if it's a string.
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
    text = (
        f"IBM Quantum job {jid} on backend {backend}, status {status}, "
        f"program {program}, tags {tags}, cost {cost}, created {created}. "
        f"Measurement samples: {num_samples}. Circuit (OPENQASM 3.0):\n{qasm}"
    )
    return _make_records(
        project, f"{project}:{jid}", text, "ibm_job", "circuits",
        "proprietary", "sensitive", provenance, risk_tier=2,
    )


def _extract_ibm_pair(info, result, project: str,
                      provenance: str) -> Tuple[List[Record], Dict[str, int]]:
    """Pair an info blob with its result blob (either may be non-dict / a Qiskit
    object repr string). Robust: never raises on odd result shapes."""
    if not isinstance(info, dict):
        return [], {"_non_dict_info": 1}
    num_samples = 0
    if isinstance(result, dict):
        try:
            res = result.get("results") or []
            if res and isinstance(res, list):
                data = (res[0].get("data") or {}) if isinstance(res[0], dict) else {}
                c = data.get("c") or data.get("register") or data.get("meas") or {}
                samples = c.get("samples") if isinstance(c, dict) else None
                num_samples = len(samples) if isinstance(samples, list) else 0
        except Exception:
            num_samples = 0
    elif isinstance(result, str):
        # QAP-style: a Qiskit object repr like "...num_shots=8192, num_bits=127..."
        m = re.search(r"num_shots=(\d+)", result)
        num_samples = int(m.group(1)) if m else 0
    return _job_record_from_info(info, project, num_samples, provenance)


def extract_ibm_jobs_zip(zip_path: str, project: str = "ibm-quantum",
                         provenance: str = "") -> Tuple[List[Record], Dict[str, int]]:
    records: List[Record] = []
    counts: Dict[str, int] = {}
    infos: Dict[str, dict] = {}
    results: Dict[str, dict] = {}
    try:
        zf = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile):
        counts["_bad_zip"] = counts.get("_bad_zip", 0) + 1
        return records, counts
    try:
        for name in zf.namelist():
            if name.endswith("-info.json"):
                key = name[: -len("-info.json")]
                try:
                    infos[key] = json.loads(zf.read(name))
                except Exception:
                    counts["_bad_info"] = counts.get("_bad_info", 0) + 1
            elif name.endswith("-result.json"):
                key = name[: -len("-result.json")]
                try:
                    results[key] = json.loads(zf.read(name))
                except Exception:
                    counts["_bad_result"] = counts.get("_bad_result", 0) + 1
        for key, info in infos.items():
            recs, c = _extract_ibm_pair(
                info, results.get(key, {}), project, provenance or zip_path
            )
            records.extend(recs)
            merge_counts(counts, c)
    finally:
        zf.close()
    return records, counts


def extract_ibm_jobs_dir(dir_path: str, project: str = "ibm-quantum",
                         provenance: str = "") -> Tuple[List[Record], Dict[str, int]]:
    records: List[Record] = []
    counts: Dict[str, int] = {}
    if not os.path.isdir(dir_path):
        return records, counts
    infos: Dict[str, dict] = {}
    results: Dict[str, dict] = {}
    for fn in os.listdir(dir_path):
        p = os.path.join(dir_path, fn)
        if not os.path.isfile(p):
            continue
        if fn.endswith("-info.json"):
            key = fn[: -len("-info.json")]
            try:
                with open(p, "rb") as f:
                    infos[key] = json.loads(f.read())
            except Exception:
                counts["_bad_info"] = counts.get("_bad_info", 0) + 1
        elif fn.endswith("-result.json"):
            key = fn[: -len("-result.json")]
            try:
                with open(p, "rb") as f:
                    results[key] = json.loads(f.read())
            except Exception:
                counts["_bad_result"] = counts.get("_bad_result", 0) + 1
    for key, info in infos.items():
        recs, c = _extract_ibm_pair(info, results.get(key, {}), project, provenance or dir_path)
        records.extend(recs)
        merge_counts(counts, c)
    return records, counts


# ── wormhole manifest ────────────────────────────────────────────────────────

def extract_manifest(path: str, project: str = "wormhole-suite",
                      provenance: str = "") -> Tuple[List[Record], Dict[str, int]]:
    records: List[Record] = []
    counts: Dict[str, int] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        counts["_manifest_missing"] = counts.get("_manifest_missing", 0) + 1
        return records, counts

    # top-level prose record
    top = (
        f"Wormhole experiment suite: {data.get('project','?')}. "
        f"Purpose: {data.get('purpose','?')}. "
        f"Target backend: {data.get('target_backend','?')} "
        f"({data.get('backend_rationale','')}). "
        f"Owner: {data.get('owner','?')}. Created: {data.get('created_at','?')}."
    )
    recs, c = _make_records(project, f"{project}:manifest", top, "manifest",
                            "experiment-design", "proprietary", "sensitive",
                            provenance or path, risk_tier=1)
    records.extend(recs)
    merge_counts(counts, c)

    # one record per circuit
    for circ in data.get("circuits", []) or []:
        cid = circ.get("id", "?")
        text = (
            f"Circuit {cid}: {circ.get('name','?')}. "
            f"Purpose: {circ.get('purpose','?')}. "
            f"Measures: {circ.get('measures','?')}. "
            f"What you get: {circ.get('what_you_get','?')}. "
            f"Circuits count {circ.get('circuits_count','?')}, "
            f"shots per circuit {circ.get('shots_per_circuit','?')}, "
            f"total shots {circ.get('total_shots','?')}, "
            f"estimated runtime (min) {circ.get('estimated_runtime_minutes','?')}. "
            f"Acceptance: {json.dumps(circ.get('acceptance',{}), ensure_ascii=False)}."
        )
        recs, c = _make_records(project, f"{project}:circuit-{cid}", text,
                                "manifest", "experiment-design", "proprietary",
                                "sensitive", provenance or path, risk_tier=1)
        records.extend(recs)
        merge_counts(counts, c)
    return records, counts


# ── workload CSV -> aggregate summary ───────────────────────────────────────

def extract_workload_csv(path: str, project: str = "ibm-quantum",
                          provenance: str = "") -> Tuple[List[Record], Dict[str, int]]:
    import csv as _csv

    records: List[Record] = []
    counts: Dict[str, int] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
    except OSError:
        counts["_csv_missing"] = counts.get("_csv_missing", 0) + 1
        return records, counts
    if not rows:
        return records, counts
    # aggregate (account/user redacted by redact_text via bare-acct pattern)
    by_backend: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    total_usage = 0.0
    for r in rows:
        b = r.get("QPU") or r.get("backend") or "?"
        by_backend[b] = by_backend.get(b, 0) + 1
        s = r.get("Status", "?")
        by_status[s] = by_status.get(s, 0) + 1
        try:
            total_usage += float(r.get("Usage (seconds)") or 0)
        except ValueError:
            pass
    text = (
        f"IBM Quantum workload summary from {os.path.basename(path)}: "
        f"{len(rows)} jobs. By backend: {by_backend}. By status: {by_status}. "
        f"Total usage (seconds): {total_usage:.0f}. "
        f"Accounts/user identifiers redacted."
    )
    recs, c = _make_records(project, f"{project}:csv-{os.path.basename(path)}",
                            text, "workload_csv", "jobs-summary",
                            "proprietary", "sensitive", provenance or path, risk_tier=2)
    records.extend(recs)
    merge_counts(counts, c)
    return records, counts


# ── PDF (optional) ───────────────────────────────────────────────────────────

def extract_pdf(path: str, project: str = "research-papers",
                 provenance: str = "") -> Tuple[List[Record], Dict[str, int]]:
    records: List[Record] = []
    counts: Dict[str, int] = {}
    try:
        import fitz  # type: ignore
    except Exception:
        counts["_pdf_no_fitz"] = counts.get("_pdf_no_fitz", 0) + 1
        return records, counts
    try:
        doc = fitz.open(path)
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
    except Exception:
        counts["_pdf_read_fail"] = counts.get("_pdf_read_fail", 0) + 1
        return records, counts
    if not text.strip():
        return records, counts
    recs, c = _make_records(project, f"{project}:{os.path.basename(path)}",
                            text, "pdf", "research-prose", "proprietary",
                            "sensitive", provenance or path, risk_tier=2)
    records.extend(recs)
    merge_counts(counts, c)
    return records, counts


# ── self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import tempfile

    sys.stdout.reconfigure(encoding="utf-8")

    def _ok(c, m):
        print(("OK  " if c else "FAIL") + " " + m)
        assert c, m

    tmp = tempfile.mkdtemp(prefix="qcorpus_ext_")

    # repo with a job artifact that should be skipped by the repo walker
    rp = os.path.join(tmp, "mini_repo")
    os.makedirs(os.path.join(rp, "docs"))
    with open(os.path.join(rp, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Mini\n\nIBMid-695001BQB4 ran here on ibm_fez.\n\nSecond paragraph stays.")
    with open(os.path.join(rp, "docs", "notes.md"), "w", encoding="utf-8") as f:
        f.write("Notes about ghp_AAAAAAAAAAAAAAAAAAAAAAAA token leak.")
    # job artifact -> skipped by repo walker
    with open(os.path.join(rp, "job-d5a6-info.json"), "w", encoding="utf-8") as f:
        json.dump({"id": "d5a6", "backend": "ibm_fez"}, f)
    recs, counts = extract_repo(rp, "mini", "MIT", "public", "local:mini")
    texts = " ".join(r.text for r in recs)
    _ok(len(recs) >= 2, f"repo produced records: {len(recs)}")
    _ok("IBMid-695001BQB4" not in texts, "ibmid redacted in repo extract")
    _ok("ghp_AAAAAAAAAAAAAAAAAAAAAAAA" not in texts, "token redacted in repo extract")
    _ok("ibm_fez" in texts, "backend preserved")
    _ok(not any(r.doc_id.startswith("mini:job-d5a6") for r in recs), "job artifact skipped by repo walker")
    print("  repo redaction counts:", counts)

    # manifest
    mp = os.path.join(tmp, "manifest.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump({
            "project": "QuantaCore", "purpose": "ER=EPR wormhole verification",
            "target_backend": "ibm_kingston", "backend_rationale": "0 queue",
            "owner": "q927", "created_at": "2026-04-30T03:09:00Z",
            "circuits": [
                {"id": 1, "name": "OTOC", "purpose": "scrambling",
                 "measures": "C(t)", "what_you_get": "Lyapunov",
                 "circuits_count": 7, "shots_per_circuit": 8192,
                 "total_shots": 57344, "estimated_runtime_minutes": 15,
                 "acceptance": {"lambda_L_min": 0.5}},
            ],
        }, f)
    recs, c = extract_manifest(mp)
    _ok(len(recs) == 2, f"manifest -> top + 1 circuit = {len(recs)}")
    _ok("ibm_kingston" in recs[0].text and "OTOC" in recs[1].text, "manifest content kept")

    # zip with a job missing its result file (adversarial)
    zp = os.path.join(tmp, "w.zip")
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("job-AAA-info.json", json.dumps({
            "id": "AAA", "backend": "ibm_fez", "state": {"status": "Completed"},
            "tags": ["Composer"], "cost": 600, "created": "2025-12-31T00:25:30Z",
            "program": {"id": "sampler"},
            "params": {"pubs": [["OPENQASM 3.0; rz(pi/2) $0;"]]},
        }))
        # NO matching result file
    recs, c = extract_ibm_jobs_zip(zp)
    _ok(len(recs) == 1, f"job with missing result still ingested: {len(recs)}")
    _ok("Measurement samples: 0" in recs[0].text, "missing result -> 0 samples")
    _ok("ibm_fez" in recs[0].text, "backend kept")

    # bad zip -> no crash
    recs, c = extract_ibm_jobs_zip(os.path.join(tmp, "nope.zip"))
    _ok(recs == [] and c.get("_bad_zip") == 1, "bad zip handled")

    # QAP-schema job: job_id/program_id/top-level status + a Qiskit-repr STRING
    # result (not a dict). Must not crash; num_shots parsed from the repr.
    qap_info = {"job_id": "d4mfl02v0j9c73e69300", "status": "Completed",
                "created": "2025-12-31T00:25:30Z", "ended": "2025-12-31T00:25:37Z",
                "backend": "ibm_fez", "program_id": "sampler",
                "params": {"circuit": "OPENQASM 3.0; qreg q[2]; h q[0];"}}
    qap_result_str = ("PrimitiveResult([SamplerPubResult(data=DataBin(meas=BitArray("
                      "<shape=(), num_shots=8192, num_bits=127>)))])")
    recs, c = _extract_ibm_pair(qap_info, qap_result_str, "ibm-quantum", "qap")
    _ok(len(recs) >= 1, f"QAP-schema job ingested: {len(recs)}")
    _ok("d4mfl02v0j9c73e69300" in recs[0].text, "QAP job_id kept")
    _ok("ibm_fez" in recs[0].text and "sampler" in recs[0].text, "QAP backend/program kept")
    _ok("Measurement samples: 8192" in recs[0].text, "num_shots parsed from repr string")
    _ok("h q[0]" in recs[0].text, "QAP params.circuit QASM kept")

    # QAP-style loose dir with a string-result file -> extract_ibm_jobs_dir
    qd = os.path.join(tmp, "qap_jobs")
    os.makedirs(qd)
    with open(os.path.join(qd, "JOB1-info.json"), "w", encoding="utf-8") as f:
        json.dump(qap_info, f)
    with open(os.path.join(qd, "JOB1-result.json"), "w", encoding="utf-8") as f:
        f.write('"' + qap_result_str + '"')  # JSON-encoded string
    recs, c = extract_ibm_jobs_dir(qd, "ibm-quantum", qd)
    _ok(len(recs) >= 1 and "ibm_fez" in recs[0].text, "QAP loose dir ingested without crash")

    # non-dict info -> skipped gracefully
    recs, c = _extract_ibm_pair("not a dict", {}, "ibm-quantum", "x")
    _ok(recs == [] and c.get("_non_dict_info") == 1, "non-dict info skipped")

    # workload csv summary with account redaction
    cp = os.path.join(tmp, "wl.csv")
    with open(cp, "w", encoding="utf-8", newline="") as f:
        f.write("WorkloadId,Status,Instance,Region,Mode,Created,Completed,QPU,Usage (seconds),User,Tags,Account\n")
        f.write("d1,completed,crn,us-east,job,2025-11-13T22:29:20Z,2025-11-13T22:29:46Z,ibm_torino,3,kub chme,,06175211d06f464ba15a52c048b1712a\n")
        f.write("d2,canceled,crn,us-east,job,2025-11-13T22:27:56Z,2025-11-13T22:28:01Z,ibm_torino,2,kub chme,,06175211d06f464ba15a52c048b1712a\n")
    recs, c = extract_workload_csv(cp)
    _ok(len(recs) == 1, "csv -> 1 summary record")
    _ok("06175211d06f464ba15a52c048b1712a" not in recs[0].text, "csv account redacted")
    _ok("ibm_torino" in recs[0].text, "csv backend kept")
    _ok("2 jobs" in recs[0].text, "csv aggregate count")

    # chunking: big text splits into multiple records, same doc_id
    big = ("para " * 50 + "\n\n") * 30
    recs, _ = _make_records("p", "p:doc", big, "repo", "docs", "MIT", "public", "x")
    _ok(len(recs) > 1, f"big text chunked into {len(recs)} records")
    _ok(len({r.doc_id for r in recs}) == 1, "chunked records share doc_id")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("SELF-TEST PASSED")