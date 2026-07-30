"""
quantum_corpus.build
=====================
End-to-end orchestrator: ingest all configured sources -> redact -> write to a
SEPARATE quantum_corpus.db -> split-by-document -> count tokens with the
project tokenizer -> build a BM25 RAG index -> run smoke queries -> print an
honest report.

Nothing here touches metatron.db or the public Hugging Face Space. The DB path
defaults to ``quantum_corpus.db`` beside this module (override with
``TMT_QUANTUM_CORPUS_DB`` or ``--db``).

CLI::

    python -m quantum_corpus.build            # full build + report
    python -m quantum_corpus.build --no-reset # add to existing corpus
    python -m quantum_corpus.build --report    # just print stats from existing db
    python -m quantum_corpus.build --query "what does the OTOC circuit measure?"
    python -m quantum_corpus.build --smoke-tests  # run all sub-module self-tests

Source paths default to the user's machine (D:\\<repos>, E:\\Descargas) and are
overridable via env vars (TMT_QC_GRE, TMT_QC_QPYTH, TMT_QC_QAP, TMT_QC_TMTQV,
TMT_QC_DESCARGAS). Each source is wrapped so one failure never aborts the build
(per [[verify-beyond-selftests]] adversarial posture).
"""

from __future__ import annotations

import os
import sys
import glob as _glob

# repo root on sys.path so tokenizer + config import cleanly
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from quantum_corpus import schema, redact, extract, split, tokenize_count, rag


# ── default source config (user's machine) ──────────────────────────────────

def _env(name, default):
    return os.environ.get(name, default)

REPO_SOURCES = [
    # (env, path, project, license, sensitivity, provenance)
    ("TMT_QC_GRE", _env("TMT_QC_GRE", r"D:\Geometric Resonance Engine"),
     "GRE", "MIT", "public", "https://github.com/quantumdynamics927-dotcom/Geometric-Resonance-Engine"),
    ("TMT_QC_QPYTH", _env("TMT_QC_QPYTH", r"D:\QPyth"),
     "QPyth", "proprietary", "internal", "https://github.com/quantumdynamics927-dotcom/QPyth"),
    ("TMT_QC_QAP", _env("TMT_QC_QAP", r"D:\Quantum Assurance Pyramid"),
     "QAP", "proprietary", "internal", "https://github.com/quantumdynamics927-dotcom/quantum_secure_gateway-"),
    ("TMT_QC_TMTQV", _env("TMT_QC_TMTQV", r"D:\TMT_Quantum_Vault-"),
     "TMT_Quantum_Vault", "GPLv3", "sensitive", "https://github.com/quantumdynamics927-dotcom/TMT_Quantum_Vault-"),
]

DESCARGAS = _env("TMT_QC_DESCARGAS", r"E:\Descargas")
QAP_JOBS = os.path.join(_env("TMT_QC_QAP", r"D:\Quantum Assurance Pyramid"), "data", "quantum_jobs")


def _collect_sources():
    """Return a list of (label, callable->(records,counts)). Order = ingest order."""
    tasks = []

    for env, path, project, lic, sens, prov in REPO_SOURCES:
        if not os.path.isdir(path):
            tasks.append((f"repo:{project}", lambda p=path, pr=project, l=lic, s=sens, u=prov:
                          ([], {"_missing": 1})))
            continue
        tasks.append((f"repo:{project}",
                      lambda p=path, pr=project, l=lic, s=sens, u=prov:
                      extract.extract_repo(p, pr, l, s, u)))

    # QAP embedded quantum_jobs (loose job files)
    if os.path.isdir(QAP_JOBS):
        tasks.append(("ibm_jobs:QAP/data/quantum_jobs",
                      lambda: extract.extract_ibm_jobs_dir(QAP_JOBS, "ibm-quantum", QAP_JOBS)))

    # E:\Descargas IBM job zips + loose job files
    if os.path.isdir(DESCARGAS):
        zips = sorted(_glob.glob(os.path.join(DESCARGAS, "workloads*.zip")))
        for z in zips:
            tasks.append((f"ibm_jobs:{os.path.basename(z)}",
                          lambda z=z: extract.extract_ibm_jobs_zip(z, "ibm-quantum", z)))
        # loose job files, if any
        loose = _glob.glob(os.path.join(DESCARGAS, "job-*-info.json"))
        if loose:
            tasks.append(("ibm_jobs:Descargas-loose",
                          lambda: extract.extract_ibm_jobs_dir(DESCARGAS, "ibm-quantum", DESCARGAS)))

        # wormhole manifest
        mpath = os.path.join(DESCARGAS, "wormhole_experiment_manifest.json")
        if os.path.isfile(mpath):
            tasks.append(("manifest:wormhole",
                          lambda: extract.extract_manifest(mpath, "wormhole-suite", mpath)))

        # workload CSV summaries
        for c in sorted(_glob.glob(os.path.join(DESCARGAS, "all_time-workloads*.csv"))):
            tasks.append((f"workload_csv:{os.path.basename(c)}",
                          lambda c=c: extract.extract_workload_csv(c, "ibm-quantum", c)))

        # NOTE: E:\Descargas PDFs are NOT auto-globbed. That folder is a Downloads
        # grab-bag containing credentials (proton-recovery-phrase.pdf), CVs, NDAs,
        # invoices and bank/tax docs. Per the user's own constraint (no raw
        # credentials / private PII in the corpus) we only ingest PDFs from an
        # explicitly-curated directory set via TMT_QC_PDF_DIR.
        pdf_dir = _env("TMT_QC_PDF_DIR", "")
        if pdf_dir and os.path.isdir(pdf_dir):
            for p in sorted(_glob.glob(os.path.join(pdf_dir, "*.pdf"))):
                tasks.append((f"pdf:{os.path.basename(p)}",
                              lambda p=p: extract.extract_pdf(p, "research-papers", p)))

    return tasks


def run_build(db_path: str, reset: bool = True, verbose: bool = True) -> dict:
    """Run the full ingest -> split -> count -> index pipeline. Returns a report dict."""
    if reset and os.path.exists(db_path):
        os.remove(db_path)
    schema.init_db(db_path)

    total_records = 0
    all_counts: dict = {}
    per_source: dict = {}

    for label, fn in _collect_sources():
        try:
            recs, counts = fn()
        except Exception as e:  # never fatal
            per_source[label] = {"records": 0, "error": str(e)}
            if verbose:
                print(f"  [FAIL] {label}: {e}")
            continue
        ins, dup = schema.write_records(db_path, recs)
        total_records += ins
        per_source[label] = {"records": ins, "duplicates": dup}
        for k, v in counts.items():
            all_counts[k] = all_counts.get(k, 0) + v
        if verbose:
            print(f"  [ok]   {label}: {ins} records (+{dup} dup)")

    # split-by-document + token count
    rows = schema.fetch_all(db_path)
    assignments = split.assign_splits(rows)
    # token counts per row
    tok_counts = []
    for r, a in zip(rows, assignments):
        tok_counts.append({"id": a["id"], "split": a["split"],
                            "token_count": tokenize_count.count_tokens(r["text"])})
    schema.update_split_and_tokens(db_path, tok_counts)

    s = schema.stats(db_path)

    # RAG index + smoke
    idx = rag.RAGIndex.build(rows)
    smoke = {
        "otoc": idx.query("what does the OTOC circuit measure?", k=3),
        "ibm_fez": idx.query("ibm_fez backend job sampler", k=3),
        "sierpinski": idx.query("sierpinski golden coin quantum walk", k=3),
        "core13": idx.query("Core-13 coordination lattice agents", k=3),
    }
    # trim snippets for the report
    smoke = {q: [{**h, "snippet": h["snippet"][:80]} for h in hits]
             for q, hits in smoke.items()}

    report = {
        "db_path": db_path,
        "total_records": total_records,
        "stats": s,
        "per_source": per_source,
        "redactions": all_counts,
        "rag_docs": len(idx),
        "smoke": {k: [{kk: vv for kk, vv in h.items() if kk != "snippet"} | {"snippet": h["snippet"][:80]}
                      for h in v] for k, v in smoke.items()},
    }
    return report


def print_report(report: dict) -> None:
    s = report["stats"]
    print("\n" + "=" * 64)
    print(f"QUANTUM CORPUS BUILD  ->  {report['db_path']}")
    print("=" * 64)
    print(f"Total records (deduped): {report['total_records']}  | RAG docs: {report['rag_docs']}")
    print(f"Total tokens (body, project tokenizer): {s['total_tokens']:,}")
    print("\nBy split:")
    for sp in ("train", "val", "test"):
        d = s["by_split"].get(sp, {"records": 0, "tokens": 0})
        print(f"  {sp:5s}: {d['records']:6d} records | {d['tokens']:>10,} tokens")
    print("\nBy source_type:")
    for k, v in sorted(s["by_source"].items()):
        print(f"  {k:14s}: {v}")
    print("\nBy project:")
    for k, v in sorted(s["by_project"].items()):
        print(f"  {k:22s}: {v}")
    print("\nPer-source ingest:")
    for k, v in sorted(report["per_source"].items()):
        print(f"  {k:34s}: {v}")
    rc = report["redactions"]
    print("\nRedactions applied (identifiers/credentials stripped):")
    if not rc:
        print("  (none recorded)")
    for k in sorted(rc):
        print(f"  {k:24s}: {rc[k]}")
    print("\nRAG smoke (top hit per query):")
    for q, hits in report["smoke"].items():
        if hits:
            h = hits[0]
            print(f"  {q:12s} -> [{h['project']}:{h['source_type']}] score={h['score']}  {h['snippet']}")
        else:
            print(f"  {q:12s} -> (no hits)")
    print("=" * 64)


def run_smoke_tests() -> int:
    """Run every sub-module's __main__ self-test as a real subprocess; return
    the number of modules that failed (non-zero exit)."""
    import subprocess
    mods = ["quantum_corpus.schema", "quantum_corpus.redact", "quantum_corpus.extract",
            "quantum_corpus.split", "quantum_corpus.tokenize_count", "quantum_corpus.rag"]
    fails = 0
    for m in mods:
        print(f"\n--- self-test: {m} ---")
        try:
            r = subprocess.run([sys.executable, "-m", m], cwd=_REPO_ROOT,
                               capture_output=True, text=True, encoding="utf-8")
            print(r.stdout.rstrip())
            if r.stderr.strip():
                print("  STDERR:", r.stderr.rstrip()[:500])
            if r.returncode == 0:
                print(f"  OK {m}")
            else:
                print(f"  FAIL {m} (exit {r.returncode})")
                fails += 1
        except Exception as e:
            print(f"  FAIL {m}: {type(e).__name__}: {e}")
            fails += 1
    return fails


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    db_path = schema.default_db_path()

    if "--smoke-tests" in argv:
        return run_smoke_tests()

    if "--query" in argv:
        i = argv.index("--query")
        q = argv[i + 1] if i + 1 < len(argv) else ""
        schema.init_db(db_path)
        rows = schema.fetch_all(db_path)
        idx = rag.RAGIndex.build(rows)
        print(f"RAG index: {len(idx)} docs  (db: {db_path})")
        for h in idx.query(q, k=8):
            print(f"\n[{h['score']}] {h['project']}:{h['source_type']}  ({h['doc_id']})\n  {h['snippet']}")
        return 0

    if "--report" in argv:
        schema.init_db(db_path)
        s = schema.stats(db_path)
        rows = schema.fetch_all(db_path)
        print_report({"db_path": db_path, "total_records": s["total"], "stats": s,
                      "per_source": {}, "redactions": {}, "rag_docs": len(rag.RAGIndex.build(rows)),
                      "smoke": {}})
        return 0

    reset = "--no-reset" not in argv
    report = run_build(db_path, reset=reset)
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 fix
    raise SystemExit(main())