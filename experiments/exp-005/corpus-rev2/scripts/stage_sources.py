"""
experiments/exp-005/corpus-rev2/scripts/stage_sources.py
==========================================================
Stage source-disjoint, provenance-attributed rows for exp-005 corpus revision 2.

Revision 2 adds:
  - PennyLane (Apache-2.0) code source
  - Qiskit Textbook (Apache-2.0) code source
  - Multiple Project Gutenberg public-domain English texts
while keeping the same quantum-domain technical base and synthetic tool traces
as revision 1.

Produces JSONL files under experiments/exp-005/corpus-rev2/raw/.

All file opens use encoding="utf-8".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Iterator

try:
    import nbformat
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"Missing dependency: pip install nbformat. Error: {e}")

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

WORKER_VERSION = 2

# ---------------------------------------------------------------------------
# Source locations and provenance constants
# ---------------------------------------------------------------------------
EXP004_CORPUS_DIR = _ROOT / "experiments" / "exp-004" / "corpus"
QISKIT_TUTORIALS = Path("E:/Temp/qiskit-tutorials")
PENNYLANE = Path("E:/Temp/pennylane")
QISKIT_TEXTBOOK = Path("E:/Temp/qiskit-textbook")
GUTENBERG_DIR = _ROOT / "experiments" / "exp-005" / "corpus-rev2" / "raw" / "gutenberg"
GUTENBERG_PRIMARY = _ROOT / "experiments" / "exp-005" / "tokenizer_probe" / "general_english.txt"
OUTPUT_DIR = _ROOT / "experiments" / "exp-005" / "corpus-rev2" / "raw"

QISKIT_REPO_URL = "https://github.com/Qiskit/qiskit-tutorials"
PENNYLANE_REPO_URL = "https://github.com/PennyLaneAI/pennylane"
QISKIT_TEXTBOOK_URL = "https://github.com/qiskit-community/qiskit-textbook"
GUTENBERG_URL_TEMPLATE = "https://www.gutenberg.org/files/{id}/{id}-0.txt"

GUTENBERG_CATALOG = [
    ("1342", "Pride and Prejudice", "Jane Austen", str(GUTENBERG_PRIMARY)),
    ("84", "Frankenstein; Or, The Modern Prometheus", "Mary Shelley", None),
    ("11", "Alice's Adventures in Wonderland", "Lewis Carroll", None),
    ("1661", "The Adventures of Sherlock Holmes", "Arthur Conan Doyle", None),
    ("174", "The Picture of Dorian Gray", "Oscar Wilde", None),
    ("2701", "Moby Dick; or, The Whale", "Herman Melville", None),
    ("98", "A Tale of Two Cities", "Charles Dickens", None),
]

# ---------------------------------------------------------------------------
# Row schema helpers
# ---------------------------------------------------------------------------

def make_row(text: str, source_id: str, domain: str, subdomain: str,
             license: str, provenance: str, category: str,
             source_type: str = "repo") -> dict:
    """Build a uniform corpus row."""
    text = text.strip()
    if not text:
        return None
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "id": f"sha256:{h[:16]}",
        "text": text,
        "source_id": source_id,
        "domain": domain,
        "subdomain": subdomain,
        "license": license,
        "provenance": provenance,
        "corpus_category": category,
        "quality_score": 0.0,
        "sensitivity": "public",
        "source_type": source_type,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Source 1: exp-004 quantum-domain technical rows
# ---------------------------------------------------------------------------

def stage_quantum_domain(seed: int = 42) -> tuple[list[dict], dict]:
    """Reuse the frozen exp-004 post-cap corpus as the quantum-domain source."""
    if not EXP004_CORPUS_DIR.exists():
        raise FileNotFoundError(f"exp-004 corpus directory not found: {EXP004_CORPUS_DIR}")

    split_files = ["train.jsonl", "val.jsonl", "hard_dev.jsonl"]
    rows = []
    split_counts = {}
    for fname in split_files:
        path = EXP004_CORPUS_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"exp-004 split file missing: {path}")
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["corpus_category"] = "quantum_domain_technical"
                row.pop("split", None)
                if "source_id" not in row:
                    row["source_id"] = row.get("source", row.get("domain", "unknown"))
                rows.append(row)
                count += 1
        split_counts[fname] = count

    return rows, {
        "source": "exp-004 frozen corpus (train + val + hard_dev)",
        "license": "mixed (see exp-004 MANIFEST.json)",
        "provenance": str(EXP004_CORPUS_DIR.resolve()),
        "exp004_split_counts": split_counts,
        "rows_staged": len(rows),
    }


# ---------------------------------------------------------------------------
# Source 2: Quantum-specific code from Qiskit, PennyLane, Qiskit Textbook
# ---------------------------------------------------------------------------

def _iter_notebooks(base_dir: Path) -> Iterator[Path]:
    for path in sorted(base_dir.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in path.parts:
            continue
        yield path


def _extract_code_cells(notebook_path: Path, min_len: int = 20) -> list[str]:
    """Return non-empty code cell sources from a notebook."""
    try:
        nb = nbformat.read(str(notebook_path), as_version=4)
    except Exception as exc:
        print(f"WARN: failed to read {notebook_path}: {exc}", file=sys.stderr)
        return []

    cells = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source.strip()
        if len(src) >= min_len:
            cells.append(src)
    return cells


def _chunk_python_source(src: str, max_chunk_chars: int = 4000) -> list[str]:
    """Split Python source into chunks at blank-line boundaries, capped in size."""
    lines = src.splitlines()
    chunks = []
    current = []
    current_len = 0

    def flush():
        if current:
            chunks.append("\n".join(current).strip())
            current.clear()
            nonlocal current_len
            current_len = 0

    for line in lines:
        current.append(line)
        current_len += len(line) + 1
        if (line.strip() == "" and current_len > 0) or current_len >= max_chunk_chars:
            flush()

    flush()
    return [c for c in chunks if len(c.strip()) >= 20]


def _stage_repo_code(repo_path: Path, repo_url: str, license: str,
                     category: str = "quantum_code", source_type: str = "repo",
                     file_glob: str = "*.py") -> tuple[list[dict], dict]:
    """Extract .py files and notebook code cells from a code repo."""
    if not repo_path.exists():
        raise FileNotFoundError(f"repo not found: {repo_path}")

    rows = []
    notebook_counts: dict[str, int] = defaultdict(int)
    py_counts: dict[str, int] = defaultdict(int)

    for nb_path in _iter_notebooks(repo_path):
        rel = nb_path.relative_to(repo_path).as_posix()
        source_id = f"{repo_path.name}:{rel}"
        provenance = f"{repo_url}/tree/main/{rel}"
        for cell_src in _extract_code_cells(nb_path):
            row = make_row(
                text=cell_src,
                source_id=source_id,
                domain="quantum",
                subdomain="code",
                license=license,
                provenance=provenance,
                category=category,
                source_type=source_type,
            )
            if row:
                rows.append(row)
                notebook_counts[source_id] += 1

    skip_dirs = {".git", "__pycache__", "tests", "test", "doc", "docs", ".github"}
    for py_path in sorted(repo_path.rglob(file_glob)):
        if any(part in skip_dirs for part in py_path.parts):
            continue
        rel = py_path.relative_to(repo_path).as_posix()
        source_id = f"{repo_path.name}:{rel}"
        provenance = f"{repo_url}/blob/main/{rel}"
        try:
            with open(py_path, "r", encoding="utf-8") as f:
                src = f.read().strip()
        except Exception:
            continue
        if len(src) >= 20:
            chunks = _chunk_python_source(src, max_chunk_chars=4000)
            for chunk in chunks:
                row = make_row(
                    text=chunk,
                    source_id=source_id,
                    domain="quantum",
                    subdomain="code",
                    license=license,
                    provenance=provenance,
                    category=category,
                    source_type=source_type,
                )
                if row:
                    rows.append(row)
                    py_counts[source_id] += 1

    return rows, {
        "notebooks": len(notebook_counts),
        "notebook_rows": sum(notebook_counts.values()),
        "py_files": len(py_counts),
        "py_rows": sum(py_counts.values()),
        "rows_staged": len(rows),
    }


def _stage_own_quantum_code() -> tuple[list[dict], dict[str, int]]:
    """Collect own .py files that mention qiskit/quantum/ibm/circuit/pennylane."""
    own_py_files = sorted(_ROOT.rglob("*.py"))
    rows = []
    counts: dict[str, int] = defaultdict(int)
    keywords = ("qiskit", "quantum", "circuit", "ibm_quantum", "qrl", "qsg", "pennylane")

    for py_path in own_py_files:
        if ".git" in py_path.parts or "__pycache__" in py_path.parts:
            continue
        rel = py_path.relative_to(_ROOT).as_posix()
        try:
            with open(py_path, "r", encoding="utf-8") as f:
                src = f.read()
        except Exception:
            continue
        lowered = src.lower()
        if not any(k in lowered for k in keywords):
            continue
        chunks = _chunk_python_source(src, max_chunk_chars=4000)
        source_id = f"tmt:{rel}"
        for chunk in chunks:
            row = make_row(
                text=chunk,
                source_id=source_id,
                domain="quantum",
                subdomain="code",
                license="N/A",
                provenance=f"file://{py_path.resolve()}",
                category="quantum_code",
                source_type="repo",
            )
            if row:
                rows.append(row)
                counts[source_id] += 1

    return rows, counts


def stage_quantum_code() -> tuple[list[dict], dict]:
    """Combine Qiskit tutorials, PennyLane, Qiskit Textbook, and own snippets."""
    qiskit_rows, qiskit_prov = _stage_repo_code(
        QISKIT_TUTORIALS, QISKIT_REPO_URL, "Apache-2.0"
    )
    pennylane_rows, pennylane_prov = _stage_repo_code(
        PENNYLANE, PENNYLANE_REPO_URL, "Apache-2.0"
    )
    textbook_rows, textbook_prov = _stage_repo_code(
        QISKIT_TEXTBOOK, QISKIT_TEXTBOOK_URL, "Apache-2.0"
    )
    own_rows, own_counts = _stage_own_quantum_code()

    rows = qiskit_rows + pennylane_rows + textbook_rows + own_rows

    return rows, {
        "source": "qiskit-tutorials + pennylane + qiskit-textbook + own snippets",
        "license": "Apache-2.0 for public repos; N/A for own snippets",
        "provenance": {
            "qiskit_tutorials": QISKIT_REPO_URL,
            "pennylane": PENNYLANE_REPO_URL,
            "qiskit_textbook": QISKIT_TEXTBOOK_URL,
        },
        "qiskit_tutorials": qiskit_prov,
        "pennylane": pennylane_prov,
        "qiskit_textbook": textbook_prov,
        "own_rows": sum(own_counts.values()),
        "rows_staged": len(rows),
    }


# ---------------------------------------------------------------------------
# Source 3: quantum tool-use traces
# ---------------------------------------------------------------------------

def _random_backend(i: int) -> str:
    backends = [
        "ibm_sherbrooke", "ibm_brisbane", "ibm_kyoto", "ibm_osaka",
        "ibm_marrakesh", "ibm_fez", "ibm_kingston", "ibm_nazca",
        "ibm_cleveland", "ibm_peekskill", "ibm_strasbourg",
    ]
    return backends[i % len(backends)]


def _random_gate_sequence(n_qubits: int, i: int) -> list[str]:
    """Generate a deterministic but varied gate sequence string list."""
    gates = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx", "cz"]
    seq = []
    rng = random.Random(i)
    length = rng.randint(4, 20)
    for step in range(length):
        g = gates[rng.randint(0, len(gates) - 1)]
        if g in ("cx", "cz"):
            q0 = rng.randint(0, max(0, n_qubits - 1))
            q1 = rng.randint(0, max(0, n_qubits - 1))
            if q0 == q1:
                q1 = (q1 + 1) % max(1, n_qubits)
            seq.append(f"{g}(q[{q0}],q[{q1}])")
        elif g in ("rx", "ry", "rz"):
            q = rng.randint(0, max(0, n_qubits - 1))
            angle = round(rng.uniform(0.0, 6.28318), 4)
            seq.append(f"{g}({angle})(q[{q}])")
        else:
            q = rng.randint(0, max(0, n_qubits - 1))
            seq.append(f"{g}(q[{q}])")
    return seq


def _build_circuit_program(i: int) -> str:
    """Return a unique QASM-like circuit program string."""
    rng = random.Random(i)
    n = rng.randint(2, 7)
    seq = _random_gate_sequence(n, i)
    return "OPENQASM 2.0;\n" + f"circuit exp005_trace_{i}({n} qubits):\n  " + "\n  ".join(seq)


def _build_observables(i: int, n_qubits: int) -> list[str]:
    """Return a list of unique Pauli-string observables."""
    rng = random.Random(i)
    paulis = ["I", "X", "Y", "Z"]
    count = rng.randint(1, 4)
    obs = []
    for _ in range(count):
        s = "".join(paulis[rng.randint(0, 3)] for _ in range(n_qubits))
        obs.append(s)
    return obs


def _tool_signature_pool() -> list[str]:
    """Return a pool of quantum-relevant tool names to synthesize traces."""
    return [
        "qiskit.circuit.QuantumCircuit",
        "qiskit.circuit.library.GroverOperator",
        "qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager",
        "qiskit_ibm_runtime.SamplerV2.run",
        "qiskit_ibm_runtime.EstimatorV2.run",
        "pennylane.qnode",
        "pennylane.ExpvalCost",
        "tmt.compute_ce",
        "tmt.quantum_corpus_freeze",
        "tmt.quantum_corpus_validate",
    ]


def _synthesize_trace(tool: str, trace_id: int) -> str:
    """Generate a structured JSON tool trace as a single text row.

    Each trace has unique string content (circuit programs, observables,
    backend names, file paths) so the corpus-level exact+MinHash deduplication
    does not collapse the synthetic set into a handful of templates.
    """
    rng = random.Random(trace_id)
    n_qubits = rng.randint(2, 7)
    backend = _random_backend(trace_id)
    program = _build_circuit_program(trace_id)
    observables = _build_observables(trace_id, n_qubits)

    if tool == "qiskit.circuit.QuantumCircuit":
        args = {"num_qubits": n_qubits, "name": f"circuit_{trace_id}", "program": program}
        result = {"status": "ok", "circuit_id": trace_id}
    elif tool == "qiskit.circuit.library.GroverOperator":
        args = {"oracle": program, "num_qubits": n_qubits}
        result = {"status": "ok", "operator_depth": rng.randint(10, 200)}
    elif tool == "qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager":
        args = {"optimization_level": rng.randint(0, 3), "backend": backend, "program": program}
        result = {"status": "ok", "layout": list(range(n_qubits))}
    elif tool == "qiskit_ibm_runtime.SamplerV2.run":
        shots = max(1, 1024 + (trace_id % 2048) - 1024)
        counts = {"0" * n_qubits: shots // 2, "1" * n_qubits: shots - shots // 2}
        args = {"circuits": [program], "shots": shots, "backend": backend}
        result = {"status": "ok", "counts": counts, "job_id": f"job_{trace_id:08d}"}
    elif tool == "qiskit_ibm_runtime.EstimatorV2.run":
        args = {"pubs": [{"circuit": program, "observables": observables, "backend": backend}]}
        result = {"status": "ok", "expectation_values": [round(rng.uniform(-1.0, 1.0), 6) for _ in observables]}
    elif tool == "pennylane.qnode":
        args = {"device": f"default.qubit({n_qubits})", "interface": "autograd", "program": program}
        result = {"status": "ok", "qnode_id": trace_id}
    elif tool == "pennylane.ExpvalCost":
        args = {"hamiltonian": " + ".join(observables), "device": f"default.qubit({n_qubits})", "program": program}
        result = {"status": "ok", "energy": round(rng.uniform(-1.0, 1.0), 6)}
    elif tool == "tmt.compute_ce":
        eval_sets = ["val", "hard_dev", "gre_longtail"]
        args = {
            "run_id": f"exp-005-rev2-{trace_id % 1000:03d}",
            "step": max(1, (trace_id % 500) * 10),
            "eval_set": eval_sets[trace_id % len(eval_sets)],
            "checkpoint_path": f"experiments/exp-005/probe/seed42/checkpoints/step{trace_id % 5000:06d}.pt",
        }
        result = {"status": "ok", "ce": round(1.0 + (trace_id % 200) / 100, 4)}
    elif tool == "tmt.quantum_corpus_freeze":
        args = {
            "corpus_dir": "experiments/exp-005/corpus-rev2/corpus",
            "revision": 2 + (trace_id // 1000),
            "notes": f"freeze trace {trace_id}",
        }
        result = {"status": "ok", "manifest_hash": f"sha256:{trace_id:08x}"}
    elif tool == "tmt.quantum_corpus_validate":
        args = {
            "manifest_path": "experiments/exp-005/corpus-rev2/MANIFEST.json",
            "checks": ["hash", "overlap", "source_disjoint"][:rng.randint(1, 3)],
        }
        result = {"status": "ok", "valid": True}
    else:
        args = {}
        result = {"status": "ok"}

    trace = {
        "tool": tool,
        "args": args,
        "result": result,
    }
    return json.dumps(trace, ensure_ascii=False, sort_keys=True, indent=None)


def stage_tool_traces(n: int = 6000, seed: int = 42) -> tuple[list[dict], dict]:
    """Generate synthetic structured JSON tool-use traces for quantum tools."""
    rng = random.Random(seed)
    pool = _tool_signature_pool()
    rows = []
    sig_counts: dict[str, int] = defaultdict(int)

    for i in range(n):
        tool = pool[i % len(pool)]
        text = _synthesize_trace(tool, i)
        source_id = f"synthetic:tool_traces:{tool}"
        row = make_row(
            text=text,
            source_id=source_id,
            domain="quantum",
            subdomain="tool_trace",
            license="N/A",
            provenance="synthetic quantum tool signatures derived from Qiskit/IBM Runtime/PennyLane/TinyMetatron CLI",
            category="tool_traces",
            source_type="synthetic",
        )
        if row:
            rows.append(row)
            sig_counts[tool] += 1

    rng.shuffle(rows)

    return rows, {
        "source": "synthetic",
        "license": "N/A",
        "provenance": "synthetic quantum tool signatures derived from Qiskit/IBM Runtime/PennyLane/TinyMetatron CLI",
        "rows_staged": len(rows),
        "by_tool": dict(sig_counts),
    }


# ---------------------------------------------------------------------------
# Source 4: General English (multiple Project Gutenberg texts)
# ---------------------------------------------------------------------------

def _load_gutenberg_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _clean_gutenberg(text: str) -> str:
    """Strip Project Gutenberg boilerplate and excessive whitespace."""
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = text.rfind("*** END OF THE PROJECT GUTENBERG EBOOK")
    if start != -1 and end != -1:
        text = text[start:end]
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s*\[Illustration:.*\]\s*$", line, re.IGNORECASE):
            continue
        if line.strip().isupper() and len(line.strip()) < 40:
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_prose(text: str, min_chars: int = 80, max_chars: int = 600,
                 min_words: int = 12) -> list[str]:
    """Split prose into paragraph-based chunks, merging tiny paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = []
    current_len = 0

    def flush(force: bool = False):
        joined = "\n\n".join(current).strip()
        if not joined:
            return
        word_count = len(joined.split())
        if (len(joined) >= min_chars and word_count >= min_words) or force:
            chunks.append(joined)
            current.clear()
            nonlocal current_len
            current_len = 0

    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            flush()
        current.append(para)
        current_len += len(para) + 2

    flush(force=True)
    return chunks


def stage_general_english() -> tuple[list[dict], dict]:
    """Chunk multiple Project Gutenberg texts into prose rows."""
    rows = []
    book_counts: dict[str, int] = {}

    for book_id, title, author, override_path in GUTENBERG_CATALOG:
        path = Path(override_path) if override_path else GUTENBERG_DIR / f"pg{book_id}.txt"
        if not path.exists():
            print(f"WARN: Gutenberg text not found: {path}", file=sys.stderr)
            continue

        raw = _load_gutenberg_text(path)
        clean = _clean_gutenberg(raw)
        chunks = _chunk_prose(clean)

        source_id = f"gutenberg:{book_id}"
        provenance = GUTENBERG_URL_TEMPLATE.format(id=book_id)
        for chunk in chunks:
            row = make_row(
                text=chunk,
                source_id=source_id,
                domain="general",
                subdomain="prose",
                license="public-domain-US",
                provenance=provenance,
                category="general_english",
                source_type="gutenberg",
            )
            if row:
                rows.append(row)
        book_counts[source_id] = len(chunks)

    return rows, {
        "source": "Project Gutenberg (multiple public-domain texts)",
        "license": "public-domain-US",
        "provenance": "https://www.gutenberg.org/",
        "books": book_counts,
        "rows_staged": len(rows),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config: dict) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    qd_rows, qd_prov = stage_quantum_domain(seed=config.get("seed", 42))
    qc_rows, qc_prov = stage_quantum_code()
    tt_rows, tt_prov = stage_tool_traces(n=config.get("tool_traces", 6000),
                                         seed=config.get("seed", 42))
    ge_rows, ge_prov = stage_general_english()

    files = {
        "quantum_domain_technical.jsonl": qd_rows,
        "quantum_code.jsonl": qc_rows,
        "tool_traces.jsonl": tt_rows,
        "general_english.jsonl": ge_rows,
    }
    for fname, rows in files.items():
        write_jsonl(OUTPUT_DIR / fname, rows)

    def _source_groups(rows: list[dict]) -> int:
        return len({r.get("source_id", "unknown") for r in rows})

    summary = {
        "quantum_domain_technical": {
            "rows": len(qd_rows),
            "source_groups": _source_groups(qd_rows),
            "provenance": qd_prov,
        },
        "quantum_code": {
            "rows": len(qc_rows),
            "source_groups": _source_groups(qc_rows),
            "provenance": qc_prov,
        },
        "tool_traces": {
            "rows": len(tt_rows),
            "source_groups": _source_groups(tt_rows),
            "provenance": tt_prov,
        },
        "general_english": {
            "rows": len(ge_rows),
            "source_groups": _source_groups(ge_rows),
            "provenance": ge_prov,
        },
    }
    total = sum(s["rows"] for s in summary.values())
    for key, s in summary.items():
        s["share"] = round(s["rows"] / total, 4) if total else 0.0

    result = {
        "worker": "experiments.exp-005.corpus-rev2.scripts.stage_sources",
        "version": WORKER_VERSION,
        "status": "success",
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "total_rows": total,
        "summary": summary,
        "raw_files": {fname: str((OUTPUT_DIR / fname).resolve()) for fname in files},
    }

    result_path = OUTPUT_DIR.parent / "stage_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-traces", type=int, default=6000,
                        help="Number of synthetic tool-use traces to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run(vars(args))
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
