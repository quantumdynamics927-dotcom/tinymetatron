"""
experiments/exp-005/corpus-rev3/scripts/stage_sources.py
==========================================================
Stage source-disjoint, provenance-attributed rows for exp-005 corpus revision 3.

Revision 3 adds NEW quantum-domain technical text sources to reach the 45% target:
  - PennyLane documentation (RST files in doc/)
  - Qiskit tutorials documentation (markdown cells from notebooks)
  - Qiskit Textbook narrative content (markdown cells)
  - Additional sources as available

While keeping FROZEN the revision 2 sources:
  - Quantum-specific code: 41,981 rows (qiskit-tutorials + pennylane + qiskit-textbook + own)
  - Tool-use traces: 4,223 rows (synthetic)
  - General English: 6,256 rows (7 Gutenberg texts)
  - exp-004 quantum-domain base: 14,190 rows

Produces JSONL files under experiments/exp-005/corpus-rev3/raw/.

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

WORKER_VERSION = 3

# ---------------------------------------------------------------------------
# Source locations and provenance constants
# ---------------------------------------------------------------------------
EXP004_CORPUS_DIR = _ROOT / "experiments" / "exp-004" / "corpus"
REV2_CORPUS_DIR = _ROOT / "experiments" / "exp-005" / "corpus-rev2" / "corpus"
QISKIT_TUTORIALS = Path("E:/Temp/qiskit-tutorials")
PENNYLANE = Path("E:/Temp/pennylane")
QISKIT_TEXTBOOK = Path("E:/Temp/qiskit-textbook")
OUTPUT_DIR = _ROOT / "experiments" / "exp-005" / "corpus-rev3" / "raw"

PENNYLANE_REPO_URL = "https://github.com/PennyLaneAI/pennylane"
QISKIT_TUTORIALS_URL = "https://github.com/Qiskit/qiskit-tutorials"
QISKIT_TEXTBOOK_URL = "https://github.com/qiskit-community/qiskit-textbook"

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
# Source 1: exp-004 quantum-domain technical rows (frozen base)
# ---------------------------------------------------------------------------

def stage_quantum_domain_base() -> tuple[list[dict], dict]:
    """Reuse the frozen exp-004 post-cap corpus as the quantum-domain base."""
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
# Source 2: NEW quantum-domain technical documentation text
# ---------------------------------------------------------------------------

def _iter_rst_files(base_dir: Path, skip_dirs: set[str] | None = None) -> Iterator[Path]:
    if skip_dirs is None:
        skip_dirs = {".git", "__pycache__", "_build", "_static", "_templates", ".github"}
    for path in sorted(base_dir.rglob("*.rst")):
        if any(part in skip_dirs for part in path.parts):
            continue
        yield path


def _clean_rst(text: str) -> str:
    """Strip RST markup to get readable prose."""
    # Remove code blocks
    text = re.sub(r"\n\.\. code-block::.*?\n(?:\n|$)", "\n", text, flags=re.DOTALL)
    text = re.sub(r"\n\s+::\n(?:\n\s+.+)+", "\n", text, flags=re.DOTALL)
    # Remove directives
    text = re.sub(r"\n\.\. (note|warning|tip|seealso|versionadded|versionchanged|deprecated)::.*?(?=\n\S|\Z)", "\n", text, flags=re.DOTALL)
    # Remove cross-references
    text = re.sub(r":(?:ref|doc|mod|func|class|meth|attr|exc|data|const):`[^`]+`", "", text)
    text = re.sub(r":[a-z]+:`[^`]+`", "", text)
    # Remove inline markup
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # bold
    text = re.sub(r"\*([^*]+)\*", r"\1", text)      # italic
    text = re.sub(r"``([^`]+)``", r"\1", text)      # literal
    # Remove section markers
    text = re.sub(r"^[=-~^_*+#]{3,}$", "", text, flags=re.MULTILINE)
    # Normalize whitespace
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


def stage_pennylane_docs() -> tuple[list[dict], dict]:
    """Extract prose from PennyLane RST documentation."""
    doc_dir = PENNYLANE / "doc"
    if not doc_dir.exists():
        raise FileNotFoundError(f"PennyLane doc directory not found: {doc_dir}")

    rows = []
    file_counts: dict[str, int] = defaultdict(int)

    for rst_path in _iter_rst_files(doc_dir):
        rel = rst_path.relative_to(PENNYLANE).as_posix()
        source_id = f"pennylane:doc:{rel}"
        provenance = f"{PENNYLANE_REPO_URL}/blob/main/{rel}"
        try:
            with open(rst_path, "r", encoding="utf-8") as f:
                raw = f.read()
        except Exception:
            continue

        clean = _clean_rst(raw)
        if len(clean) < 80:
            continue

        chunks = _chunk_prose(clean)
        for chunk in chunks:
            row = make_row(
                text=chunk,
                source_id=source_id,
                domain="quantum",
                subdomain="documentation",
                license="Apache-2.0",
                provenance=provenance,
                category="quantum_domain_technical",
                source_type="docs",
            )
            if row:
                rows.append(row)
                file_counts[source_id] += 1

    return rows, {
        "source": "PennyLane documentation (RST files)",
        "license": "Apache-2.0",
        "provenance": PENNYLANE_REPO_URL,
        "files_processed": len(file_counts),
        "rows_staged": len(rows),
        "by_file": dict(file_counts),
    }


def _extract_markdown_cells(notebook_path: Path, min_len: int = 50) -> list[str]:
    """Return non-empty markdown cell sources from a notebook."""
    try:
        nb = nbformat.read(str(notebook_path), as_version=4)
    except Exception as exc:
        print(f"WARN: failed to read {notebook_path}: {exc}", file=sys.stderr)
        return []

    cells = []
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        src = cell.source.strip()
        # Skip pure code blocks in markdown
        if src.startswith("```") and src.endswith("```"):
            continue
        if len(src) >= min_len:
            cells.append(src)
    return cells


def _clean_markdown(text: str) -> str:
    """Strip markdown formatting to get readable prose."""
    # Remove code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove links but keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove images
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", text)
    # Remove headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stage_qiskit_tutorials_docs() -> tuple[list[dict], dict]:
    """Extract markdown prose from Qiskit tutorials notebooks."""
    if not QISKIT_TUTORIALS.exists():
        raise FileNotFoundError(f"Qiskit tutorials not found: {QISKIT_TUTORIALS}")

    rows = []
    file_counts: dict[str, int] = defaultdict(int)

    for nb_path in sorted(QISKIT_TUTORIALS.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in nb_path.parts:
            continue
        rel = nb_path.relative_to(QISKIT_TUTORIALS).as_posix()
        source_id = f"qiskit-tutorials:doc:{rel}"
        provenance = f"{QISKIT_TUTORIALS_URL}/blob/main/{rel}"
        for cell_src in _extract_markdown_cells(nb_path):
            clean = _clean_markdown(cell_src)
            if len(clean) < 80:
                continue
            chunks = _chunk_prose(clean)
            for chunk in chunks:
                row = make_row(
                    text=chunk,
                    source_id=source_id,
                    domain="quantum",
                    subdomain="documentation",
                    license="Apache-2.0",
                    provenance=provenance,
                    category="quantum_domain_technical",
                    source_type="docs",
                )
                if row:
                    rows.append(row)
                    file_counts[source_id] += 1

    return rows, {
        "source": "Qiskit tutorials documentation (notebook markdown)",
        "license": "Apache-2.0",
        "provenance": QISKIT_TUTORIALS_URL,
        "notebooks_processed": len(file_counts),
        "rows_staged": len(rows),
        "by_notebook": dict(file_counts),
    }


def stage_qiskit_textbook_narrative() -> tuple[list[dict], dict]:
    """Extract narrative markdown from Qiskit Textbook (content/ or qiskit-textbook-src/)."""
    # Qiskit textbook uses Jekyll with content in content/ or qiskit-textbook-src/
    content_dirs = [QISKIT_TEXTBOOK / "content", QISKIT_TEXTBOOK / "qiskit-textbook-src"]
    content_dir = next((d for d in content_dirs if d.exists()), None)

    if content_dir is None:
        print("WARN: Qiskit Textbook content directory not found", file=sys.stderr)
        return [], {"source": "Qiskit Textbook narrative (not found)", "rows_staged": 0}

    rows = []
    file_counts: dict[str, int] = defaultdict(int)

    # Look for .md files
    for md_path in sorted(content_dir.rglob("*.md")):
        if ".github" in md_path.parts:
            continue
        rel = md_path.relative_to(QISKIT_TEXTBOOK).as_posix()
        source_id = f"qiskit-textbook:doc:{rel}"
        provenance = f"{QISKIT_TEXTBOOK_URL}/blob/main/{rel}"
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                raw = f.read()
        except Exception:
            continue

        clean = _clean_markdown(raw)
        if len(clean) < 80:
            continue

        chunks = _chunk_prose(clean)
        for chunk in chunks:
            row = make_row(
                text=chunk,
                source_id=source_id,
                domain="quantum",
                subdomain="documentation",
                license="Apache-2.0",
                provenance=provenance,
                category="quantum_domain_technical",
                source_type="docs",
            )
            if row:
                rows.append(row)
                file_counts[source_id] += 1

    return rows, {
        "source": "Qiskit Textbook narrative (markdown content)",
        "license": "Apache-2.0",
        "provenance": QISKIT_TEXTBOOK_URL,
        "files_processed": len(file_counts),
        "rows_staged": len(rows),
        "by_file": dict(file_counts),
    }


# ---------------------------------------------------------------------------
# Source 3: Frozen revision 2 sources (copied from rev2 corpus splits)
# ---------------------------------------------------------------------------

def _load_rev2_category(category: str) -> list[dict]:
    """Load a category from revision 2 frozen corpus splits (train/val/hard_dev)."""
    rows = []
    for split_file in ["train.jsonl", "val.jsonl", "hard_dev.jsonl"]:
        path = REV2_CORPUS_DIR / split_file
        if not path.exists():
            raise FileNotFoundError(f"Revision 2 split file missing: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("corpus_category") == category:
                    rows.append(row)
    return rows


def stage_frozen_quantum_code() -> tuple[list[dict], dict]:
    """Load frozen quantum_code from revision 2."""
    rows = _load_rev2_category("quantum_code")
    return rows, {
        "source": "FROZEN from revision 2 (qiskit-tutorials + pennylane + qiskit-textbook + own snippets)",
        "license": "Apache-2.0 for public repos; N/A for own snippets",
        "provenance": "revision 2 corpus",
        "rows_staged": len(rows),
        "note": "Frozen - not re-extracted",
    }


def stage_frozen_tool_traces() -> tuple[list[dict], dict]:
    """Load frozen tool_traces from revision 2."""
    rows = _load_rev2_category("tool_traces")
    return rows, {
        "source": "FROZEN from revision 2 (synthetic quantum tool signatures)",
        "license": "N/A",
        "provenance": "revision 2 corpus",
        "rows_staged": len(rows),
        "note": "Frozen - not re-generated",
    }


def stage_frozen_general_english() -> tuple[list[dict], dict]:
    """Load frozen general_english from revision 2."""
    rows = _load_rev2_category("general_english")
    return rows, {
        "source": "FROZEN from revision 2 (Project Gutenberg 7 texts)",
        "license": "public-domain-US",
        "provenance": "revision 2 corpus",
        "rows_staged": len(rows),
        "note": "Frozen - not re-extracted",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config: dict) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # NEW quantum-domain technical sources
    pd_rows, pd_prov = stage_pennylane_docs()
    qt_rows, qt_prov = stage_qiskit_tutorials_docs()
    qtb_rows, qtb_prov = stage_qiskit_textbook_narrative()

    # Frozen base sources
    qd_base_rows, qd_base_prov = stage_quantum_domain_base()
    qc_rows, qc_prov = stage_frozen_quantum_code()
    tt_rows, tt_prov = stage_frozen_tool_traces()
    ge_rows, ge_prov = stage_frozen_general_english()

    # Combine quantum-domain: base + new docs
    qd_rows = qd_base_rows + pd_rows + qt_rows + qtb_rows

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
            "provenance": {
                "exp004_base": qd_base_prov,
                "pennylane_docs": pd_prov,
                "qiskit_tutorials_docs": qt_prov,
                "qiskit_textbook_narrative": qtb_prov,
            },
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
        "worker": "experiments.exp-005.corpus-rev3.scripts.stage_sources",
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run(vars(args))
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())