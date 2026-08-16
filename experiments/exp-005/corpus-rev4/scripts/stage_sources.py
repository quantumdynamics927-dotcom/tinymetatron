"""
experiments/exp-005/corpus-rev4/scripts/stage_sources.py
=========================================================
Stage corpus revision 4: add Qiskit main docs + Cirq + QuTiP v5.

Revision 4 adds MORE quantum-domain technical text sources to continue
building toward the 45% quantum-domain share target. The frozen base
from revisions 2/3 is preserved:
  - Quantum-code: 41,924 rows (frozen)
  - Tool-use traces: 2,832 rows (frozen)
  - General English: 2,685 rows (frozen)
  - Quantum-domain technical: 15,652 rows (from rev3)

NEW sources (revision 4):
  - Qiskit main documentation (MDX/Markdown)
  - Cirq documentation (Sphinx RST)
  - QuTiP documentation v5 (Sphinx RST)

Produces JSONL files under experiments/exp-005/corpus-rev4/raw/.

All file opens use encoding="utf-8".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

try:
    import nbformat
except ImportError as exc:
    raise SystemExit(f"MISSING: pip install nbformat. {exc}")

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

WORKER_VERSION = 4

# ── Source roots ─────────────────────────────────────────────────────────────

# Pre-existing local clones
_PENNYLANE     = Path("E:/Temp/pennylane")
_QISKIT_TUT    = Path("E:/Temp/qiskit-tutorials")
_QISKIT_TB     = Path("E:/Temp/qiskit-textbook")
_EXP004_CORPUS = _ROOT / "experiments" / "exp-004" / "corpus"
_REV2_CORPUS   = _ROOT / "experiments" / "exp-005" / "corpus-rev2" / "corpus"

# Rev4 clone targets (cloned if absent)
_QISKIT_DOCS   = _ROOT / "experiments" / "exp-005" / "corpus-rev4" / "repos" / "qiskit-documentation"
_CIRQ          = _ROOT / "experiments" / "exp-005" / "corpus-rev4" / "repos" / "cirq"
_QUTIP         = _ROOT / "experiments" / "exp-005" / "corpus-rev4" / "repos" / "qutip"

OUTPUT_DIR     = _ROOT / "experiments" / "exp-005" / "corpus-rev4" / "raw"

REPOS = {
    "qiskit-documentation": {
        "url": "https://github.com/Qiskit/qiskit.git",
        "local": _QISKIT_DOCS,
        "subdir": "docs",          # docs/ is the Sphinx doc root
        "license": "Apache-2.0",
    },
    "cirq": {
        "url": "https://github.com/quantumlib/Cirq.git",
        "local": _CIRQ,
        "subdir": "cirq docs",    # cirq/docs/ is the Sphinx doc root
        "license": "Apache-2.0",
    },
    "qutip": {
        "url": "https://github.com/qutip/qutip.git",
        "local": _QUTIP,
        "subdir": "doc",           # qutip/doc/ is the Sphinx doc root
        "license": "BSD-3-Clause",
    },
}


# ── Row schema ────────────────────────────────────────────────────────────────

def make_row(text: str, source_id: str, domain: str, subdomain: str,
             license: str, provenance: str, category: str,
             source_type: str = "repo") -> dict | None:
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


# ── Narrative Extractor — three frontends ────────────────────────────────────

class NarrativeExtractor:
    """
    Extract prose narrative from quantum-library documentation sources.
    Three frontend formats: MDX/Markdown, notebook markdown cells, Sphinx RST.
    """

    def extract_file(self, path: Path) -> list[dict]:
        """Auto-detect format from extension and extract."""
        ext = path.suffix.lower()
        if ext in (".md", ".mdx"):
            return self.from_markdown(path)
        elif ext == ".ipynb":
            return self.from_notebook(path)
        elif ext == ".rst":
            return self.from_rst(path)
        return []

    # ── Frontend 1: MDX / Markdown ─────────────────────────────────────────

    def from_markdown(self, path: Path) -> list[dict]:
        """Extract prose from MDX or Markdown documentation files."""
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        text = self._clean_mdx(raw)
        if len(text) < 80:
            return []

        chunks = self._chunk_prose(text)
        rows = []
        rel = path.as_posix()
        provenance = f"file://{path.resolve().as_posix()}"
        source_id = f"narrative:md:{rel}"

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
        return rows

    def _clean_mdx(self, text: str) -> str:
        """Strip MDX and Markdown markup to get readable prose."""
        # Remove JSX components <Component ... />
        text = re.sub(r"<[A-Z][a-zA-Z]*[^>]*/>", "", text)
        text = re.sub(r"<[A-Z][a-zA-Z]*[^>]*>.*?</[A-Z][a-zA-Z]*>", "", text, flags=re.DOTALL)
        # Remove MDX/JSX expressions {expression}
        text = re.sub(r"\{[^}]+\}", "", text)
        # Remove code blocks (fenced)
        text = re.sub(r"```[\s\S]*?```", "", text)
        # Remove inline code
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # Remove images
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
        # Remove links but keep text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove headers markers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove horizontal rules
        text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)
        # Remove table rows
        text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[-:|]+$", "", text, flags=re.MULTILINE)
        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ── Frontend 2: Notebook markdown cells ───────────────────────────────────

    def from_notebook(self, path: Path) -> list[dict]:
        """Extract prose from Jupyter notebook markdown cells."""
        try:
            nb = nbformat.read(str(path), as_version=4)
        except Exception as exc:
            print(f"WARN: notebook read failed {path}: {exc}", file=sys.stderr)
            return []

        rows = []
        rel = path.as_posix()
        provenance = f"file://{path.resolve().as_posix()}"
        source_id = f"narrative:nb:{rel}"

        for cell in nb.cells:
            if cell.cell_type != "markdown":
                continue
            src = cell.source.strip()
            if len(src) < 50:
                continue
            clean = self._clean_mdx(src)  # notebooks use Markdown syntax
            if len(clean) < 80:
                continue
            chunks = self._chunk_prose(clean)
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
        return rows

    # ── Frontend 3: Sphinx RST ────────────────────────────────────────────────

    def from_rst(self, path: Path) -> list[dict]:
        """Extract prose from Sphinx RST documentation files."""
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        text = self._clean_rst(raw)
        if len(text) < 80:
            return []

        chunks = self._chunk_prose(text)
        rows = []
        rel = path.as_posix()
        provenance = f"file://{path.resolve().as_posix()}"
        source_id = f"narrative:rst:{rel}"

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
        return rows

    def _clean_rst(self, text: str) -> str:
        """Strip RST markup to get readable prose."""
        # Remove code blocks
        text = re.sub(r"\n\.\. code-block::.*?\n(?:\n|$)", "\n", text, flags=re.DOTALL)
        text = re.sub(r"\n\s+::\n(?:\n\s+.+)+", "\n", text, flags=re.DOTALL)
        # Remove directives (.. note::, .. warning::, etc.)
        text = re.sub(
            r"\n\.\. (?:note|warning|tip|seealso|versionadded|versionchanged|deprecated|toctree|autoclass|autofunction|autodata|automodule|image|figure)::.*?(?=\n\S|\Z)",
            "\n", text, flags=re.DOTALL,
        )
        # Remove cross-references
        text = re.sub(r":(?:ref|doc|mod|func|class|meth|attr|exc|data|const):`[^`]+`", "", text)
        text = re.sub(r":[a-z]+:`[^`]+`", "", text)
        # Remove inline markup
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*(?!\s)([^*]+)(?<!\s)\*", r"\1", text)
        text = re.sub(r"``([^`]+)``", r"\1", text)
        # Remove section overlines
        text = re.sub(r"^[=-~^_*+#]{3,}$", "", text, flags=re.MULTILINE)
        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ── Shared chunking ─────────────────────────────────────────────────────

    def _chunk_prose(
        self, text: str, min_chars: int = 80, max_chars: int = 600,
        min_words: int = 12,
    ) -> list[str]:
        """Split prose into paragraph-based chunks."""
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

        for para in paragraphs:
            if current_len + len(para) > max_chars and current:
                flush()
            current.append(para)
            current_len += len(para) + 2

        flush(force=True)
        return chunks


# ── Repo cloning ──────────────────────────────────────────────────────────────

def clone_or_update(name: str, repo: dict) -> Path | None:
    """Clone a repo if not present, or verify it exists."""
    local = repo["local"]
    if local.exists() and (local / ".git").exists():
        print(f"  {name}: using existing clone at {local}")
        return local
    parent = local.parent
    parent.mkdir(parents=True, exist_ok=True)
    print(f"  {name}: cloning {repo['url']} ...")
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", repo["url"], str(local)],
            check=True, capture_output=True, text=True,
        )
        print(f"  {name}: cloned to {local}")
        return local
    except Exception as exc:
        print(f"  {name}: clone failed: {exc}")
        return None


# ── Source stagings ───────────────────────────────────────────────────────────

EXTRACTOR = NarrativeExtractor()

# RST skip dirs common to Sphinx builds
RST_SKIP = {".git", "__pycache__", "_build", "_static", "_templates", ".github", ".doctrees"}


def _iter_rst(base_dir: Path) -> Iterator[Path]:
    for path in sorted(base_dir.rglob("*.rst")):
        if any(part in RST_SKIP for part in path.parts):
            continue
        yield path


def _iter_md(base_dir: Path) -> Iterator[Path]:
    for ext in ("*.md", "*.mdx"):
        for path in sorted(base_dir.rglob(ext)):
            if any(part in RST_SKIP for part in path.parts):
                continue
            yield path


def stage_qiskit_docs() -> tuple[list[dict], dict]:
    """Stage Qiskit main documentation — RST files in docs/ and Python docstrings."""
    docs_dir = _QISKIT_DOCS / "docs"
    if not docs_dir.exists():
        return [], {"source": "Qiskit docs (not found)", "rows_staged": 0}

    rows = []
    file_counts: dict[str, int] = {}

    # RST files in docs/
    for path in _iter_rst(docs_dir):
        rel = path.relative_to(_QISKIT_DOCS).as_posix()
        source_id = f"qiskit:docs:{rel}"
        provenance = f"https://github.com/Qiskit/qiskit/blob/main/{rel}"
        for row in EXTRACTOR.from_rst(path):
            rows.append(row)
            file_counts[row["source_id"]] = file_counts.get(row["source_id"], 0) + 1

    # Markdown/MDX files in docs/
    for path in _iter_md(docs_dir):
        rel = path.relative_to(_QISKIT_DOCS).as_posix()
        source_id = f"qiskit:docs:{rel}"
        provenance = f"https://github.com/Qiskit/qiskit/blob/main/{rel}"
        for row in EXTRACTOR.from_markdown(path):
            rows.append(row)
            file_counts[row["source_id"]] = file_counts.get(row["source_id"], 0) + 1

    return rows, {
        "source": "Qiskit main documentation (docs/ RST + MDX + Markdown)",
        "license": "Apache-2.0",
        "provenance": "https://github.com/Qiskit/qiskit",
        "files_processed": len(file_counts),
        "rows_staged": len(rows),
    }


def stage_cirq_docs() -> tuple[list[dict], dict]:
    """Stage Cirq documentation (notebooks + markdown in cirq/docs/)."""
    docs_dir = _CIRQ / "docs"
    if not docs_dir.exists():
        return [], {"source": "Cirq docs (not found)", "rows_staged": 0}

    rows = []
    file_counts: dict[str, int] = {}

    # Notebooks
    for path in docs_dir.rglob("*.ipynb"):
        if any(part in RST_SKIP for part in path.parts):
            continue
        rel = path.relative_to(_CIRQ).as_posix()
        source_id = f"cirq:docs:{rel}"
        provenance = f"https://github.com/quantumlib/Cirq/blob/main/{rel}"
        for row in EXTRACTOR.from_notebook(path):
            rows.append(row)
            file_counts[row["source_id"]] = file_counts.get(row["source_id"], 0) + 1

    # Markdown files
    for path in docs_dir.rglob("*.md"):
        if any(part in RST_SKIP for part in path.parts):
            continue
        rel = path.relative_to(_CIRQ).as_posix()
        source_id = f"cirq:docs:{rel}"
        provenance = f"https://github.com/quantumlib/Cirq/blob/main/{rel}"
        for row in EXTRACTOR.from_markdown(path):
            rows.append(row)
            file_counts[row["source_id"]] = file_counts.get(row["source_id"], 0) + 1

    return rows, {
        "source": "Cirq documentation (notebooks + markdown)",
        "license": "Apache-2.0",
        "provenance": "https://github.com/quantumlib/Cirq",
        "files_processed": len(file_counts),
        "rows_staged": len(rows),
    }


def stage_qutip_docs() -> tuple[list[dict], dict]:
    """Stage QuTiP v5 documentation (RST in qutip/doc/)."""
    doc_dir = _QUTIP / "doc"
    if not doc_dir.exists():
        return [], {"source": "QuTiP docs (not found)", "rows_staged": 0}

    rows = []
    file_counts: dict[str, int] = {}
    for path in _iter_rst(doc_dir):
        rel = path.relative_to(_QUTIP).as_posix()
        source_id = f"qutip:docs:{rel}"
        provenance = f"https://github.com/qutip/qutip/blob/main/{rel}"
        for row in EXTRACTOR.from_rst(path):
            rows.append(row)
            file_counts[row["source_id"]] = file_counts.get(row["source_id"], 0) + 1

    return rows, {
        "source": "QuTiP documentation v5 (RST)",
        "license": "BSD-3-Clause",
        "provenance": "https://github.com/qutip/qutip",
        "files_processed": len(file_counts),
        "rows_staged": len(rows),
    }


# ── Frozen base (from rev3, as atomic units) ─────────────────────────────────

def stage_frozen_rev3() -> tuple[list[dict], dict]:
    """Load ALL categories from the rev3 frozen splits."""
    rev3_dir = _ROOT / "experiments" / "exp-005" / "corpus-rev3" / "split"
    rows_by_cat: dict[str, list[dict]] = {
        "quantum_domain_technical": [],
        "quantum_code": [],
        "tool_traces": [],
        "general_english": [],
    }

    for split_file in ["train.jsonl", "val.jsonl", "hard_dev.jsonl"]:
        path = rev3_dir / split_file
        if not path.exists():
            raise FileNotFoundError(f"Rev3 split missing: {path}")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cat = row.get("corpus_category", "unknown")
                if cat in rows_by_cat:
                    rows_by_cat[cat].append(row)

    total = sum(len(v) for v in rows_by_cat.values())
    return rows_by_cat, {
        "source": "FROZEN rev3 splits (train+val+hard_dev)",
        "rows_staged": total,
        "by_category": {k: len(v) for k, v in rows_by_cat.items()},
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run(config: dict) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clone rev4 repos
    print("\n=== Cloning / verifying repos ===")
    for name, repo in REPOS.items():
        clone_or_update(name, repo)

    # Stage NEW quantum-domain sources
    print("\n=== Staging Qiskit main documentation ===")
    qk_rows, qk_prov = stage_qiskit_docs()
    print(f"  Qiskit docs: {qk_prov}")

    print("\n=== Staging Cirq documentation ===")
    cirq_rows, cirq_prov = stage_cirq_docs()
    print(f"  Cirq docs: {cirq_prov}")

    print("\n=== Staging QuTiP v5 documentation ===")
    qutip_rows, qutip_prov = stage_qutip_docs()
    print(f"  QuTiP docs: {qutip_prov}")

    # Load frozen rev3 base (all categories)
    print("\n=== Loading frozen rev3 base ===")
    rev3_rows_by_cat, rev3_prov = stage_frozen_rev3()
    print(f"  Rev3 base: {rev3_prov}")

    # Combine: rev3 base (all categories) + NEW technical rows
    # The NEW technical rows (Qiskit/Cirq/QuTiP) are added to quantum_domain_technical
    qd_rows = (
        rev3_rows_by_cat["quantum_domain_technical"]
        + qk_rows + cirq_rows + qutip_rows
    )
    qc_rows = rev3_rows_by_cat["quantum_code"]
    tt_rows = rev3_rows_by_cat["tool_traces"]
    ge_rows = rev3_rows_by_cat["general_english"]

    files = {
        "quantum_domain_technical.jsonl": qd_rows,
        "quantum_code.jsonl": qc_rows,
        "tool_traces.jsonl": tt_rows,
        "general_english.jsonl": ge_rows,
    }
    for fname, rows in files.items():
        write_jsonl(OUTPUT_DIR / fname, rows)

    def _src_groups(rows: list[dict]) -> int:
        return len({r.get("source_id", "unknown") for r in rows})

    summary = {
        "quantum_domain_technical": {
            "rows": len(qd_rows),
            "source_groups": _src_groups(qd_rows),
            "new_sources": {
                "qiskit_docs": qk_prov,
                "cirq_docs": cirq_prov,
                "qutip_docs": qutip_prov,
            },
        },
        "quantum_code": {
            "rows": len(qc_rows),
            "source_groups": _src_groups(qc_rows),
            "note": "frozen from rev3",
        },
        "tool_traces": {
            "rows": len(tt_rows),
            "source_groups": _src_groups(tt_rows),
            "note": "frozen from rev3",
        },
        "general_english": {
            "rows": len(ge_rows),
            "source_groups": _src_groups(ge_rows),
            "note": "frozen from rev3",
        },
    }
    total = sum(s["rows"] for s in summary.values())
    for key, s in summary.items():
        s["share"] = round(s["rows"] / total, 4) if total else 0.0

    result = {
        "worker": "experiments.exp-005.corpus-rev4.scripts.stage_sources",
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

    print(f"\n=== Stage complete: {total} total rows ===")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run(vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
