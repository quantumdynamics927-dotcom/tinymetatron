"""
quantum_corpus.freeze
=====================
Reproducible corpus freeze: source-disjoint capped split + manifest + gate.

Exposes the exp-004 revision-2 corpus policy (``source_disjoint_capped_v1``):

  * whole source groups are assigned to exactly one partition (source-disjoint);
  * a deterministic per-source row cap (``max_rows_per_source``) is applied
    BEFORE allocation, so the retained rows do not depend on which split a
    source lands in;
  * rows beyond the cap are preserved in ``excluded_by_cap.jsonl`` (recorded,
    not deleted) for audit and optional long-tail eval segments;
  * the manifest carries ``corpus_revision``, ``max_source_share_gate_threshold``,
    and cap/exclusion provenance + hashes;
  * an opt-in ``max_source_row_share <= 0.25`` gate — tiny synthetic corpora
    cannot satisfy it by construction, so it is only enforced when
    ``--max-source-share`` is passed.

The GRE long-tail evaluation segment
(``eval_segments/gre_runtime_jobdata_longtail.jsonl``) is a SEPARATE challenge
metric and is NOT part of the primary train/val/hard_dev corpus.

CLI::

    quantum-corpus-freeze --corpus-dir <deduped> --output-dir <out> \
        --max-rows-per-source 400 --max-source-share 0.25 --revision 2
    quantum-corpus-validate --manifest <MANIFEST.json> --corpus-dir <out>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Split policy identifier recorded in split_meta.json and MANIFEST.json.
# v1 = whole source groups assigned to exactly one partition.
# capped_v1 = same, plus a deterministic per-source row cap applied before
# allocation so no single source can dominate a partition.
SPLIT_POLICY = "source_disjoint_capped_v1"

# Default per-source row cap. Configurable via --max-rows-per-source.
DEFAULT_MAX_ROWS_PER_SOURCE = 400

# Default max-source-share gate threshold. Configurable via --max-source-share.
DEFAULT_MAX_SOURCE_SHARE = 0.25

# A meaningful 3-way source-disjoint split needs at least this many groups.
MIN_SOURCE_GROUPS = 3

# Split file stem -> partition name used in the manifest's overlap/counts dicts.
_SPLIT_NAMES = {"train": "train", "val": "val", "hard_dev": "hard_dev"}


# ── Split (source_disjoint_capped_v1) ────────────────────────────────────────

def _source_of(row: dict) -> str:
    """Source identifier for a row (matches workers.corpus.split._src)."""
    return row.get("source_id", row.get("source", row.get("domain", "unknown")))


def split_source_disjoint_capped(
    rows: list[dict],
    seed: int = 42,
    train_pct: float = 0.80,
    val_pct: float = 0.10,
    max_rows_per_source: int = DEFAULT_MAX_ROWS_PER_SOURCE,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    """
    Split rows into train/val/hard_dev with the source_disjoint_capped_v1 policy.

    Every source group is assigned wholly to exactly one partition; no source is
    sliced across train/val/hard_dev. Before allocation, each source group is
    capped at ``max_rows_per_source`` rows (deterministic: rows are shuffled with
    the split seed, then the first N are retained). Rows beyond the cap are
    returned as ``excluded`` (preserved, not deleted).

    Returns ``(train, val, hard_dev, excluded, meta)`` where ``meta`` records the
    policy, seed, cap, pre/post-cap row counts, capped sources, and the
    source/text overlap + max_source_row_share metrics.
    """
    if not rows:
        raise ValueError("split_source_disjoint_capped: no rows to split")

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[_source_of(row)].append(row)

    n_groups = len(by_source)
    if n_groups < MIN_SOURCE_GROUPS:
        raise ValueError(
            f"source-disjoint split requires >={MIN_SOURCE_GROUPS} independent "
            f"source groups, got {n_groups}. Refusing to slice sources across "
            f"partitions — real corpus freezing must fail when groups are too "
            f"few for a meaningful split."
        )

    # Deterministic ordering: shuffle the group keys, then rows within each
    # group, both driven by the fixed seed.
    rng = random.Random(seed)
    group_keys = list(by_source.keys())
    rng.shuffle(group_keys)
    for k in group_keys:
        rng.shuffle(by_source[k])

    # Deterministic per-source cap. Applied BEFORE allocation so the retained
    # rows do not depend on which split a source lands in. Rows beyond the cap
    # are excluded from training/evaluation but preserved (recorded, not
    # deleted) for audit and optional long-tail eval segments.
    pre_cap_rows = len(rows)
    capped_sources: list[dict] = []
    excluded_rows: list[dict] = []
    rows_dropped_by_cap = 0
    if max_rows_per_source and max_rows_per_source > 0:
        for k in group_keys:
            n = len(by_source[k])
            if n > max_rows_per_source:
                dropped = n - max_rows_per_source
                rows_dropped_by_cap += dropped
                capped_sources.append({
                    "source_id": k,
                    "original_rows": n,
                    "retained_rows": max_rows_per_source,
                    "excluded_rows": dropped,
                })
                excluded_rows.extend(by_source[k][max_rows_per_source:])
                by_source[k] = by_source[k][:max_rows_per_source]

    # Whole-group allocation. The ratio target applies to SOURCE GROUPS, not
    # rows: each partition receives ~train_pct / val_pct of the source groups so
    # every partition has genuine source diversity. Row counts are approximate
    # (whole groups can't be sliced) and are recorded in the result. Targeting
    # rows instead would let a few huge groups swallow a partition (e.g. a val
    # split 85% from one document), which is not a meaningful held-out set.
    target_train_groups = train_pct * n_groups
    target_val_groups = val_pct * n_groups

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    hard_dev_rows: list[dict] = []

    i = 0
    acc_train_groups = 0
    while i < n_groups and acc_train_groups < target_train_groups and (n_groups - i) > 2:
        g = group_keys[i]
        train_rows.extend(by_source[g])
        acc_train_groups += 1
        i += 1

    acc_val_groups = 0
    while i < n_groups and acc_val_groups < target_val_groups and (n_groups - i) > 1:
        g = group_keys[i]
        val_rows.extend(by_source[g])
        acc_val_groups += 1
        i += 1

    for j in range(i, n_groups):
        g = group_keys[j]
        hard_dev_rows.extend(by_source[g])

    # Verify the source-disjoint + text-disjoint invariants.
    train_sources = {_source_of(r) for r in train_rows}
    val_sources = {_source_of(r) for r in val_rows}
    hard_sources = {_source_of(r) for r in hard_dev_rows}
    train_texts = {r.get("text", "") for r in train_rows}
    val_texts = {r.get("text", "") for r in val_rows}
    hard_texts = {r.get("text", "") for r in hard_dev_rows}

    assert train_sources.isdisjoint(val_sources), "train/val source overlap!"
    assert train_sources.isdisjoint(hard_sources), "train/hard_dev source overlap!"
    assert val_sources.isdisjoint(hard_sources), "val/hard_dev source overlap!"
    assert train_texts.isdisjoint(val_texts), "train/val text overlap!"
    assert train_texts.isdisjoint(hard_texts), "train/hard_dev text overlap!"
    assert val_texts.isdisjoint(hard_texts), "val/hard_dev text overlap!"

    source_counts = {
        "train": len(train_sources),
        "val": len(val_sources),
        "hard_dev": len(hard_sources),
    }
    source_overlap = {
        "train_val": len(train_sources & val_sources),
        "train_hard_dev": len(train_sources & hard_sources),
        "val_hard_dev": len(val_sources & hard_sources),
    }
    text_overlap = {
        "train_val": len(train_texts & val_texts),
        "train_hard_dev": len(train_texts & hard_texts),
        "val_hard_dev": len(val_texts & hard_texts),
    }

    # Largest single source's share of each partition's rows. A partition whose
    # rows come mostly from one source is not a representative held-out set, so
    # this is gated (max_source_row_share <= 0.25) at freeze time.
    def _max_share(dist: dict, n_rows: int) -> float:
        return round(max(dist.values()) / n_rows, 4) if dist and n_rows else 0.0

    train_src_counts = Counter(_source_of(r) for r in train_rows)
    val_src_counts = Counter(_source_of(r) for r in val_rows)
    hard_src_counts = Counter(_source_of(r) for r in hard_dev_rows)

    max_source_row_share = {
        "train": _max_share(train_src_counts, len(train_rows)),
        "val": _max_share(val_src_counts, len(val_rows)),
        "hard_dev": _max_share(hard_src_counts, len(hard_dev_rows)),
    }

    meta = {
        "split_policy": SPLIT_POLICY,
        "split_seed": seed,
        "max_rows_per_source": max_rows_per_source,
        "pre_cap_rows": pre_cap_rows,
        "post_cap_rows": len(train_rows) + len(val_rows) + len(hard_dev_rows),
        "capped_sources": capped_sources,
        "rows_dropped_by_cap": rows_dropped_by_cap,
        "max_source_row_share": max_source_row_share,
        "source_counts": source_counts,
        "source_overlap": source_overlap,
        "text_overlap": text_overlap,
    }

    return train_rows, val_rows, hard_dev_rows, excluded_rows, meta


# ── Manifest ─────────────────────────────────────────────────────────────────

def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower())


def _load_split_meta(corpus_dir: Path) -> dict:
    """Read split_meta.json written by the freeze CLI, if present."""
    meta_path = corpus_dir / "split_meta.json"
    if meta_path.exists():
        try:
            return json.loads(open(meta_path, encoding="utf-8").read())
        except Exception:
            return {}
    return {}


def build_manifest(
    corpus_dir: Path,
    revision: int | None = None,
    max_source_share_threshold: float | None = None,
    scope: str | None = None,
) -> dict:
    """
    Build a frozen-corpus manifest from a directory with train/val/hard_dev
    JSONL split files. Mirrors workers.corpus.version.build_manifest.

    Only the three primary splits are part of the frozen corpus.
    excluded_by_cap.jsonl (rows dropped by the per-source cap) is an audit
    artifact, not a split, and must not enter the corpus hash or the
    unique/subdomain tallies.
    """
    splits = {}
    split_rows: dict[str, list[dict]] = {}
    all_rows = []

    for f in sorted(corpus_dir.glob("*.jsonl")):
        if f.stem not in _SPLIT_NAMES:
            continue
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        splits[f.name] = {
            "rows": len(rows),
            "sha256": _hash_file(f),
        }
        all_rows.extend(rows)
        split_rows[f.stem] = rows

    # Aggregate corpus hash
    corp_h = hashlib.sha256()
    for f in sorted(corpus_dir.glob("*.jsonl")):
        if f.stem not in _SPLIT_NAMES:
            continue
        corp_h.update(_hash_file(f).encode())
    corpus_hash = corp_h.hexdigest()[:16]

    # Unique normalized facts
    norms = set(_normalize(r["text"]) for r in all_rows)

    # Subdomain distribution
    subdomains = {}
    for r in all_rows:
        sd = r.get("subdomain", "unknown")
        subdomains[sd] = subdomains.get(sd, 0) + 1

    # Source/text disjointness metrics, computed directly from the frozen split
    # files (independent verification of the split's claims).
    train_sources = {_source_of(r) for r in split_rows.get("train", [])}
    val_sources = {_source_of(r) for r in split_rows.get("val", [])}
    hard_sources = {_source_of(r) for r in split_rows.get("hard_dev", [])}
    train_texts = {r.get("text", "") for r in split_rows.get("train", [])}
    val_texts = {r.get("text", "") for r in split_rows.get("val", [])}
    hard_texts = {r.get("text", "") for r in split_rows.get("hard_dev", [])}

    source_counts = {
        "train": len(train_sources),
        "val": len(val_sources),
        "hard_dev": len(hard_sources),
    }
    source_overlap = {
        "train_val": len(train_sources & val_sources),
        "train_hard_dev": len(train_sources & hard_sources),
        "val_hard_dev": len(val_sources & hard_sources),
    }
    text_overlap = {
        "train_val": len(train_texts & val_texts),
        "train_hard_dev": len(train_texts & hard_texts),
        "val_hard_dev": len(val_texts & hard_texts),
    }

    # Largest single source's share of each partition's rows, computed directly
    # from the frozen split files (independent verification of the split's
    # claim). A partition dominated by one source is not a representative
    # held-out set, so this is gated at freeze time.
    train_src_counts = Counter(_source_of(r) for r in split_rows.get("train", []))
    val_src_counts = Counter(_source_of(r) for r in split_rows.get("val", []))
    hard_src_counts = Counter(_source_of(r) for r in split_rows.get("hard_dev", []))

    def _max_share(counts, n_rows: int) -> float:
        return round(max(counts.values()) / n_rows, 4) if counts and n_rows else 0.0

    max_source_row_share = {
        "train": _max_share(train_src_counts, len(split_rows.get("train", []))),
        "val": _max_share(val_src_counts, len(split_rows.get("val", []))),
        "hard_dev": _max_share(hard_src_counts, len(split_rows.get("hard_dev", []))),
    }

    # Rows excluded by the per-source cap are preserved (not deleted) for audit
    # and optional long-tail eval segments. Record their location + hash.
    excluded_path = corpus_dir / "excluded_by_cap.jsonl"
    excluded_by_cap = None
    if excluded_path.exists():
        excluded_by_cap = {
            "path": str(excluded_path.resolve()),
            "rows": sum(1 for _ in open(excluded_path, encoding="utf-8")),
            "sha256": _hash_file(excluded_path),
        }

    meta = _load_split_meta(corpus_dir)

    manifest = {
        "corpus_hash": corpus_hash,
        "splits": splits,
        "total_rows": len(all_rows),
        "unique_normalized": len(norms),
        "subdomains": subdomains,
        "split_policy": meta.get("split_policy", SPLIT_POLICY),
        "split_seed": meta.get("split_seed", 42),
        "max_rows_per_source": meta.get("max_rows_per_source"),
        "pre_cap_rows": meta.get("pre_cap_rows"),
        "post_cap_rows": meta.get("post_cap_rows"),
        "capped_sources": meta.get("capped_sources", []),
        "max_source_row_share": max_source_row_share,
        "excluded_by_cap": excluded_by_cap,
        "source_counts": source_counts,
        "source_overlap": source_overlap,
        "text_overlap": text_overlap,
    }
    if revision is not None:
        manifest["corpus_revision"] = int(revision)
    if max_source_share_threshold is not None:
        manifest["max_source_share_gate_threshold"] = float(max_source_share_threshold)
    if scope:
        manifest["scope"] = scope
    return manifest


# ── Gate ─────────────────────────────────────────────────────────────────────

def gate_max_source_share(manifest: dict, threshold: float = DEFAULT_MAX_SOURCE_SHARE) -> tuple[bool, float | None]:
    """
    Evaluate the max-source-share gate: the largest single source's share of any
    primary partition must be <= threshold. Returns (passed, actual).

    Opt-in by design: a tiny synthetic corpus (smoke fixture) cannot satisfy a
    dominance gate by construction, so callers only enforce it on real corpora.
    """
    max_share = manifest.get("max_source_row_share", {})
    actual = max(max_share.values()) if max_share else None
    if actual is None:
        return False, None
    return actual <= threshold, actual


# ── Validate ─────────────────────────────────────────────────────────────────

def validate_manifest(manifest_path: Path, corpus_dir: Path) -> dict:
    """
    Validate a frozen MANIFEST.json against its corpus directory: recompute the
    split-file hashes and corpus_hash, compare against the manifest, and report
    the source/text overlap totals and max_source_row_share vs the manifest's
    gate threshold.
    """
    manifest = json.loads(open(manifest_path, encoding="utf-8").read())

    checks = {}
    for name, info in manifest.get("splits", {}).items():
        f = corpus_dir / name
        if not f.exists():
            checks[name] = {"status": "missing", "expected": info["sha256"]}
            continue
        actual = _hash_file(f)
        checks[name] = {
            "status": "ok" if actual == info["sha256"] else "MISMATCH",
            "expected": info["sha256"],
            "actual": actual,
        }

    # Recompute the aggregate corpus hash the same way build_manifest does.
    corp_h = hashlib.sha256()
    for f in sorted(corpus_dir.glob("*.jsonl")):
        if f.stem not in _SPLIT_NAMES:
            continue
        corp_h.update(_hash_file(f).encode())
    actual_corpus_hash = corp_h.hexdigest()[:16]
    expected_corpus_hash = manifest.get("corpus_hash")
    corpus_hash_ok = actual_corpus_hash == expected_corpus_hash

    source_overlap = manifest.get("source_overlap", {})
    text_overlap = manifest.get("text_overlap", {})
    max_share = manifest.get("max_source_row_share", {})
    threshold = manifest.get("max_source_share_gate_threshold")
    actual_max_share = max(max_share.values()) if max_share else None
    gate_passed = actual_max_share is not None and (
        threshold is None or actual_max_share <= threshold
    )

    return {
        "manifest": str(manifest_path),
        "corpus_dir": str(corpus_dir),
        "corpus_hash": {
            "expected": expected_corpus_hash,
            "actual": actual_corpus_hash,
            "ok": corpus_hash_ok,
        },
        "split_hashes": checks,
        "all_split_hashes_ok": all(c["status"] == "ok" for c in checks.values()),
        "source_overlap_total": sum(int(v) for v in source_overlap.values()),
        "text_overlap_total": sum(int(v) for v in text_overlap.values()),
        "max_source_row_share": actual_max_share,
        "max_source_share_gate_threshold": threshold,
        "gate_passed": gate_passed,
        "valid": corpus_hash_ok and all(c["status"] == "ok" for c in checks.values()),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, rows: list[dict]) -> int:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _load_input_rows(corpus_dir: Path) -> list[dict]:
    """Load deduped.jsonl (or the first jsonl) from the corpus dir."""
    input_path = corpus_dir / "deduped.jsonl"
    if not input_path.exists():
        candidates = list(corpus_dir.glob("*.jsonl"))
        input_path = candidates[0] if candidates else None
    if not input_path or not input_path.exists():
        raise FileNotFoundError(f"No deduped.jsonl found in {corpus_dir}")
    rows = []
    for line in open(input_path, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def freeze(
    corpus_dir: Path,
    output_dir: Path,
    seed: int = 42,
    train_pct: float = 0.80,
    val_pct: float = 0.10,
    max_rows_per_source: int = DEFAULT_MAX_ROWS_PER_SOURCE,
    max_source_share: float | None = None,
    revision: int | None = None,
    scope: str | None = None,
) -> dict:
    """
    Run the full freeze: split (source_disjoint_capped_v1) -> write splits ->
    build manifest -> optional max-source-share gate. Returns a summary dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_input_rows(corpus_dir)

    train, val, hard_dev, excluded, meta = split_source_disjoint_capped(
        rows, seed=seed, train_pct=train_pct, val_pct=val_pct,
        max_rows_per_source=max_rows_per_source,
    )

    n_train = _write_jsonl(output_dir / "train.jsonl", train)
    n_val = _write_jsonl(output_dir / "val.jsonl", val)
    n_hard = _write_jsonl(output_dir / "hard_dev.jsonl", hard_dev)
    n_excluded = _write_jsonl(output_dir / "excluded_by_cap.jsonl", excluded)

    with open(output_dir / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    manifest = build_manifest(
        output_dir,
        revision=revision,
        max_source_share_threshold=max_source_share,
        scope=scope,
    )
    manifest_path = output_dir / "MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    gate = None
    if max_source_share is not None:
        passed, actual = gate_max_source_share(manifest, threshold=max_source_share)
        gate = {"passed": passed, "actual": actual, "threshold": max_source_share}
        if not passed:
            raise ValueError(
                f"max-source-share gate FAILED: max_source_row_share={actual} "
                f"(expected <= {max_source_share}). Corpus not frozen."
            )

    return {
        "manifest_path": str(manifest_path),
        "corpus_hash": manifest["corpus_hash"],
        "total_rows": manifest["total_rows"],
        "splits": {k: v["rows"] for k, v in manifest["splits"].items()},
        "excluded_by_cap_rows": n_excluded,
        "max_source_row_share": manifest["max_source_row_share"],
        "gate": gate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quantum-corpus-freeze",
        description="Reproducible corpus freeze: source-disjoint capped split + manifest + optional gate.",
    )
    parser.add_argument("--corpus-dir", required=True,
                        help="Directory containing deduped.jsonl (or a single jsonl) to split")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for train/val/hard_dev/excluded_by_cap + MANIFEST.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-pct", type=float, default=0.80)
    parser.add_argument("--val-pct", type=float, default=0.10)
    parser.add_argument("--max-rows-per-source", type=int, default=DEFAULT_MAX_ROWS_PER_SOURCE,
                        help=f"Cap rows per source group before splitting (default {DEFAULT_MAX_ROWS_PER_SOURCE})")
    parser.add_argument("--max-source-share", type=float, default=None,
                        help=f"Enforce the max-source-share gate with this threshold "
                             f"(default {DEFAULT_MAX_SOURCE_SHARE} when set). Omit to skip the gate "
                             f"(smoke/synthetic corpora).")
    parser.add_argument("--revision", type=int, default=None,
                        help="Corpus revision number written into MANIFEST.json")
    parser.add_argument("--scope", default=None,
                        help="Experiment scope written into MANIFEST.json")
    args = parser.parse_args(argv)

    try:
        summary = freeze(
            Path(args.corpus_dir), Path(args.output_dir),
            seed=args.seed, train_pct=args.train_pct, val_pct=args.val_pct,
            max_rows_per_source=args.max_rows_per_source,
            max_source_share=args.max_source_share,
            revision=args.revision, scope=args.scope,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    return 0


def validate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quantum-corpus-validate",
        description="Validate a frozen MANIFEST.json against its corpus directory.",
    )
    parser.add_argument("--manifest", required=True, help="Path to MANIFEST.json")
    parser.add_argument("--corpus-dir", required=True,
                        help="Directory containing the split JSONL files")
    args = parser.parse_args(argv)

    try:
        result = validate_manifest(Path(args.manifest), Path(args.corpus_dir))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
