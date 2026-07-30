"""
manage_data.py
==============
Data-management CLI for the TinyMetatron SLM.

Public interface (per IMPLEMENTATION_CONTRACT.md section 3):
    main(argv) -> int

Subcommands:
    generate  --domain X --count N [--db_path P]
              Synthesize N deterministic rows from a small built-in per-domain
              template list (cybersecurity / software / general; ~15 templates
              each) and insert them via db.add_texts (which applies the auto
              quality scorer from quality.score_quality).
    import    --file F --domain X [--db_path P]
              Read lines from F, insert via db.add_texts with
              quality_threshold=0.0 (keep everything; the scorer still records
              the quality score for later filtering).
    export    --domain X --min_quality Q [--out F] [--db_path P]
              Fetch matching rows via db.fetch_training_rows and print (or
              write) them as ``id|domain|quality|text``.
    stats     [--db_path P]
              Print db.stats (total, by_domain, avg_quality, used_in_training).
    clean     --min_quality Q [--db_path P]
              Call db.delete_low_quality(path, min_quality) and print the count
              deleted.

All DB access goes through the sibling ``db`` module; nothing here opens
SQLite directly. ``CONFIG`` is imported from the frozen ``config.py`` for the
default db path (contract rule 0.2: no hardcoded constants).

References (patent context): the Metatron architecture routes tokens through
13 polyhedral experts; this CLI curates which texts reach that forward path.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from config import CONFIG
import db


# ── Built-in per-domain synthesis templates ─────────────────────────────────
# ~15 deterministic, technically-flavored sentence templates per domain. They
# are real prose (not "lorem ipsum") so quality.score_quality assigns them a
# meaningful score. ``{i}`` placeholders are filled with a deterministic index
# so cycling beyond the template count still yields distinct rows.

_CYBERSECURITY_TEMPLATES = [
    "The firewall enforces a zero-trust policy with mfa, tls encryption and "
    "anomaly detection to block phishing and ransomware payloads.",
    "Ransomware encrypts files on disk and demands payment; regular patches "
    "and offline backups reduce the blast radius of an outbreak.",
    "A zero-day exploit targets an unpatched vulnerability before the vendor "
    "ships a fix, so intrusion detection and threat hunting are critical.",
    "Public key infrastructure binds identities to certificates via a chain "
    "of trust anchored at a root ca; tls validates that chain on every "
    "handshake.",
    "Multi-factor authentication combines something you know, have and are, "
    "raising the cost of credential theft far beyond a single leaked "
    "password.",
    "A siem aggregates logs from endpoints, network and cloud to correlate "
    "alerts and surface anomalies a human analyst can triage.",
    "Network segmentation isolates critical assets so a lateral-movement "
    "attempt from a compromised workstation cannot reach the database tier.",
    "Salted password hashing with a slow kdf resists rainbow-table and brute-"
    "force attacks; a unique nonce per record defeats reuse of precomputed "
    "tables.",
    "A red-team exercise simulates a real adversary to test blue-team "
    "detection and response, exposing gaps that tabletop drills miss.",
    "Phishing emails spoof sender headers and urgent language to harvest "
    "credentials; training and inline banners cut click-through rates.",
    "An ids inspects packet streams for known signatures while an ips blocks "
    "in real time; both feed a forensics pipeline for post-incident review.",
    "Hardening a host means closing unused ports, disabling legacy "
    "protocols and enforcing least-privilege accounts before exposure.",
    "An oauth token carries delegated authorization scopes with an expiry, "
    "so a leaked token is bounded in both privilege and lifetime.",
    "Botnet command-and-control traffic hides in dns or https tunnels; "
    "beacon detection relies on timing and volume rather than signatures.",
    "A certificate transparency log makes mis-issued tls certificates publicly "
    "auditable, letting monitors catch a rogue ca fast.",
]

_SOFTWARE_TEMPLATES = [
    "A transformer model projects queries, keys and values, scales by the "
    "inverse root of the head dimension and applies softmax over attention "
    "scores before the output projection.",
    "Recursion solves a problem in terms of smaller subproblems; a base case "
    "terminates the call stack so depth stays bounded and the algorithm "
    "converges.",
    "A unit-test exercises one behavior in isolation; together with coverage "
    "metrics and a ci pipeline it catches regressions before they reach "
    "production.",
    "Encapsulation hides internal state behind a method api so a class can "
    "refactor its representation without breaking its callers.",
    "A container packages an app with its runtime and libraries so the same "
    "image deploys unchanged across dev, staging and kubernetes clusters.",
    "Polymorphism lets a callback or interface dispatch to different concrete "
    "implementations at runtime without a branch in the calling code.",
    "An index on a database column accelerates point queries but slows "
    "writes; a query planner chooses between a full scan and an index seek.",
    "Acid transactions guarantee atomic, consistent, isolated and durable "
    "updates even when several concurrent processes write at once.",
    "Async await suspends a coroutine on i/o without blocking a thread, so a "
    "small pool can serve many concurrent requests with low memory overhead.",
    "A gradient flows backward through the layer via the chain rule; an "
    "optimizer like adam steps the parameters using momentum and a per-axis "
    "learning-rate scale.",
    "An iterator yields elements lazily so a generator can stream a large "
    "sequence without materializing the whole list in memory.",
    "A binary tree keeps searches logarithmic when balanced; rotations in an "
    "avl or red-black tree restore balance after insert and delete.",
    "A microservice owns one bounded context and exposes an api; a schema "
    "and a migration tool keep its storage backward compatible.",
    "Dropout randomly zeros activations during training to regularize the "
    "model and reduce overfitting on the training set.",
    "A cache trades memory for latency; an lru eviction policy keeps the "
    "hot working set resident so repeated lookups hit memory not disk.",
]

_GENERAL_TEMPLATES = [
    "The quick brown fox jumps over the lazy dog near the river bank every "
    "morning before the sun clears the ridge.",
    "A small dataset of clean examples often teaches more than a vast "
    "noisy one, because consistent patterns are easier to learn.",
    "Reading documentation end to end is slow but builds a mental map that "
    "speeds up every later debugging session.",
    "A short walk between tasks resets attention and reduces the chance of "
    "carrying a wrong context into the next problem.",
    "Writing the failing test first clarifies the desired behavior and "
    "keeps the implementation honest as it grows.",
    "A clear commit message explains why a change was made, not just what "
    "changed, so future readers can follow the reasoning.",
    "Breaking a large task into small verifiable steps turns an intimidating "
    "goal into a sequence of checkable wins.",
    "Naming a variable after its meaning rather than its type makes the code "
    "read like a sentence and eases later review.",
    "Simplicity is a feature: a smaller surface area means fewer places for "
    "bugs to hide and fewer paths to document.",
    "A diagram drawn before coding often reveals a hidden assumption that "
    "would otherwise become a late-stage refactor.",
    "Consistent formatting across a codebase lets readers focus on logic "
    "instead of style, so review time goes to substance.",
    "Logging the right amount is a skill: too little hides the bug, too "
    "much buries it under noise.",
    "A backup you have never restored from is a hope, not a backup; test "
    "recovery before you need it for real.",
    "Restating a problem in your own words is often the fastest way to find "
    "the mismatch between what was asked and what was built.",
    "Slow is smooth and smooth is fast: a careful first pass avoids the "
    "rework that haste always costs later.",
]

_DOMAIN_TEMPLATES = {
    "cybersecurity": _CYBERSECURITY_TEMPLATES,
    "software": _SOFTWARE_TEMPLATES,
    "general": _GENERAL_TEMPLATES,
}


def _synthesize(domain: str, count: int) -> list[str]:
    """
    Produce ``count`` deterministic texts for ``domain``.

    Cycles through the domain's template list; when count exceeds the template
    count, an index is appended so rows remain distinct while still
    deterministic. Unknown domains fall back to the general template list.
    """
    templates = _DOMAIN_TEMPLATES.get(domain, _GENERAL_TEMPLATES)
    n_tpl = len(templates)
    out: list[str] = []
    for i in range(count):
        base = templates[i % n_tpl]
        if i < n_tpl:
            out.append(base)
        else:
            # Append a deterministic suffix to keep rows distinct once we
            # cycle past the template list.
            out.append(f"{base} (variant {i - n_tpl + 1})")
    return out


# ── Subcommand handlers ─────────────────────────────────────────────────────

def _cmd_generate(args: argparse.Namespace) -> int:
    domain = args.domain
    if domain not in _DOMAIN_TEMPLATES:
        print(f"warn: unknown domain '{domain}', falling back to 'general'",
              file=sys.stderr)
    texts = _synthesize(domain, args.count)
    # Ensure the schema exists (idempotent) before inserting.
    db.init_db(args.db_path)
    added, rejected = db.add_texts(args.db_path, texts, domain,
                                   quality_threshold=0.0)
    print(f"generated {args.count} rows for domain '{domain}': "
          f"added={added} rejected={rejected}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    domain = args.domain
    # Read lines, strip trailing newline; skip blank lines.
    try:
        with open(args.file, "r", encoding="utf-8") as fh:
            texts = [line.rstrip("\n") for line in fh if line.strip()]
    except FileNotFoundError:
        print(f"error: import file not found: '{args.file}'", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: cannot read import file '{args.file}': {exc}",
              file=sys.stderr)
        return 1
    db.init_db(args.db_path)
    # Contract: import uses quality_threshold=0.0 (keep everything; the
    # scorer still records a quality score for later filtering).
    added, rejected = db.add_texts(args.db_path, texts, domain,
                                    quality_threshold=0.0)
    print(f"imported {len(texts)} lines from '{args.file}' for domain "
          f"'{domain}': added={added} rejected={rejected}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    domain = args.domain
    db.init_db(args.db_path)
    # Export every matching row regardless of used_in_training: fetch both
    # the unused and used sets and merge them.
    limit = 10_000_000
    rows = db.fetch_training_rows(args.db_path, domain, args.min_quality,
                                  limit, used=False)
    rows += db.fetch_training_rows(args.db_path, domain, args.min_quality,
                                   limit, used=True)
    # Stable order by id.
    rows.sort(key=lambda r: r["id"])

    lines = []
    for r in rows:
        text = r["text"]
        # Collapse newlines inside text so the pipe format stays one row per
        # line and remains parseable.
        text_flat = " ".join(str(text).splitlines())
        lines.append(f"{r['id']}|{r['domain']}|{r['quality_score']:.4f}|"
                     f"{text_flat}")

    payload = "\n".join(lines)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(payload)
            if payload:
                fh.write("\n")
        print(f"exported {len(rows)} rows for domain '{domain}' "
              f"(min_quality={args.min_quality}) to '{args.out}'")
    else:
        print(f"# exported {len(rows)} rows for domain '{domain}' "
              f"(min_quality={args.min_quality})")
        if payload:
            print(payload)
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    db.init_db(args.db_path)
    s = db.stats(args.db_path)
    print(f"db: {args.db_path}")
    print(f"  total            : {s['total']}")
    print(f"  used_in_training : {s['used_in_training']}")
    print(f"  avg_quality      : {s['avg_quality']:.4f}")
    by_domain = s["by_domain"]
    if by_domain:
        print("  by_domain        :")
        for domain in sorted(by_domain):
            print(f"    {domain:16s}: {by_domain[domain]}")
    else:
        print("  by_domain        : (empty)")
    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    db.init_db(args.db_path)
    deleted = db.delete_low_quality(args.db_path, args.min_quality)
    print(f"cleaned {deleted} row(s) with quality < {args.min_quality} "
          f"from '{args.db_path}'")
    return 0


# ── Argument parser ─────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manage_data",
        description="TinyMetatron SLM data-management CLI",
    )
    # Global default for the DB path (overridable per-subcommand so each
    # subcommand can be pointed at a temp DB in tests).
    default_db = CONFIG["db_path"]

    sub = parser.add_subparsers(dest="command", required=True,
                                metavar="<subcommand>")

    # generate
    p_gen = sub.add_parser("generate", help="synthesize N deterministic rows")
    p_gen.add_argument("--domain", required=True,
                       choices=["cybersecurity", "software", "general"],
                       help="domain label for the synthesized rows")
    p_gen.add_argument("--count", type=int, required=True,
                       help="number of rows to synthesize")
    p_gen.add_argument("--db_path", default=default_db,
                       help=f"sqlite db path (default: {default_db})")
    p_gen.set_defaults(func=_cmd_generate)

    # import
    p_imp = sub.add_parser("import", help="import lines from a file")
    p_imp.add_argument("--file", required=True, help="input text file path")
    p_imp.add_argument("--domain", required=True,
                       help="domain label for imported rows")
    p_imp.add_argument("--db_path", default=default_db,
                       help=f"sqlite db path (default: {default_db})")
    p_imp.set_defaults(func=_cmd_import)

    # export
    p_exp = sub.add_parser("export", help="export matching rows")
    p_exp.add_argument("--domain", required=True, help="domain to export")
    p_exp.add_argument("--min_quality", type=float, required=True,
                       help="minimum quality score (inclusive)")
    p_exp.add_argument("--out", default=None,
                       help="write to file instead of stdout")
    p_exp.add_argument("--db_path", default=default_db,
                       help=f"sqlite db path (default: {default_db})")
    p_exp.set_defaults(func=_cmd_export)

    # stats
    p_stats = sub.add_parser("stats", help="print db.stats")
    p_stats.add_argument("--db_path", default=default_db,
                         help=f"sqlite db path (default: {default_db})")
    p_stats.set_defaults(func=_cmd_stats)

    # clean
    p_clean = sub.add_parser("clean", help="delete low-quality rows")
    p_clean.add_argument("--min_quality", type=float, required=True,
                         help="delete rows with quality_score < min_quality")
    p_clean.add_argument("--db_path", default=default_db,
                         help=f"sqlite db path (default: {default_db})")
    p_clean.set_defaults(func=_cmd_clean)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point: parse argv, dispatch to the selected subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


# ── Self-test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import tempfile

    sys.stdout.reconfigure(encoding="utf-8")  # rule 0.4 / Windows cp1252 fix

    # Real CLI dispatch by default; pass --selftest to run the embedded self-test.
    _argv = sys.argv[1:]
    if "--selftest" not in _argv and len(_argv) > 0:
        sys.exit(main(_argv))

    def _ok(cond: bool, msg: str) -> None:
        print(("OK  " if cond else "FAIL") + " " + msg)
        assert cond, msg

    tmpdir = tempfile.mkdtemp(prefix="tinymetatron_managedata_")
    db_path = os.path.join(tmpdir, "test.db")

    # 1. generate --count 5 --domain cybersecurity
    rc = main(["generate", "--count", "5", "--domain", "cybersecurity",
               "--db_path", db_path])
    _ok(rc == 0, "generate returned 0")
    _ok(os.path.exists(db_path), "generate created the db file")

    s = db.stats(db_path)
    print("after generate:", s)
    _ok(s["total"] == 5, f"stats total == 5 (got {s['total']})")
    _ok(s["by_domain"].get("cybersecurity") == 5,
        "by_domain cybersecurity == 5")
    _ok(s["avg_quality"] > 0.0, "avg_quality > 0")

    # 2. stats subcommand (must exit 0 and print without error)
    rc = main(["stats", "--db_path", db_path])
    _ok(rc == 0, "stats returned 0")

    # 3. export --domain cybersecurity --min_quality 0.0 --out <file>
    out_path = os.path.join(tmpdir, "export.txt")
    rc = main(["export", "--domain", "cybersecurity", "--min_quality", "0.0",
               "--out", out_path, "--db_path", db_path])
    _ok(rc == 0, "export returned 0")
    _ok(os.path.exists(out_path), "export wrote the output file")
    with open(out_path, "r", encoding="utf-8") as fh:
        export_lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    _ok(len(export_lines) == 5,
        f"export file has 5 rows (got {len(export_lines)})")
    # each line is id|domain|quality|text with exactly 3 pipes
    for ln in export_lines:
        parts = ln.split("|", 3)
        _ok(len(parts) == 4, f"export line has 4 pipe-fields: {ln[:40]!r}")
        _ok(parts[1] == "cybersecurity", "export line domain is cybersecurity")

    # 4. clean --min_quality (set high enough to delete at least one row)
    s_before = db.stats(db_path)
    # pick a threshold just below the max quality so we delete some but maybe
    # not all; use the average as a safe middle threshold.
    threshold = s_before["avg_quality"]
    rc = main(["clean", "--min_quality", str(threshold),
               "--db_path", db_path])
    _ok(rc == 0, "clean returned 0")
    s_after = db.stats(db_path)
    _ok(s_after["total"] <= s_before["total"],
        "clean did not increase total")
    print(f"clean: before={s_before['total']} after={s_after['total']} "
          f"(threshold={threshold:.4f})")

    # 5. import a tiny temp file into a fresh domain
    imp_path = os.path.join(tmpdir, "import.txt")
    with open(imp_path, "w", encoding="utf-8") as fh:
        fh.write("The transformer attends over a sequence of token embeddings.\n")
        fh.write("\n")  # blank line, must be skipped
        fh.write("A gradient descent step follows the slope of the loss.\n")
    rc = main(["import", "--file", imp_path, "--domain", "software",
               "--db_path", db_path])
    _ok(rc == 0, "import returned 0")
    s = db.stats(db_path)
    _ok(s["by_domain"].get("software", 0) == 2,
        f"import added 2 software rows (got {s['by_domain'].get('software')})")

    # cleanup
    try:
        os.remove(db_path)
    except OSError:
        pass
    for p in (out_path, imp_path):
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    print("\nSELF-TEST PASSED")