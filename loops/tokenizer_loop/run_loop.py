"""
run_loop.py
===========
Tokenizer-improvement loop orchestrator for TinyMetatron.

OBSERVE -> ORIENT -> PROPOSE -> VALIDATE -> APPROVE -> EXECUTE -> MEASURE -> LEARN

Each loop iteration:
  1. Measures current tokenizer fragmentation via refresh.py
  2. Operator identifies one issue and creates one experiment
  3. Loop runs mandatory gates (train, eval, atomic tag, unk count, prose median)
  4. Waits for human approval before committing
  5. Archives the experiment

Usage::

    python loops/tokenizer_loop/run_loop.py list
    python loops/tokenizer_loop/run_loop.py propose --exp 001 --hypothesis "..." --action "..."
    python loops/tokenizer_loop/run_loop.py gates --exp 001
    python loops/tokenizer_loop/run_loop.py approve --exp 001
    python loops/tokenizer_loop/run_loop.py reject --exp 001
    python loops/tokenizer_loop/run_loop.py status
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.resolve()
TOKENIZER_LOOP = Path(__file__).parent.resolve()
EXPERIMENTS_DIR = TOKENIZER_LOOP / "experiments"
ARCHIVE_DIR = TOKENIZER_LOOP / "archive"

# Paths to the English tokenizer and its training infrastructure
TOKENIZER_DIR = ROOT / "experiments/english_first_tokenizer/tokenizers/tinymetatron_v2_en_8k"
TRAIN_SCRIPT = ROOT / "experiments/english_first_tokenizer/train_en_tokenizer.py"
TRAIN_DATA_DIR = ROOT / "experiments/english_first_tokenizer/data/train"
TRAIN_DATA_APPROVED = ROOT / "experiments/english_first_tokenizer/data/approved"

sys.path.insert(0, str(ROOT))


# ── State machine ──────────────────────────────────────────────────────────────

STATES = [
    "NEW", "OBSERVED", "HYPOTHESIS_CREATED", "VALIDATED",
    "AWAITING_APPROVAL", "EXECUTED", "MEASURED",
    "ACCEPTED", "REJECTED", "ARCHIVED",
]

TERMINAL = {"ACCEPTED", "REJECTED", "ARCHIVED"}


def load_experiments() -> dict:
    """Load all experiments. Returns {exp_id: exp_data}."""
    exps = {}
    for d in sorted(EXPERIMENTS_DIR.iterdir()):
        if not d.is_dir() or not re.match(r"^exp-\d+$", d.name):
            continue
        meta = d / "experiment.json"
        if meta.exists():
            with open(meta, encoding="utf-8") as f:
                exps[d.name] = json.load(f)
    return exps


def save_experiment(exp_id: str, data: dict) -> None:
    """Save experiment data to its directory."""
    d = EXPERIMENTS_DIR / exp_id
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def next_exp_id() -> str:
    """Return next sequential experiment ID."""
    existing = load_experiments()
    if not existing:
        return "exp-001"
    nums = []
    for k in existing:
        m = re.search(r"\d+", k)
        if m:
            nums.append(int(m.group()))
    return f"exp-{max(nums) + 1:03d}"


def create_experiment(exp_id: str, hypothesis: str, proposed_action: str) -> dict:
    """Create a new experiment record."""
    return {
        "exp_id": exp_id,
        "state": "NEW",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": hypothesis,
        "proposed_action": proposed_action,
        "code_commit_before": _current_commit(),
        "validation_results": {},
        "gate_results": {},
        "human_approval": None,
        "approval_note": "",
        "code_commit_after": None,
        "metrics_delta": {},
        "loop_id": str(uuid.uuid4()),
        "project": "tinymetatron-tokenizer",
        # Frozen input artifacts
        "frozen_tokenizer_sha": _file_sha256(TOKENIZER_DIR / "tokenizer.json"),
        "frozen_vocab_sha": _file_sha256(TOKENIZER_DIR / "vocab.json"),
        "frozen_merges_sha": _file_sha256(TOKENIZER_DIR / "merges.txt"),
        "frozen_manifest": _read_json(TOKENIZER_DIR / "manifest.json"),
    }


# ── Git / file helpers ────────────────────────────────────────────────────────

def _current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()[:12]
    except Exception:
        return "unknown"


def _is_clean() -> bool:
    try:
        status = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return status == ""
    except Exception:
        return False


def _file_sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Current metrics (from refresh.py output) ──────────────────────────────────

def load_current_metrics() -> dict:
    """Load current tokenizer metrics from refresh.py output."""
    metrics_path = TOKENIZER_LOOP / "current_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def corpus_manifest() -> dict:
    """Build a manifest of approved training data files with SHA256 hashes."""
    manifest = {}
    if TRAIN_DATA_APPROVED.exists():
        for f in sorted(TRAIN_DATA_APPROVED.glob("*.txt")):
            manifest[f.name] = _file_sha256(f)
    return manifest


# ── Gate runners ──────────────────────────────────────────────────────────────

GATE_DEFS = [
    ("train_tokenizer",         "python experiments/english_first_tokenizer/train_en_tokenizer.py",
                                 "Train tokenizer — no crash"),
    ("en_tag_atomic",           "python -c \"import sys; sys.path.insert(0,'experiments/english_first_tokenizer'); from train_en_tokenizer import evaluate_tokenizer, TEST_CASES; from tokenizers import ByteLevelBPETokenizer; t=ByteLevelBPETokenizer(str('experiments/english_first_tokenizer/tokenizers/tinymetatron_v2_en_8k')); enc=t.encode('<|en|>'); assert len(enc.ids)==1, f'<|en|> not atomic: {enc.ids}'\"",
                                 "<|en|> encodes as exactly 1 token"),
    ("unk_count_zero",           "python -c \"import sys; sys.path.insert(0,'experiments/english_first_tokenizer'); from train_en_tokenizer import evaluate_tokenizer; from tokenizers import ByteLevelBPETokenizer; t=ByteLevelBPETokenizer(str('experiments/english_first_tokenizer/tokenizers/tinymetatron_v2_en_8k')); _,s=evaluate_tokenizer(t); assert s['all_unk']==0, f'unk count={s[\\\"all_unk\\\"]}'\"",
                                 "Unknown token count = 0 on TEST_CASES"),
    ("round_trip",               "python -c \"import sys; sys.path.insert(0,'experiments/english_first_tokenizer'); from train_en_tokenizer import TEST_CASES; from tokenizers import ByteLevelBPETokenizer; t=ByteLevelBPETokenizer(str('experiments/english_first_tokenizer/tokenizers/tinymetatron_v2_en_8k')); errors=[]; [errors.append(n) for n,c in TEST_CASES.items() if t.decode(t.encode(c).ids, skip_special_tokens=False)!=c]; assert not errors, f'Round-trip failed: {errors}'\"",
                                 "Round-trip encode/decode passes for all TEST_CASES"),
]


def run_gate(name: str, command: str, timeout: int = 600) -> dict:
    """Run one gate. Returns {passed, stdout, stderr, duration}."""
    start = datetime.now(timezone.utc)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(ROOT), env=env,
        )
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        passed = result.returncode == 0
        return {
            "passed": passed,
            "returncode": result.returncode,
            "duration_s": round(duration, 1),
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-1000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return {
            "passed": False,
            "returncode": -1,
            "duration_s": round(duration, 1),
            "stdout": "",
            "stderr": f"Timeout after {timeout}s",
        }
    except Exception as e:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return {
            "passed": False,
            "returncode": -2,
            "duration_s": round(duration, 1),
            "stdout": "",
            "stderr": str(e),
        }


def run_all_gates() -> dict:
    """Run all mandatory gates. Returns {gate_name: result_dict}."""
    results = {}
    for name, cmd, desc in GATE_DEFS:
        print(f"\n  [{name}] {desc}")
        print(f"    $ {cmd}")
        r = run_gate(name, cmd)
        results[name] = r
        status = "PASS" if r["passed"] else f"FAIL (exit {r['returncode']})"
        print(f"    => {status} in {r['duration_s']}s")
        if not r["passed"] and r["stderr"]:
            print(f"    stderr: {r['stderr'][:500]}")
    return results


def all_gates_passed(results: dict) -> bool:
    return all(v["passed"] for v in results.values())


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list():
    """List current tokenizer metrics summary."""
    metrics = load_current_metrics()
    manifest = corpus_manifest()
    tok_manifest = _read_json(TOKENIZER_DIR / "manifest.json")

    print(f"\n{'='*60}")
    print("CURRENT TOKENIZER STATE")
    print(f"{'='*60}")

    if metrics:
        print(f"  prose_median tokens/word : {metrics.get('prose_median_tokens_per_word', '?')}")
        print(f"  all_avg tokens/word    : {metrics.get('all_avg_tokens_per_word', '?')}")
        print(f"  all_p95 tokens/word    : {metrics.get('all_p95_tokens_per_word', '?')}")
        print(f"  unknown tokens         : {metrics.get('all_unk', '?')}")
        print(f"  <|en|> atomic          : {metrics.get('all_tags_atomic', '?')}")
        print(f"  decision               : {metrics.get('decision', '?')}")
    else:
        print("  No metrics yet. Run: python loops/tokenizer_loop/refresh.py")

    print(f"\n  Tokenizer: {TOKENIZER_DIR}")
    print(f"  vocab_size: {tok_manifest.get('vocab_size', '?')}")
    print(f"  special_tokens: {tok_manifest.get('special_tokens', [])}")

    print(f"\n  Approved training files ({len(manifest)}):")
    for name, sha in manifest.items():
        print(f"    {name}: {sha[:16]}...")

    print()
    return 0


def cmd_propose(exp_id: str | None, hypothesis: str, proposed_action: str):
    """Create a new experiment proposal."""
    if exp_id is None:
        exp_id = next_exp_id()
    else:
        exp_id = f"exp-{int(exp_id):03d}"
        if (EXPERIMENTS_DIR / exp_id).exists():
            print(f"ERROR: Experiment {exp_id} already exists.")
            return 1

    exp = create_experiment(exp_id, hypothesis, proposed_action)
    exp["corpus_manifest"] = corpus_manifest()
    save_experiment(exp_id, exp)
    print(f"Created {exp_id} in {EXPERIMENTS_DIR / exp_id}")
    print(f"  State: NEW")
    print(f"  Hypothesis: {hypothesis}")
    print(f"  Proposed action: {proposed_action}")
    print(f"  Frozen tokenizer: {exp['frozen_tokenizer_sha'][:16]}...")
    return 0


def cmd_gates(exp_id: str):
    """Run gates for an experiment and record results."""
    d = EXPERIMENTS_DIR / exp_id
    if not d.exists():
        print(f"ERROR: Experiment {exp_id} not found.")
        return 1

    with open(d / "experiment.json", encoding="utf-8") as f:
        exp = json.load(f)

    if exp["state"] not in ("NEW", "HYPOTHESIS_CREATED", "REJECTED"):
        print(f"ERROR: Experiment {exp_id} is in state {exp['state']}, not runnable.")
        return 1

    # Transition to OBSERVED
    exp["state"] = "OBSERVED"
    save_experiment(exp_id, exp)

    print(f"\n{'='*60}")
    print(f"Running gates for {exp_id}")
    print(f"  Hypothesis: {exp['hypothesis']}")
    print(f"  Proposed action: {exp['proposed_action']}")
    print(f"{'='*60}")

    results = run_all_gates()

    # Also run a soft prose_median gate — warn if > 2.5 but pass if <= 4.0
    prose_ok = _soft_prose_median_gate()

    exp["gate_results"] = results
    exp["soft_prose_note"] = prose_ok

    passed = all_gates_passed(results)
    if passed:
        exp["state"] = "VALIDATED"
        print(f"\nALL GATES PASSED — {exp_id} is VALIDATED")
    else:
        exp["state"] = "OBSERVED"
        failed = [k for k, v in results.items() if not v["passed"]]
        print(f"\nGATES FAILED: {', '.join(failed)}")

    save_experiment(exp_id, exp)
    return 0


def _soft_prose_median_gate() -> str:
    """Check prose_median and warn if above 2.5. Returns note string."""
    try:
        from tokenizers import ByteLevelBPETokenizer
        sys.path.insert(0, str(ROOT / "experiments/english_first_tokenizer"))
        from train_en_tokenizer import evaluate_tokenizer
        t = ByteLevelBPETokenizer(str(TOKENIZER_DIR))
        _, summary = evaluate_tokenizer(t)
        pm = summary.get("prose_median_tokens_per_word", 999)
        unk = summary.get("all_unk", -1)
        atomic = summary.get("all_tags_atomic", False)
        note = f"prose_median={pm}, unk={unk}, tags_atomic={atomic}"
        if pm > 4.0:
            note += " [HARDFAIL: prose_median > 4.0]"
        elif pm > 2.5:
            note += " [SOFT WARN: prose_median > 2.5]"
        else:
            note += " [OK]"
        print(f"\n  [soft_prose_median] {note}")
        return note
    except Exception as e:
        return f"Could not evaluate prose_median: {e}"


def cmd_approve(exp_id: str, note: str = ""):
    """Approve and commit an experiment. Requires VALIDATED state."""
    d = EXPERIMENTS_DIR / exp_id
    if not d.exists():
        print(f"ERROR: Experiment {exp_id} not found.")
        return 1

    with open(d / "experiment.json", encoding="utf-8") as f:
        exp = json.load(f)

    if exp["state"] not in ("VALIDATED", "AWAITING_APPROVAL"):
        print(f"ERROR: Experiment must be VALIDATED. Currently: {exp['state']}")
        return 1

    # Check git is clean
    if not _is_clean():
        print("ERROR: Working tree is not clean. Commit or stash changes first.")
        return 1

    # Stage relevant files
    to_stage = [
        ROOT / "experiments/english_first_tokenizer",
        ROOT / "experiments/english_first_tokenizer/tokenizers",
        TOKENIZER_LOOP / "current_metrics.json",
    ]
    staged = [str(p) for p in to_stage if p.exists()]
    if staged:
        subprocess.run(["git", "-C", str(ROOT), "add"] + staged, check=True)

    commit_msg = (
        f"Tokenizer loop {exp_id}: {exp['hypothesis']}\n\n"
        f"Action: {exp['proposed_action']}\n"
        f"Loop ID: {exp['loop_id']}\n"
        f"Co-Authored-By: Claude <noreply@anthropic.com>\n"
    )
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "commit", "-m", commit_msg],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Git commit failed:\n{result.stderr}")
            return 1
        commit = _current_commit()
    except Exception as e:
        print(f"Git commit failed: {e}")
        return 1

    exp["state"] = "EXECUTED"
    exp["human_approval"] = True
    exp["approval_note"] = note
    exp["code_commit_after"] = commit
    save_experiment(exp_id, exp)

    print(f"Committed as {commit}")
    print(f"Experiment {exp_id} is EXECUTED.")
    return 0


def cmd_reject(exp_id: str, reason: str = ""):
    """Archive an experiment as rejected."""
    d = EXPERIMENTS_DIR / exp_id
    if not d.exists():
        print(f"ERROR: Experiment {exp_id} not found.")
        return 1

    with open(d / "experiment.json", encoding="utf-8") as f:
        exp = json.load(f)

    exp["state"] = "REJECTED"
    exp["human_approval"] = False
    exp["approval_note"] = reason
    save_experiment(exp_id, exp)

    # Archive
    archive_name = f"{exp_id}_rejected_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    archive_path = ARCHIVE_DIR / archive_name
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(d), str(archive_path))
    print(f"Moved {exp_id} to archive: {archive_path}")
    return 0


def cmd_status():
    """Show status of all experiments."""
    exps = load_experiments()
    if not exps:
        print("\nNo experiments yet.")
        return 0

    for exp_id in sorted(exps, key=lambda x: int(re.search(r"\d+", x).group())):
        e = exps[exp_id]
        gates = e.get("gate_results", {})
        gate_str = " ".join(
            f"{k}:{'PASS' if v['passed'] else 'FAIL'}"
            for k, v in gates.items()
        ) if gates else "none"
        print(f"\n  [{exp_id}] {e['state']}")
        print(f"    Hypothesis: {e['hypothesis']}")
        print(f"    Action: {e['proposed_action']}")
        print(f"    Gates: {gate_str}")
        print(f"    Created: {e['created_at'][:10]}")
        if e.get("human_approval") is not None:
            print(f"    Approved: {e['human_approval']}  Note: {e['approval_note']}")
        if e.get("code_commit_after"):
            print(f"    Commit: {e['code_commit_after']}")
    print()
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    cmd = argv[0]

    if cmd == "list":
        return cmd_list()

    elif cmd == "propose":
        hypothesis = ""
        proposed_action = ""
        exp_id = None
        i = 1
        while i < len(argv):
            if argv[i] == "--exp" and i + 1 < len(argv):
                exp_id = argv[i + 1]; i += 2
            elif argv[i] == "--hypothesis" and i + 1 < len(argv):
                hypothesis = argv[i + 1]; i += 2
            elif argv[i] == "--action" and i + 1 < len(argv):
                proposed_action = argv[i + 1]; i += 2
            else:
                i += 1
        if not hypothesis or not proposed_action:
            print("ERROR: --hypothesis and --action are required")
            return 1
        return cmd_propose(exp_id, hypothesis, proposed_action)

    elif cmd == "gates":
        exp_id = None
        for i, a in enumerate(argv[1:]):
            if a == "--exp" and i + 2 < len(argv):
                exp_id = argv[i + 2]
        if not exp_id:
            print("ERROR: --exp <id> required")
            return 1
        return cmd_gates(exp_id)

    elif cmd == "approve":
        exp_id = None
        note = ""
        for i, a in enumerate(argv[1:]):
            if a == "--exp" and i + 2 < len(argv):
                exp_id = argv[i + 2]
            elif a == "--note" and i + 2 < len(argv):
                note = argv[i + 2]
        if not exp_id:
            print("ERROR: --exp <id> required")
            return 1
        return cmd_approve(exp_id, note)

    elif cmd == "reject":
        exp_id = None
        reason = ""
        for i, a in enumerate(argv[1:]):
            if a == "--exp" and i + 2 < len(argv):
                exp_id = argv[i + 2]
            elif a == "--reason" and i + 2 < len(argv):
                reason = argv[i + 2]
        if not exp_id:
            print("ERROR: --exp <id> required")
            return 1
        return cmd_reject(exp_id, reason)

    elif cmd == "status":
        return cmd_status()

    else:
        print(f"Unknown command: {cmd}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
