"""
Retrieval-Improvement Loop Orchestrator

OBSERVE -> ORIENT -> PROPOSE -> VALIDATE -> APPROVE -> EXECUTE -> MEASURE -> LEARN
"""
from __future__ import annotations

import datetime, json, os, re, shutil, subprocess, sys, textwrap, time
from datetime import timezone
from pathlib import Path
from typing import Optional

import yaml

try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "loops" / "retrieval_loop" / "experiments"
FAILURE_CARDS = REPO_ROOT / "quantum_corpus" / "eval" / "retrieval_failure_cards.jsonl"
CORPUS_DB = os.environ.get("TMT_QUANTUM_CORPUS_DB", r"E:\Temp\qcorpus\quantum_corpus.db")

sys.path.insert(0, str(REPO_ROOT))

# ── helpers ───────────────────────────────────────────────────────────────────
def _current_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"

def _commit_pending():
    try:
        r = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=REPO_ROOT, capture_output=True, text=True
        )
        return bool(r.stdout.strip())
    except Exception:
        return False

def load_failure_cards():
    if not FAILURE_CARDS.exists():
        return []
    with open(FAILURE_CARDS, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def load_experiment(exp_id: str) -> Optional[dict]:
    path = EXP_DIR / exp_id / "experiment.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_experiment(exp_id: str, exp: dict):
    path = EXP_DIR / exp_id / "experiment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(exp, f, ensure_ascii=False, indent=2)

def load_all_experiments():
    if not EXP_DIR.exists():
        return {}
    exps = {}
    for d in EXP_DIR.iterdir():
        if d.is_dir() and (d / "experiment.json").exists():
            with open(d / "experiment.json", encoding="utf-8") as f:
                exps[d.name] = json.load(f)
    return exps

def cards_summary(cards):
    open_cards = [c for c in cards if not c.get("in_top5")]
    addressed = len(cards) - len(open_cards)
    by_cat: dict[str, int] = {}
    for c in open_cards:
        cat = c.get("category", "unknown")
        by_cat[cat] = by_cat.get(cat, 0) + 1
    return addressed, len(open_cards), open_cards, by_cat

def _print_status_rich(cards, exps):
    console = Console()
    # Failure card summary
    addressed, total_open, open_cards, by_cat = cards_summary(cards)
    t = Table(title="Failure Cards")
    t.add_column("Category", style="cyan")
    t.add_column("Open", style="red")
    t.add_column("Addressed", style="green")
    for cat, n in sorted(by_cat.items()):
        t.add_row(cat, str(n), str(sum(1 for c in open_cards if c.get("category") != cat)))
    t.add_row("TOTAL", str(total_open), str(addressed))
    console.print(t)

# ── Gate infrastructure ──────────────────────────────────────────────────────
GATES = {
    "gate_regression_check": {
        "description": "Retrieval dev eval — no regression",
        "command": [
            sys.executable, "-m", "quantum_corpus.eval.runner",
            "dev", "--mode", "ask", "--retriever", "hybrid",
        ],
        "timeout": 600,
    },
    "gate_canary": {
        "description": "Canary leakage check",
        "command": [
            sys.executable, "-m", "quantum_corpus.eval.runner",
            "run_canaries",
        ],
        "timeout": 120,
    },
    "gate_conscious_dna_alias": {
        "description": "Conscious-DNA alias regression",
        "command": [
            sys.executable, "-m", "quantum_corpus.eval.test_conscious_dna_alias",
        ],
        "timeout": 120,
    },
    "gate_bm25_parity": {
        "description": "BM25-only runner parity",
        "command": [
            sys.executable, "-m", "quantum_corpus.eval.runner",
            "dev", "--mode", "bm25",
        ],
        "timeout": 300,
    },
}

def run_gate(name: str, command: list, timeout: int = 600) -> dict:
    """Run one gate. Returns {passed, stdout, stderr, duration}."""
    start = datetime.now(timezone.utc)
    env = os.environ.copy()
    env["TMT_QUANTUM_CORPUS_DB"] = CORPUS_DB
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(REPO_ROOT),
        )
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        passed = result.returncode == 0
        return {
            "passed": passed,
            "returncode": result.returncode,
            "duration_s": round(duration, 1),
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-500:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        return {
            "passed": False,
            "returncode": -1,
            "duration_s": round(duration, 1),
            "stdout": "",
            "stderr": "timeout",
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

def run_all_gates(gate_keys=None):
    results = {}
    for name, gate in GATES.items():
        if gate_keys and name not in gate_keys:
            continue
        cmd = gate["command"]
        print(f"  Running {name}...")
        r = run_gate(name, cmd, gate.get("timeout", 600))
        results[name] = r
        status = "PASS" if r["passed"] else f"FAIL (exit {r['returncode']})"
        print(f"    => {status} in {r['duration_s']}s")
    return results

def all_gates_passed(results: dict) -> bool:
    return all(v["passed"] for v in results.values())

# ── Commands ──────────────────────────────────────────────────────────────────
def cmd_list():
    cards = load_failure_cards()
    addressed, total_open, open_cards, by_cat = cards_summary(cards)
    print(f"Failure card summary:")
    print(f"  Total     : {len(cards)}")
    print(f"  Addressed : {addressed}  (in_top5=True)")
    print(f"  Open      : {total_open}")
    print(f"  Open IDs  : {', '.join(c.get('id','?') for c in open_cards[:10])}")
    print(f"  By category:")
    for cat, n in sorted(by_cat.items()):
        print(f"    {cat}: {n}")
    return 0

def cmd_propose(exp_id: str, hypothesis: str, proposed_action: str, root_cause: str):
    exp = load_experiment(exp_id)
    if exp is None:
        exp = {
            "exp_id": exp_id,
            "state": "NEW",
            "created_at": datetime.datetime.now(timezone.utc).isoformat(),
            "root_cause": root_cause,
            "hypothesis": hypothesis,
            "proposed_action": proposed_action,
            "code_commit_before": _current_commit(),
            "validation_results": {},
            "gate_results": {},
            "human_approval": None,
            "approval_note": "",
            "code_commit_after": None,
            "metrics_delta": {},
            "loop_id": None,
            "project": "tinymetatron-retrieval",
        }
        save_experiment(exp_id, exp)
        print(f"Created experiment {exp_id}")
    else:
        print(f"Experiment {exp_id} already exists")
    print(f"Hypothesis: {hypothesis}")
    return 0

def cmd_gates(exp_id: str):
    exp = load_experiment(exp_id)
    if exp is None:
        print(f"ERROR: Experiment {exp_id} not found.")
        return 1
    if exp["state"] not in ("HYPOTHESIS_CREATED", "OBSERVED"):
        print(f"Experiment {exp_id} is {exp['state']} — cannot run gates from this state")
        return 1

    print(f"\nRunning gates for {exp_id}...")
    results = run_all_gates()

    # Save gate results
    exp["gate_results"] = results
    passed = all_gates_passed(results)
    if passed:
        exp["state"] = "VALIDATED"
        print(f"\nALL GATES PASSED — {exp_id} is VALIDATED")
    else:
        exp["state"] = "OBSERVED"
        failed = [k for k, v in results.items() if not v["passed"]]
        print(f"\nGATES FAILED: {', '.join(failed)}")

    save_experiment(exp_id, exp)
    return 0 if passed else 1

def cmd_approve(exp_id: str, note: str = ""):
    exp = load_experiment(exp_id)
    if exp is None:
        print(f"ERROR: Experiment {exp_id} not found.")
        return 1
    if exp["state"] != "VALIDATED":
        print(f"Experiment {exp_id} is {exp['state']} — must be VALIDATED before approval.")
        return 1
    exp["human_approval"] = datetime.datetime.now(timezone.utc).isoformat()
    exp["approval_note"] = note
    exp["state"] = "AWAITING_APPROVAL"
    save_experiment(exp_id, exp)
    print(f"Experiment {exp_id} approved. Awaiting execution.")
    return 0

def cmd_execute(exp_id: str):
    """Commit the change — implementation-specific, caller must verify."""
    exp = load_experiment(exp_id)
    if exp is None:
        print(f"ERROR: Experiment {exp_id} not found.")
        return 1
    if exp["state"] != "AWAITING_APPROVAL":
        print(f"Experiment {exp_id} is {exp['state']} — must be AWAITING_APPROVAL before execution.")
        return 1
    exp["state"] = "EXECUTED"
    exp["code_commit_after"] = _current_commit()
    save_experiment(exp_id, exp)
    print(f"Experiment {exp_id} executed.")
    return 0

def cmd_reject(exp_id: str):
    exp = load_experiment(exp_id)
    if exp is None:
        print(f"ERROR: Experiment {exp_id} not found.")
        return 1
    exp["state"] = "REJECTED"
    save_experiment(exp_id, exp)
    archive_dir = EXP_DIR / "archive" / exp_id
    shutil.move(str(EXP_DIR / exp_id), str(archive_dir))
    print(f"Experiment {exp_id} archived as REJECTED.")
    return 0

def cmd_status():
    cards = load_failure_cards()
    addressed, total_open, open_cards, by_cat = cards_summary(cards)
    exps = load_all_experiments()

    print(f"Failure card summary:")
    print(f"  Total     : {len(cards)}")
    print(f"  Addressed : {addressed}")
    print(f"  Open      : {total_open}")
    print(f"  Open IDs  : {', '.join(c.get('id','?') for c in open_cards)}")
    print(f"  By category:")
    for cat, n in sorted(by_cat.items()):
        print(f"    {cat}: {n}")

    print(f"\nExperiments:")
    for exp_id in sorted(exps, key=lambda x: int(re.search(r"\d+", x).group())):
        e = exps[exp_id]
        gates = e.get("gate_results", {})
        parts = []
        for k, v in gates.items():
            s = "[+]" if v["passed"] else "[-]"
            parts.append("{0}:{1}".format(k, s))
        gate_str = " ".join(parts) if gates else "none"
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

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 0

    cmd = sys.argv[1]

    if cmd == "list":
        return cmd_list()
    elif cmd == "status":
        return cmd_status()
    elif cmd == "propose":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--exp", required=True)
        p.add_argument("--hypothesis", required=True)
        p.add_argument("--action", dest="action", required=True,
                       help="One-line description of the change")
        p.add_argument("--root-cause", required=True)
        args = p.parse_args(sys.argv[2:])
        return cmd_propose(args.exp, args.hypothesis, args.action, args.root_cause)
    elif cmd == "gates":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--exp", required=True)
        args = p.parse_args(sys.argv[2:])
        return cmd_gates(args.exp)
    elif cmd == "approve":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--exp", required=True)
        p.add_argument("--note", default="")
        args = p.parse_args(sys.argv[2:])
        return cmd_approve(args.exp, args.note)
    elif cmd == "reject":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--exp", required=True)
        args = p.parse_args(sys.argv[2:])
        return cmd_reject(args.exp)
    elif cmd == "execute":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--exp", required=True)
        args = p.parse_args(sys.argv[2:])
        return cmd_execute(args.exp)
    else:
        print(f"Unknown command: {cmd}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
