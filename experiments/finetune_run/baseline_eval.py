"""
Baseline zero-shot evaluation of Qwen2.5-3B-Instruct (Ollama) on quantum_corpus RAG eval set.

Run:
    python experiments/finetune_run/baseline_eval.py --max-records N --output report.json

Requires:
    - Ollama running with qwen2.5:3b-instruct pulled
    - quantum_corpus eval data at quantum_corpus/eval/qa_val.jsonl
    - CPU inference (no GPU needed for zero-shot eval)
"""

from __future__ import annotations

import json
import time
import statistics
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

# ── Ollama API ────────────────────────────────────────────────────────────────

OLLAMA_BASE = "http://localhost:11434"
MODEL_NAME = "qwen2.5:3b-instruct"

SYSTEM_PROMPT = """You are a quantum research assistant.
Answer questions based ONLY on the provided retrieval context.
If the context does not contain enough information to answer, say so.
Cite specific records when available. Do not fabricate information."""


def ollama_chat(prompt: str, system: str = SYSTEM_PROMPT, timeout: int = 120) -> str:
    """Call Ollama chat API and return the assistant message."""
    import urllib.request
    import urllib.error

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 512}
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.load(resp)
            return result["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        return f"[HTTP ERROR {e.code}]: {e.reason}"
    except urllib.error.URLError as e:
        return f"[URL ERROR]: {e.reason}"
    except Exception as e:
        return f"[ERROR]: {e}"


# ── Evaluation helpers ────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    qa_id: str
    category: str
    question: str
    gold_answer_snippet: str
    model_answer: str
    verdict: str          # "correct" | "incorrect" | "abstained" | "error"
    abstained: bool
    response_time_ms: int
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_answer(model_answer: str, question: str, gold_answer_requirements: str,
                   expected_abstention: bool) -> tuple[str, bool]:
    """Judge a model answer against a question.

    Returns (verdict, abstained).

    Abstention heuristics:
    - Answer contains "do not know", "not in the context", "cannot answer",
      "not enough information", "i'm not sure", "doesn't mention"
    """
    abstention_phrases = [
        "do not know", "don't know", "not know",
        "not in the context", "not provided in the context",
        "not enough information", "cannot answer", "cannot be determined",
        "i'm not sure", "i am not sure", "i could not find",
        "could not find", "doesn't mention", "does not mention",
        "context does not", "no information about",
    ]

    answer_lower = model_answer.lower()
    abstained = any(phrase in answer_lower for phrase in abstention_phrases)

    if abstained:
        if expected_abstention:
            return "correct", True
        else:
            # Topically related but didn't find — penalize as incorrect
            return "incorrect", True

    # Check if answer contains the key facts from gold requirements
    # gold_answer_requirements is a string like "Cite record 36895 and state backend is ibm_fez"
    key_terms = [t.strip(".,").lower()
                 for t in gold_answer_requirements.split()
                 if len(t.strip(".,")) > 2]

    if not key_terms:
        # No specific terms to check — anything non-abstaining is "correct"
        return "correct", False

    matched = sum(1 for term in key_terms if term in answer_lower)
    if matched >= len(key_terms) * 0.5:
        return "correct", False
    else:
        return "incorrect", False


# ── Main eval ────────────────────────────────────────────────────────────────

def run_baseline_eval(
    qa_path: Path,
    output_path: Path,
    max_records: Optional[int] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Run zero-shot eval and save results."""

    # Load QA records
    records = []
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if max_records:
        records = records[:max_records]

    results: List[EvalResult] = []
    correct = 0
    incorrect = 0
    abstained = 0
    errors = 0

    print(f"Running zero-shot baseline on {len(records)} questions...")
    print(f"Model: {MODEL_NAME} (Ollama)")
    print(f"Source: {qa_path}")
    print()

    for i, rec in enumerate(records):
        qa_id = rec.get("id", f"q{i}")
        question = rec["question"]
        gold_req = rec.get("answer_requirements", "")
        expected_abstain = rec.get("expected_abstention", False)
        category = rec.get("category", "unknown")

        print(f"[{i+1}/{len(records)}] {qa_id}: ", end="", flush=True)

        if dry_run:
            model_answer = "[dry-run]"
            verdict = "correct"
            abstained = expected_abstain
            response_time_ms = 0
        else:
            t0 = time.perf_counter()
            model_answer = ollama_chat(question)
            response_time_ms = int((time.perf_counter() - t0) * 1000)

            verdict, abstained = evaluate_answer(
                model_answer, question, gold_req, expected_abstain
            )

        if verdict == "correct":
            correct += 1
        elif verdict == "incorrect":
            incorrect += 1
        elif verdict == "abstained":
            abstained += 1
        else:
            errors += 1

        result = EvalResult(
            qa_id=qa_id,
            category=category,
            question=question,
            gold_answer_snippet=gold_req,
            model_answer=model_answer,
            verdict=verdict,
            abstained=abstained,
            response_time_ms=response_time_ms,
            notes=f"expected_abstention={expected_abstain}"
        )
        results.append(result)

        status = verdict.upper()
        print(f"{status} ({response_time_ms}ms)")

        # Progress dot every 10
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(records)}  "
                  f"correct={correct}  incorrect={incorrect}  abstained={abstained}")

    # Summary
    total = len(results)
    n_correct = correct
    n_incorrect = incorrect
    n_abstained = abstained
    accuracy = n_correct / total if total else 0

    # Metrics
    metrics = {
        "model": MODEL_NAME,
        "inference_engine": "ollama",
        "ollama_version": "0.32.5",
        "quantization": "q4_K_M (Ollama default)",
        "total_questions": total,
        "correct": n_correct,
        "incorrect": n_incorrect,
        "abstained": n_abstained,
        "accuracy": round(accuracy, 4),
        "abstention_rate": round(n_abstained / total, 4) if total else 0,
        "false_abstention": sum(1 for r in results
                                 if not r.abstained and "expected_abstention=True" in r.notes),
        "mean_response_time_ms": round(
            statistics.mean(r.response_time_ms for r in results if r.response_time_ms > 0) or 0, 1
        ),
    }

    # Per-category breakdown
    categories = {}
    for cat in set(r.category for r in results):
        cat_results = [r for r in results if r.category == cat]
        cat_correct = sum(1 for r in cat_results if r.verdict == "correct")
        categories[cat] = {
            "total": len(cat_results),
            "correct": cat_correct,
            "accuracy": round(cat_correct / len(cat_results), 4) if cat_results else 0
        }
    metrics["per_category"] = categories

    report = {
        "metrics": metrics,
        "results": [r.to_dict() for r in results]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("BASELINE EVAL SUMMARY")
    print(f"  Model:          {MODEL_NAME}")
    print(f"  Questions:      {total}")
    print(f"  Accuracy:       {accuracy:.1%}")
    print(f"  Correct:        {n_correct}")
    print(f"  Incorrect:      {n_incorrect}")
    print(f"  Abstained:     {n_abstained}")
    print(f"  Mean resp time: {metrics['mean_response_time_ms']}ms")
    print(f"  Report:         {output_path}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot Qwen2.5 baseline eval")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Max records to evaluate (default: all)")
    parser.add_argument("--output", type=str,
                        default="experiments/finetune_run/experiments/exp-001/baseline_eval.json",
                        help="Output JSON path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip API calls, use placeholder answers")
    args = parser.parse_args()

    # Repo root: baseline_eval.py is at experiments/finetune_run/baseline_eval.py
    # Two parent dirs up from this file = repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    qa_path = repo_root / "quantum_corpus" / "eval" / "qa_val.jsonl"
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = run_baseline_eval(
        qa_path=qa_path,
        output_path=output_path,
        max_records=args.max_records,
        dry_run=args.dry_run
    )
