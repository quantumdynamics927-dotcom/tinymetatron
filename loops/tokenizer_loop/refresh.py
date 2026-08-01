"""
refresh.py
==========
Measure current tokenizer fragmentation and write results to
loops/tokenizer_loop/current_metrics.json.

Run this BEFORE starting a new tokenizer loop iteration to get an accurate
view of current tokenizer quality.

Usage::

    python loops/tokenizer_loop/refresh.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.resolve()
TOKENIZER_DIR = ROOT / "experiments/english_first_tokenizer/tokenizers/tinymetatron_v2_en_8k"
METRICS_OUT = ROOT / "loops/tokenizer_loop/current_metrics.json"

sys.path.insert(0, str(ROOT / "experiments/english_first_tokenizer"))

from train_en_tokenizer import TEST_CASES, evaluate_tokenizer


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    print(f"Loading tokenizer from: {TOKENIZER_DIR}")
    if not TOKENIZER_DIR.exists():
        print(f"ERROR: Tokenizer not found at {TOKENIZER_DIR}")
        return 1

    try:
        from tokenizers import ByteLevelBPETokenizer
    except ImportError:
        print("FATAL: tokenizers library not installed.")
        print("  pip install tokenizers")
        return 1

    # Load the full tokenizer from tokenizer.json to get exact config
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))
    results, summary = evaluate_tokenizer(tokenizer)

    # Enrich summary with artifact hashes
    summary["measured_at"] = datetime.now(timezone.utc).isoformat()
    summary["tokenizer_json_sha256"] = _sha256(TOKENIZER_DIR / "tokenizer.json")
    summary["vocab_sha256"] = _sha256(TOKENIZER_DIR / "vocab.json")
    summary["merges_sha256"] = _sha256(TOKENIZER_DIR / "merges.txt")

    # Write per-case results too for traceability
    summary["per_case_results"] = {
        name: {
            "token_count": r["token_count"],
            "ratio_words": r["ratio_words"],
            "unk_count": r["unk_count"],
            "tag_atomic": r["tag_atomic"],
        }
        for name, r in results.items()
    }

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nWritten: {METRICS_OUT}")

    # Print a compact summary
    print(f"\n{'='*60}")
    print("CURRENT METRICS SUMMARY")
    print(f"{'='*60}")
    print(f"  prose_median   : {summary['prose_median_tokens_per_word']}")
    print(f"  all_avg        : {summary['all_avg_tokens_per_word']}")
    print(f"  all_p95        : {summary['all_p95_tokens_per_word']}")
    print(f"  unknown tokens : {summary['all_unk']}")
    print(f"  <|en|> atomic  : {summary['all_tags_atomic']}")
    print(f"  decision       : {summary['decision']}")

    if summary["prose_median_tokens_per_word"] > 2.5:
        print(f"\n  NOTE: prose_median ({summary['prose_median_tokens_per_word']}) > 2.5")
        print(f"        This is a soft gate threshold. Consider increasing corpus size")
        print(f"        or vocabulary if approaching 4.0.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
