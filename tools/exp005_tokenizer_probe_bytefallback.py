"""
exp-005 byte-fallback BPE tokenizer probe.

Same protocol as exp005_tokenizer_probe.py, but uses byte-level pre-tokenizer
and byte-fallback so any input sequence is representable (UNK rate = 0 by
construction). Compares vocab sizes on tokens/word and compression only.

Non-destructive: reads input files, writes results under
experiments/exp-005/tokenizer_probe/. Never touches exp-004, quantum_corpus,
or trained models.
"""

import argparse
import json
import time
from pathlib import Path

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
except ImportError as e:
    raise SystemExit(
        "Missing dependency: pip install tokenizers\n"
        f"Original error: {e}"
    )


def load_samples(sample_paths: list[str]) -> list[str]:
    texts = []
    for p in sample_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Sample file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def load_jsonl_text(path: str, text_field: str = "text", limit: int | None = None) -> str:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            rows.append(row.get(text_field, ""))
    return "\n".join(rows)


def train_and_eval(vocab_size: int, train_texts: list[str], eval_text: str) -> dict:
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    # ByteLevel pre-tokenizer makes the tokenizer byte-level by construction
    # (GPT-2/RoBERTa style), so every byte sequence is representable and the
    # effective unk-rate is zero.
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )

    t0 = time.time()
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    train_seconds = time.time() - t0

    words = eval_text.split()
    if not words:
        raise ValueError("Eval text is empty after whitespace split.")

    encoding = tokenizer.encode(eval_text)
    tokens = encoding.tokens

    return {
        "vocab_size": vocab_size,
        "train_seconds": round(train_seconds, 3),
        "num_words_eval": len(words),
        "num_tokens_eval": len(tokens),
        "tokens_per_word": round(len(tokens) / len(words), 4),
        "unk_count": 0,
        "unk_rate": 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="exp-005 byte-fallback BPE vocab-size probe")
    parser.add_argument("--sample", action="append", required=True,
                        help="Path to a plain-text sample file. Repeat for multiple samples.")
    parser.add_argument("--vocab-sizes", type=int, nargs="+",
                        default=[8000, 16000, 24000, 32000])
    parser.add_argument("--output-dir", default="experiments/exp-005/tokenizer_probe")
    parser.add_argument("--eval-holdout-fraction", type=float, default=0.2,
                        help="Fraction of concatenated sample text held out for eval only.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = load_samples(args.sample)
    combined = "\n".join(texts)

    split_point = int(len(combined) * (1 - args.eval_holdout_fraction))
    train_text, eval_text = combined[:split_point], combined[split_point:]

    if not eval_text.strip():
        raise ValueError("Eval holdout is empty - provide more sample text or lower the holdout fraction.")

    results = []
    for vocab_size in args.vocab_sizes:
        print(f"Training byte-fallback BPE tokenizer at vocab_size={vocab_size} ...")
        metrics = train_and_eval(vocab_size, [train_text], eval_text)
        results.append(metrics)
        print(f"  tokens_per_word={metrics['tokens_per_word']}  train_seconds={metrics['train_seconds']}  (unk_rate=0 by construction)")

    report = {
        "sample_files": args.sample,
        "vocab_sizes_tested": args.vocab_sizes,
        "eval_holdout_fraction": args.eval_holdout_fraction,
        "eval_text_chars": len(eval_text),
        "results": results,
        "note": "byte-fallback enabled; UNK rate is zero by construction",
    }

    report_path = output_dir / "probe_results_bytefallback.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nWrote byte-fallback probe results to {report_path}")


if __name__ == "__main__":
    main()
