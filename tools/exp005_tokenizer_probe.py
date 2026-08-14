"""
exp-005 tokenizer probe - measures unk-rate, tokens/word, and train time for
candidate BPE vocabulary sizes on a set of read-only text samples.

Non-destructive: only reads input files and writes results under
experiments/exp-005/tokenizer_probe/. Never touches experiments/exp-004/ or
quantum_corpus/.

Usage:
    python tools/exp005_tokenizer_probe.py \\
        --sample general_english.txt --sample quantum_domain.txt --sample code_or_tools.txt \\
        --vocab-sizes 8000 16000 24000 32000 \\
        --output-dir experiments/exp-005/tokenizer_probe

Requires: pip install tokenizers
"""

import argparse
import json
import time
from pathlib import Path

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
except ImportError as e:
    raise SystemExit(
        "Missing dependency: pip install tokenizers\n"
        f"Original error: {e}"
    )


def load_samples(sample_paths):
    texts = []
    for p in sample_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Sample file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def load_jsonl_text(path, text_field="text", limit=None):
    """Read text rows from a JSONL corpus file (read-only reference use)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            rows.append(row.get(text_field, ""))
    return "\n".join(rows)


def train_and_eval(vocab_size, train_texts, eval_text):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    )

    t0 = time.time()
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    train_seconds = time.time() - t0

    words = eval_text.split()
    if not words:
        raise ValueError("Eval text is empty after whitespace split.")

    encoding = tokenizer.encode(eval_text)
    tokens = encoding.tokens
    unk_id = tokenizer.token_to_id("[UNK]")
    unk_count = sum(1 for tid in encoding.ids if tid == unk_id)

    return {
        "vocab_size": vocab_size,
        "train_seconds": round(train_seconds, 3),
        "num_words_eval": len(words),
        "num_tokens_eval": len(tokens),
        "tokens_per_word": round(len(tokens) / len(words), 4),
        "unk_count": unk_count,
        "unk_rate": round(unk_count / len(tokens), 6) if tokens else None,
    }


def main():
    parser = argparse.ArgumentParser(description="exp-005 tokenizer vocab-size probe")
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
        print(f"Training BPE tokenizer at vocab_size={vocab_size} ...")
        metrics = train_and_eval(vocab_size, [train_text], eval_text)
        results.append(metrics)
        unk_rate = metrics["unk_rate"]
        tpw = metrics["tokens_per_word"]
        secs = metrics["train_seconds"]
        print(f"  unk_rate={unk_rate}  tokens_per_word={tpw}  train_seconds={secs}")

    report = {
        "sample_files": args.sample,
        "vocab_sizes_tested": args.vocab_sizes,
        "eval_holdout_fraction": args.eval_holdout_fraction,
        "eval_text_chars": len(eval_text),
        "results": results,
    }

    report_path = output_dir / "probe_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nWrote probe results to {report_path}")


if __name__ == "__main__":
    main()
