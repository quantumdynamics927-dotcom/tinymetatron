"""
experiments/finetune_run/experiments/exp-001/data_pipeline/format_sft.py
=======================================================================
Formats quantum_corpus QA pairs into instruction-tuning format for QLoRA fine-tuning.

Input:  quantum_corpus/eval/qa_val.jsonl (or any QA subset)
Output: JSONL of instruction-tuning samples {messages: [{role, content}]}

Compatible with:
  - HuggingFace transformers Trainer (chat template applied at runtime)
  - Unsloth FastLanguageModel.from_template()
  - TRL SFTTrainer

This is a SCHEMA AND INTERFACE stub — blocked: awaiting GPU.
Do not execute training. This defines the data contract only.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Path to the QA source data (absolute, resolved relative to repo root)
QA_VAL_PATH = Path(__file__).parent.parent.parent.parent.parent / "quantum_corpus" / "eval" / "qa_val.jsonl"

# Output path for formatted SFT data
SFT_OUTPUT_PATH = Path(__file__).parent / "sft_data.jsonl"

# System prompt for RAG-grounded QA (matches quantum_corpus/answer.py behavior)
SYSTEM_PROMPT = """You are a quantum research assistant.
Answer questions based ONLY on the provided retrieval context.
If the context does not contain enough information to answer, say so.
Cite specific records when available. Do not fabricate information."""


def format_instruction_sample(qa_record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one QA record to instruction-tuning format.

    Input QA record schema (from quantum_corpus/eval/qa_val.jsonl):
        {
            "id": str,
            "category": str,
            "question": str,
            "gold_record_ids": List[int],
            "gold_source_identities": List[str],
            "answer_requirements": str,   # may be empty
            "expected_abstention": bool,
            "notes": str
        }

    Output SFT sample schema:
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "<question>"},
                {"role": "assistant", "content": "<answer>"}
            ],
            "metadata": {
                "qa_id": str,
                "category": str,
                "expected_abstention": bool,
                "gold_record_ids": List[int]
            }
        }
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": qa_record["question"]},
        # Assistant turn is empty — SFT trains on the target response.
        # The gold answer is injected at training time from metadata.
        {"role": "assistant", "content": ""}
    ]

    return {
        "messages": messages,
        "metadata": {
            "qa_id": qa_record["id"],
            "category": qa_record["category"],
            "expected_abstention": qa_record["expected_abstention"],
            "gold_record_ids": qa_record["gold_record_ids"],
        }
    }


def load_qa_records(path: Path) -> List[Dict[str, Any]]:
    """Load QA records from jsonl."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def filter_supervised_subset(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only answerable records (not expected to abstain).

    Abstention cases are useful for DPO/RM training later, but not for SFT.
    """
    return [r for r in records if not r.get("expected_abstention", False)]


def format_all(
    input_path: Path = QA_VAL_PATH,
    output_path: Path = SFT_OUTPUT_PATH,
    include_abstention: bool = False
) -> int:
    """Format all QA records and write SFT JSONL.

    Returns: number of records written.
    """
    records = load_qa_records(input_path)

    if not include_abstention:
        records = filter_supervised_subset(records)

    with open(output_path, "w", encoding="utf-8") as out:
        for r in records:
            sample = format_instruction_sample(r)
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return len(records)


# ── Schema contract ────────────────────────────────────────────────────────
# Training code (when GPU is available) should:
#   1. Load this JSONL
#   2. Apply chat template via tokenizer.apply_chat_template(messages)
#   3. Tokenize with max_seq_length=2048, truncation=True
#   4. Train with causal LM objective (labels = input_ids)
#
# The assistant turn content ("") is the training target (loss on those tokens only).
# System and user turns are not masked from loss (standard causal LM).
