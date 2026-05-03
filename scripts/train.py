"""
Fine-tuning script for the aipet-llm project.

Fine-tunes a small causal LM on synthetic pet-brain prompt/completion pairs
using HuggingFace Transformers + Trainer.

Usage:
    uv run python scripts/train.py [options]
    uv run python scripts/train.py --dry-run   # 1-step smoke test (no GPU required)

Requirements (optional[train] + torch):
    uv add --optional train torch
    uv sync --extra train
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Graceful import of heavy dependencies
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401  – imported for side-effects / device detection
    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    _TORCH_AVAILABLE = False

try:
    import transformers  # noqa: F401
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
    _TRANSFORMERS_AVAILABLE = True
except ModuleNotFoundError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from datasets import Dataset  # noqa: F401
    _DATASETS_AVAILABLE = True
except ModuleNotFoundError:
    _DATASETS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_LENGTH = 512
DEFAULT_MODEL = "HuggingFaceTB/SmolLM-360M"
DEFAULT_TRAIN_DATA = "data/train.jsonl"
DEFAULT_EVAL_DATA = "data/eval.jsonl"
DEFAULT_OUTPUT_DIR = "models/checkpoints"
DEFAULT_EPOCHS = 3


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM on aipet pet-brain data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model ID or local path to fine-tune.",
    )
    parser.add_argument(
        "--train-data",
        default=DEFAULT_TRAIN_DATA,
        dest="train_data",
        help="Path to training JSONL file (each line: {\"prompt\": ..., \"completion\": ...}).",
    )
    parser.add_argument(
        "--eval-data",
        default=DEFAULT_EVAL_DATA,
        dest="eval_data",
        help="Path to evaluation JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        dest="output_dir",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Train for exactly 1 step to verify the pipeline (useful without a GPU).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Per-device batch size. Defaults to 1 on CPU/MPS, 4 on CUDA.",
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        dest="no_mps",
        help="Disable MPS (Apple Silicon GPU) and fall back to CPU.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, returning a list of dicts."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {exc}") from exc
    return records


def build_hf_dataset(records: list[dict], tokenizer, max_length: int = MAX_LENGTH) -> "Dataset":
    """
    Tokenise prompt+completion pairs.

    Labels are set equal to input_ids, with the prompt tokens masked to -100
    so that the cross-entropy loss is computed only on the completion tokens.
    """
    if not _DATASETS_AVAILABLE:
        raise ImportError("The 'datasets' package is required. Install with: uv sync --extra train")

    all_input_ids: list[list[int]] = []
    all_attention_masks: list[list[int]] = []
    all_labels: list[list[int]] = []

    for record in records:
        prompt: str = record["prompt"]
        completion: str = record["completion"]
        full_text: str = prompt + completion

        # Tokenise the full sequence
        full_enc = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids: list[int] = full_enc["input_ids"]
        attention_mask: list[int] = full_enc["attention_mask"]

        # Tokenise only the prompt to find where completion tokens start.
        # add_special_tokens=False avoids double BOS when we re-use the prompt portion.
        prompt_enc = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,
            return_tensors=None,
        )
        prompt_len: int = len(prompt_enc["input_ids"])

        # Build labels: mask prompt tokens with -100
        labels: list[int] = [-100] * prompt_len + input_ids[prompt_len:]

        # Ensure labels and input_ids are the same length (can differ by 1 due
        # to BOS being prepended only for full_text tokenisation)
        labels = labels[: len(input_ids)]
        if len(labels) < len(input_ids):
            labels = labels + [-100] * (len(input_ids) - len(labels))

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    dataset = Dataset.from_dict(
        {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
        }
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Main training entry point."""
    if not _TORCH_AVAILABLE:
        print(
            "ERROR: PyTorch is not installed. Install it with:\n"
            "  uv add --optional train torch\n"
            "  uv sync --extra train",
            file=sys.stderr,
        )
        sys.exit(1)

    if not _TRANSFORMERS_AVAILABLE:
        print(
            "ERROR: 'transformers' is not installed. Install it with:\n"
            "  uv sync --extra train",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Device / batch-size resolution
    # ------------------------------------------------------------------
    use_cuda = _TORCH_AVAILABLE and torch.cuda.is_available()
    use_mps = (
        _TORCH_AVAILABLE
        and not args.no_mps
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    if args.no_mps and use_mps:
        import os as _os
        _os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Default batch sizes: 4 on CUDA, 1 on CPU/MPS to avoid OOM
    default_batch = 4 if use_cuda else 1
    batch_size = args.batch_size if args.batch_size is not None else default_batch
    # Use gradient accumulation so effective batch size stays ~4
    grad_accum = max(1, 4 // batch_size)

    device_label = "CUDA" if use_cuda else ("MPS" if use_mps else "CPU")
    print(f"Device: {device_label}  batch_size={batch_size}  grad_accum={grad_accum}")

    # ------------------------------------------------------------------
    # Load tokeniser + model
    # ------------------------------------------------------------------
    print(f"Loading tokeniser from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Many causal LMs have no pad token; use EOS as a stand-in.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    # ------------------------------------------------------------------
    # Load + tokenise datasets
    # ------------------------------------------------------------------
    print(f"Loading training data from: {args.train_data}")
    train_records = load_jsonl(args.train_data)

    print(f"Loading eval data from: {args.eval_data}")
    eval_records = load_jsonl(args.eval_data)

    if args.dry_run:
        # Keep just 8 examples so tokenisation is fast
        train_records = train_records[:8]
        eval_records = eval_records[:4]

    print(f"Tokenising {len(train_records)} training examples …")
    train_dataset = build_hf_dataset(train_records, tokenizer)

    print(f"Tokenising {len(eval_records)} eval examples …")
    eval_dataset = build_hf_dataset(eval_records, tokenizer)

    # ------------------------------------------------------------------
    # TrainingArguments
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dry_run:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            max_steps=1,
            eval_steps=1,
            save_steps=1,
            eval_strategy="steps",
            save_strategy="steps",
            logging_steps=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            # Keep memory usage minimal during dry-run
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
        )
    else:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            eval_steps=200,
            save_steps=200,
            eval_strategy="steps",
            save_strategy="steps",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=50,
            weight_decay=0.01,
            fp16=use_cuda,
            use_cpu=args.no_mps and not use_cuda,
        )

    # ------------------------------------------------------------------
    # Data collator — pads batches dynamically
    # ------------------------------------------------------------------
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training …")
    train_result = trainer.train()

    # ------------------------------------------------------------------
    # Save model + tokeniser
    # ------------------------------------------------------------------
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    train_loss = train_result.training_loss

    # Retrieve the last recorded eval loss from the trainer log history
    eval_loss: float | None = None
    for entry in reversed(trainer.state.log_history):
        if "eval_loss" in entry:
            eval_loss = entry["eval_loss"]
            break

    best_checkpoint = trainer.state.best_model_checkpoint or args.output_dir

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Final train loss : {train_loss:.4f}")
    if eval_loss is not None:
        print(f"  Best eval loss   : {eval_loss:.4f}")
    else:
        print("  Best eval loss   : N/A")
    print(f"  Checkpoint path  : {best_checkpoint}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
