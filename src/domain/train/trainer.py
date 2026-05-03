"""Fine-tuning logic — pure training functions, no CLI concerns."""

from __future__ import annotations

import json
import os

try:
    import torch
    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    _TORCH_AVAILABLE = False

try:
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
    from datasets import Dataset
    _DATASETS_AVAILABLE = True
except ModuleNotFoundError:
    _DATASETS_AVAILABLE = False

MAX_LENGTH = 512
DEFAULT_MODEL = "HuggingFaceTB/SmolLM-360M"
DEFAULT_TRAIN_DATA = "data/train.jsonl"
DEFAULT_EVAL_DATA = "data/eval.jsonl"
DEFAULT_OUTPUT_DIR = "models/checkpoints"
DEFAULT_EPOCHS = 3


def load_jsonl(path: str) -> list[dict]:
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
    """Tokenise prompt+completion pairs with prompt tokens masked in labels."""
    if not _DATASETS_AVAILABLE:
        raise ImportError("Install with: uv sync --extra train")

    all_input_ids: list[list[int]] = []
    all_attention_masks: list[list[int]] = []
    all_labels: list[list[int]] = []

    for record in records:
        prompt: str = record["prompt"]
        completion: str = record["completion"]
        full_text: str = prompt + completion

        full_enc = tokenizer(full_text, max_length=max_length, truncation=True, padding=False, return_tensors=None)
        input_ids: list[int] = full_enc["input_ids"]
        attention_mask: list[int] = full_enc["attention_mask"]

        prompt_enc = tokenizer(prompt, max_length=max_length, truncation=True, padding=False, add_special_tokens=False, return_tensors=None)
        prompt_len: int = len(prompt_enc["input_ids"])

        labels: list[int] = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[: len(input_ids)]
        if len(labels) < len(input_ids):
            labels = labels + [-100] * (len(input_ids) - len(labels))

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    dataset = Dataset.from_dict({"input_ids": all_input_ids, "attention_mask": all_attention_masks, "labels": all_labels})
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def train(
    model: str = DEFAULT_MODEL,
    train_data: str = DEFAULT_TRAIN_DATA,
    eval_data: str = DEFAULT_EVAL_DATA,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    epochs: int = DEFAULT_EPOCHS,
    dry_run: bool = False,
    batch_size: int | None = None,
    no_mps: bool = False,
) -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed. Run: uv sync --extra train")
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers not installed. Run: uv sync --extra train")

    use_cuda = torch.cuda.is_available()
    use_mps = (
        not no_mps
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    if no_mps and use_mps:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    default_batch = 4 if use_cuda else 1
    effective_batch = batch_size if batch_size is not None else default_batch
    grad_accum = max(1, 4 // effective_batch)

    device_label = "CUDA" if use_cuda else ("MPS" if use_mps else "CPU")
    print(f"Device: {device_label}  batch_size={effective_batch}  grad_accum={grad_accum}")

    print(f"Loading tokeniser from: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from: {model}")
    hf_model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)

    print(f"Loading training data from: {train_data}")
    train_records = load_jsonl(train_data)
    print(f"Loading eval data from: {eval_data}")
    eval_records = load_jsonl(eval_data)

    if dry_run:
        train_records = train_records[:8]
        eval_records = eval_records[:4]

    print(f"Tokenising {len(train_records)} training examples …")
    train_dataset = build_hf_dataset(train_records, tokenizer)
    print(f"Tokenising {len(eval_records)} eval examples …")
    eval_dataset = build_hf_dataset(eval_records, tokenizer)

    os.makedirs(output_dir, exist_ok=True)

    if dry_run:
        training_args = TrainingArguments(
            output_dir=output_dir,
            max_steps=1, eval_steps=1, save_steps=1,
            eval_strategy="steps", save_strategy="steps", logging_steps=1,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, report_to="none",
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
        )
    else:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            eval_steps=200, save_steps=200,
            eval_strategy="steps", save_strategy="steps", logging_steps=50,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, report_to="none",
            per_device_train_batch_size=effective_batch,
            per_device_eval_batch_size=effective_batch,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=50, weight_decay=0.01,
            fp16=use_cuda, use_cpu=no_mps and not use_cuda,
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=hf_model,
        padding=True, pad_to_multiple_of=8, label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=hf_model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training …")
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    eval_loss: float | None = None
    for entry in reversed(trainer.state.log_history):
        if "eval_loss" in entry:
            eval_loss = entry["eval_loss"]
            break

    best_checkpoint = trainer.state.best_model_checkpoint or output_dir
    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Final train loss : {train_result.training_loss:.4f}")
    print(f"  Best eval loss   : {eval_loss:.4f}" if eval_loss is not None else "  Best eval loss   : N/A")
    print(f"  Checkpoint path  : {best_checkpoint}")
    print("=" * 60)
