"""Fine-tuning logic — pure training functions, no CLI concerns."""

from __future__ import annotations

import json
import os
from collections import Counter

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
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainerControl,
        TrainerState,
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
DEFAULT_EPOCHS = 5
DEFAULT_PATIENCE = 3
DEFAULT_WARMUP_RATIO = 0.05


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


def compute_sample_weights(records: list[dict]) -> list[float]:
    """Compute inverse-frequency sample weights so each action class is sampled equally."""
    actions: list[str] = []
    for r in records:
        try:
            completion = json.loads(r["completion"])
            actions.append(completion.get("action", "IDLE"))
        except Exception:
            actions.append("IDLE")

    counts = Counter(actions)
    n = len(actions)
    n_classes = len(counts)
    return [n / (n_classes * counts[a]) for a in actions]


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


class _ActionQualityCallback(TrainerCallback):
    """Logs per-action accuracy on a small eval sample at each eval step."""

    def __init__(self, eval_records: list[dict], tokenizer, n_sample: int = 20):
        self._records = eval_records[:n_sample]
        self._tokenizer = tokenizer

    def on_evaluate(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        model=None,
        **kwargs,
    ) -> None:
        if model is None or not self._records:
            return

        from infrastructure.prompt import parse_response

        model.eval()
        action_counts: Counter[str] = Counter()
        correct = 0
        total = 0

        for record in self._records:
            prompt = record["prompt"]
            try:
                expected_action = json.loads(record["completion"]).get("action", "")
            except Exception:
                continue

            inputs = self._tokenizer(prompt, return_tensors="pt")
            if _TORCH_AVAILABLE:
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

            try:
                resp = parse_response(raw)
                predicted = resp.action.value
                action_counts[predicted] += 1
                if predicted == expected_action:
                    correct += 1
            except Exception:
                action_counts["[INVALID]"] += 1
            total += 1

        if total > 0:
            pct = correct / total
            print(f"\n[Quality] step={state.global_step}  action_acc={correct}/{total} ({pct:.1%})")
            for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
                print(f"  {action}: {count}")


class _WeightedTrainer(Trainer):
    """Trainer subclass that supports inverse-frequency weighted random sampling."""

    def __init__(self, *args, sample_weights: list[float] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_weights = sample_weights

    def get_train_dataloader(self):
        if self._sample_weights is None:
            return super().get_train_dataloader()

        from torch.utils.data import DataLoader, WeightedRandomSampler

        weights_tensor = torch.tensor(self._sample_weights, dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=len(self.train_dataset),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=0,
        )


def train(
    model: str = DEFAULT_MODEL,
    train_data: str = DEFAULT_TRAIN_DATA,
    eval_data: str = DEFAULT_EVAL_DATA,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
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
    # On MPS, set the fallback env var so ops unsupported by Metal fall back to CPU.
    if use_mps:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    default_batch = 4 if use_cuda else 1
    effective_batch = batch_size if batch_size is not None else default_batch
    grad_accum = max(1, 4 // effective_batch)

    device_label = "CUDA" if use_cuda else ("MPS" if use_mps else "CPU")
    print(f"Device: {device_label}  batch_size={effective_batch}  grad_accum={grad_accum}")

    print(f"Loading tokeniser from: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in float16 on MPS to halve peak VRAM; CPU and CUDA use their normal defaults.
    load_dtype = torch.float16 if use_mps else None
    print(f"Loading model from: {model}" + (f"  dtype=float16" if use_mps else ""))
    hf_model = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=True, torch_dtype=load_dtype
    )
    # Gradient checkpointing trades compute for memory — on by default for non-CUDA
    # devices where RAM is the binding constraint.
    if not use_cuda:
        hf_model.gradient_checkpointing_enable()

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

    sample_weights = compute_sample_weights(train_records)

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
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            fp16=use_cuda,
            use_cpu=no_mps and not use_cuda and not use_mps,
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=hf_model,
        padding=True, pad_to_multiple_of=8, label_pad_token_id=-100,
    )

    callbacks = [_ActionQualityCallback(eval_records, tokenizer, n_sample=20)]
    if not dry_run:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = _WeightedTrainer(
        model=hf_model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        sample_weights=sample_weights,
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
