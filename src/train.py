"""
Fine-tune SmolLM2-135M with LoRA on the game dataset.
Run:  python -m src.train
"""
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.config import lora_cfg, model_cfg, train_cfg
from src.dataset import build_dataset

DATA_PATH = Path(__file__).parent.parent / "data" / "game_dataset.parquet"


def tokenize(batch, tokenizer):
    full = [p + " " + r for p, r in zip(batch["prompt"], batch["response"])]
    enc = tokenizer(
        full,
        truncation=True,
        max_length=model_cfg.max_input_length + model_cfg.max_output_length,
        padding="max_length",
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc


def main():
    # ── dataset ───────────────────────────────────────────────────────────────
    if DATA_PATH.exists():
        ds = load_dataset("parquet", data_files=str(DATA_PATH), split="train")
    else:
        ds = build_dataset(2000, DATA_PATH)

    split = ds.train_test_split(test_size=0.1, seed=42)

    # ── tokenizer & model ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.base_model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    peft_cfg = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # ── tokenise splits ───────────────────────────────────────────────────────
    tok_fn = lambda b: tokenize(b, tokenizer)
    train_ds = split["train"].map(
        tok_fn, batched=True, remove_columns=split["train"].column_names
    )
    eval_ds = split["test"].map(
        tok_fn, batched=True, remove_columns=split["test"].column_names
    )

    # ── training args ─────────────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_epochs,
        per_device_train_batch_size=train_cfg.batch_size,
        per_device_eval_batch_size=train_cfg.batch_size,
        learning_rate=train_cfg.learning_rate,
        warmup_steps=train_cfg.warmup_steps,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        evaluation_strategy="steps",
        eval_steps=train_cfg.save_steps,
        fp16=train_cfg.fp16,
        report_to="none",
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    trainer.train()
    trainer.save_model(train_cfg.output_dir)
    tokenizer.save_pretrained(train_cfg.output_dir)
    print(f"\nAdapter saved → {train_cfg.output_dir}")


if __name__ == "__main__":
    main()
