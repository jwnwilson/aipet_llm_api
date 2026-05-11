"""Fine-tuning logic — pure training functions, no CLI concerns."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path

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
    # Stubs so class definitions below parse without transformers installed.
    TrainerCallback = object  # type: ignore[assignment,misc]
    Trainer = object  # type: ignore[assignment,misc]

try:
    from datasets import Dataset
    _DATASETS_AVAILABLE = True
except ModuleNotFoundError:
    _DATASETS_AVAILABLE = False

try:
    from peft import LoraConfig, TaskType, get_peft_model
    _PEFT_AVAILABLE = True
except ModuleNotFoundError:
    _PEFT_AVAILABLE = False

MAX_LENGTH = 512
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"
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
        raise ImportError("Install with: uv sync")

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
        # +1 accounts for the BOS token prepended by the full tokenization but absent
        # in the prompt-only pass (add_special_tokens=False), so the label boundary
        # correctly lands at the first completion token rather than the last prompt token.
        prompt_len: int = len(prompt_enc["input_ids"]) + 1

        labels: list[int] = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[: len(input_ids)]
        if len(labels) < len(input_ids):
            labels = labels + [-100] * (len(input_ids) - len(labels))

        if all(l == -100 for l in labels):
            continue  # skip: fully-masked labels would produce NaN loss

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    # No set_format — keep as Python lists so DataCollatorForSeq2Seq can pad and
    # convert to tensors in one step. Converting to torch first then back through
    # numpy for padding produces the "list of numpy.ndarrays is extremely slow" warning.
    return Dataset.from_dict({"input_ids": all_input_ids, "attention_mask": all_attention_masks, "labels": all_labels})


class _ProgressCallback(TrainerCallback):
    """Writes a JSON sidecar after each log step so remote pollers can read training progress."""

    def __init__(self, progress_path: "Path") -> None:
        import time as _time
        self._path = progress_path
        self._start = _time.time()

    def on_log(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        import time as _time
        if not logs:
            return
        entry: dict = {
            "step": state.global_step,
            "max_steps": state.max_steps,
            "epoch": round(state.epoch or 0.0, 2),
            "elapsed_s": int(_time.time() - self._start),
        }
        for k, v in logs.items():
            if isinstance(v, float):
                entry[k] = round(v, 4)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(entry))
        except OSError:
            pass


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

        from adapters.prompt import parse_response

        model.eval()
        action_counts: Counter[str] = Counter()
        correct = 0
        total = 0

        prompts: list[str] = []
        expected_actions: list[str] = []
        for record in self._records:
            try:
                expected_actions.append(json.loads(record["completion"]).get("action", ""))
                prompts.append(record["prompt"])
            except Exception:
                continue

        if not prompts:
            return

        # Batch all prompts in one generate call — left-pad so sequences are
        # right-aligned and the model attends to the correct tokens.
        orig_padding_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"
        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        self._tokenizer.padding_side = orig_padding_side

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

        prompt_len = inputs["input_ids"].shape[1]
        for i, expected_action in enumerate(expected_actions):
            new_tokens = generated[i][prompt_len:]
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
        # MPS and CPU cannot share tensors across worker processes;
        # num_workers > 0 only makes sense on CUDA.
        num_workers = 2 if torch.cuda.is_available() else 0
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=num_workers,
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
    progress_path: str | None = None,
) -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed. Run: uv sync")
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers not installed. Run: uv sync")

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

    # Match load dtype to the training precision so the fp16/bf16 gradient scaler
    # never sees a tensor in the wrong dtype (e.g. bfloat16 model + fp16 scaler → crash).
    # T4 and older (SM<8.0) lack hardware bf16 support; use fp16 there instead.
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16
    if use_bf16:
        load_dtype = torch.bfloat16
    elif use_fp16:
        load_dtype = torch.float16
    else:
        load_dtype = None  # CPU / MPS: keep model's native dtype

    if use_cuda:
        # Must be set before any CUDA allocation — including model loading.
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # Estimate parameter count from config before loading so we can choose the
    # right loading strategy without wasting GPU memory on the wrong path.
    # Formula: embedding table + transformer layers (rough but reliable enough).
    _use_qlora = False
    if use_cuda and _PEFT_AVAILABLE:
        try:
            from transformers import AutoConfig
            _cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
            _h = getattr(_cfg, "hidden_size", 0)
            _L = getattr(_cfg, "num_hidden_layers", 0)
            _V = getattr(_cfg, "vocab_size", 0)
            _use_qlora = (_V * _h + _L * 12 * _h * _h) > 1_000_000_000
        except Exception:
            pass

    if not _PEFT_AVAILABLE:
        print("WARNING: peft not installed — full fine-tune only (install with: uv sync)")

    _lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    if _use_qlora:
        # QLoRA: load base in 4-bit (0.85 GB for 1.7B vs 3.4 GB in fp16).
        # prepare_model_for_kbit_training handles gradient checkpointing and
        # input_require_grads in the correct order for quantised models.
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        _bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=load_dtype or torch.float16,
        )
        print(f"Loading model from: {model}  dtype=4-bit NF4 (QLoRA)")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model, quantization_config=_bnb, device_map={"": 0},
            trust_remote_code=True, attn_implementation="eager",
        )
        hf_model = prepare_model_for_kbit_training(
            hf_model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        hf_model = get_peft_model(hf_model, _lora_cfg)
        use_lora = True
    else:
        dtype_label = "bfloat16" if use_bf16 else ("float16" if use_fp16 else "default")
        print(f"Loading model from: {model}  dtype={dtype_label}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True, dtype=load_dtype,
            attn_implementation="eager",
        )
        hf_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        use_lora = _PEFT_AVAILABLE
        if use_lora:
            print("Applying LoRA adapters …")
            hf_model = get_peft_model(hf_model, _lora_cfg)
            hf_model.enable_input_require_grads()

    trainable = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in hf_model.parameters())
    print(f"Trainable params: {trainable/1e6:.1f}M / {total/1e9:.2f}B "
          f"({'QLoRA' if _use_qlora else 'LoRA' if use_lora else 'full'})")

    # Auto-reduce batch for large models when user hasn't overridden it.
    # QLoRA keeps the base in 4-bit (~0.85 GB for 1.7B), so batch=4 fits on a T4.
    if batch_size is None and use_cuda and total > 1_000_000_000:
        effective_batch = 4 if _use_qlora else 1
        grad_accum = 2 if _use_qlora else 8
        print(f"Large model — batch={effective_batch}, grad_accum={grad_accum}")

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
            save_total_limit=2,
            per_device_train_batch_size=effective_batch,
            per_device_eval_batch_size=8 if use_cuda else effective_batch,
            gradient_accumulation_steps=grad_accum,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            fp16=use_fp16,
            bf16=use_bf16,
            optim="adamw_torch",
            use_cpu=no_mps and not use_cuda and not use_mps,
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=hf_model,
        padding=True, pad_to_multiple_of=8, label_pad_token_id=-100,
    )

    callbacks = [_ActionQualityCallback(eval_records, tokenizer, n_sample=20)]
    if progress_path:
        callbacks.append(_ProgressCallback(Path(progress_path)))
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

    if use_lora:
        # Merge adapter weights into the base model so the saved checkpoint is a
        # standard HF model — no PEFT dependency needed for inference or export.
        print("Merging LoRA adapters into base model …")
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(output_dir)
        # Remove intermediate checkpoint-XXXX dirs — they held adapter-only weights
        # that are now superseded by the merged model, freeing ~N×28MB.
        for _d in Path(output_dir).iterdir():
            if _d.is_dir() and _d.name.startswith("checkpoint-"):
                shutil.rmtree(_d)
                print(f"  Removed {_d.name}")
    else:
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
