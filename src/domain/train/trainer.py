"""Fine-tuning logic — pure training functions, no CLI concerns."""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)

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

from domain.train.config import (
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_DATA,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PATIENCE,
    DEFAULT_TRAIN_DATA,
    DEFAULT_WARMUP_RATIO,
    MAX_LENGTH,
)


def _strip_bnb_config(config_file: Path) -> None:
    if not config_file.exists():
        return
    cfg = json.loads(config_file.read_text())
    qc = cfg.get("quantization_config") or {}
    if qc.get("quant_method") == "bitsandbytes" or qc.get("quant_type") in ("nf4", "fp4"):
        cfg.pop("quantization_config")
        config_file.write_text(json.dumps(cfg, indent=2))


def _dequantize_linear4bit(model: "AutoModelForCausalLM") -> None:
    """Replace any remaining Linear4bit modules with plain nn.Linear in float16.

    Called after merge_and_unload() as a safety net for PEFT versions that do
    not fully dequantize all layers, which would otherwise write .absmax tensors
    to disk and break llama.cpp conversion.
    """
    try:
        import bitsandbytes.nn as bnb_nn
    except ImportError:
        return

    replaced = 0
    skipped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, bnb_nn.Linear4bit):
            continue
        parent_path, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        with torch.no_grad():
            w = module.weight.dequantize()
        # After merge_and_unload(), quant_state may be None — dequantize() then
        # returns the raw packed uint8 bytes recast as the storage dtype rather than
        # the full float16 weight.  Detect this by checking element count: packed NF4
        # stores 2 weights per byte so the tensor is half the expected size.
        expected_numel = module.out_features * module.in_features
        if w.numel() != expected_numel:
            skipped += 1
            continue
        w = w.to(torch.float16).reshape(module.out_features, module.in_features)
        linear = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None, dtype=torch.float16)
        linear.weight = torch.nn.Parameter(w)
        if module.bias is not None:
            linear.bias = torch.nn.Parameter(module.bias.to(torch.float16))
        setattr(parent, child_name, linear)
        replaced += 1

    if replaced:
        log.info("Explicitly dequantized %d remaining Linear4bit layer(s) to float16.", replaced)
    if skipped:
        raise RuntimeError(
            f"  {skipped} Linear4bit layer(s) could not be dequantized: quant_state was "
            "lost during merge_and_unload(). This indicates a bitsandbytes/PEFT version "
            "incompatibility with QLoRA merge. Retrain without QLoRA (raise _use_qlora "
            "threshold) or use a compatible bitsandbytes version."
        )


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
            dist = "  ".join(f"{a}:{c}" for a, c in sorted(action_counts.items(), key=lambda x: -x[1]))
            log.info("[Quality] step=%d  action_acc=%d/%d (%.1f%%)  %s", state.global_step, correct, total, pct * 100, dist)


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
    force_qlora: bool | None = None,
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
    log.info("Device: %s  batch_size=%d  grad_accum=%d", device_label, effective_batch, grad_accum)

    log.info("Loading tokeniser from: %s", model)
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
    if force_qlora is True:
        _use_qlora = use_cuda and _PEFT_AVAILABLE
    elif force_qlora is None and use_cuda and _PEFT_AVAILABLE:
        try:
            from transformers import AutoConfig
            _cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
            _h = getattr(_cfg, "hidden_size", 0)
            _L = getattr(_cfg, "num_hidden_layers", 0)
            _V = getattr(_cfg, "vocab_size", 0)
            _use_qlora = (_V * _h + _L * 12 * _h * _h) > 3_000_000_000
        except Exception:
            pass
    # force_qlora is False → _use_qlora stays False (standard LoRA on any device)

    if not _PEFT_AVAILABLE:
        log.warning("peft not installed — full fine-tune only (install with: uv sync)")

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
        log.info("Loading model from: %s  dtype=4-bit NF4 (QLoRA)", model)
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
        log.info("Loading model from: %s  dtype=%s", model, dtype_label)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True, dtype=load_dtype,
            attn_implementation="eager",
        )
        hf_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        use_lora = _PEFT_AVAILABLE
        if use_lora:
            log.info("Applying LoRA adapters …")
            hf_model = get_peft_model(hf_model, _lora_cfg)
            hf_model.enable_input_require_grads()

    trainable = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in hf_model.parameters())
    log.info("Trainable params: %.1fM / %.2fB (%s)", trainable / 1e6, total / 1e9,
             "QLoRA" if _use_qlora else "LoRA" if use_lora else "full")

    # Auto-reduce batch for large models when user hasn't overridden it.
    # QLoRA keeps the base in 4-bit (~0.85 GB for 1.7B), so batch=4 fits on a T4.
    if batch_size is None and use_cuda and total > 1_000_000_000:
        effective_batch = 4 if _use_qlora else 1
        grad_accum = 2 if _use_qlora else 8
        log.info("Large model — batch=%d  grad_accum=%d", effective_batch, grad_accum)

    log.info("Loading training data from: %s", train_data)
    train_records = load_jsonl(train_data)
    log.info("Loading eval data from: %s", eval_data)
    eval_records = load_jsonl(eval_data)

    if dry_run:
        train_records = train_records[:8]
        eval_records = eval_records[:4]

    log.info("Tokenising %d training examples …", len(train_records))
    train_dataset = build_hf_dataset(train_records, tokenizer)
    log.info("Tokenising %d eval examples …", len(eval_records))
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

    effective_progress_path = progress_path or str(Path(output_dir) / "progress.json")
    callbacks = [
        _ActionQualityCallback(eval_records, tokenizer, n_sample=20),
        _ProgressCallback(Path(effective_progress_path)),
    ]
    if not dry_run:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = _WeightedTrainer(
        model=hf_model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        sample_weights=sample_weights,
    )

    log.info("Starting training …")
    train_result = trainer.train()

    if use_lora:
        if _use_qlora:
            # QLoRA: merge_and_unload() on a 4-bit model loses quant_state, so
            # dequantize()-in-place produces corrupt weights.  Instead:
            # 1. Save only the tiny LoRA adapter from the trained model.
            # 2. Reload the base in float16 (no quantization).
            # 3. Merge the adapter onto the clean float16 base and save.
            log.info("QLoRA — saving adapter then re-merging onto float16 base …")
            adapter_dir = Path(output_dir) / "_adapter"
            trainer.model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))

            log.info("Reloading base model %s in float16 for clean merge …", model)
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
            )
            peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
            log.info("Merging LoRA adapters into float16 base …")
            merged = peft_model.merge_and_unload()
        else:
            log.info("Merging LoRA adapters into base model …")
            merged = trainer.model.merge_and_unload()

        merged.save_pretrained(output_dir)
        _strip_bnb_config(Path(output_dir) / "config.json")
        for _d in Path(output_dir).iterdir():
            if _d.is_dir() and _d.name.startswith("checkpoint-"):
                shutil.rmtree(_d)
                log.info("Removed checkpoint dir: %s", _d.name)
        if _use_qlora:
            shutil.rmtree(adapter_dir, ignore_errors=True)
    else:
        trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    eval_loss: float | None = None
    for entry in reversed(trainer.state.log_history):
        if "eval_loss" in entry:
            eval_loss = entry["eval_loss"]
            break

    best_checkpoint = trainer.state.best_model_checkpoint or output_dir
    eval_loss_str = f"{eval_loss:.4f}" if eval_loss is not None else "N/A"
    log.info(
        "Training complete  train_loss=%.4f  eval_loss=%s  checkpoint=%s",
        train_result.training_loss, eval_loss_str, best_checkpoint,
    )
