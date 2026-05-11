"""Evaluation logic — measures schema-valid response rate against an eval set."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from adapters.prompt import parse_response

PASS_THRESHOLD = 0.95


def load_hf_pipeline(checkpoint: str) -> Any:
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError("Install with: uv sync") from exc

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        model_dtype = torch.float32

    print(f"Loading HF checkpoint from {checkpoint} …")
    return pipeline(
        "text-generation",
        model=checkpoint,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=False,
        device=0 if use_cuda else -1,
        dtype=model_dtype,
    )


def infer_hf(pipe: Any, prompt: str) -> str:
    outputs = pipe(prompt, return_full_text=False)
    return outputs[0]["generated_text"]


def load_llama_cpp_adapter(model_path: str) -> Any:
    from adapters.inference import LlamaCppInferenceAdapter
    return LlamaCppInferenceAdapter(model_path=model_path)


def infer_llama_cpp(adapter: Any, prompt: str) -> str:
    try:
        import llama_cpp
    except ImportError as exc:
        raise ImportError("Install llama-cpp-python.") from exc

    llm = adapter._get_llm()  # noqa: SLF001
    completion = llm(prompt, max_tokens=128, temperature=0.1, stop=[])
    return completion["choices"][0]["text"]


def evaluate(eval_data: Path, infer_fn: Callable[[str], str]) -> int:
    """Run evaluation; returns exit code 0 (pass ≥95%) or 1 (fail)."""
    total = valid = invalid = 0
    action_counts: Counter[str] = Counter()

    with eval_data.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                example: dict[str, str] = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                print(f"  [!] line {line_no}: cannot parse JSONL line: {exc}", file=sys.stderr)
                continue

            total += 1
            try:
                raw_response = infer_fn(example.get("prompt", ""))
                response = parse_response(raw_response)
                valid += 1
                action_counts[response.action.value] += 1
            except Exception as exc:  # noqa: BLE001
                invalid += 1
                if invalid <= 5:
                    print(f"  [!] line {line_no}: invalid response — {exc}", file=sys.stderr)

    if total == 0:
        print("ERROR: no examples found in eval data.", file=sys.stderr)
        return 1

    pct = valid / total
    status = "PASS" if pct >= PASS_THRESHOLD else "FAIL"
    print(f"Valid: {valid}/{total} ({pct:.1%})  [{status}]")
    print("\nAction distribution:")
    if action_counts:
        for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            print(f"  {action:<12} {count:>5}  {'#' * min(40, count)}")
    else:
        print("  (no valid responses)")

    return 0 if pct >= PASS_THRESHOLD else 1
