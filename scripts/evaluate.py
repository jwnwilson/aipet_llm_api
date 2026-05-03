"""Evaluate schema-valid response rate of a trained aipet model.

Usage (HF checkpoint):
    uv run python scripts/evaluate.py --checkpoint models/checkpoints --eval-data data/eval.jsonl

Usage (GGUF model):
    uv run python scripts/evaluate.py --model-path models/aipet.gguf --eval-data data/eval.jsonl

Exit code 0 if valid-parse rate >= 95%, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when run as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure.prompt import parse_response  # noqa: E402

PASS_THRESHOLD = 0.95  # 95 %


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _load_hf_model(checkpoint: str) -> Any:
    """Load a HuggingFace causal-LM pipeline from *checkpoint*."""
    try:
        from transformers import pipeline  # type: ignore[import]
    except ImportError as exc:
        print(
            "ERROR: 'transformers' is not installed. "
            "Run `uv pip install transformers` or add it to pyproject.toml.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(f"Loading HF checkpoint from {checkpoint} …")
    pipe = pipeline(
        "text-generation",
        model=checkpoint,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=False,
    )
    return pipe


def _infer_hf(pipe: Any, prompt: str) -> str:
    """Run a single inference call through a HF text-generation pipeline."""
    outputs = pipe(prompt, return_full_text=False)
    return outputs[0]["generated_text"]


def _load_llama_cpp_model(model_path: str) -> Any:
    """Load a GGUF model via LlamaCppInferenceAdapter (lazy — model not yet loaded)."""
    from src.infrastructure.inference import LlamaCppInferenceAdapter  # noqa: E402

    return LlamaCppInferenceAdapter(model_path=model_path)


def _infer_llama_cpp(adapter: Any, prompt: str) -> str:
    """Run a raw prompt through llama-cpp and return the raw text response."""
    try:
        import llama_cpp  # type: ignore[import]
    except ImportError as exc:
        print(
            "ERROR: 'llama-cpp-python' is not installed. "
            "Run `uv pip install llama-cpp-python`.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    # Access the underlying llama_cpp.Llama instance via the adapter's helper.
    llm = adapter._get_llm()  # noqa: SLF001
    completion = llm(prompt, max_tokens=128, temperature=0.1, stop=[])
    return completion["choices"][0]["text"]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(eval_data: Path, infer_fn: Any) -> int:
    """Run evaluation and return exit code (0 = pass, 1 = fail)."""
    total = 0
    valid = 0
    invalid = 0
    action_counts: Counter[str] = Counter()

    with eval_data.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                example: dict[str, str] = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                print(
                    f"  [!] line {line_no}: cannot parse JSONL line: {exc}",
                    file=sys.stderr,
                )
                continue

            prompt: str = example.get("prompt", "")
            total += 1

            try:
                raw_response = infer_fn(prompt)
                response = parse_response(raw_response)
                valid += 1
                action_counts[response.action.value] += 1
            except Exception as exc:  # noqa: BLE001
                invalid += 1
                if invalid <= 5:
                    # Print a sample of failures to help diagnose issues.
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
            bar = "#" * min(40, count)
            print(f"  {action:<12} {count:>5}  {bar}")
    else:
        print("  (no valid responses)")

    return 0 if pct >= PASS_THRESHOLD else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate schema-valid response rate of a trained aipet model."
    )
    parser.add_argument(
        "--checkpoint",
        default="models/checkpoints",
        help="Path to HuggingFace checkpoint directory (default: models/checkpoints).",
    )
    parser.add_argument(
        "--eval-data",
        default="data/eval.jsonl",
        help="Path to evaluation JSONL file (default: data/eval.jsonl).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path to GGUF model file. If provided, uses LlamaCppInferenceAdapter "
            "instead of the HF checkpoint."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    eval_data = Path(args.eval_data)
    if not eval_data.exists():
        print(f"ERROR: eval data not found: {eval_data}", file=sys.stderr)
        sys.exit(1)

    if args.model_path is not None:
        # Use GGUF model via llama-cpp.
        print(f"Using GGUF model: {args.model_path}")
        adapter = _load_llama_cpp_model(args.model_path)

        def infer_fn(prompt: str) -> str:
            return _infer_llama_cpp(adapter, prompt)

    else:
        # Use HuggingFace checkpoint.
        pipe = _load_hf_model(args.checkpoint)

        def infer_fn(prompt: str) -> str:  # type: ignore[misc]
            return _infer_hf(pipe, prompt)

    exit_code = evaluate(eval_data, infer_fn)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
