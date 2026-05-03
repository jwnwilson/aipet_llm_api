"""CLI: evaluate schema-valid response rate of a trained aipet model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from domain.train.evaluate import (
    evaluate,
    infer_hf,
    infer_llama_cpp,
    load_hf_pipeline,
    load_llama_cpp_adapter,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate schema-valid response rate.")
    parser.add_argument("--checkpoint", default="models/checkpoints")
    parser.add_argument("--eval-data", default="data/eval.jsonl", dest="eval_data")
    parser.add_argument("--model-path", default=None, dest="model_path",
                        help="GGUF model path; if set, uses llama-cpp instead of HF checkpoint")
    args = parser.parse_args(argv)

    eval_data = Path(args.eval_data)
    if not eval_data.exists():
        print(f"ERROR: eval data not found: {eval_data}", file=sys.stderr)
        sys.exit(1)

    if args.model_path is not None:
        print(f"Using GGUF model: {args.model_path}")
        adapter = load_llama_cpp_adapter(args.model_path)
        infer_fn = lambda prompt: infer_llama_cpp(adapter, prompt)  # noqa: E731
    else:
        pipe = load_hf_pipeline(args.checkpoint)
        infer_fn = lambda prompt: infer_hf(pipe, prompt)  # noqa: E731

    sys.exit(evaluate(eval_data, infer_fn))


if __name__ == "__main__":
    main()
