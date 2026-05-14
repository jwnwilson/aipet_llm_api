"""CLI: evaluate schema-valid response rate of a trained aipet model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from domain.train.evaluate import (
    evaluate,
    infer_hf,
    infer_llama_cpp,
    load_hf_pipeline,
    load_llama_cpp_adapter,
)


def _make_adapter(backend: str):
    if backend == "kaggle":
        from adapters.compute.kaggle import KaggleTrainingAdapter
        return KaggleTrainingAdapter()
    if backend == "ssh":
        from adapters.compute.ssh import SshTrainingAdapter
        return SshTrainingAdapter()
    if backend == "colab":
        from adapters.compute.colab.adapter import ColabTrainingAdapter
        return ColabTrainingAdapter()
    raise SystemExit(f"Unknown remote backend: {backend!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate schema-valid response rate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-id", default=None, dest="run_id",
        help="Evaluate a specific pipeline run by UUID — derives checkpoint and eval-data paths",
    )
    parser.add_argument("--checkpoint", default="models/checkpoints")
    parser.add_argument("--eval-data", default="data/eval.jsonl", dest="eval_data")
    parser.add_argument(
        "--model-path", default=None, dest="model_path",
        help="GGUF model path; if set, uses llama-cpp instead of HF checkpoint",
    )
    parser.add_argument(
        "--quality", action="store_true", default=False,
        help=(
            "Run the full quality report (per-stat accuracy, target accuracy, "
            "action distribution). Requires --model-path (GGUF)."
        ),
    )
    parser.add_argument(
        "--quality-output", default="data/quality_report.json", dest="quality_output",
        help="Path to write the JSON quality report (only used with --quality)",
    )
    parser.add_argument(
        "--remote-backend",
        default=os.getenv("REMOTE_BACKEND", ""),
        dest="remote_backend",
        choices=["", "kaggle", "ssh", "colab"],
        help="Remote adapter to pull checkpoint from (env: REMOTE_BACKEND)",
    )
    parser.add_argument(
        "--remote-run-id",
        default=os.getenv("REMOTE_RUN_ID", ""),
        dest="remote_run_id",
        help="Opaque run ID for the remote adapter, e.g. Kaggle kernel slug (env: REMOTE_RUN_ID)",
    )
    args = parser.parse_args(argv)

    # Validate remote args: both must be set together.
    if bool(args.remote_run_id) != bool(args.remote_backend):
        print("ERROR: --remote-run-id and --remote-backend must both be set", file=sys.stderr)
        sys.exit(1)

    # Download checkpoint from remote if requested.
    if args.remote_run_id and args.remote_backend:
        print(f"Downloading checkpoint from {args.remote_backend} run: {args.remote_run_id} …")
        adapter = _make_adapter(args.remote_backend)
        dest = Path("models/checkpoints") / args.remote_run_id.replace("/", "_")
        args.checkpoint = adapter.download(args.remote_run_id, dest)
        print(f"Checkpoint at: {args.checkpoint}")

    # Derive paths from DB run UUID (local-only, unchanged behaviour).
    if args.run_id and not args.remote_run_id:
        args.checkpoint = f"data/workflow/{args.run_id}/checkpoint"
        args.eval_data = f"data/workflow/{args.run_id}/eval.jsonl"

    eval_data = Path(args.eval_data)
    if not eval_data.exists():
        print(f"ERROR: eval data not found: {eval_data}", file=sys.stderr)
        sys.exit(1)

    if args.model_path is not None:
        print(f"Using GGUF model: {args.model_path}")
        adapter = load_llama_cpp_adapter(args.model_path)
        infer_fn = lambda prompt: infer_llama_cpp(adapter, prompt)  # noqa: E731
    else:
        if args.quality:
            print("ERROR: --quality requires --model-path (GGUF model)", file=sys.stderr)
            sys.exit(1)
        pipe = load_hf_pipeline(args.checkpoint)
        infer_fn = lambda prompt: infer_hf(pipe, prompt)  # noqa: E731

    # Always run the schema-validity evaluation.
    exit_code, _valid_pct = evaluate(eval_data, infer_fn)

    # Optionally run the full quality report.
    if args.quality:
        from domain.train.quality_report import print_report, run_quality_report

        print("\nRunning quality report …")
        report = run_quality_report(adapter.infer)
        print_report(report)

        output_path = Path(args.quality_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\nQuality report written → {output_path}")

        if not report["pass"]:
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
