"""CLI: export a HuggingFace checkpoint to a quantised GGUF for RPi deployment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from domain.train.export import export


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
        description="Export HF checkpoint to quantised GGUF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="models/checkpoints")
    parser.add_argument("--output", default="models/aipet.gguf")
    parser.add_argument("--quantize", default="Q4_K_M")
    parser.add_argument("--llama-cpp-dir", default="llama.cpp", dest="llama_cpp_dir")
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
        dest = Path(args.output).parent / "checkpoint"
        args.checkpoint = adapter.download(args.remote_run_id, dest)
        print(f"Checkpoint at: {args.checkpoint}")

    checkpoint = Path(args.checkpoint)
    if not args.remote_run_id and not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    export(
        checkpoint=checkpoint,
        output=Path(args.output),
        quantize=args.quantize,
        llama_cpp_dir=Path(args.llama_cpp_dir),
    )


if __name__ == "__main__":
    main()
