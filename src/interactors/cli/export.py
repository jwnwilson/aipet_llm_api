"""CLI: export a HuggingFace checkpoint to a quantised GGUF for RPi deployment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from domain.train.export import export


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export HF checkpoint to quantised GGUF.")
    parser.add_argument("--checkpoint", default="models/checkpoints")
    parser.add_argument("--output", default="models/aipet.gguf")
    parser.add_argument("--quantize", default="Q4_K_M")
    parser.add_argument("--llama-cpp-dir", default="llama.cpp", dest="llama_cpp_dir")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
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
