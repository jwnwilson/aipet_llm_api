"""CLI: fine-tune a causal LM on aipet pet-brain data."""

from __future__ import annotations

import argparse
import sys

from src.domain.train.trainer import (
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_DATA,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRAIN_DATA,
    train,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM on aipet pet-brain data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-data", default=DEFAULT_TRAIN_DATA, dest="train_data")
    parser.add_argument("--eval-data", default=DEFAULT_EVAL_DATA, dest="eval_data")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, dest="output_dir")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--dry-run", action="store_true", default=False, dest="dry_run",
                        help="Train for 1 step only (smoke test, no GPU required)")
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--no-mps", action="store_true", default=False, dest="no_mps")
    args = parser.parse_args(argv)

    try:
        train(
            model=args.model,
            train_data=args.train_data,
            eval_data=args.eval_data,
            output_dir=args.output_dir,
            epochs=args.epochs,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            no_mps=args.no_mps,
        )
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
