"""CLI: generate synthetic training and eval datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from domain.train.dataset import EVAL_SIZE, SEED, TRAIN_SIZE, generate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic aipet training data.")
    parser.add_argument("--data-dir", default="data", help="Output directory (default: data)")
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--eval-size", type=int, default=EVAL_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    ok = generate(
        data_dir=Path(args.data_dir),
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
