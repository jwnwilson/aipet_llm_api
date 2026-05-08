"""Temporal activities — one per pipeline stage, each wrapping a domain function."""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

from temporalio import activity
from temporalio.exceptions import ApplicationError

from domain.train.dataset import EVAL_SIZE, SEED, TRAIN_SIZE
from domain.train.trainer import DEFAULT_EPOCHS, DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_PATIENCE


# ---------------------------------------------------------------------------
# Config / result dataclasses (Temporal serialises these as JSON)
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    data_dir: str = "data"
    train_size: int = TRAIN_SIZE
    eval_size: int = EVAL_SIZE
    seed: int = SEED


@dataclass
class DatasetPaths:
    train: str = ""
    eval: str = ""


@dataclass
class TrainConfig:
    model: str = DEFAULT_MODEL
    train_data: str = "data/train.jsonl"
    eval_data: str = "data/eval.jsonl"
    output_dir: str = DEFAULT_OUTPUT_DIR
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE


@dataclass
class CheckpointPath:
    path: str = ""


@dataclass
class EvalConfig:
    checkpoint: str = ""
    eval_data: str = "data/eval.jsonl"


@dataclass
class EvalResult:
    valid_pct: float = 0.0
    passed: bool = False


@dataclass
class GGUFPath:
    path: str = ""


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------


@activity.defn
async def generate_dataset_activity(config: DatasetConfig) -> DatasetPaths:
    from domain.train.dataset import generate

    try:
        ok = generate(
            data_dir=Path(config.data_dir),
            train_size=config.train_size,
            eval_size=config.eval_size,
            seed=config.seed,
        )
    except Exception as exc:
        raise ApplicationError(f"generate_dataset failed: {exc}") from exc

    if not ok:
        raise ApplicationError("Dataset generation failed: invalid examples or distribution out of bounds")

    return DatasetPaths(
        train=str(Path(config.data_dir) / "train.jsonl"),
        eval=str(Path(config.data_dir) / "eval.jsonl"),
    )


@activity.defn
async def train_activity(config: TrainConfig) -> CheckpointPath:
    from domain.train.trainer import train

    try:
        train(
            model=config.model,
            train_data=config.train_data,
            eval_data=config.eval_data,
            output_dir=config.output_dir,
            epochs=config.epochs,
            patience=config.patience,
        )
    except Exception as exc:
        raise ApplicationError(f"train failed: {exc}") from exc

    return CheckpointPath(path=config.output_dir)


@activity.defn
async def evaluate_activity(config: EvalConfig) -> EvalResult:
    from domain.train.evaluate import PASS_THRESHOLD, evaluate, infer_hf, load_hf_pipeline

    try:
        pipe = load_hf_pipeline(config.checkpoint)
        infer_fn = lambda prompt: infer_hf(pipe, prompt)  # noqa: E731

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = evaluate(Path(config.eval_data), infer_fn)

        output = buf.getvalue()
        valid_pct = _parse_valid_pct(output)
        passed = exit_code == 0
        if valid_pct is None:
            valid_pct = 1.0 if passed else 0.0
    except Exception as exc:
        raise ApplicationError(f"evaluate failed: {exc}") from exc

    return EvalResult(valid_pct=valid_pct, passed=passed)


def _parse_valid_pct(output: str) -> float | None:
    """Extract valid fraction from a line like 'Valid: 190/200 (95.0%)  [PASS]'."""
    for line in output.splitlines():
        if line.startswith("Valid:") and "(" in line and "%)" in line:
            try:
                pct_str = line.split("(")[1].split("%")[0].strip()
                return float(pct_str) / 100.0
            except (IndexError, ValueError):
                pass
    return None


@activity.defn
async def export_activity(checkpoint: CheckpointPath) -> GGUFPath:
    from domain.train.export import export as export_gguf

    output_path = Path("models/aipet.gguf")
    try:
        export_gguf(checkpoint=Path(checkpoint.path), output=output_path)
    except SystemExit as exc:
        raise ApplicationError(f"export failed: llama.cpp setup issue (exit {exc.code})") from exc
    except Exception as exc:
        raise ApplicationError(f"export failed: {exc}") from exc

    return GGUFPath(path=str(output_path))
