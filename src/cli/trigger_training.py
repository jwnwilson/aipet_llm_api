"""CLI: trigger a TrainingPipelineWorkflow execution on Temporal."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid


async def _trigger(
    experiment_name: str,
    epochs: int,
    patience: int,
    warmup_ratio: float,
    skip_generate: bool,
    dry_run: bool,
    remote_backend: str,
    model: str,
    train_size: int,
    eval_size: int,
) -> None:
    from pathlib import Path

    from temporalio.client import Client

    from temporal.worker import TASK_QUEUE
    from temporal.workflows import ExperimentConfig, TrainingPipelineWorkflow

    temporal_host = os.environ.get("TEMPORAL_HOST", "localhost:7233")
    client = await Client.connect(temporal_host)

    run_id = str(uuid.uuid4())
    run_data_dir = f"data/workflow/{run_id}"
    Path(run_data_dir).mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        experiment_name=experiment_name,
        run_id=run_id,
        epochs=epochs,
        patience=patience,
        warmup_ratio=warmup_ratio,
        skip_generate=skip_generate,
        dry_run=dry_run,
        remote_backend=remote_backend,
        model=model,
        train_size=train_size,
        eval_size=eval_size,
        data_dir=run_data_dir,
        output_dir=f"{run_data_dir}/checkpoint",
        gguf_output=f"{run_data_dir}/model.gguf",
    )

    workflow_id = f"training-{experiment_name}-{uuid.uuid4().hex[:8]}"
    handle = await client.start_workflow(
        TrainingPipelineWorkflow.run,
        config,
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    print(f"Workflow started")
    print(f"  ID         : {handle.id}")
    print(f"  Run ID     : {run_id}")
    print(f"  Backend    : {remote_backend or 'local'}")
    print(f"  Model      : {model}")
    print(f"  Data dir   : {run_data_dir}")
    print(f"  UI         : http://localhost:8233/namespaces/default/workflows/{handle.id}")


def main(argv: list[str] | None = None) -> None:
    from domain.train.dataset import EVAL_SIZE, TRAIN_SIZE
    from domain.train.trainer import DEFAULT_EPOCHS, DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_PATIENCE, DEFAULT_WARMUP_RATIO

    parser = argparse.ArgumentParser(
        description="Trigger a TrainingPipelineWorkflow on Temporal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment-name", required=True, dest="experiment_name",
                        help="Name/tag for this experiment (used as part of the workflow ID)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO, dest="warmup_ratio")
    parser.add_argument("--skip-generate", action="store_true", dest="skip_generate",
                        help="Reuse the existing dataset — useful for hyperparameter experiments")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Train for 1 step only (smoke test)")
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE, dest="train_size")
    parser.add_argument("--eval-size", type=int, default=EVAL_SIZE, dest="eval_size")
    parser.add_argument(
        "--remote-backend",
        dest="remote_backend",
        choices=["local", "kaggle", "ssh", "colab"],
        default="local",
        help="Where to run fine-tuning: local machine, Kaggle GPU, SSH remote host, or Google Colab",
    )
    parser.add_argument(
        "--model",
        "--remote-model",
        dest="model",
        default=DEFAULT_MODEL,
        help="Base model to fine-tune (use a larger model with --remote-backend kaggle/ssh)",
    )
    args = parser.parse_args(argv)

    try:
        asyncio.run(_trigger(
            experiment_name=args.experiment_name,
            epochs=args.epochs,
            patience=args.patience,
            warmup_ratio=args.warmup_ratio,
            skip_generate=args.skip_generate,
            dry_run=args.dry_run,
            remote_backend=args.remote_backend if args.remote_backend != "local" else "",
            model=args.model,
            train_size=args.train_size,
            eval_size=args.eval_size,
        ))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
