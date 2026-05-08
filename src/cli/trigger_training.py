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
    remote_backend: str,
    model: str,
) -> None:
    from temporalio.client import Client

    from temporal.worker import TASK_QUEUE
    from temporal.workflows import ExperimentConfig, TrainingPipelineWorkflow

    temporal_host = os.environ.get("TEMPORAL_HOST", "localhost:7233")
    client = await Client.connect(temporal_host)

    config = ExperimentConfig(
        experiment_name=experiment_name,
        epochs=epochs,
        patience=patience,
        warmup_ratio=warmup_ratio,
        skip_generate=skip_generate,
        remote_backend=remote_backend,
        model=model,
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
    print(f"  Run ID     : {handle.result_run_id}")
    print(f"  Backend    : {remote_backend or 'local'}")
    print(f"  Model      : {model}")
    print(f"  UI         : http://localhost:8233/namespaces/default/workflows/{handle.id}")


def main(argv: list[str] | None = None) -> None:
    from domain.train.trainer import DEFAULT_EPOCHS, DEFAULT_MODEL, DEFAULT_PATIENCE, DEFAULT_WARMUP_RATIO

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
    parser.add_argument(
        "--remote-backend",
        dest="remote_backend",
        choices=["local", "kaggle", "ssh"],
        default="local",
        help="Where to run fine-tuning: local machine, Kaggle GPU, or SSH remote host",
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
            remote_backend=args.remote_backend if args.remote_backend != "local" else "",
            model=args.model,
        ))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
