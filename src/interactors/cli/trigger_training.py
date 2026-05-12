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
    model_id: str | None = None,
) -> None:
    from pathlib import Path

    from temporalio.client import Client

    from interactors.temporal.worker import TASK_QUEUE
    from interactors.temporal.workflows import ExperimentConfig, TrainingPipelineWorkflow

    temporal_host = os.environ.get("TEMPORAL_HOST", "localhost:7233")
    client = await Client.connect(temporal_host)

    model_name = ""
    if model_id is not None:
        from adapters.database.engine import make_engine
        from adapters.database.model_store import SQLAlchemyModelStore
        from adapters.database.run_store import SQLAlchemyRunStore
        from domain.models import RunConfig

        engine = make_engine()
        db_model = SQLAlchemyModelStore(engine).get(model_id)
        if db_model is None:
            print(f"ERROR: model '{model_id}' not found in database.", file=sys.stderr)
            sys.exit(1)
        model_name = db_model.name
        workflow_id = f"training-{model_id}-{uuid.uuid4().hex[:8]}"
        run_record = SQLAlchemyRunStore(engine).create(
            RunConfig(model_id=model_id, workflow_id=workflow_id)
        )
        run_id = run_record.id
        print(f"RunRecord created: run_id={run_id}")
    else:
        run_id = str(uuid.uuid4())
        workflow_id = f"training-{experiment_name}-{uuid.uuid4().hex[:8]}"

    run_data_dir = f"data/workflow/{run_id}"
    Path(run_data_dir).mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        experiment_name=experiment_name,
        model_id=model_id or "",
        model_name=model_name,
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
    print(f"  Model name : {model_name or '(untracked)'}")
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
    parser.add_argument(
        "--model-id",
        dest="model_id",
        default=None,
        help=(
            "ID of a model record in the training_models table. "
            "When provided, a RunRecord is created in the database and the run is linked to the model."
        ),
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
            model_id=args.model_id,
        ))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
