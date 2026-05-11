"""Temporal worker — registers all activities and the workflow, then polls for tasks."""

from __future__ import annotations

import asyncio
import os

from temporalio.client import Client
from temporalio.worker import Worker

from temporal.activities import (
    configure_model_store,
    configure_run_store,
    configure_storage,
    evaluate_activity,
    export_activity,
    finalise_run_activity,
    generate_dataset_activity,
    save_gguf_path_activity,
    train_activity,
)
from temporal.workflows import TrainingPipelineWorkflow

TASK_QUEUE = "aipet-training"


async def main() -> None:
    from infrastructure.database import init_db, make_engine
    from infrastructure.models.model_store import SQLAlchemyModelStore
    from infrastructure.models.run_store import SQLAlchemyRunStore
    from infrastructure.storage import LocalStorageAdapter

    engine = make_engine()
    init_db(engine)
    configure_model_store(SQLAlchemyModelStore(engine))
    configure_run_store(SQLAlchemyRunStore(engine))
    configure_storage(LocalStorageAdapter())

    temporal_host = os.environ.get("TEMPORAL_HOST", "localhost:7233")
    client = await Client.connect(temporal_host)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[TrainingPipelineWorkflow],
        activities=[
            generate_dataset_activity,
            train_activity,
            evaluate_activity,
            export_activity,
            finalise_run_activity,
            save_gguf_path_activity,
        ],
    )

    print(f"Worker started — task_queue={TASK_QUEUE}  host={temporal_host}")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
