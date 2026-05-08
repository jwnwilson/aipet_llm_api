"""Temporal worker — registers all activities and the workflow, then polls for tasks."""

from __future__ import annotations

import asyncio
import os

from temporalio.client import Client
from temporalio.worker import Worker

from temporal.activities import (
    evaluate_activity,
    export_activity,
    generate_dataset_activity,
    train_activity,
)
from temporal.workflows import TrainingPipelineWorkflow

TASK_QUEUE = "aipet-training"


async def main() -> None:
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
        ],
    )

    print(f"Worker started — task_queue={TASK_QUEUE}  host={temporal_host}")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
