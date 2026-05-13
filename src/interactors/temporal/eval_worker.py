"""Temporal eval worker — handles evaluation and GGUF export.

Run alongside worker.py (orchestration) and training_worker.py. All three poll
the same task queue; Temporal routes each activity to whichever worker registered it.
Can be deployed on the target RPi or a separate inference machine (uses llama-cpp,
not training libs).
"""

from __future__ import annotations

import asyncio
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from temporalio.client import Client
from temporalio.worker import Worker

from interactors.temporal.activities import (
    configure_model_store,
    configure_run_store,
    configure_storage,
    evaluate_activity,
    export_activity,
)

TASK_QUEUE = "aipet-training"


async def main() -> None:
    from adapters.database import init_db, make_engine
    from adapters.database.model_store import SQLAlchemyModelStore
    from adapters.database.run_store import SQLAlchemyRunStore

    engine = make_engine()
    init_db(engine)
    configure_model_store(SQLAlchemyModelStore(engine))
    configure_run_store(SQLAlchemyRunStore(engine))

    if os.getenv("AWS_S3_BUCKET"):
        from adapters.storage.s3 import S3StorageAdapter
        configure_storage(S3StorageAdapter())
    else:
        from adapters.storage.local import LocalStorageAdapter
        configure_storage(LocalStorageAdapter())

    temporal_host = os.environ.get("TEMPORAL_HOST", "localhost:7233")
    client = await Client.connect(temporal_host)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[],
        activities=[
            evaluate_activity,
            export_activity,
        ],
    )

    print(f"Eval worker started — task_queue={TASK_QUEUE}  host={temporal_host}")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
