"""FastAPI application factory for the aipet inference service."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


def _make_storage_adapter():
    """Return S3StorageAdapter when AWS_S3_BUCKET is set, otherwise LocalStorageAdapter."""
    if os.getenv("AWS_S3_BUCKET"):
        from adapters.storage.s3 import S3StorageAdapter
        return S3StorageAdapter()
    from adapters.storage.local import LocalStorageAdapter
    return LocalStorageAdapter()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from adapters.database import init_db, make_engine
    from adapters.database.model_store import SQLAlchemyModelStore
    from adapters.database.run_store import SQLAlchemyRunStore
    from adapters.inference import LlamaCppInferenceAdapter
    from interactors.api.deps import (
        clear_adapter,
        configure,
        configure_model_store,
        configure_run_store,
    )
    from interactors.temporal.activities import (
        configure_run_store as configure_activity_run_store,
        configure_storage,
    )

    engine = make_engine()
    init_db(engine)

    store = SQLAlchemyModelStore(engine)
    configure_model_store(store)

    run_store = SQLAlchemyRunStore(engine)
    configure_run_store(run_store)
    configure_activity_run_store(run_store)

    storage = _make_storage_adapter()
    configure_storage(storage)

    active = store.active()
    if active and active.gguf_path:
        local_path = Path("models/cache") / active.id / "model.gguf"
        try:
            storage.download(active.gguf_path, local_path)
            model_path = str(local_path)
            logger.info("Loading active model %s from storage key %s", active.id, active.gguf_path)
        except Exception:
            logger.warning(
                "Could not load active model %s from storage; falling back to MODEL_PATH",
                active.id,
                exc_info=True,
            )
            model_path = os.getenv("MODEL_PATH", "models/aipet.gguf")
    else:
        model_path = os.getenv("MODEL_PATH", "models/aipet.gguf")

    configure(LlamaCppInferenceAdapter(model_path=model_path))

    try:
        yield
    finally:
        clear_adapter()


from interactors.api.routes.inference import router as inference_router  # noqa: E402
from interactors.api.routes.models import router as models_router  # noqa: E402
from interactors.api.routes.runs import router as runs_router  # noqa: E402

app = FastAPI(title="aipet-llm inference service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference_router)
app.include_router(models_router)
app.include_router(runs_router)
