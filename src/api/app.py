"""FastAPI application factory for the aipet inference service."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from domain.ports import InferencePort

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adapter singleton — wired at startup or via configure() in tests
# ---------------------------------------------------------------------------

_adapter: InferencePort | None = None


def get_adapter() -> InferencePort:
    """FastAPI dependency that returns the active InferencePort adapter."""
    if _adapter is None:
        raise RuntimeError("InferencePort adapter has not been configured.")
    return _adapter


def configure(adapter: InferencePort) -> None:
    """Wire in a concrete InferencePort implementation.

    Called by the lifespan handler on startup, by the activate endpoint for
    hot-swapping, and by tests to inject a stub.
    """
    global _adapter
    _adapter = adapter


# ---------------------------------------------------------------------------
# Lifespan — load / unload the real adapter around the server's lifetime
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from infrastructure.database import init_db, make_engine
    from infrastructure.inference import LlamaCppInferenceAdapter
    from infrastructure.models.model_store import SQLAlchemyModelStore
    from infrastructure.models.run_store import SQLAlchemyRunStore
    from infrastructure.storage import LocalStorageAdapter
    from api.training_routes import configure_model_store
    from api.training_routes import configure_run_store as configure_route_run_store
    from temporal.activities import configure_run_store, configure_storage

    engine = make_engine()
    init_db(engine)
    store = SQLAlchemyModelStore(engine)
    configure_model_store(store)

    run_store = SQLAlchemyRunStore(engine)
    configure_run_store(run_store)
    configure_route_run_store(run_store)

    storage = LocalStorageAdapter()
    configure_storage(storage)

    # Startup strategy: prefer the DB-flagged active model; fall back to env var.
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
        global _adapter
        _adapter = None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

from api.routes import router  # noqa: E402
from api.training_routes import router as training_router  # noqa: E402

app = FastAPI(title="aipet-llm inference service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(training_router)
