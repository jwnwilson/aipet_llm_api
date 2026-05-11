"""FastAPI application factory for the aipet inference service."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from domain.ports import InferencePort

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

    Called by the lifespan handler on startup and by tests to inject a stub.
    """
    global _adapter
    _adapter = adapter


# ---------------------------------------------------------------------------
# Lifespan — load / unload the real adapter around the server's lifetime
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from infrastructure.inference import LlamaCppInferenceAdapter
    from infrastructure.database import init_db, make_engine
    from infrastructure.model_store import SQLAlchemyModelStore
    from api.training_routes import configure_model_store

    model_path = os.getenv("MODEL_PATH", "models/aipet.gguf")
    configure(LlamaCppInferenceAdapter(model_path=model_path))

    engine = make_engine()
    init_db(engine)
    configure_model_store(SQLAlchemyModelStore(engine))

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
