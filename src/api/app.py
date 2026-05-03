"""FastAPI application factory for the aipet inference service."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from src.domain.ports import InferencePort

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
    from src.infrastructure.inference import LlamaCppInferenceAdapter

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

from src.api.routes import router  # noqa: E402 — imported after lifespan defined

app = FastAPI(title="aipet-llm inference service", lifespan=lifespan)
app.include_router(router)
