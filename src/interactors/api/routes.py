"""API route definitions for the aipet inference service."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException

from interactors.api.app import get_adapter
from domain.models import InferenceRequest, InferenceResponse
from domain.ports import InferencePort

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/infer", response_model=InferenceResponse)
def infer(
    request: InferenceRequest,
    adapter: InferencePort = Depends(get_adapter),
) -> InferenceResponse:
    """Run inference for the given scene and pet stats.

    Returns a pet action chosen by the model.
    Schema validation failures (malformed body) are handled automatically
    by FastAPI and result in HTTP 422.
    """
    try:
        return adapter.infer(request)
    except Exception:
        logger.exception("Unexpected error during inference")
        raise HTTPException(status_code=500, detail={"error": "inference_failed"})


@router.get("/health")
def health() -> dict:
    """Liveness check — confirms the service is running and shows the active model."""
    return {
        "status": "ok",
        "model": os.getenv("MODEL_PATH", "models/aipet.gguf"),
    }
