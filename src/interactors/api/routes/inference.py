"""Inference and health endpoints."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException

from interactors.api.deps import get_adapter
from domain.models import InferenceRequest, InferenceResponse
from domain.ports import InferencePort

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/infer", response_model=InferenceResponse)
def infer(
    request: InferenceRequest,
    adapter: InferencePort = Depends(get_adapter),
) -> InferenceResponse:
    try:
        return adapter.infer(request)
    except Exception:
        logger.exception("Unexpected error during inference")
        raise HTTPException(status_code=500, detail={"error": "inference_failed"})


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": os.getenv("MODEL_PATH", "models/aipet.gguf"),
    }
