"""Inference and health endpoints."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException

from domain.models import InferenceRequest, InferenceResponse
from domain.ports import InferencePort
from interactors.api.auth import require_auth
from interactors.api.deps import get_adapter

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/infer", response_model=InferenceResponse, dependencies=[Depends(require_auth)])
def infer(
    request: InferenceRequest,
    adapter: InferencePort = Depends(get_adapter),
) -> InferenceResponse:
    try:
        return adapter.infer(request)
    except Exception:
        log.exception("Unexpected error during inference")
        raise HTTPException(status_code=500, detail={"error": "inference_failed"})


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": os.getenv("MODEL_PATH", "models/aipet.gguf"),
    }
