"""Inference and health endpoints."""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import APIRouter, Depends, HTTPException

from domain.models import InferenceRequest, InferenceResponse
from domain.ports import InferencePort
from interactors.api.auth import require_approved
from interactors.api.deps import get_adapter

log = logging.getLogger(__name__)

router = APIRouter()

# One inference at a time — concurrent calls on RPi thrash the CPU.
_infer_semaphore = asyncio.Semaphore(1)


@router.post("/infer", response_model=InferenceResponse, dependencies=[Depends(require_approved)])
async def infer(
    request: InferenceRequest,
    adapter: InferencePort = Depends(get_adapter),
) -> InferenceResponse:
    if os.getenv("INFERENCE_DISABLED", "").lower() in ("1", "true", "yes"):
        raise HTTPException(status_code=503, detail={"error": "inference_disabled"})
    loop = asyncio.get_event_loop()
    async with _infer_semaphore:
        try:
            return await loop.run_in_executor(None, adapter.infer, request)
        except Exception:
            log.exception("Unexpected error during inference")
            raise HTTPException(status_code=500, detail={"error": "inference_failed"})


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": os.getenv("MODEL_PATH", "models/aipet.gguf"),
    }
