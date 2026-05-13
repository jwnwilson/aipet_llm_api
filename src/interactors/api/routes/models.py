"""Model CRUD and management endpoints."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from domain.models import TrainingModel, TrainingModelConfig
from domain.ports import ModelStorePort
from interactors.api.auth import require_auth
from interactors.api.deps import get_model_store

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/models",
    tags=["models"],
    dependencies=[Depends(require_auth)],
)


@router.get("", response_model=list[TrainingModel])
def list_models(store: ModelStorePort = Depends(get_model_store)) -> list[TrainingModel]:
    return store.list()


@router.post("", response_model=TrainingModel, status_code=201)
def create_model(
    config: TrainingModelConfig,
    store: ModelStorePort = Depends(get_model_store),
) -> TrainingModel:
    return store.create(config)


@router.get("/{model_id}", response_model=TrainingModel)
def get_model(
    model_id: str,
    store: ModelStorePort = Depends(get_model_store),
) -> TrainingModel:
    model = store.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.put("/{model_id}", response_model=TrainingModel)
def update_model(
    model_id: str,
    config: TrainingModelConfig,
    store: ModelStorePort = Depends(get_model_store),
) -> TrainingModel:
    model = store.update(model_id, config)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.delete("/{model_id}", status_code=204)
def delete_model(
    model_id: str,
    store: ModelStorePort = Depends(get_model_store),
) -> None:
    deleted = store.delete(model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found")


@router.post("/{model_id}/activate", response_model=TrainingModel)
def activate_model(
    model_id: str,
    store: ModelStorePort = Depends(get_model_store),
) -> TrainingModel:
    model = store.activate(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if not model.gguf_path:
        raise HTTPException(
            status_code=409,
            detail="Model has no exported GGUF yet — run a training pipeline first",
        )

    from adapters.inference import LlamaCppInferenceAdapter
    from adapters.storage import LocalStorageAdapter
    from interactors.api.deps import configure
    from interactors.temporal.activities import _get_storage

    try:
        storage = _get_storage()
    except RuntimeError:
        storage = LocalStorageAdapter()

    local_path = Path("models/cache") / model.id / "model.gguf"
    try:
        storage.download(model.gguf_path, local_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model from storage: {exc}") from exc

    configure(LlamaCppInferenceAdapter(model_path=str(local_path)))
    log.info("Activated model %s — gguf_path=%s", model.id, model.gguf_path)
    return model
