"""Training management API routes — model configs and Temporal run status."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from domain.models import TrainingModel, TrainingModelConfig
from domain.ports import ModelStorePort
from infrastructure.database import get_session
from infrastructure.models.model_store import SQLAlchemyModelStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

_store: ModelStorePort | None = None


def get_model_store() -> ModelStorePort:
    if _store is None:
        raise RuntimeError("ModelStorePort has not been configured.")
    return _store


def configure_model_store(store: ModelStorePort) -> None:
    global _store
    _store = store


# ---------------------------------------------------------------------------
# Model CRUD
# ---------------------------------------------------------------------------

@router.get("/models", response_model=list[TrainingModel])
def list_models(store: ModelStorePort = Depends(get_model_store)) -> list[TrainingModel]:
    return store.list()


@router.post("/models", response_model=TrainingModel, status_code=201)
def create_model(
    config: TrainingModelConfig,
    store: ModelStorePort = Depends(get_model_store),
) -> TrainingModel:
    return store.create(config)


@router.get("/models/{model_id}", response_model=TrainingModel)
def get_model(
    model_id: str,
    store: ModelStorePort = Depends(get_model_store),
) -> TrainingModel:
    model = store.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.put("/models/{model_id}", response_model=TrainingModel)
def update_model(
    model_id: str,
    config: TrainingModelConfig,
    store: ModelStorePort = Depends(get_model_store),
) -> TrainingModel:
    model = store.update(model_id, config)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.delete("/models/{model_id}", status_code=204)
def delete_model(
    model_id: str,
    store: ModelStorePort = Depends(get_model_store),
) -> None:
    deleted = store.delete(model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found")


# ---------------------------------------------------------------------------
# Trigger training run
# ---------------------------------------------------------------------------

@router.post("/models/{model_id}/trigger", status_code=202)
async def trigger_run(
    model_id: str,
    store: ModelStorePort = Depends(get_model_store),
) -> dict[str, str]:
    model = store.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        from temporalio.client import Client
        from temporal.worker import TASK_QUEUE
        from temporal.workflows import ExperimentConfig, TrainingPipelineWorkflow

        temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
        client = await Client.connect(temporal_host)

        config = ExperimentConfig(
            experiment_name=model.name,
            epochs=model.epochs,
            patience=model.patience,
            warmup_ratio=model.warmup_ratio,
            skip_generate=model.skip_generate,
            remote_backend="" if model.remote_backend == "local" else model.remote_backend,
            model=model.base_model,
        )

        workflow_id = f"training-{model.name}-{uuid.uuid4().hex[:8]}"
        await client.start_workflow(
            TrainingPipelineWorkflow.run,
            config,
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )

        return {"workflow_id": workflow_id}
    except Exception:
        logger.exception("Failed to trigger training workflow for model %s", model_id)
        raise HTTPException(status_code=500, detail="Failed to start training workflow")


# ---------------------------------------------------------------------------
# Run status (via Temporal)
# ---------------------------------------------------------------------------

@router.get("/runs")
async def list_runs() -> list[dict[str, Any]]:
    try:
        from temporalio.client import Client

        temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
        client = await Client.connect(temporal_host)

        runs = []
        async for wf in client.list_workflows("WorkflowType='TrainingPipelineWorkflow'"):
            runs.append({
                "workflow_id": wf.id,
                "run_id": wf.run_id,
                "status": wf.status.name if wf.status else "UNKNOWN",
                "start_time": wf.start_time.isoformat() if wf.start_time else None,
                "close_time": wf.close_time.isoformat() if wf.close_time else None,
            })
        return runs
    except Exception:
        logger.exception("Failed to list runs from Temporal")
        raise HTTPException(status_code=500, detail="Failed to list runs")


@router.get("/runs/{workflow_id}")
async def get_run(workflow_id: str) -> dict[str, Any]:
    try:
        from temporalio.client import Client

        temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
        client = await Client.connect(temporal_host)

        handle = client.get_workflow_handle(workflow_id)
        desc = await handle.describe()

        return {
            "workflow_id": workflow_id,
            "run_id": desc.run_id,
            "status": desc.status.name if desc.status else "UNKNOWN",
            "start_time": desc.start_time.isoformat() if desc.start_time else None,
            "close_time": desc.close_time.isoformat() if desc.close_time else None,
        }
    except Exception:
        logger.exception("Failed to fetch run %s from Temporal", workflow_id)
        raise HTTPException(status_code=404, detail="Run not found")
