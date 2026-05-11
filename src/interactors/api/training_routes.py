"""Training management API routes — model configs and run management."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from domain.models import RunConfig, RunRecord, RunStatus, TrainingModel, TrainingModelConfig
from domain.ports import ModelStorePort, RunStorePort

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

_store: ModelStorePort | None = None
_run_store: RunStorePort | None = None


def get_model_store() -> ModelStorePort:
    if _store is None:
        raise RuntimeError("ModelStorePort has not been configured.")
    return _store


def configure_model_store(store: ModelStorePort) -> None:
    global _store
    _store = store


def get_run_store() -> RunStorePort:
    if _run_store is None:
        raise RuntimeError("RunStorePort has not been configured.")
    return _run_store


def configure_run_store(store: RunStorePort) -> None:
    global _run_store
    _run_store = store


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
# List runs for a model
# ---------------------------------------------------------------------------

@router.get("/models/{model_id}/runs", response_model=list[RunRecord])
def list_model_runs(
    model_id: str,
    run_store: RunStorePort = Depends(get_run_store),
) -> list[RunRecord]:
    return run_store.list(model_id=model_id)


# ---------------------------------------------------------------------------
# Activate model (hot-swap inference adapter)
# ---------------------------------------------------------------------------

@router.post("/models/{model_id}/activate", response_model=TrainingModel)
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
    from interactors.api.app import configure
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
    logger.info("Activated model %s — gguf_path=%s", model.id, model.gguf_path)
    return model


# ---------------------------------------------------------------------------
# Trigger training run
# ---------------------------------------------------------------------------

@router.post("/models/{model_id}/trigger", status_code=202)
async def trigger_run(
    model_id: str,
    store: ModelStorePort = Depends(get_model_store),
    run_store: RunStorePort = Depends(get_run_store),
) -> dict[str, str]:
    model = store.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        from temporalio.client import Client
        from interactors.temporal.worker import TASK_QUEUE
        from interactors.temporal.workflows import ExperimentConfig, TrainingPipelineWorkflow

        temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
        client = await Client.connect(temporal_host)

        workflow_id = f"training-{model.id}-{uuid.uuid4().hex[:8]}"
        run = run_store.create(RunConfig(model_id=model.id, workflow_id=workflow_id))
        run_id = run.id

        Path(f"data/workflow/{run_id}").mkdir(parents=True, exist_ok=True)

        config = ExperimentConfig(
            experiment_name=model.name,
            model_id=model.id,
            run_id=run_id,
            epochs=model.epochs,
            patience=model.patience,
            warmup_ratio=model.warmup_ratio,
            skip_generate=model.skip_generate,
            remote_backend="" if model.remote_backend == "local" else model.remote_backend,
            model=model.base_model,
            data_dir=f"data/workflow/{run_id}",
            output_dir=f"data/workflow/{run_id}/checkpoint",
            gguf_output=f"data/workflow/{run_id}/model.gguf",
        )

        await client.start_workflow(
            TrainingPipelineWorkflow.run,
            config,
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )

        logger.info("Training triggered: model=%s run_id=%s workflow_id=%s", model_id, run_id, workflow_id)
        return {"workflow_id": workflow_id, "run_id": run_id}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to trigger training workflow for model %s", model_id)
        raise HTTPException(status_code=500, detail="Failed to start training workflow")


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

@router.get("/runs", response_model=list[RunRecord])
def list_runs(run_store: RunStorePort = Depends(get_run_store)) -> list[RunRecord]:
    return run_store.list()


@router.get("/runs/{run_id}", response_model=RunRecord)
def get_run(run_id: str, run_store: RunStorePort = Depends(get_run_store)) -> RunRecord:
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("/runs/{run_id}/activate", response_model=RunRecord)
def activate_run(
    run_id: str,
    run_store: RunStorePort = Depends(get_run_store),
) -> RunRecord:
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status != RunStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Run has not completed successfully (status={run.status.value})",
        )

    from adapters.inference import LlamaCppInferenceAdapter
    from adapters.storage import LocalStorageAdapter
    from interactors.api.app import configure
    from interactors.temporal.activities import _get_storage

    try:
        storage = _get_storage()
    except RuntimeError:
        storage = LocalStorageAdapter()

    gguf_key = f"workflow/{run_id}/model.gguf"
    local_path = Path(f"data/workflow/{run_id}/model.gguf")
    try:
        storage.download(gguf_key, local_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load run model from storage: {exc}") from exc

    configure(LlamaCppInferenceAdapter(model_path=str(local_path)))
    logger.info("Activated run %s — gguf=%s", run_id, local_path)
    return run


# ---------------------------------------------------------------------------
# Re-evaluate and download-export existing runs
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    remote_backend: str = ""
    remote_run_id: str = ""


class ExportRequest(BaseModel):
    remote_backend: str = ""
    remote_run_id: str = ""



@router.post("/runs/{run_id}/evaluate", status_code=202)
async def evaluate_run(
    run_id: str,
    body: EvaluateRequest = EvaluateRequest(),
    run_store: RunStorePort = Depends(get_run_store),
    store: ModelStorePort = Depends(get_model_store),
) -> dict[str, str]:
    """Start an async eval workflow for an existing run. Poll GET /api/runs/{run_id} for status."""
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    model = store.get(run.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found for this run")

    remote_backend = body.remote_backend or model.remote_backend
    if remote_backend == "local":
        remote_backend = ""

    workflow_id = f"evaluate-{run_id}-{uuid.uuid4().hex[:8]}"
    try:
        from temporalio.client import Client
        from interactors.temporal.worker import TASK_QUEUE
        from interactors.temporal.workflows import EvaluateWorkflow, EvaluateWorkflowConfig

        client = await Client.connect(os.getenv("TEMPORAL_HOST", "localhost:7233"))
        run_store.update(run_id, RunConfig(model_id=run.model_id, workflow_id=workflow_id))
        run_store.update_status(run_id, RunStatus.RUNNING)

        await client.start_workflow(
            EvaluateWorkflow.run,
            EvaluateWorkflowConfig(
                run_id=run_id,
                remote_backend=remote_backend,
                remote_run_id=body.remote_run_id,
                eval_data=model.eval_data,
                checkpoint_path=f"data/workflow/{run_id}/checkpoint",
                output_dir=f"data/workflow/{run_id}",
            ),
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )

        logger.info("Eval workflow started: run_id=%s workflow_id=%s", run_id, workflow_id)
        return {"run_id": run_id, "workflow_id": workflow_id}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to start evaluate workflow for run %s", run_id)
        raise HTTPException(status_code=500, detail="Failed to start evaluation workflow")


@router.post("/runs/{run_id}/export", status_code=202)
async def export_run(
    run_id: str,
    body: ExportRequest = ExportRequest(),
    run_store: RunStorePort = Depends(get_run_store),
    store: ModelStorePort = Depends(get_model_store),
) -> dict[str, str]:
    """Start an async export workflow for an existing run. Poll GET /api/runs/{run_id} for status."""
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    model = store.get(run.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found for this run")

    remote_backend = body.remote_backend or model.remote_backend
    if remote_backend == "local":
        remote_backend = ""

    workflow_id = f"export-{run_id}-{uuid.uuid4().hex[:8]}"
    try:
        from temporalio.client import Client
        from interactors.temporal.worker import TASK_QUEUE
        from interactors.temporal.workflows import ExportWorkflow, ExportWorkflowConfig

        client = await Client.connect(os.getenv("TEMPORAL_HOST", "localhost:7233"))
        run_store.update(run_id, RunConfig(model_id=run.model_id, workflow_id=workflow_id))
        run_store.update_status(run_id, RunStatus.RUNNING)

        await client.start_workflow(
            ExportWorkflow.run,
            ExportWorkflowConfig(
                run_id=run_id,
                model_id=model.id,
                remote_backend=remote_backend,
                remote_run_id=body.remote_run_id,
                checkpoint_path=f"data/workflow/{run_id}/checkpoint",
                gguf_output=f"data/workflow/{run_id}/model.gguf",
            ),
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )

        logger.info("Export workflow started: run_id=%s workflow_id=%s", run_id, workflow_id)
        return {"run_id": run_id, "workflow_id": workflow_id}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to start export workflow for run %s", run_id)
        raise HTTPException(status_code=500, detail="Failed to start export workflow")
