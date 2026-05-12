"""E2E test: Kaggle remote backend — full training pipeline.

Submits a real Kaggle kernel, polls until complete, downloads the checkpoint,
runs local eval, exports to GGUF, and validates inference.  Nothing is mocked.

Two scenarios:
  - standard LoRA  (kaggle-e2e-lora)
  - QLoRA          (kaggle-e2e-qlora) — Kaggle runs QLoRA natively on the GPU

Markers
-------
  @pytest.mark.kaggle  — requires KAGGLE_USERNAME + KAGGLE_KEY or KAGGLE_API_TOKEN
  @pytest.mark.slow    — submits a real Kaggle kernel; takes ~5-10 minutes

Requirements
------------
  make setup-llama   — builds llama.cpp (convert_hf_to_gguf.py + llama-quantize)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

log = logging.getLogger(__name__)

from adapters.compute.kaggle import KaggleTrainingAdapter
from adapters.inference import LlamaCppInferenceAdapter
from adapters.storage.local import LocalStorageAdapter
from domain.actions import Action
from domain.models import InferenceRequest, PetStats, SceneData, SceneObject
from domain.train.export import export as export_gguf
from interactors.temporal.activities import (
    configure_storage,
    evaluate_activity,
    export_activity,
    finalise_run_activity,
    generate_dataset_activity,
    save_gguf_path_activity,
    train_activity,
    update_run_status_activity,
)
from interactors.temporal.workflows import ExperimentConfig, PipelineResult, TrainingPipelineWorkflow

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

TINY_MODEL = "HuggingFaceTB/SmolLM2-135M"

_ACTIVITIES = [
    generate_dataset_activity,
    train_activity,
    evaluate_activity,
    export_activity,
    finalise_run_activity,
    save_gguf_path_activity,
    update_run_status_activity,
]

_MOCK_RUN_ID = "testuser/aipet-kaggle-e2e"

# ---------------------------------------------------------------------------
# Mock-only tests (fast — no real Kaggle job)
# ---------------------------------------------------------------------------


def _make_mock_adapter(checkpoint_dir: Path) -> MagicMock:
    adapter = MagicMock()
    adapter.submit.return_value = _MOCK_RUN_ID
    adapter.status.return_value = "done"
    adapter.logs.return_value = ""
    adapter.eval.return_value = (0.97, True)
    adapter.download.return_value = str(checkpoint_dir)
    return adapter


@pytest.mark.asyncio
async def test_kaggle_workflow_e2e_pass(tmp_path: Path) -> None:
    """Happy path: Kaggle job submits, completes, eval passes, GGUF is exported."""
    mock_adapter = _make_mock_adapter(tmp_path / "checkpoint")
    mock_storage = MagicMock()
    configure_storage(mock_storage)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-kaggle-queue",
            workflows=[TrainingPipelineWorkflow],
            activities=_ACTIVITIES,
        ):
            with (
                patch("domain.train.dataset.generate", return_value=True),
                patch("interactors.temporal.activities._make_remote_adapter", return_value=mock_adapter),
                patch("domain.train.export.export"),
            ):
                config = ExperimentConfig(
                    experiment_name="kaggle-e2e-test",
                    remote_backend="kaggle",
                    train_size=10,
                    eval_size=5,
                    epochs=1,
                    dry_run=True,
                )
                result: PipelineResult = await env.client.execute_workflow(
                    TrainingPipelineWorkflow.run,
                    config,
                    id="test-kaggle-e2e-pass",
                    task_queue="test-kaggle-queue",
                )

    assert result.passed is True
    assert result.checkpoint.run_id == _MOCK_RUN_ID
    assert result.checkpoint.remote_backend == "kaggle"
    assert abs(result.eval_result.valid_pct - 0.97) < 1e-6
    mock_adapter.submit.assert_called_once()
    mock_adapter.status.assert_called()
    mock_adapter.eval.assert_called_once_with(_MOCK_RUN_ID, "data/eval.jsonl")
    mock_adapter.download.assert_called_once()
    mock_storage.upload.assert_called_once()


@pytest.mark.asyncio
async def test_kaggle_workflow_e2e_eval_fail(tmp_path: Path) -> None:
    """Kaggle job completes but eval score is below threshold — no export."""
    mock_adapter = _make_mock_adapter(tmp_path / "checkpoint")
    mock_adapter.eval.return_value = (0.70, False)
    mock_storage = MagicMock()
    configure_storage(mock_storage)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-kaggle-queue-fail",
            workflows=[TrainingPipelineWorkflow],
            activities=_ACTIVITIES,
        ):
            with (
                patch("domain.train.dataset.generate", return_value=True),
                patch("interactors.temporal.activities._make_remote_adapter", return_value=mock_adapter),
                patch("domain.train.export.export"),
            ):
                config = ExperimentConfig(
                    experiment_name="kaggle-e2e-fail",
                    remote_backend="kaggle",
                    train_size=10,
                    eval_size=5,
                    epochs=1,
                    dry_run=True,
                )
                result: PipelineResult = await env.client.execute_workflow(
                    TrainingPipelineWorkflow.run,
                    config,
                    id="test-kaggle-e2e-fail",
                    task_queue="test-kaggle-queue-fail",
                )

    assert result.passed is False
    assert abs(result.eval_result.valid_pct - 0.70) < 1e-6
    assert result.gguf_path.path == ""
    mock_adapter.download.assert_not_called()
    mock_storage.upload.assert_not_called()


# ---------------------------------------------------------------------------
# Real pipeline runner
# ---------------------------------------------------------------------------


async def _run_pipeline(tmp_path: Path, scenario: str, task_queue: str) -> PipelineResult:
    """Submit a real Kaggle training job and run the full Temporal pipeline."""
    data_dir = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoint"
    gguf_raw = tmp_path / "model_raw.gguf"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)

    configure_storage(LocalStorageAdapter(base_dir=storage_dir))

    config = ExperimentConfig(
        experiment_name=f"kaggle-e2e-{scenario}",
        model=TINY_MODEL,
        model_name=f"kaggle-e2e-{scenario}",
        remote_backend="kaggle",
        train_size=10,
        eval_size=5,
        epochs=1,
        dry_run=True,
        data_dir=str(data_dir),
        output_dir=str(checkpoint_dir),
        gguf_output=str(gguf_raw),
    )

    t_start = time.monotonic()
    log.info("[%s] submitting Kaggle job  model=%s", scenario, TINY_MODEL)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[TrainingPipelineWorkflow],
            activities=_ACTIVITIES,
        ):
            result: PipelineResult = await env.client.execute_workflow(
                TrainingPipelineWorkflow.run,
                config,
                id=f"test-{task_queue}",
                task_queue=task_queue,
            )

    log.info(
        "[%s] workflow done  run_id=%s  eval=%.1f%%  passed=%s  total=%.1fs",
        scenario, result.checkpoint.run_id,
        result.eval_result.valid_pct * 100, result.passed,
        time.monotonic() - t_start,
    )
    return result


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------


def _assert_pipeline(result: PipelineResult, tmp_path: Path, scenario: str) -> None:
    # 1. Training submitted.
    log.info("[%s] step 1/4: run_id=%s", scenario, result.checkpoint.run_id)
    assert result.checkpoint.run_id, "run_id must be non-empty"
    assert result.checkpoint.remote_backend == "kaggle"

    # 2. Evaluation ran (passing not required).
    log.info("[%s] step 2/4: valid_pct=%.1f%%  passed=%s",
             scenario, result.eval_result.valid_pct * 100, result.eval_result.passed)
    assert 0.0 <= result.eval_result.valid_pct <= 1.0

    # 3. Download checkpoint and export to GGUF (always, regardless of eval score).
    checkpoint_dir = tmp_path / "dl_checkpoint"
    gguf_path = tmp_path / "model.gguf"

    log.info("[%s] step 3/4: downloading checkpoint from Kaggle", scenario)
    t0 = time.monotonic()
    checkpoint_path = KaggleTrainingAdapter().download(result.checkpoint.run_id, checkpoint_dir)
    log.info("[%s] downloaded to %s  elapsed=%.1fs", scenario, checkpoint_path, time.monotonic() - t0)

    t0 = time.monotonic()
    export_gguf(checkpoint=Path(checkpoint_path), output=gguf_path)
    assert gguf_path.exists()
    log.info("[%s] step 3/4 OK — %.1f MB  elapsed=%.1fs",
             scenario, gguf_path.stat().st_size / 1024**2, time.monotonic() - t0)

    # 4. Inference.
    log.info("[%s] step 4/4: inference", scenario)
    response = LlamaCppInferenceAdapter(str(gguf_path)).infer(InferenceRequest(
        scene=SceneData(objects=[SceneObject(id="bowl_0", type="bowl", distance=5.0)], tick=0),
        pet_stats=PetStats(hunger=0.9, tiredness=0.1, boredom=0.1, social=0.1, toilet=0.1),
    ))
    assert response.action in Action
    log.info("[%s] step 4/4 OK — action=%s", scenario, response.action.value)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.kaggle
@pytest.mark.slow
@pytest.mark.asyncio
async def test_kaggle_pipeline_lora_e2e(
    tmp_path: Path, llama_cpp_ready: Path, kaggle_credentials: None
) -> None:
    """Full pipeline: real Kaggle kernel → download → export → inference (standard LoRA)."""
    log.info("=== test_kaggle_pipeline_lora_e2e START ===")
    result = await _run_pipeline(tmp_path, scenario="lora", task_queue="real-kaggle-lora-e2e")
    _assert_pipeline(result, tmp_path, scenario="lora")
    log.info("=== test_kaggle_pipeline_lora_e2e PASSED ===")


@pytest.mark.kaggle
@pytest.mark.slow
@pytest.mark.asyncio
async def test_kaggle_pipeline_qlora_e2e(
    tmp_path: Path, llama_cpp_ready: Path, kaggle_credentials: None
) -> None:
    """Full pipeline: real Kaggle kernel (QLoRA) → download → export → inference."""
    log.info("=== test_kaggle_pipeline_qlora_e2e START ===")
    result = await _run_pipeline(tmp_path, scenario="qlora", task_queue="real-kaggle-qlora-e2e")
    _assert_pipeline(result, tmp_path, scenario="qlora")
    log.info("=== test_kaggle_pipeline_qlora_e2e PASSED ===")
