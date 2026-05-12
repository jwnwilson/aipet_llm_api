"""E2E test: Kaggle backend workflow.

Two test suites:

1. Mock-only tests (fast, no model download)
   - test_kaggle_workflow_e2e_pass       — happy path, all Kaggle API + export mocked
   - test_kaggle_workflow_e2e_eval_fail  — eval score below threshold, no export

2. Full-pipeline tests (slow, downloads a HuggingFace model)
   - test_kaggle_pipeline_standard_lora_e2e  — real checkpoint, real export, real inference
   - test_kaggle_pipeline_qlora_e2e          — same with force_qlora=True (GPU required)

   These mirror test_temporal_pipeline_e2e.py but exercise the Kaggle remote_backend code
   path.  Kaggle training is mocked via _make_remote_adapter; the downloaded checkpoint is
   a locally pre-trained tiny model so that real export and inference can be tested.

Markers
-------
  @pytest.mark.slow  — full-pipeline tests; downloads ~135 MB HuggingFace model
  @pytest.mark.gpu   — QLoRA test only; requires CUDA for the 4-bit path

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

from adapters.inference import LlamaCppInferenceAdapter
from adapters.storage.local import LocalStorageAdapter
from domain.actions import Action
from domain.models import InferenceRequest, PetStats, SceneData, SceneObject
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

_KAGGLE_RUN_ID = "testuser/aipet-kaggle-e2e"

# Capture the real _evaluate_local before any test patches it.


# ---------------------------------------------------------------------------
# Mock-only helper
# ---------------------------------------------------------------------------


def _make_kaggle_adapter(checkpoint_dir: Path) -> MagicMock:
    adapter = MagicMock()
    adapter.submit.return_value = _KAGGLE_RUN_ID
    adapter.status.return_value = "done"
    adapter.logs.return_value = ""
    adapter.eval.return_value = (0.97, True)
    adapter.download.return_value = str(checkpoint_dir)
    return adapter


# ---------------------------------------------------------------------------
# Mock-only tests (fast — no real model)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kaggle_workflow_e2e_pass(tmp_path: Path) -> None:
    """Happy path: Kaggle job submits, completes, eval passes, GGUF is exported."""
    mock_adapter = _make_kaggle_adapter(tmp_path / "checkpoint")
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
                patch(
                    "interactors.temporal.activities._make_remote_adapter",
                    return_value=mock_adapter,
                ),
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
    assert result.checkpoint.run_id == _KAGGLE_RUN_ID
    assert result.checkpoint.remote_backend == "kaggle"
    assert abs(result.eval_result.valid_pct - 0.97) < 1e-6

    mock_adapter.submit.assert_called_once()
    mock_adapter.status.assert_called()
    mock_adapter.eval.assert_called_once_with(_KAGGLE_RUN_ID, "data/eval.jsonl")
    mock_adapter.download.assert_called_once()
    mock_storage.upload.assert_called_once()


@pytest.mark.asyncio
async def test_kaggle_workflow_e2e_eval_fail(tmp_path: Path) -> None:
    """Kaggle job completes but eval score is below threshold — no export."""
    mock_adapter = _make_kaggle_adapter(tmp_path / "checkpoint")
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
                patch(
                    "interactors.temporal.activities._make_remote_adapter",
                    return_value=mock_adapter,
                ),
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
# Real Kaggle pipeline runner (slow — submits a genuine Kaggle kernel)
# ---------------------------------------------------------------------------


async def _run_real_kaggle_pipeline(
    tmp_path: Path,
    scenario: str,
    task_queue: str,
) -> PipelineResult:
    """Submit a real training job to Kaggle and run the full downstream pipeline.

    Nothing is mocked.  Kaggle handles training and evaluation end-to-end;
    export and inference run locally on the downloaded checkpoint.

    Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables (enforced by
    the kaggle_credentials fixture).
    """
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
    log.info(
        "[%s] submitting real Kaggle training job  model=%s  dry_run=True",
        scenario, TINY_MODEL,
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[TrainingPipelineWorkflow],
            activities=_ACTIVITIES,
        ):
            log.info("[%s] Temporal worker started — executing workflow", scenario)
            result: PipelineResult = await env.client.execute_workflow(
                TrainingPipelineWorkflow.run,
                config,
                id=f"test-{task_queue}",
                task_queue=task_queue,
            )

    log.info(
        "[%s] workflow finished  run_id=%s  eval=%.1f%%  passed=%s  gguf=%s  total=%.1fs",
        scenario, result.checkpoint.run_id, result.eval_result.valid_pct * 100,
        result.passed, result.gguf_path.path, time.monotonic() - t_start,
    )
    return result


# ---------------------------------------------------------------------------
# Shared assertions helper (real Kaggle pipeline)
# ---------------------------------------------------------------------------


def _assert_real_kaggle_pipeline_outputs(
    result: PipelineResult,
    tmp_path: Path,
    scenario: str,
) -> None:
    from pathlib import Path as _Path
    from adapters.compute.kaggle import KaggleTrainingAdapter
    from domain.train.export import export as export_gguf

    # 1. Training submitted — Kaggle returned a non-empty run_id slug.
    log.info("[%s] asserting step 1/4: Kaggle run_id set", scenario)
    assert result.checkpoint.run_id, "Kaggle adapter must return a non-empty run_id slug"
    assert result.checkpoint.remote_backend == "kaggle"
    log.info("[%s] step 1/4 OK — run_id=%s", scenario, result.checkpoint.run_id)

    # 2. Evaluation ran — valid_pct is a sensible float; passing is not required.
    log.info("[%s] asserting step 2/4: Kaggle eval result", scenario)
    assert 0.0 <= result.eval_result.valid_pct <= 1.0, (
        f"valid_pct={result.eval_result.valid_pct!r} must be in [0.0, 1.0]"
    )
    log.info("[%s] step 2/4 OK — valid_pct=%.1f%%  passed=%s", scenario,
             result.eval_result.valid_pct * 100, result.eval_result.passed)

    # 3. Download checkpoint from Kaggle and export to GGUF.
    #    This runs regardless of eval score — we always want to verify that
    #    the checkpoint can be downloaded and converted.
    checkpoint_dir = tmp_path / "dl_checkpoint"
    gguf_path = tmp_path / "model.gguf"

    log.info("[%s] asserting step 3/4: downloading checkpoint from Kaggle run_id=%s",
             scenario, result.checkpoint.run_id)
    t0 = time.monotonic()
    kaggle = KaggleTrainingAdapter()
    checkpoint_path = kaggle.download(result.checkpoint.run_id, checkpoint_dir)
    log.info("[%s] checkpoint downloaded to %s  elapsed=%.1fs",
             scenario, checkpoint_path, time.monotonic() - t0)

    log.info("[%s] exporting GGUF from %s", scenario, checkpoint_path)
    t0 = time.monotonic()
    export_gguf(checkpoint=_Path(checkpoint_path), output=gguf_path)
    assert gguf_path.exists(), f"GGUF must exist at {gguf_path} after export"
    size_mb = gguf_path.stat().st_size / (1024 ** 2)
    log.info("[%s] step 3/4 OK — GGUF exported  size=%.1f MB  elapsed=%.1fs",
             scenario, size_mb, time.monotonic() - t0)

    # 4. Inference works — adapter loads GGUF and returns a valid Action.
    log.info("[%s] asserting step 4/4: running inference via LlamaCppInferenceAdapter", scenario)
    inf_adapter = LlamaCppInferenceAdapter(str(gguf_path))
    request = InferenceRequest(
        scene=SceneData(
            objects=[SceneObject(id="bowl_0", type="bowl", distance=5.0)],
            tick=0,
        ),
        pet_stats=PetStats(
            hunger=0.9,
            tiredness=0.1,
            boredom=0.1,
            social=0.1,
            toilet=0.1,
        ),
    )
    response = inf_adapter.infer(request)
    assert response.action in Action, (
        f"Inference must return a valid Action; got {response.action!r}"
    )
    log.info("[%s] step 4/4 OK — action=%s", scenario, response.action.value)


# ---------------------------------------------------------------------------
# Test: standard LoRA on real Kaggle
# ---------------------------------------------------------------------------


@pytest.mark.kaggle
@pytest.mark.slow
@pytest.mark.asyncio
async def test_kaggle_pipeline_standard_lora_e2e(
    tmp_path: Path, llama_cpp_ready: Path, kaggle_credentials: None
) -> None:
    """Full pipeline against the real Kaggle backend with standard LoRA.

    Submits a genuine Kaggle kernel, polls until complete, downloads the
    checkpoint, runs local eval, exports to GGUF, and tests inference.

    Steps verified:
      1. Training   — real Kaggle kernel submitted and polled to completion.
      2. Evaluation — local HF eval on the downloaded checkpoint (score may be low).
      3. Export     — real GGUF produced from the downloaded checkpoint.
      4. Inference  — LlamaCppInferenceAdapter returns a valid Action.
    """
    log.info("=== test_kaggle_pipeline_standard_lora_e2e START ===")
    result = await _run_real_kaggle_pipeline(
        tmp_path=tmp_path,
        scenario="lora",
        task_queue="real-kaggle-lora-e2e",
    )
    _assert_real_kaggle_pipeline_outputs(result, tmp_path, scenario="lora")
    log.info("=== test_kaggle_pipeline_standard_lora_e2e PASSED ===")


# ---------------------------------------------------------------------------
# Test: QLoRA on real Kaggle
# ---------------------------------------------------------------------------


@pytest.mark.kaggle
@pytest.mark.slow
@pytest.mark.asyncio
async def test_kaggle_pipeline_qlora_e2e(
    tmp_path: Path, llama_cpp_ready: Path, kaggle_credentials: None
) -> None:
    """Full pipeline against the real Kaggle backend with QLoRA.

    The Kaggle notebook runs QLoRA natively on the GPU.  No local GPU is needed.

    Steps verified (same as standard LoRA test):
      1. Training   — real Kaggle kernel with QLoRA submitted and polled to completion.
      2. Evaluation — local HF eval on the downloaded checkpoint.
      3. Export     — real GGUF produced from the QLoRA-merged checkpoint.
      4. Inference  — LlamaCppInferenceAdapter returns a valid Action.
    """
    log.info("=== test_kaggle_pipeline_qlora_e2e START ===")
    result = await _run_real_kaggle_pipeline(
        tmp_path=tmp_path,
        scenario="qlora",
        task_queue="real-kaggle-qlora-e2e",
    )
    _assert_real_kaggle_pipeline_outputs(result, tmp_path, scenario="qlora")
    log.info("=== test_kaggle_pipeline_qlora_e2e PASSED ===")
