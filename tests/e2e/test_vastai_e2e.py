"""E2E test: Vast.ai remote backend — full training pipeline.

Rents a Vast.ai GPU instance, polls S3 status until complete, downloads the
checkpoint, runs local eval, exports to GGUF, and validates inference.
Nothing is mocked.

Training data and the project wheel are staged to S3 before the instance
starts.  The instance writes status.txt and checkpoint.tar.gz back to S3
on completion.

Two scenarios:
  - standard LoRA  (vastai-e2e-lora)
  - QLoRA          (vastai-e2e-qlora)

Markers
-------
  @pytest.mark.vastai  — requires VAST_API_KEY + AWS credentials
  @pytest.mark.slow    — rents a real GPU instance; takes ~10-20 minutes

Requirements
------------
  make setup-llama   — builds llama.cpp (convert_hf_to_gguf.py + llama-quantize)

Environment variables
---------------------
  VAST_API_KEY            Vast.ai API key
  AWS_S3_BUCKET           S3 bucket for staging and checkpoint transfer
  AWS_ACCESS_KEY_ID       AWS credentials
  AWS_SECRET_ACCESS_KEY
  AWS_DEFAULT_REGION      (optional, default: us-east-1)
  VASTAI_GPU_QUERY        (optional, default: num_gpus=1 gpu_name=RTX_3090 reliability>0.99)
  VASTAI_IMAGE            (optional, Docker image for the instance)
  VASTAI_DISK_GB          (optional, disk size in GB, default: 50)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

log = logging.getLogger(__name__)

from adapters.compute.vastai import VastAiTrainingAdapter
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

# ---------------------------------------------------------------------------
# Real pipeline runner
# ---------------------------------------------------------------------------


async def _run_pipeline(tmp_path: Path, scenario: str, task_queue: str) -> PipelineResult:
    """Rent a real Vast.ai instance and run the full Temporal pipeline.

    Training and evaluation both run on the same instance: training_script.py
    runs HF eval after training and writes eval_result.json to S3 before the
    instance exits.  VastAiTrainingAdapter.eval() reads that file directly —
    no second instance is rented.
    """
    data_dir = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoint"
    gguf_raw = tmp_path / "model_raw.gguf"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)

    configure_storage(LocalStorageAdapter(base_dir=storage_dir))

    config = ExperimentConfig(
        experiment_name=f"vastai-e2e-{scenario}",
        model=TINY_MODEL,
        model_name=f"vastai-e2e-{scenario}",
        remote_backend="vastai",
        train_size=10,
        eval_size=5,
        epochs=1,
        dry_run=True,
        data_dir=str(data_dir),
        output_dir=str(checkpoint_dir),
        gguf_output=str(gguf_raw),
    )

    t_start = time.monotonic()
    log.info("[%s] renting Vast.ai instance  model=%s", scenario, TINY_MODEL)

    async with await WorkflowEnvironment.start_local() as env:
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
    # 1. Training submitted — S3 run_id prefix returned.
    log.info("[%s] step 1/4: run_id=%s", scenario, result.checkpoint.run_id)
    assert result.checkpoint.run_id, "run_id must be non-empty"
    assert result.checkpoint.remote_backend == "vastai"

    # 2. Evaluation ran via local fallback (passing not required).
    log.info("[%s] step 2/4: valid_pct=%.1f%%  passed=%s",
             scenario, result.eval_result.valid_pct * 100, result.eval_result.passed)
    assert 0.0 <= result.eval_result.valid_pct <= 1.0

    # 3. Download checkpoint from S3 and export to GGUF (always, regardless of eval score).
    checkpoint_dir = tmp_path / "dl_checkpoint"
    gguf_path = tmp_path / "model.gguf"

    log.info("[%s] step 3/4: downloading checkpoint from S3 run_id=%s", scenario, result.checkpoint.run_id)
    t0 = time.monotonic()
    checkpoint_path = VastAiTrainingAdapter().download(result.checkpoint.run_id, checkpoint_dir)
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


@pytest.mark.vastai
@pytest.mark.slow
@pytest.mark.asyncio
async def test_vastai_pipeline_lora_e2e(
    tmp_path: Path, llama_cpp_ready: Path, vastai_credentials: None
) -> None:
    """Full pipeline: real Vast.ai instance → S3 download → export → inference (standard LoRA)."""
    log.info("=== test_vastai_pipeline_lora_e2e START ===")
    result = await _run_pipeline(tmp_path, scenario="lora", task_queue="real-vastai-lora-e2e")
    _assert_pipeline(result, tmp_path, scenario="lora")
    log.info("=== test_vastai_pipeline_lora_e2e PASSED ===")


@pytest.mark.vastai
@pytest.mark.slow
@pytest.mark.asyncio
async def test_vastai_pipeline_qlora_e2e(
    tmp_path: Path, llama_cpp_ready: Path, vastai_credentials: None
) -> None:
    """Full pipeline: real Vast.ai instance (QLoRA) → S3 download → export → inference."""
    log.info("=== test_vastai_pipeline_qlora_e2e START ===")
    result = await _run_pipeline(tmp_path, scenario="qlora", task_queue="real-vastai-qlora-e2e")
    _assert_pipeline(result, tmp_path, scenario="qlora")
    log.info("=== test_vastai_pipeline_qlora_e2e PASSED ===")
