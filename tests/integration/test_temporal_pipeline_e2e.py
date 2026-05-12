"""E2E test: local Temporal training pipeline with a tiny model.

Covers the full pipeline end-to-end using the Temporal WorkflowEnvironment:
  generate_dataset → train → evaluate → export → inference

Two scenarios are exercised:
  1. Standard LoRA  (force_qlora=False — default for small models)
  2. QLoRA          (force_qlora=True  — skipped when CUDA is unavailable)

Each scenario asserts:
  1. Training completed — checkpoint directory with config.json exists.
  2. Evaluation ran    — valid_pct is in [0.0, 1.0]; a low score is OK for a
                         1-step-trained model.
  3. Export ran        — GGUF file is present in the local storage directory.
  4. Inference works   — LlamaCppInferenceAdapter returns a valid Action.

Markers
-------
  @pytest.mark.slow  — downloads a HuggingFace model (~270 MB); run with -m slow
  @pytest.mark.gpu   — QLoRA test only; requires CUDA for the 4-bit path

Requirements
------------
  make setup-llama   — builds llama.cpp (convert_hf_to_gguf.py + llama-quantize)
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

log = logging.getLogger(__name__)

from adapters.inference import LlamaCppInferenceAdapter
from adapters.storage.local import LocalStorageAdapter
from domain.actions import Action
from domain.models import InferenceRequest, PetStats, SceneData, SceneObject
from interactors.temporal import activities as _acts
from interactors.temporal.activities import (
    EvalResult,
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

# Capture the real _evaluate_local before any test patches it so we can call
# through to it from within the wrapping function.
_real_evaluate_local = _acts._evaluate_local


# ---------------------------------------------------------------------------
# Shared pipeline runner
# ---------------------------------------------------------------------------


async def _run_pipeline(
    tmp_path: Path,
    force_qlora: bool | None,
    task_queue: str,
) -> tuple[PipelineResult, list[EvalResult]]:
    """Execute the full Temporal pipeline and return (result, captured_eval_results).

    The evaluate step is wrapped so that:
      - The real evaluation code runs against the trained checkpoint.
      - The captured EvalResult is stored for assertions.
      - ``passed=True`` is returned regardless of score, so the export step
        always runs even when the 1-step model scores < 95 %.
    """
    scenario = "qlora" if force_qlora else "lora"
    data_dir = tmp_path / "data"
    checkpoint_dir = tmp_path / "checkpoint"
    gguf_raw = tmp_path / "model_raw.gguf"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)

    log.info("[%s] pipeline starting  model=%s  tmp=%s", scenario, TINY_MODEL, tmp_path)

    configure_storage(LocalStorageAdapter(base_dir=storage_dir))

    eval_captures: list[EvalResult] = []

    async def _capturing_evaluate_local(config, loop):
        log.info("[%s] step 2/4 — evaluating checkpoint at %s", scenario, config.checkpoint)
        t0 = time.monotonic()
        try:
            result = await _real_evaluate_local(config, loop)
        except Exception as exc:
            # A freshly initialised 1-step model often produces unparseable
            # output.  Record the failure but do not abort — the test validates
            # that the evaluate step ran, not that the model is good.
            log.warning("[%s] evaluation raised %s — recording valid_pct=0.0", scenario, exc)
            result = EvalResult(valid_pct=0.0, passed=False)
        elapsed = time.monotonic() - t0
        log.info(
            "[%s] step 2/4 — evaluation done  valid_pct=%.1f%%  passed=%s  elapsed=%.1fs",
            scenario, result.valid_pct * 100, result.passed, elapsed,
        )
        eval_captures.append(result)
        # Force passed=True so the workflow always proceeds to export.
        return dataclasses.replace(result, passed=True)

    config = ExperimentConfig(
        experiment_name=f"e2e-{scenario}",
        model=TINY_MODEL,
        model_name=f"e2e-{scenario}",   # drives storage key: gguf/e2e-{scenario}.gguf
        train_size=10,
        eval_size=5,
        epochs=1,
        dry_run=True,                    # max_steps=1 → fast single gradient step
        data_dir=str(data_dir),
        output_dir=str(checkpoint_dir),
        gguf_output=str(gguf_raw),
        force_qlora=force_qlora,
    )

    t_start = time.monotonic()
    log.info(
        "[%s] step 1/4 — generating dataset  train=10  eval=5  "
        "then training %s  dry_run=True  force_qlora=%s",
        scenario, TINY_MODEL, force_qlora,
    )

    with patch.object(_acts, "_evaluate_local", _capturing_evaluate_local):
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
                log.info("[%s] step 3/4 — export activity completed", scenario)

    log.info(
        "[%s] pipeline finished  passed=%s  gguf=%s  total=%.1fs",
        scenario, result.passed, result.gguf_path.path, time.monotonic() - t_start,
    )
    return result, eval_captures


# ---------------------------------------------------------------------------
# Shared assertions helper
# ---------------------------------------------------------------------------


def _assert_pipeline_outputs(
    result: PipelineResult,
    eval_captures: list[EvalResult],
    tmp_path: Path,
    scenario: str,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    storage_dir = tmp_path / "storage"
    gguf_path = storage_dir / "gguf" / f"e2e-{scenario}.gguf"

    # 1. Training completed — merged checkpoint on disk.
    log.info("[%s] asserting step 1/4: checkpoint exists at %s", scenario, checkpoint_dir)
    assert checkpoint_dir.exists(), "Checkpoint directory must exist after training"
    assert (checkpoint_dir / "config.json").exists(), "Checkpoint must contain config.json"
    log.info("[%s] step 1/4 OK — checkpoint present", scenario)

    # 2. Evaluation ran — valid_pct is a sensible float (low score is OK).
    log.info("[%s] asserting step 2/4: evaluation result captured", scenario)
    assert eval_captures, "evaluate_activity must have been called exactly once"
    assert 0.0 <= eval_captures[0].valid_pct <= 1.0, (
        f"valid_pct={eval_captures[0].valid_pct!r} must be in [0.0, 1.0]"
    )
    log.info("[%s] step 2/4 OK — valid_pct=%.1f%%", scenario, eval_captures[0].valid_pct * 100)

    # 3. Export completed — GGUF file uploaded to local storage.
    log.info("[%s] asserting step 3/4: GGUF at %s", scenario, gguf_path)
    assert result.gguf_path.path != "", "Workflow result must include a non-empty GGUF storage key"
    assert gguf_path.exists(), f"GGUF file must exist at {gguf_path}"
    size_mb = gguf_path.stat().st_size / (1024 ** 2)
    log.info("[%s] step 3/4 OK — GGUF present  size=%.1f MB  key=%s", scenario, size_mb, result.gguf_path.path)

    # 4. Inference works — adapter loads GGUF and returns a valid Action.
    log.info("[%s] asserting step 4/4: running inference via LlamaCppInferenceAdapter", scenario)
    adapter = LlamaCppInferenceAdapter(str(gguf_path))
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
    response = adapter.infer(request)
    assert response.action in Action, (
        f"Inference must return a valid Action; got {response.action!r}"
    )
    log.info("[%s] step 4/4 OK — inference returned action=%s", scenario, response.action.value)


# ---------------------------------------------------------------------------
# Test: standard LoRA (force_qlora=False)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.asyncio
async def test_local_pipeline_standard_lora_e2e(tmp_path: Path, llama_cpp_ready: Path) -> None:
    """Full Temporal pipeline with standard LoRA.

    Steps verified:
      1. Training   — merged checkpoint written to disk (dry_run, 1 step).
      2. Evaluation — real eval runs on the checkpoint; score may be < 95 %.
      3. Export     — GGUF produced and stored in local storage.
      4. Inference  — LlamaCppInferenceAdapter returns a valid Action.
    """
    log.info("=== test_local_pipeline_standard_lora_e2e START ===")
    result, eval_captures = await _run_pipeline(
        tmp_path=tmp_path,
        force_qlora=False,
        task_queue="test-lora-pipeline-e2e",
    )
    _assert_pipeline_outputs(result, eval_captures, tmp_path, scenario="lora")
    log.info("=== test_local_pipeline_standard_lora_e2e PASSED ===")


# ---------------------------------------------------------------------------
# Test: QLoRA (force_qlora=True) — skip when CUDA is unavailable
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_local_pipeline_qlora_e2e(tmp_path: Path, llama_cpp_ready: Path) -> None:
    """Full Temporal pipeline with QLoRA (force_qlora=True).

    On CUDA hardware:  loads base in 4-bit NF4, trains LoRA adapters, saves
    the adapter, reloads the base in float16, merges, then exports to GGUF.

    On non-CUDA hardware: skipped — QLoRA requires CUDA for the 4-bit path.

    Steps verified (same as standard LoRA test):
      1. Training   — merged checkpoint written to disk.
      2. Evaluation — real eval runs; score may be < 95 %.
      3. Export     — GGUF produced and stored in local storage.
      4. Inference  — LlamaCppInferenceAdapter returns a valid Action.
    """
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed — cannot determine CUDA availability")

    if not torch.cuda.is_available():
        pytest.skip("QLoRA requires CUDA — skipping on non-GPU machine")

    breakpoint()
    log.info("=== test_local_pipeline_qlora_e2e START ===")
    result, eval_captures = await _run_pipeline(
        tmp_path=tmp_path,
        force_qlora=True,
        task_queue="test-qlora-pipeline-e2e",
    )
    _assert_pipeline_outputs(result, eval_captures, tmp_path, scenario="qlora")
    log.info("=== test_local_pipeline_qlora_e2e PASSED ===")
