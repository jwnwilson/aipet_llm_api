"""E2E workflow tests — full TrainingPipelineWorkflow with mocked domain functions (dry run).

Uses Temporal's embedded time-skipping test server so no real Temporal cluster is needed,
and patches all domain functions so no ML computation runs.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

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


def _dry_run_patches(eval_passes: bool = True):
    """Return a list of patches that stub every domain I/O call."""

    def fake_evaluate(path, infer_fn):
        if eval_passes:
            print("Valid: 190/200 (95.0%)  [PASS]")
            return 0
        else:
            print("Valid: 150/200 (75.0%)  [FAIL]")
            return 1

    return [
        patch("domain.train.dataset.generate", return_value=True),
        patch("domain.train.trainer.train"),
        patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()),
        patch("domain.train.evaluate.infer_hf", return_value='{"action": "IDLE"}'),
        patch("domain.train.evaluate.evaluate", side_effect=fake_evaluate),
        patch("domain.train.export.export"),
    ]


def _configure_mock_storage() -> MagicMock:
    """Wire a mock StoragePort into the activities module and return it."""
    storage = MagicMock()
    configure_storage(storage)
    return storage


_ACTIVITIES = [
    generate_dataset_activity,
    train_activity,
    evaluate_activity,
    export_activity,
    finalise_run_activity,
    save_gguf_path_activity,
    update_run_status_activity,
]


@pytest.mark.asyncio
async def test_training_pipeline_workflow_e2e_pass():
    """Happy path: all stages succeed and a GGUF is exported."""
    _configure_mock_storage()
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[TrainingPipelineWorkflow],
            activities=_ACTIVITIES,
        ):
            patches = _dry_run_patches(eval_passes=True)
            for p in patches:
                p.start()
            try:
                config = ExperimentConfig(
                    experiment_name="dry-run-pass",
                    train_size=10,
                    eval_size=5,
                    epochs=1,
                )
                result: PipelineResult = await env.client.execute_workflow(
                    TrainingPipelineWorkflow.run,
                    config,
                    id="test-dry-run-pass",
                    task_queue="test-queue",
                )
            finally:
                for p in reversed(patches):
                    p.stop()

    assert result.passed is True
    assert result.dataset_paths.train.endswith("train.jsonl")
    assert result.dataset_paths.eval.endswith("eval.jsonl")
    assert result.checkpoint.path != ""
    assert abs(result.eval_result.valid_pct - 0.95) < 1e-6
    assert result.gguf_path.path.endswith(".gguf")


@pytest.mark.asyncio
async def test_training_pipeline_workflow_e2e_eval_fail_skips_export():
    """When eval does not reach 95%, the workflow completes but skips export."""
    _configure_mock_storage()
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-queue-fail",
            workflows=[TrainingPipelineWorkflow],
            activities=_ACTIVITIES,
        ):
            patches = _dry_run_patches(eval_passes=False)
            for p in patches:
                p.start()
            try:
                config = ExperimentConfig(
                    experiment_name="dry-run-fail",
                    train_size=10,
                    eval_size=5,
                    epochs=1,
                )
                result: PipelineResult = await env.client.execute_workflow(
                    TrainingPipelineWorkflow.run,
                    config,
                    id="test-dry-run-fail",
                    task_queue="test-queue-fail",
                )
            finally:
                for p in reversed(patches):
                    p.stop()

    assert result.passed is False
    assert abs(result.eval_result.valid_pct - 0.75) < 1e-6
    assert result.gguf_path.path == ""


@pytest.mark.asyncio
async def test_training_pipeline_workflow_e2e_skip_generate():
    """With skip_generate=True the dataset step is bypassed and existing paths are used."""
    _configure_mock_storage()
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-queue-skip",
            workflows=[TrainingPipelineWorkflow],
            activities=_ACTIVITIES,
        ):
            patches = _dry_run_patches(eval_passes=True)
            for p in patches:
                p.start()
            try:
                config = ExperimentConfig(
                    experiment_name="dry-run-skip-gen",
                    skip_generate=True,
                    data_dir="data",
                    epochs=1,
                )
                result: PipelineResult = await env.client.execute_workflow(
                    TrainingPipelineWorkflow.run,
                    config,
                    id="test-dry-run-skip-gen",
                    task_queue="test-queue-skip",
                )
            finally:
                for p in reversed(patches):
                    p.stop()

    assert result.passed is True
    assert result.dataset_paths.train == "data/train.jsonl"
    assert result.dataset_paths.eval == "data/eval.jsonl"
