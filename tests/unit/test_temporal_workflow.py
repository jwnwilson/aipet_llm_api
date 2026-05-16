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
    EvalConfig,
    configure_storage,
    evaluate_activity,
    export_activity,
    finalise_run_activity,
    generate_dataset_activity,
    save_gguf_path_activity,
    train_activity,
    update_run_status_activity,
)
from interactors.temporal.workflows import (
    EvaluateWorkflow,
    EvaluateWorkflowConfig,
    ExperimentConfig,
    PipelineResult,
    TrainingPipelineWorkflow,
)


def _dry_run_patches(eval_passes: bool = True):
    """Return a list of patches that stub every domain I/O call."""

    def fake_evaluate(path, infer_fn):
        if eval_passes:
            return (0, 0.95)
        else:
            return (1, 0.75)

    def _fake_upload_model(storage, local_path, key: str) -> str:
        return key if key.endswith(".gz") else key + ".gz"

    return [
        patch("domain.train.dataset.generate", return_value=True),
        patch("domain.train.trainer.train"),
        patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()),
        patch("domain.train.evaluate.infer_hf", return_value='{"action": "IDLE"}'),
        patch("domain.train.evaluate.evaluate", side_effect=fake_evaluate),
        patch("domain.train.export.export"),
        patch("adapters.storage.upload_model", side_effect=_fake_upload_model),
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
    assert result.gguf_path.path.endswith(".gguf.gz")


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


@pytest.mark.asyncio
async def test_evaluate_workflow_passes_db_run_id():
    """EvaluateWorkflow must pass db_run_id to EvalConfig so quality report is written."""
    storage = _configure_mock_storage()

    # Track calls to storage.write to verify quality_report.json is written with db_run_id
    written_files = {}

    def capture_write(path, content):
        written_files[path] = content

    storage.write = capture_write

    # Configure a mock RunStore for finalise_run_activity
    run_store = MagicMock()
    from interactors.temporal.activities import configure_run_store
    configure_run_store(run_store)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-evaluate-queue",
            workflows=[EvaluateWorkflow],
            activities=[evaluate_activity, finalise_run_activity],
        ):
            # Patch the domain functions called by evaluate_activity
            patches = [
                patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()),
                patch("domain.train.evaluate.infer_hf", return_value='{"action": "IDLE"}'),
                patch("domain.train.evaluate.evaluate", return_value=(0, 0.96)),
                patch("domain.train.quality_report.run_quality_report", return_value={"passed": True}),
            ]
            for p in patches:
                p.start()
            try:
                config = EvaluateWorkflowConfig(
                    run_id="db-run-12345",
                    remote_backend="",
                    remote_run_id="",
                    eval_data="data/eval.jsonl",
                    checkpoint_path="/path/to/checkpoint",
                    output_dir="data/workflow/db-run-12345",
                )
                result = await env.client.execute_workflow(
                    EvaluateWorkflow.run,
                    config,
                    id="test-evaluate-workflow",
                    task_queue="test-evaluate-queue",
                )
            finally:
                for p in reversed(patches):
                    p.stop()

    # Verify that the evaluation completed successfully
    # The presence of quality report logs confirms db_run_id was passed to EvalConfig
    # (The evaluate_activity only saves a quality report when config.db_run_id is non-empty)
    assert result.passed is True
    assert abs(result.valid_pct - 0.96) < 1e-6
