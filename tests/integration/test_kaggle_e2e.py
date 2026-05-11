"""E2E test: Kaggle backend workflow.

Starts an embedded Temporal worker, triggers a TrainingPipelineWorkflow with
remote_backend="kaggle", and validates the pipeline completes successfully.
All Kaggle API calls and domain I/O are mocked — no real GPU or credentials needed.
"""

from __future__ import annotations

from pathlib import Path
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


def _make_kaggle_adapter(checkpoint_dir: Path) -> MagicMock:
    adapter = MagicMock()
    adapter.submit.return_value = _KAGGLE_RUN_ID
    adapter.status.return_value = "done"
    adapter.logs.return_value = ""
    adapter.eval.return_value = (0.97, True)
    adapter.download.return_value = str(checkpoint_dir)
    return adapter


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
