"""Integration test: TrainingPipelineWorkflow updates RunRecord status in DB."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from adapters.database import init_db
from adapters.database.model_store import SQLAlchemyModelStore
from adapters.database.run_store import SQLAlchemyRunStore
from domain.models import RunConfig, RunStatus, TrainingModelConfig
from interactors.temporal.activities import (
    configure_model_store,
    configure_run_store,
    configure_storage,
    evaluate_activity,
    export_activity,
    finalise_run_activity,
    generate_dataset_activity,
    save_gguf_path_activity,
    train_activity,
    update_run_status_activity,
)
from interactors.temporal.workflows import ExperimentConfig, TrainingPipelineWorkflow


_ACTIVITIES = [
    generate_dataset_activity,
    train_activity,
    evaluate_activity,
    export_activity,
    finalise_run_activity,
    save_gguf_path_activity,
    update_run_status_activity,
]


def _fake_evaluate(eval_data, infer_fn):
    print("Valid: 190/200 (95.0%)  [PASS]")
    return 0


@pytest.mark.asyncio
async def test_workflow_updates_run_status_in_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    model_store = SQLAlchemyModelStore(engine)
    run_store = SQLAlchemyRunStore(engine)

    model = model_store.create(TrainingModelConfig(
        name="status-test-model",
        base_model="HuggingFaceTB/SmolLM2-360M",
        train_data="data/train.jsonl",
        eval_data="data/eval.jsonl",
        epochs=3,
        patience=2,
        warmup_ratio=0.05,
        remote_backend="local",
        skip_generate=False,
    ))

    run = run_store.create(RunConfig(model_id=model.id, workflow_id="wf-status-test"))
    assert run.status == RunStatus.PENDING

    configure_run_store(run_store)
    configure_model_store(model_store)
    configure_storage(MagicMock())

    config = ExperimentConfig(
        experiment_name="db-status-test",
        model_id=model.id,
        model_name=model.name,
        run_id=run.id,
        model=model.base_model,
        epochs=model.epochs,
        patience=model.patience,
        warmup_ratio=model.warmup_ratio,
        skip_generate=False,
        remote_backend="",
        data_dir="data/test",
        output_dir="data/test/checkpoint",
        gguf_output="data/test/model.gguf",
    )

    with ExitStack() as stack:
        stack.enter_context(patch("domain.train.dataset.generate", return_value=True))
        stack.enter_context(patch("domain.train.trainer.train"))
        stack.enter_context(patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()))
        stack.enter_context(patch("domain.train.evaluate.infer_hf", return_value='{"action": "IDLE"}'))
        stack.enter_context(patch("domain.train.evaluate.evaluate", side_effect=_fake_evaluate))
        stack.enter_context(patch("domain.train.export.export"))

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="aipet-training",
                workflows=[TrainingPipelineWorkflow],
                activities=_ACTIVITIES,
            ):
                result = await env.client.execute_workflow(
                    TrainingPipelineWorkflow.run,
                    config,
                    id="wf-status-test",
                    task_queue="aipet-training",
                )

    final_run = run_store.get(run.id)
    assert final_run.status == RunStatus.COMPLETED
    assert final_run.eval_valid_pct == pytest.approx(0.95, abs=0.01)
    assert result.passed is True
