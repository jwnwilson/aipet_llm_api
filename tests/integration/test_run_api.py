"""Integration tests for the run management API endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from interactors.api.app import app
from interactors.api.deps import get_model_store, get_run_store
from domain.models import RunConfig, RunStatus, TrainingModelConfig
from adapters.database import Base, init_db
from adapters.database.model_store import SQLAlchemyModelStore
from adapters.database.run_store import SQLAlchemyRunStore

_VALID_MODEL_CONFIG = TrainingModelConfig(
    name="test-model",
    description="",
    base_model="HuggingFaceTB/SmolLM2-360M",
    train_data="data/train.jsonl",
    eval_data="data/eval.jsonl",
    epochs=3,
    patience=2,
    warmup_ratio=0.05,
    remote_backend="local",
    skip_generate=False,
)


@pytest_asyncio.fixture
async def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    model_store = SQLAlchemyModelStore(engine)
    run_store = SQLAlchemyRunStore(engine)

    app.dependency_overrides[get_model_store] = lambda: model_store
    app.dependency_overrides[get_run_store] = lambda: run_store

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c, model_store, run_store

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client_with_model(client):
    c, model_store, run_store = client
    model = model_store.create(_VALID_MODEL_CONFIG)
    yield c, model, run_store


class TestTriggerRun:
    def _connect_mock(self):
        mock_client = AsyncMock()
        mock_client.start_workflow = AsyncMock(return_value=MagicMock())
        return AsyncMock(return_value=mock_client), mock_client

    @pytest.mark.asyncio
    async def test_trigger_returns_run_id_and_workflow_id(self, client_with_model):
        c, model, run_store = client_with_model
        connect_mock, _ = self._connect_mock()

        with (
            patch("temporalio.client.Client.connect", connect_mock),
            patch("pathlib.Path.mkdir"),
        ):
            resp = await c.post("/api/runs/trigger", json={"model_id": model.id})

        assert resp.status_code == 202
        body = resp.json()
        assert "run_id" in body
        assert "workflow_id" in body
        assert len(body["run_id"]) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_trigger_creates_pending_run_in_db(self, client_with_model):
        c, model, run_store = client_with_model
        connect_mock, _ = self._connect_mock()

        with (
            patch("temporalio.client.Client.connect", connect_mock),
            patch("pathlib.Path.mkdir"),
        ):
            resp = await c.post("/api/runs/trigger", json={"model_id": model.id})

        run_id = resp.json()["run_id"]
        run = run_store.get(run_id)
        assert run is not None
        assert run.status == RunStatus.PENDING
        assert run.model_id == model.id

    @pytest.mark.asyncio
    async def test_trigger_unknown_model_returns_404(self, client):
        c, _, _ = client
        resp = await c.post("/api/runs/trigger", json={"model_id": "no-such-model"})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_trigger_passes_model_config_to_workflow(self, client_with_model):
        c, model, run_store = client_with_model
        connect_mock, mock_wf_client = self._connect_mock()

        with (
            patch("temporalio.client.Client.connect", connect_mock),
            patch("pathlib.Path.mkdir"),
        ):
            resp = await c.post("/api/runs/trigger", json={"model_id": model.id})

        assert resp.status_code == 202
        # ExperimentConfig is the second positional arg to start_workflow
        config = mock_wf_client.start_workflow.call_args[0][1]
        assert config.model_id == model.id
        assert config.model == model.base_model
        assert config.epochs == model.epochs
        assert config.skip_generate == model.skip_generate


class TestListRuns:
    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_runs(self, client):
        c, _, _ = client
        resp = await c.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_returns_all_runs(self, client_with_model):
        c, model, run_store = client_with_model
        from domain.models import RunConfig
        run_store.create(RunConfig(model_id=model.id, workflow_id="wf-1"))
        run_store.create(RunConfig(model_id=model.id, workflow_id="wf-2"))

        resp = await c.get("/api/runs")
        assert resp.status_code == 200
        assert len(resp.json()) == 2


class TestGetRun:
    @pytest.mark.asyncio
    async def test_returns_run_by_id(self, client_with_model):
        c, model, run_store = client_with_model
        from domain.models import RunConfig
        run = run_store.create(RunConfig(model_id=model.id, workflow_id="wf-x"))

        resp = await c.get(f"/api/runs/{run.id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == run.id
        assert body["model_id"] == model.id

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_run(self, client):
        c, _, _ = client
        resp = await c.get("/api/runs/no-such-run")
        assert resp.status_code == 404


class TestActivateRun:
    @pytest.mark.asyncio
    async def test_activates_completed_run_returns_200_with_run_record(self, client_with_model):
        """Activating a completed run returns 200 and the run record body."""
        c, model, run_store = client_with_model
        from domain.models import RunConfig
        run = run_store.create(RunConfig(model_id=model.id, workflow_id="wf-y"))
        run_store.update_status(run.id, RunStatus.COMPLETED)

        with (
            patch("interactors.temporal.activities._get_storage", return_value=MagicMock()),
            patch("adapters.inference.LlamaCppInferenceAdapter"),
            patch("interactors.api.deps.configure"),
        ):
            resp = await c.post(f"/api/runs/{run.id}/activate")

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == run.id
        assert body["status"] == RunStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_rejects_pending_run_with_409(self, client_with_model):
        c, model, run_store = client_with_model
        from domain.models import RunConfig
        run = run_store.create(RunConfig(model_id=model.id, workflow_id="wf-z"))
        # status is PENDING by default

        resp = await c.post(f"/api/runs/{run.id}/activate")
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_rejects_failed_run_with_409(self, client_with_model):
        c, model, run_store = client_with_model
        from domain.models import RunConfig
        run = run_store.create(RunConfig(model_id=model.id, workflow_id="wf-fail"))
        run_store.update_status(run.id, RunStatus.FAILED)

        resp = await c.post(f"/api/runs/{run.id}/activate")
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_run(self, client):
        c, _, _ = client
        resp = await c.post("/api/runs/no-such-run/activate")
        assert resp.status_code == 404


class TestGetRunEvaluation:
    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_run(self, client):
        c, _, _ = client
        resp = await c.get("/api/runs/does-not-exist/evaluation")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_evaluation_without_report(self, client_with_model, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        c, model, run_store = client_with_model
        run = run_store.create(RunConfig(model_id=model.id, workflow_id="wf-1"))
        run_store.update_status(run.id, RunStatus.COMPLETED)
        run_store.update_eval(run.id, 0.95)

        resp = await c.get(f"/api/runs/{run.id}/evaluation")

        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == run.id
        assert body["status"] == RunStatus.COMPLETED.value
        assert body["eval_valid_pct"] == pytest.approx(0.95)
        assert body["quality_report"] is None

    @pytest.mark.asyncio
    async def test_returns_evaluation_with_report(self, client_with_model, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        c, model, run_store = client_with_model
        run = run_store.create(RunConfig(model_id=model.id, workflow_id="wf-2"))
        run_store.update_status(run.id, RunStatus.COMPLETED)
        run_store.update_eval(run.id, 0.97)

        report = {
            "per_stat_accuracy": {
                s: {"correct": 38, "total": 40, "accuracy": 0.95, "passed": True}
                for s in ["hunger", "boredom", "social", "tiredness", "toilet"]
            },
            "target_accuracy": {"correct": 18, "total": 20, "accuracy": 0.9, "passed": True},
            "priority_conflict": {"correct": 16, "total": 20, "accuracy": 0.8, "passed": True},
            "fallback_accuracy": {"correct": 19, "total": 20, "accuracy": 0.95, "passed": True},
            "action_distribution": {"EAT": 50, "SLEEP": 40},
            "max_action_share": 0.25,
            "passed": True,
        }
        report_dir = tmp_path / "data" / "workflow" / run.id
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "quality_report.json").write_text(json.dumps(report))

        resp = await c.get(f"/api/runs/{run.id}/evaluation")

        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == run.id
        assert body["status"] == RunStatus.COMPLETED.value
        assert body["eval_valid_pct"] == pytest.approx(0.97)
        assert body["quality_report"]["passed"] is True
        assert body["quality_report"]["per_stat_accuracy"]["hunger"]["correct"] == 38
        assert body["quality_report"]["action_distribution"]["EAT"] == 50
