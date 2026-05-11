"""Integration tests for the run management API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from api.app import app
from api.training_routes import (
    configure_model_store,
    configure_run_store,
    get_model_store,
    get_run_store,
)
from domain.models import RunStatus, TrainingModelConfig
from infrastructure.database import Base, init_db
from infrastructure.models.model_store import SQLAlchemyModelStore
from infrastructure.models.run_store import SQLAlchemyRunStore

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
            resp = await c.post(f"/api/models/{model.id}/trigger")

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
            resp = await c.post(f"/api/models/{model.id}/trigger")

        run_id = resp.json()["run_id"]
        run = run_store.get(run_id)
        assert run is not None
        assert run.status == RunStatus.PENDING
        assert run.model_id == model.id

    @pytest.mark.asyncio
    async def test_trigger_unknown_model_returns_404(self, client):
        c, _, _ = client
        resp = await c.post("/api/models/no-such-model/trigger")
        assert resp.status_code == 404


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
    async def test_activates_completed_run(self, client_with_model):
        c, model, run_store = client_with_model
        from domain.models import RunConfig
        run = run_store.create(RunConfig(model_id=model.id, workflow_id="wf-y"))
        run_store.update_status(run.id, RunStatus.COMPLETED)

        mock_storage = MagicMock()
        with (
            patch("temporal.activities._get_storage", return_value=mock_storage),
            patch("infrastructure.inference.LlamaCppInferenceAdapter"),
            patch("api.app.configure"),
        ):
            resp = await c.post(f"/api/runs/{run.id}/activate")

        assert resp.status_code == 200
        mock_storage.download.assert_called_once()

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


class TestListModelRuns:
    @pytest.mark.asyncio
    async def test_returns_only_runs_for_given_model(self, client_with_model):
        c, model, run_store = client_with_model
        from domain.models import RunConfig

        other_model = SQLAlchemyModelStore(
            create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        )
        run_store.create(RunConfig(model_id=model.id, workflow_id="wf-1"))
        run_store.create(RunConfig(model_id=model.id, workflow_id="wf-2"))
        run_store.create(RunConfig(model_id="other-model", workflow_id="wf-3"))

        resp = await c.get(f"/api/models/{model.id}/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 2
        assert all(r["model_id"] == model.id for r in runs)

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_runs_for_model(self, client_with_model):
        c, model, _ = client_with_model
        resp = await c.get(f"/api/models/{model.id}/runs")
        assert resp.status_code == 200
        assert resp.json() == []
