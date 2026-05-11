"""Integration tests for the training management API endpoints."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from api.app import app
from api.training_routes import configure_model_store, get_model_store
from infrastructure.database import Base
from infrastructure.models.model_store import SQLAlchemyModelStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VALID_CONFIG: dict[str, Any] = {
    "name": "test-model",
    "description": "A test training config",
    "base_model": "HuggingFaceTB/SmolLM2-360M",
    "train_data": "data/train.jsonl",
    "eval_data": "data/eval.jsonl",
    "epochs": 3,
    "patience": 2,
    "warmup_ratio": 0.05,
    "remote_backend": "local",
    "skip_generate": False,
}


@pytest_asyncio.fixture
async def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    store = SQLAlchemyModelStore(engine)
    app.dependency_overrides[get_model_store] = lambda: store
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client_with_model(client):
    """Client fixture that pre-populates one model and yields (client, model_id)."""
    resp = await client.post("/api/models", json=_VALID_CONFIG)
    assert resp.status_code == 201
    model_id = resp.json()["id"]
    yield client, model_id


# ---------------------------------------------------------------------------
# TestListModels
# ---------------------------------------------------------------------------

class TestListModels:
    @pytest.mark.asyncio
    async def test_empty_store_returns_empty_list(self, client):
        resp = await client.get("/api/models")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_populated_store_returns_all_models(self, client):
        await client.post("/api/models", json=_VALID_CONFIG)
        await client.post("/api/models", json={**_VALID_CONFIG, "name": "second-model"})

        resp = await client.get("/api/models")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    @pytest.mark.asyncio
    async def test_models_ordered_newest_first(self, client):
        await client.post("/api/models", json={**_VALID_CONFIG, "name": "first"})
        await asyncio.sleep(0.01)
        await client.post("/api/models", json={**_VALID_CONFIG, "name": "second"})

        models = (await client.get("/api/models")).json()
        assert models[0]["name"] == "second"
        assert models[1]["name"] == "first"


# ---------------------------------------------------------------------------
# TestCreateModel
# ---------------------------------------------------------------------------

class TestCreateModel:
    @pytest.mark.asyncio
    async def test_valid_payload_returns_201_with_id_and_timestamps(self, client):
        resp = await client.post("/api/models", json=_VALID_CONFIG)
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "test-model"
        assert "id" in body
        assert "created_at" in body
        assert "updated_at" in body

    @pytest.mark.asyncio
    async def test_missing_required_name_returns_422(self, client):
        payload = {k: v for k, v in _VALID_CONFIG.items() if k != "name"}
        resp = await client.post("/api/models", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_defaults_are_applied_when_optional_fields_omitted(self, client):
        resp = await client.post("/api/models", json={"name": "minimal"})
        assert resp.status_code == 201
        body = resp.json()
        assert body["epochs"] == 5
        assert body["patience"] == 3

    @pytest.mark.asyncio
    async def test_created_at_equals_updated_at_on_creation(self, client):
        resp = await client.post("/api/models", json=_VALID_CONFIG)
        body = resp.json()
        assert body["created_at"] == body["updated_at"]

    @pytest.mark.asyncio
    async def test_all_config_fields_are_persisted(self, client):
        resp = await client.post("/api/models", json=_VALID_CONFIG)
        body = resp.json()
        for field, value in _VALID_CONFIG.items():
            assert body[field] == value


# ---------------------------------------------------------------------------
# TestGetModel
# ---------------------------------------------------------------------------

class TestGetModel:
    @pytest.mark.asyncio
    async def test_known_id_returns_200_and_model(self, client_with_model):
        client, model_id = client_with_model
        resp = await client.get(f"/api/models/{model_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == model_id

    @pytest.mark.asyncio
    async def test_unknown_id_returns_404(self, client):
        resp = await client.get("/api/models/does-not-exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestUpdateModel
# ---------------------------------------------------------------------------

class TestUpdateModel:
    @pytest.mark.asyncio
    async def test_valid_update_returns_200_with_changed_fields(self, client_with_model):
        client, model_id = client_with_model
        updated = {**_VALID_CONFIG, "name": "updated-name", "epochs": 10}
        resp = await client.put(f"/api/models/{model_id}", json=updated)
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "updated-name"
        assert body["epochs"] == 10

    @pytest.mark.asyncio
    async def test_unknown_id_returns_404(self, client):
        resp = await client.put("/api/models/does-not-exist", json=_VALID_CONFIG)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_updated_at_advances_but_created_at_unchanged(self, client_with_model):
        client, model_id = client_with_model
        original = (await client.get(f"/api/models/{model_id}")).json()

        await asyncio.sleep(0.01)
        resp = await client.put(f"/api/models/{model_id}", json={**_VALID_CONFIG, "name": "new-name"})
        body = resp.json()

        assert body["created_at"] == original["created_at"]
        assert body["updated_at"] > original["updated_at"]


# ---------------------------------------------------------------------------
# TestDeleteModel
# ---------------------------------------------------------------------------

class TestDeleteModel:
    @pytest.mark.asyncio
    async def test_known_id_returns_204_and_model_is_gone(self, client_with_model):
        client, model_id = client_with_model
        resp = await client.delete(f"/api/models/{model_id}")
        assert resp.status_code == 204

        get_resp = await client.get(f"/api/models/{model_id}")
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_unknown_id_returns_404(self, client):
        resp = await client.delete("/api/models/does-not-exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestTriggerRun
# ---------------------------------------------------------------------------

def _make_temporal_mock(workflow_id: str = "training-test-model-abc12345") -> MagicMock:
    handle = MagicMock()
    handle.id = workflow_id

    client_mock = AsyncMock()
    client_mock.start_workflow = AsyncMock(return_value=handle)

    connect_mock = AsyncMock(return_value=client_mock)
    return connect_mock, client_mock


class TestTriggerRun:
    @pytest.mark.asyncio
    async def test_valid_model_triggers_workflow_and_returns_202(self, client_with_model):
        client, model_id = client_with_model
        connect_mock, temporal_client = _make_temporal_mock()

        with patch("temporalio.client.Client.connect", connect_mock):
            resp = await client.post(f"/api/models/{model_id}/trigger")

        assert resp.status_code == 202
        body = resp.json()
        assert "workflow_id" in body
        temporal_client.start_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_model_id_returns_404(self, client):
        resp = await client.post("/api/models/does-not-exist/trigger")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_temporal_error_returns_500(self, client_with_model):
        client, model_id = client_with_model
        connect_mock = AsyncMock(side_effect=RuntimeError("Temporal unavailable"))

        with patch("temporalio.client.Client.connect", connect_mock):
            resp = await client.post(f"/api/models/{model_id}/trigger")

        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# TestListRuns
# ---------------------------------------------------------------------------

class TestListRuns:
    @pytest.mark.asyncio
    async def test_returns_workflow_list_from_temporal(self, client):
        wf = MagicMock()
        wf.id = "training-foo-abc12345"
        wf.run_id = "run-uuid"
        wf.status = MagicMock()
        wf.status.name = "RUNNING"
        wf.start_time = None
        wf.close_time = None

        async def _fake_list(*_args, **_kwargs):
            yield wf

        temporal_client = AsyncMock()
        temporal_client.list_workflows = _fake_list
        connect_mock = AsyncMock(return_value=temporal_client)

        with patch("temporalio.client.Client.connect", connect_mock):
            resp = await client.get("/api/runs")

        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 1
        assert runs[0]["workflow_id"] == "training-foo-abc12345"
        assert runs[0]["status"] == "RUNNING"


# ---------------------------------------------------------------------------
# TestGetRun
# ---------------------------------------------------------------------------

class TestGetRun:
    @pytest.mark.asyncio
    async def test_known_workflow_id_returns_status(self, client):
        desc = MagicMock()
        desc.run_id = "run-uuid"
        desc.status = MagicMock()
        desc.status.name = "COMPLETED"
        desc.start_time = None
        desc.close_time = None

        handle = AsyncMock()
        handle.describe = AsyncMock(return_value=desc)

        temporal_client = AsyncMock()
        temporal_client.get_workflow_handle = MagicMock(return_value=handle)
        connect_mock = AsyncMock(return_value=temporal_client)

        with patch("temporalio.client.Client.connect", connect_mock):
            resp = await client.get("/api/runs/training-foo-abc12345")

        assert resp.status_code == 200
        assert resp.json()["status"] == "COMPLETED"

    @pytest.mark.asyncio
    async def test_temporal_error_returns_404(self, client):
        connect_mock = AsyncMock(side_effect=RuntimeError("not found"))

        with patch("temporalio.client.Client.connect", connect_mock):
            resp = await client.get("/api/runs/does-not-exist")

        assert resp.status_code == 404
