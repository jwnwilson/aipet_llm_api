"""Integration tests for the training management API endpoints."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from adapters.inference import LlamaCppInferenceAdapter

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from interactors.api.app import app
from interactors.api.deps import get_model_store, get_run_store
from adapters.database import Base, init_db
from adapters.database.model_store import SQLAlchemyModelStore
from adapters.database.run_store import SQLAlchemyRunStore

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
    init_db(engine)
    store = SQLAlchemyModelStore(engine)
    run_store = SQLAlchemyRunStore(engine)
    app.dependency_overrides[get_model_store] = lambda: store
    app.dependency_overrides[get_run_store] = lambda: run_store
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

        with (
            patch("temporalio.client.Client.connect", connect_mock),
            patch("pathlib.Path.mkdir"),
        ):
            resp = await client.post("/api/runs/trigger", json={"model_id": model_id})

        assert resp.status_code == 202
        body = resp.json()
        assert "workflow_id" in body
        assert "run_id" in body
        assert len(body["run_id"]) == 36  # UUID format
        temporal_client.start_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_model_id_returns_404(self, client):
        resp = await client.post("/api/runs/trigger", json={"model_id": "does-not-exist"})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_temporal_error_returns_500(self, client_with_model):
        client, model_id = client_with_model
        connect_mock = AsyncMock(side_effect=RuntimeError("Temporal unavailable"))

        with patch("temporalio.client.Client.connect", connect_mock):
            resp = await client.post("/api/runs/trigger", json={"model_id": model_id})

        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# TestListRuns
# ---------------------------------------------------------------------------

class TestListRuns:
    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_runs(self, client):
        resp = await client.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_returns_runs_from_db(self, client_with_model):
        client, model_id = client_with_model
        connect_mock, _ = _make_temporal_mock()

        with (
            patch("temporalio.client.Client.connect", connect_mock),
            patch("pathlib.Path.mkdir"),
        ):
            await client.post("/api/runs/trigger", json={"model_id": model_id})
            await client.post("/api/runs/trigger", json={"model_id": model_id})

        resp = await client.get("/api/runs")
        assert resp.status_code == 200
        assert len(resp.json()) == 2


# ---------------------------------------------------------------------------
# TestGetRun
# ---------------------------------------------------------------------------

class TestGetRun:
    @pytest.mark.asyncio
    async def test_known_run_id_returns_record(self, client_with_model):
        client, model_id = client_with_model
        connect_mock, _ = _make_temporal_mock()

        with (
            patch("temporalio.client.Client.connect", connect_mock),
            patch("pathlib.Path.mkdir"),
        ):
            resp = await client.post("/api/runs/trigger", json={"model_id": model_id})
        run_id = resp.json()["run_id"]

        get_resp = await client.get(f"/api/runs/{run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == run_id
        assert get_resp.json()["status"] == "pending"

    @pytest.mark.asyncio
    async def test_unknown_run_id_returns_404(self, client):
        resp = await client.get("/api/runs/does-not-exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestActivateModel
# ---------------------------------------------------------------------------

_GGUF_CONFIG = {**_VALID_CONFIG, "gguf_path": "s3/model.gguf"}


class TestActivateModel:
    @pytest.mark.asyncio
    async def test_unknown_model_returns_404(self, client):
        resp = await client.post("/api/models/does-not-exist/activate")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_model_without_gguf_returns_409_and_db_unchanged(self, client):
        resp = await client.post("/api/models", json=_VALID_CONFIG)
        model_id = resp.json()["id"]

        activate_resp = await client.post(f"/api/models/{model_id}/activate")
        assert activate_resp.status_code == 409

        # DB must NOT have been mutated — model must remain inactive
        model = (await client.get(f"/api/models/{model_id}")).json()
        assert model["is_active"] is False

    @pytest.mark.asyncio
    async def test_storage_failure_returns_500_and_db_unchanged(self, client):
        resp = await client.post("/api/models", json=_GGUF_CONFIG)
        model_id = resp.json()["id"]

        mock_storage = MagicMock()
        mock_storage.download.side_effect = RuntimeError("S3 error")

        with patch("interactors.temporal.activities._get_storage", return_value=mock_storage):
            activate_resp = await client.post(f"/api/models/{model_id}/activate")

        assert activate_resp.status_code == 500

        # DB must NOT have been mutated — model must remain inactive
        model = (await client.get(f"/api/models/{model_id}")).json()
        assert model["is_active"] is False

    @pytest.mark.asyncio
    async def test_successful_activation_sets_active_deactivates_others(self, client):
        r1 = await client.post("/api/models", json={**_GGUF_CONFIG, "name": "model-1"})
        model1_id = r1.json()["id"]
        r2 = await client.post("/api/models", json={**_GGUF_CONFIG, "name": "model-2"})
        model2_id = r2.json()["id"]

        mock_storage = MagicMock()
        mock_adapter = MagicMock()

        with (
            patch("interactors.temporal.activities._get_storage", return_value=mock_storage),
            patch("adapters.inference.LlamaCppInferenceAdapter", return_value=mock_adapter),
            patch("interactors.api.deps.get_adapter", side_effect=RuntimeError("no adapter")),
            patch("interactors.api.deps.configure"),
        ):
            resp = await client.post(f"/api/models/{model2_id}/activate")

        assert resp.status_code == 200
        assert resp.json()["id"] == model2_id

        model1 = (await client.get(f"/api/models/{model1_id}")).json()
        model2 = (await client.get(f"/api/models/{model2_id}")).json()
        assert model1["is_active"] is False
        assert model2["is_active"] is True

        mock_storage.download.assert_called_once()
        mock_adapter.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_successful_activation_releases_old_adapter(self, client):
        resp = await client.post("/api/models", json=_GGUF_CONFIG)
        model_id = resp.json()["id"]

        mock_storage = MagicMock()
        old_adapter = LlamaCppInferenceAdapter(model_path="/fake/old.gguf")
        old_adapter._llm = MagicMock()  # simulate loaded model in RAM

        with (
            patch("interactors.temporal.activities._get_storage", return_value=mock_storage),
            patch("adapters.inference.LlamaCppInferenceAdapter._load_model", return_value=MagicMock()),
            patch("adapters.inference.LlamaCppInferenceAdapter._get_llama_cpp", return_value=MagicMock()),
            patch("interactors.api.deps.get_adapter", return_value=old_adapter),
            patch("interactors.api.deps.configure"),
        ):
            resp = await client.post(f"/api/models/{model_id}/activate")

        assert resp.status_code == 200
        assert old_adapter._llm is None
