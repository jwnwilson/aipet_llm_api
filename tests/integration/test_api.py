"""Integration tests for the aipet FastAPI application."""
from __future__ import annotations

import pytest
import pytest_asyncio
import httpx
from httpx import ASGITransport

from api.app import app, configure
from domain.actions import Action
from domain.models import InferenceRequest, InferenceResponse
from domain.ports import InferencePort


class FakeInferenceAdapter(InferencePort):
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(action=Action.IDLE, target_object_id=None)


class ErrorInferenceAdapter(InferencePort):
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        raise RuntimeError("inference error")


VALID_PAYLOAD = {
    "scene": {"objects": [], "tick": 1},
    "pet_stats": {
        "hunger": 0.5,
        "boredom": 0.3,
        "social": 0.2,
        "toilet": 0.1,
        "tiredness": 0.4,
    },
}


@pytest_asyncio.fixture
async def client():
    configure(FakeInferenceAdapter())
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest_asyncio.fixture
async def error_client():
    configure(ErrorInferenceAdapter())
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


class TestInferEndpoint:
    @pytest.mark.asyncio
    async def test_valid_request_returns_200_with_valid_schema(self, client):
        response = await client.post("/infer", json=VALID_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        resp = InferenceResponse(**body)
        assert resp.action == Action.IDLE

    @pytest.mark.asyncio
    async def test_malformed_request_returns_422(self, client):
        response = await client.post("/infer", json={"bad": "data"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_adapter_exception_returns_500(self, error_client):
        response = await error_client.post("/infer", json=VALID_PAYLOAD)
        assert response.status_code == 500
        assert response.json()["detail"]["error"] == "inference_failed"


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "model" in body
