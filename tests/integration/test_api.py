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


class TrackingInferenceAdapter(InferencePort):
    """Spy adapter — records every call and returns a configurable response."""

    def __init__(self, response: InferenceResponse) -> None:
        self._response = response
        self.call_count = 0
        self.last_request: InferenceRequest | None = None

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        self.call_count += 1
        self.last_request = request
        return self._response


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

PAYLOAD_WITH_BOWL = {
    "scene": {
        "objects": [{"id": "bowl1", "type": "bowl", "distance": 1.5}],
        "tick": 2,
    },
    "pet_stats": {
        "hunger": 0.9,
        "boredom": 0.1,
        "social": 0.1,
        "toilet": 0.1,
        "tiredness": 0.1,
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


# ---------------------------------------------------------------------------
# Adapter invocation tests — verify the route calls adapter.infer() and that
# its result is forwarded verbatim (no silent fallback substitution).
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def tracking_client():
    """Client wired with a TrackingInferenceAdapter returning IDLE."""
    adapter = TrackingInferenceAdapter(
        InferenceResponse(action=Action.IDLE, target_object_id=None)
    )
    configure(adapter)
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c, adapter


@pytest_asyncio.fixture
async def eat_tracking_client():
    """Client wired with a TrackingInferenceAdapter returning EAT→bowl1."""
    adapter = TrackingInferenceAdapter(
        InferenceResponse(action=Action.EAT, target_object_id="bowl1")
    )
    configure(adapter)
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c, adapter


class TestAdapterInvocation:
    """Verify the /infer route calls adapter.infer() and forwards its result."""

    @pytest.mark.asyncio
    async def test_adapter_infer_is_called(self, tracking_client):
        client, adapter = tracking_client
        await client.post("/infer", json=VALID_PAYLOAD)
        assert adapter.call_count == 1

    @pytest.mark.asyncio
    async def test_adapter_called_once_per_request(self, tracking_client):
        client, adapter = tracking_client
        for _ in range(3):
            await client.post("/infer", json=VALID_PAYLOAD)
        assert adapter.call_count == 3

    @pytest.mark.asyncio
    async def test_adapter_receives_correct_request(self, tracking_client):
        client, adapter = tracking_client
        await client.post("/infer", json=PAYLOAD_WITH_BOWL)
        req = adapter.last_request
        assert req is not None
        assert req.pet_stats.hunger == pytest.approx(0.9)
        assert req.scene.tick == 2
        assert len(req.scene.objects) == 1
        assert req.scene.objects[0].id == "bowl1"

    @pytest.mark.asyncio
    async def test_non_idle_response_forwarded_verbatim(self, eat_tracking_client):
        # If the adapter returns EAT the route must not substitute the IDLE fallback.
        client, adapter = eat_tracking_client
        response = await client.post("/infer", json=PAYLOAD_WITH_BOWL)
        assert response.status_code == 200
        body = response.json()
        assert body["action"] == "EAT"
        assert body["target_object_id"] == "bowl1"
        assert adapter.call_count == 1

    @pytest.mark.asyncio
    async def test_adapter_result_matches_response_body(self, eat_tracking_client):
        # Round-trip: the HTTP response must deserialise to exactly what the adapter returned.
        client, adapter = eat_tracking_client
        response = await client.post("/infer", json=PAYLOAD_WITH_BOWL)
        resp = InferenceResponse(**response.json())
        assert resp.action == adapter._response.action
        assert resp.target_object_id == adapter._response.target_object_id
