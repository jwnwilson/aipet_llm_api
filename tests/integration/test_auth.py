"""Integration tests — auth enforced on all routes except GET /health."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from domain.actions import Action
from domain.models import InferenceRequest, InferenceResponse, UserContext
from domain.ports import AuthPort, InferencePort
from interactors.api.app import app
from interactors.api.app import app
from interactors.api.deps import clear_auth, configure, configure_auth

VALID_TOKEN = "valid-test-token"

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


class _FakeInferenceAdapter(InferencePort):
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(action=Action.IDLE)


class _FakeAuthAdapter(AuthPort):
    def authenticate(self, token: str) -> UserContext | None:
        if token == VALID_TOKEN:
            return UserContext(user_id="u1", email="u@example.com", roles=["user"])
        return None


@pytest.fixture(autouse=True)
def _auth_bypass():
    from interactors.api.auth import require_admin, require_approved
    app.dependency_overrides.pop(require_approved, None)
    app.dependency_overrides.pop(require_admin, None)
    configure_auth(_FakeAuthAdapter())
    yield
    clear_auth()
    app.dependency_overrides[require_approved] = lambda: None
    app.dependency_overrides[require_admin] = lambda: None


@pytest_asyncio.fixture
async def client():
    configure(_FakeInferenceAdapter())
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


VALID_HEADERS = {"Authorization": f"Bearer {VALID_TOKEN}"}


class TestHealthIsPublic:
    @pytest.mark.asyncio
    async def test_no_auth_returns_200(self, client):
        assert (await client.get("/health")).status_code == 200

    @pytest.mark.asyncio
    async def test_with_valid_auth_returns_200(self, client):
        assert (await client.get("/health", headers=VALID_HEADERS)).status_code == 200


class TestInferRequiresAuth:
    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client):
        assert (await client.post("/infer", json=VALID_PAYLOAD)).status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_token_returns_401(self, client):
        resp = await client.post(
            "/infer", json=VALID_PAYLOAD, headers={"Authorization": "Bearer wrong"}
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_token_returns_200(self, client):
        resp = await client.post("/infer", json=VALID_PAYLOAD, headers=VALID_HEADERS)
        assert resp.status_code == 200


class TestModelsRequiresAuth:
    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client):
        assert (await client.get("/api/models")).status_code == 401

    @pytest.mark.asyncio
    async def test_no_auth_on_post_returns_401(self, client):
        assert (await client.post("/api/models", json={})).status_code == 401


class TestRunsRequiresAuth:
    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client):
        assert (await client.get("/api/runs")).status_code == 401

    @pytest.mark.asyncio
    async def test_no_auth_on_get_by_id_returns_401(self, client):
        assert (await client.get("/api/runs/some-id")).status_code == 401
