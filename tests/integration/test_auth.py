"""Integration tests — auth enforced on all routes except GET /health."""

from __future__ import annotations

import pytest
import pytest_asyncio
import httpx
from httpx import ASGITransport

from domain.actions import Action
from domain.models import InferenceRequest, InferenceResponse, UserContext
from domain.ports import AuthPort, InferencePort, UserStorePort
from interactors.api.app import app
from interactors.api.deps import configure, configure_auth, configure_user_store, clear_user_store

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
            return UserContext(user_id="u1", email="u@example.com")
        return None


class _FakeUserStore(UserStorePort):
    def is_approved(self, user_id: str) -> bool:
        return user_id == "u1"

    def approve(self, user_id: str, email: str | None = None) -> None:
        pass

    def list_approved(self) -> list[UserContext]:
        return []

    def revoke(self, user_id: str) -> None:
        pass


@pytest.fixture(autouse=True)
def _auth_bypass():
    # Override the conftest _auth_bypass: remove dependency override and
    # use real (fake) adapters so auth is actually enforced.
    from interactors.api.auth import require_approved
    from interactors.api.deps import clear_auth
    app.dependency_overrides.pop(require_approved, None)
    configure_auth(_FakeAuthAdapter())
    configure_user_store(_FakeUserStore())
    yield
    clear_auth()
    clear_user_store()
    app.dependency_overrides[require_approved] = lambda: None


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
