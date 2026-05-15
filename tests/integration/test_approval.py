"""Integration tests — require_approved enforces allowlist on all protected routes."""
from __future__ import annotations

import pytest
import pytest_asyncio
import httpx
from httpx import ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from adapters.database import Base, init_db
from adapters.database.model_store import SQLAlchemyModelStore
from adapters.database.run_store import SQLAlchemyRunStore
from domain.models import UserContext
from domain.ports import AuthPort, UserStorePort
from interactors.api.app import app
from interactors.api.deps import (
    configure_auth,
    configure_user_store,
    get_model_store,
    get_run_store,
    get_user_store,
)

VALID_TOKEN = "valid-token"
VALID_USER = UserContext(user_id="auth0|testuser", email="test@example.com")


class _FakeAuth(AuthPort):
    def authenticate(self, token: str) -> UserContext | None:
        return VALID_USER if token == VALID_TOKEN else None


class _InMemoryUserStore(UserStorePort):
    def __init__(self) -> None:
        self._approved: set[str] = set()

    def is_approved(self, user_id: str) -> bool:
        return user_id in self._approved

    def approve(self, user_id: str, email: str | None = None) -> None:
        self._approved.add(user_id)

    def list_approved(self) -> list[UserContext]:
        return [UserContext(user_id=uid, status="approved") for uid in self._approved]

    def revoke(self, user_id: str) -> None:
        self._approved.discard(user_id)


@pytest.fixture(autouse=True)
def _setup():
    from interactors.api.auth import require_approved
    from interactors.api.deps import clear_auth, clear_user_store
    app.dependency_overrides.pop(require_approved, None)
    configure_auth(_FakeAuth())
    configure_user_store(_InMemoryUserStore())
    yield
    clear_auth()
    clear_user_store()
    app.dependency_overrides[require_approved] = lambda: None


@pytest_asyncio.fixture
async def client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    app.dependency_overrides[get_model_store] = lambda: SQLAlchemyModelStore(engine)
    app.dependency_overrides[get_run_store] = lambda: SQLAlchemyRunStore(engine)
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.pop(get_model_store, None)
    app.dependency_overrides.pop(get_run_store, None)


VALID_HEADERS = {"Authorization": f"Bearer {VALID_TOKEN}"}


class TestUnapprovedUser:
    @pytest.mark.asyncio
    async def test_models_returns_403(self, client) -> None:
        assert (await client.get("/api/models", headers=VALID_HEADERS)).status_code == 403

    @pytest.mark.asyncio
    async def test_runs_returns_403(self, client) -> None:
        assert (await client.get("/api/runs", headers=VALID_HEADERS)).status_code == 403

    @pytest.mark.asyncio
    async def test_no_token_returns_401(self, client) -> None:
        assert (await client.get("/api/models")).status_code == 401


class TestApprovedUser:
    @pytest.mark.asyncio
    async def test_models_returns_200_when_approved(self, client) -> None:
        get_user_store().approve(VALID_USER.user_id)
        assert (await client.get("/api/models", headers=VALID_HEADERS)).status_code == 200

    @pytest.mark.asyncio
    async def test_runs_returns_200_when_approved(self, client) -> None:
        get_user_store().approve(VALID_USER.user_id)
        assert (await client.get("/api/runs", headers=VALID_HEADERS)).status_code == 200


class TestAdminEndpoint:
    @pytest.mark.asyncio
    async def test_no_token_returns_401(self, client) -> None:
        resp = await client.post("/api/admin/users", json={"user_id": "auth0|x"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_token_returns_401(self, client) -> None:
        resp = await client.post(
            "/api/admin/users",
            json={"user_id": "auth0|x"},
            headers={"Authorization": "Bearer bad-token"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_approve_user(self, client) -> None:
        resp = await client.post(
            "/api/admin/users",
            json={"user_id": "auth0|new", "email": "new@example.com"},
            headers=VALID_HEADERS,
        )
        assert resp.status_code == 201
        assert get_user_store().is_approved("auth0|new")

    @pytest.mark.asyncio
    async def test_list_approved_users(self, client) -> None:
        get_user_store().approve("auth0|existing", "existing@example.com")
        resp = await client.get("/api/admin/users", headers=VALID_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert any(u["user_id"] == "auth0|existing" for u in data)
        assert all(u["status"] == "approved" for u in data)

    @pytest.mark.asyncio
    async def test_revoke_user(self, client) -> None:
        get_user_store().approve("auth0|todelete")
        resp = await client.delete(
            "/api/admin/users/auth0%7Ctodelete",
            headers=VALID_HEADERS,
        )
        assert resp.status_code == 204
        assert not get_user_store().is_approved("auth0|todelete")


class TestListPendingUsers:
    @pytest.mark.asyncio
    async def test_returns_only_unapproved_users(self, client, monkeypatch) -> None:
        get_user_store().approve("auth0|alpha")

        import interactors.api.routes.admin as admin_module
        monkeypatch.setattr(
            admin_module,
            "list_auth0_users",
            lambda domain, client_id, client_secret: [
                {"user_id": "auth0|alpha", "email": "alpha@example.com"},
                {"user_id": "auth0|beta", "email": "beta@example.com"},
            ],
        )

        resp = await client.get("/api/admin/users?status=pending", headers=VALID_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["user_id"] == "auth0|beta"
        assert data[0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_no_token_returns_401(self, client) -> None:
        resp = await client.get("/api/admin/users?status=pending")
        assert resp.status_code == 401
