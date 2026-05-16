"""Integration tests — require_approved and require_admin enforce Auth0 roles."""
from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from adapters.database import Base, init_db
from adapters.database.model_store import SQLAlchemyModelStore
from adapters.database.run_store import SQLAlchemyRunStore
from domain.models import UserContext
from domain.ports import AuthPort
from interactors.api.app import app
from interactors.api.deps import configure_auth, get_model_store, get_run_store

USER_TOKEN = "user-token"
ADMIN_TOKEN = "admin-token"
NO_ROLE_TOKEN = "no-role-token"

USER = UserContext(user_id="auth0|user", email="user@example.com", roles=["user"])
ADMIN = UserContext(user_id="auth0|admin", email="admin@example.com", roles=["user", "admin"])
NO_ROLE = UserContext(user_id="auth0|norole", email="norole@example.com", roles=[])


class _FakeAuth(AuthPort):
    def authenticate(self, token: str) -> UserContext | None:
        return {USER_TOKEN: USER, ADMIN_TOKEN: ADMIN, NO_ROLE_TOKEN: NO_ROLE}.get(token)


@pytest.fixture(autouse=True)
def _setup():
    from interactors.api.auth import require_admin, require_approved
    from interactors.api.deps import clear_auth
    app.dependency_overrides.pop(require_approved, None)
    app.dependency_overrides.pop(require_admin, None)
    configure_auth(_FakeAuth())
    yield
    clear_auth()
    app.dependency_overrides[require_approved] = lambda: None
    app.dependency_overrides[require_admin] = lambda: None


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


USER_HEADERS = {"Authorization": f"Bearer {USER_TOKEN}"}
ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
NO_ROLE_HEADERS = {"Authorization": f"Bearer {NO_ROLE_TOKEN}"}


class TestUnapprovedUser:
    @pytest.mark.asyncio
    async def test_models_returns_403_without_user_role(self, client) -> None:
        assert (await client.get("/api/models", headers=NO_ROLE_HEADERS)).status_code == 403

    @pytest.mark.asyncio
    async def test_runs_returns_403_without_user_role(self, client) -> None:
        assert (await client.get("/api/runs", headers=NO_ROLE_HEADERS)).status_code == 403

    @pytest.mark.asyncio
    async def test_no_token_returns_401(self, client) -> None:
        assert (await client.get("/api/models")).status_code == 401


class TestApprovedUser:
    @pytest.mark.asyncio
    async def test_models_returns_200_with_user_role(self, client) -> None:
        assert (await client.get("/api/models", headers=USER_HEADERS)).status_code == 200

    @pytest.mark.asyncio
    async def test_runs_returns_200_with_user_role(self, client) -> None:
        assert (await client.get("/api/runs", headers=USER_HEADERS)).status_code == 200


class TestAdminEndpoint:
    @pytest.mark.asyncio
    async def test_no_token_returns_401(self, client) -> None:
        assert (await client.post("/api/admin/users", json={"user_id": "auth0|x"})).status_code == 401

    @pytest.mark.asyncio
    async def test_user_role_only_returns_403(self, client) -> None:
        resp = await client.post(
            "/api/admin/users",
            json={"user_id": "auth0|x"},
            headers=USER_HEADERS,
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_approve_user_calls_assign_role(self, client, monkeypatch) -> None:
        import interactors.api.routes.admin as admin_module
        assigned: list[tuple] = []
        monkeypatch.setattr(admin_module, "assign_role", lambda *a, **kw: assigned.append(a))
        resp = await client.post(
            "/api/admin/users",
            json={"user_id": "auth0|new"},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 201
        assert any("auth0|new" in call for call in assigned)

    @pytest.mark.asyncio
    async def test_list_approved_returns_users_with_role(self, client, monkeypatch) -> None:
        import interactors.api.routes.admin as admin_module
        monkeypatch.setattr(
            admin_module,
            "list_users_with_role",
            lambda *a, **kw: [{"user_id": "auth0|alpha", "email": "alpha@example.com"}],
        )
        resp = await client.get("/api/admin/users", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert any(u["user_id"] == "auth0|alpha" for u in data)
        assert all(u["status"] == "approved" for u in data)

    @pytest.mark.asyncio
    async def test_revoke_user_calls_revoke_role(self, client, monkeypatch) -> None:
        import interactors.api.routes.admin as admin_module
        revoked: list[tuple] = []
        monkeypatch.setattr(admin_module, "revoke_role", lambda *a, **kw: revoked.append(a))
        resp = await client.delete("/api/admin/users/auth0%7Ctodelete", headers=ADMIN_HEADERS)
        assert resp.status_code == 204
        assert any("auth0|todelete" in call for call in revoked)


class TestListPendingUsers:
    @pytest.mark.asyncio
    async def test_returns_only_users_without_role(self, client, monkeypatch) -> None:
        import interactors.api.routes.admin as admin_module
        monkeypatch.setattr(
            admin_module,
            "list_auth0_users",
            lambda *a, **kw: [
                {"user_id": "auth0|alpha", "email": "alpha@example.com"},
                {"user_id": "auth0|beta", "email": "beta@example.com"},
            ],
        )
        monkeypatch.setattr(
            admin_module,
            "list_users_with_role",
            lambda *a, **kw: [{"user_id": "auth0|alpha"}],
        )
        resp = await client.get("/api/admin/users?status=pending", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["user_id"] == "auth0|beta"
        assert data[0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_no_token_returns_401(self, client) -> None:
        assert (await client.get("/api/admin/users?status=pending")).status_code == 401
