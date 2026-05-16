"""Unit tests for require_approved and require_admin dependencies."""
from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from domain.models import UserContext
from domain.ports import AuthPort
from interactors.api.auth import require_admin, require_approved
from interactors.api.deps import configure_auth


class _StubAuth(AuthPort):
    def __init__(self, result: UserContext | None) -> None:
        self._result = result

    def authenticate(self, token: str) -> UserContext | None:
        return self._result


def _make_client(auth_result: UserContext | None, route_dep) -> TestClient:
    configure_auth(_StubAuth(auth_result))
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(route_dep)])
    def protected() -> dict:
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


USER = UserContext(user_id="auth0|abc", email="user@example.com", roles=["user"])
ADMIN = UserContext(user_id="auth0|abc", email="admin@example.com", roles=["user", "admin"])
NO_ROLE = UserContext(user_id="auth0|abc", email="user@example.com", roles=[])


class TestRequireApproved:
    def test_missing_header_returns_401(self) -> None:
        client = _make_client(USER, require_approved)
        assert client.get("/protected").status_code == 401

    def test_invalid_token_returns_401(self) -> None:
        client = _make_client(None, require_approved)
        assert client.get("/protected", headers={"Authorization": "Bearer bad"}).status_code == 401

    def test_no_user_role_returns_403(self) -> None:
        client = _make_client(NO_ROLE, require_approved)
        assert client.get("/protected", headers={"Authorization": "Bearer tok"}).status_code == 403

    def test_user_role_returns_200(self) -> None:
        client = _make_client(USER, require_approved)
        assert client.get("/protected", headers={"Authorization": "Bearer tok"}).status_code == 200

    def test_admin_role_returns_200(self) -> None:
        client = _make_client(ADMIN, require_approved)
        assert client.get("/protected", headers={"Authorization": "Bearer tok"}).status_code == 200


class TestRequireAdmin:
    def test_missing_header_returns_401(self) -> None:
        client = _make_client(ADMIN, require_admin)
        assert client.get("/protected").status_code == 401

    def test_invalid_token_returns_401(self) -> None:
        client = _make_client(None, require_admin)
        assert client.get("/protected", headers={"Authorization": "Bearer bad"}).status_code == 401

    def test_no_admin_role_returns_403(self) -> None:
        client = _make_client(USER, require_admin)
        assert client.get("/protected", headers={"Authorization": "Bearer tok"}).status_code == 403

    def test_admin_role_returns_200(self) -> None:
        client = _make_client(ADMIN, require_admin)
        assert client.get("/protected", headers={"Authorization": "Bearer tok"}).status_code == 200
