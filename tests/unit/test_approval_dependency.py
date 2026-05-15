"""Unit tests for get_current_user and require_approved dependencies."""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from domain.models import UserContext
from domain.ports import AuthPort, UserStorePort
from interactors.api.auth import get_current_user, require_approved
from interactors.api.deps import configure_auth, configure_user_store


class _StubAuth(AuthPort):
    def __init__(self, result: UserContext | None) -> None:
        self._result = result

    def authenticate(self, token: str) -> UserContext | None:
        return self._result


class _StubUserStore(UserStorePort):
    def __init__(self, approved_ids: set[str]) -> None:
        self._approved = approved_ids

    def is_approved(self, user_id: str) -> bool:
        return user_id in self._approved

    def approve(self, user_id: str, email: str | None = None) -> None:
        self._approved.add(user_id)

    def list_approved(self) -> list[UserContext]:
        return []

    def revoke(self, user_id: str) -> None:
        self._approved.discard(user_id)


VALID_USER = UserContext(user_id="auth0|abc", email="user@example.com")


def _make_client(auth_result: UserContext | None, approved_ids: set[str]) -> TestClient:
    configure_auth(_StubAuth(auth_result))
    configure_user_store(_StubUserStore(approved_ids))
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_approved)])
    def protected() -> dict:
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


class TestRequireApproved:
    def test_missing_header_returns_401(self) -> None:
        client = _make_client(VALID_USER, {"auth0|abc"})
        assert client.get("/protected").status_code == 401

    def test_invalid_token_returns_401(self) -> None:
        client = _make_client(None, set())
        assert client.get("/protected", headers={"Authorization": "Bearer bad"}).status_code == 401

    def test_authenticated_but_unapproved_returns_403(self) -> None:
        client = _make_client(VALID_USER, set())
        assert client.get("/protected", headers={"Authorization": "Bearer good"}).status_code == 403

    def test_approved_user_returns_200(self) -> None:
        client = _make_client(VALID_USER, {"auth0|abc"})
        assert client.get("/protected", headers={"Authorization": "Bearer good"}).status_code == 200