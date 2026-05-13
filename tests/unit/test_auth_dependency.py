"""Unit tests for the require_auth FastAPI dependency."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from domain.models import UserContext
from domain.ports import AuthPort
from interactors.api.deps import configure_auth


class _StubAuthPort(AuthPort):
    def __init__(self, result: UserContext | None) -> None:
        self._result = result

    def authenticate(self, token: str) -> UserContext | None:
        return self._result


_VALID_USER = UserContext(user_id="u1", email="u@example.com")


def _make_client(auth_port: AuthPort) -> TestClient:
    from interactors.api.auth import require_auth

    configure_auth(auth_port)
    test_app = FastAPI()

    @test_app.get("/protected", dependencies=[Depends(require_auth)])
    def protected() -> dict:
        return {"ok": True}

    return TestClient(test_app, raise_server_exceptions=False)


class TestRequireAuth:
    def test_missing_header_returns_401(self):
        client = _make_client(_StubAuthPort(_VALID_USER))
        assert client.get("/protected").status_code == 401

    def test_non_bearer_scheme_returns_401(self):
        client = _make_client(_StubAuthPort(_VALID_USER))
        assert client.get("/protected", headers={"Authorization": "ApiKey abc"}).status_code == 401

    def test_bearer_with_no_token_returns_401(self):
        client = _make_client(_StubAuthPort(_VALID_USER))
        assert client.get("/protected", headers={"Authorization": "Bearer "}).status_code == 401

    def test_invalid_token_returns_401(self):
        client = _make_client(_StubAuthPort(None))
        assert client.get("/protected", headers={"Authorization": "Bearer bad"}).status_code == 401

    def test_valid_token_returns_200(self):
        client = _make_client(_StubAuthPort(_VALID_USER))
        assert client.get("/protected", headers={"Authorization": "Bearer valid"}).status_code == 200

    def test_valid_token_response_body(self):
        client = _make_client(_StubAuthPort(_VALID_USER))
        response = client.get("/protected", headers={"Authorization": "Bearer valid"})
        assert response.json() == {"ok": True}
