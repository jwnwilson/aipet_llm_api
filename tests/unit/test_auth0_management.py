"""Unit tests for auth0_management helpers."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adapters.auth.auth0_management import revoke_role

DOMAIN = "test.auth0.com"
CLIENT_ID = "client_id"
CLIENT_SECRET = "client_secret"
USER_ID = "auth0|abc123"
ROLE_NAME = "user"
ROLE_ID = "rol_xyz"


def _mock_response(json_data, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


def _make_httpx_mock(token: str, role_id: str):
    """Return a side-effect function for httpx.post/get/request calls."""

    def side_effect(method_or_url, url=None, **kwargs):
        # httpx.request("DELETE", url, ...) — method is positional
        if isinstance(method_or_url, str) and method_or_url.upper() == "DELETE":
            target_url = url
        else:
            target_url = method_or_url

        if "/oauth/token" in str(target_url):
            return _mock_response({"access_token": token})
        if "/api/v2/roles" in str(target_url) and "/users" not in str(target_url):
            return _mock_response([{"id": role_id, "name": ROLE_NAME}])
        if f"/api/v2/users/{USER_ID}/roles" in str(target_url):
            return _mock_response({})
        raise ValueError(f"Unexpected URL: {target_url}")

    return side_effect


class TestRevokeRole:
    def test_revoke_role_sends_delete_with_json_body(self):
        """Regression: httpx.delete() does not accept json=; must use httpx.request()."""
        token = "mgmt_token"
        calls = []

        def fake_post(url, **kwargs):
            calls.append(("POST", url, kwargs))
            if "/oauth/token" in url:
                return _mock_response({"access_token": token})
            raise ValueError(url)

        def fake_get(url, **kwargs):
            calls.append(("GET", url, kwargs))
            if "/api/v2/roles" in url:
                return _mock_response([{"id": ROLE_ID, "name": ROLE_NAME}])
            raise ValueError(url)

        def fake_request(method, url, **kwargs):
            calls.append((method, url, kwargs))
            return _mock_response({})

        with (
            patch("adapters.auth.auth0_management.httpx.post", side_effect=fake_post),
            patch("adapters.auth.auth0_management.httpx.get", side_effect=fake_get),
            patch("adapters.auth.auth0_management.httpx.request", side_effect=fake_request),
        ):
            revoke_role(DOMAIN, CLIENT_ID, CLIENT_SECRET, USER_ID, ROLE_NAME)

        delete_calls = [c for c in calls if c[0].upper() == "DELETE"]
        assert len(delete_calls) == 1, "Expected exactly one DELETE call"

        method, url, kwargs = delete_calls[0]
        assert f"/api/v2/users/{USER_ID}/roles" in url
        assert kwargs.get("json") == {"roles": [ROLE_ID]}

    def test_revoke_role_uses_correct_role_id(self):
        """Role lookup must translate role name to ID before revoking."""
        token = "tok"
        captured = {}

        def fake_post(url, **kwargs):
            if "/oauth/token" in url:
                return _mock_response({"access_token": token})
            raise ValueError(url)

        def fake_get(url, **kwargs):
            if "/api/v2/roles" in url:
                return _mock_response([{"id": "rol_correct", "name": ROLE_NAME}])
            raise ValueError(url)

        def fake_request(method, url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _mock_response({})

        with (
            patch("adapters.auth.auth0_management.httpx.post", side_effect=fake_post),
            patch("adapters.auth.auth0_management.httpx.get", side_effect=fake_get),
            patch("adapters.auth.auth0_management.httpx.request", side_effect=fake_request),
        ):
            revoke_role(DOMAIN, CLIENT_ID, CLIENT_SECRET, USER_ID, ROLE_NAME)

        assert captured["json"] == {"roles": ["rol_correct"]}
