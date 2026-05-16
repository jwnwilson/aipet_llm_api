"""Thin wrapper around the Auth0 Management API v2."""
from __future__ import annotations

import httpx


def list_auth0_users(domain: str, client_id: str, client_secret: str) -> list[dict]:
    """Return all Auth0 users as dicts with keys user_id and email.

    Fetches a short-lived M2M token using client credentials, then pages through
    GET /api/v2/users (100 per page) until exhausted.
    """
    token = _get_mgmt_token(domain, client_id, client_secret)
    users: list[dict] = []
    page = 0
    while True:
        resp = httpx.get(
            f"https://{domain}/api/v2/users",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "fields": "user_id,email",
                "include_fields": "true",
                "per_page": 100,
                "page": page,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        batch: list[dict] = resp.json()
        if not batch:
            break
        users.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return users


def list_users_with_role(domain: str, client_id: str, client_secret: str, role_name: str) -> list[dict]:
    """Return all users assigned the given role as dicts with keys user_id and email."""
    token = _get_mgmt_token(domain, client_id, client_secret)
    role_id = _get_role_id(token, domain, role_name)
    users: list[dict] = []
    page = 0
    while True:
        resp = httpx.get(
            f"https://{domain}/api/v2/roles/{role_id}/users",
            headers={"Authorization": f"Bearer {token}"},
            params={"per_page": 100, "page": page},
            timeout=10.0,
        )
        resp.raise_for_status()
        batch: list[dict] = resp.json()
        if not batch:
            break
        users.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return users


def assign_role(domain: str, client_id: str, client_secret: str, user_id: str, role_name: str) -> None:
    """Assign a role to a user by role name."""
    token = _get_mgmt_token(domain, client_id, client_secret)
    role_id = _get_role_id(token, domain, role_name)
    resp = httpx.post(
        f"https://{domain}/api/v2/users/{user_id}/roles",
        headers={"Authorization": f"Bearer {token}"},
        json={"roles": [role_id]},
        timeout=10.0,
    )
    resp.raise_for_status()


def revoke_role(domain: str, client_id: str, client_secret: str, user_id: str, role_name: str) -> None:
    """Remove a role from a user by role name."""
    token = _get_mgmt_token(domain, client_id, client_secret)
    role_id = _get_role_id(token, domain, role_name)
    resp = httpx.delete(
        f"https://{domain}/api/v2/users/{user_id}/roles",
        headers={"Authorization": f"Bearer {token}"},
        json={"roles": [role_id]},
        timeout=10.0,
    )
    resp.raise_for_status()


def _get_role_id(token: str, domain: str, role_name: str) -> str:
    resp = httpx.get(
        f"https://{domain}/api/v2/roles",
        headers={"Authorization": f"Bearer {token}"},
        params={"name_filter": role_name},
        timeout=10.0,
    )
    resp.raise_for_status()
    roles = resp.json()
    for role in roles:
        if role["name"] == role_name:
            return role["id"]
    raise ValueError(f"Auth0 role '{role_name}' not found")


def _get_mgmt_token(domain: str, client_id: str, client_secret: str) -> str:
    resp = httpx.post(
        f"https://{domain}/oauth/token",
        json={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "audience": f"https://{domain}/api/v2/",
        },
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]
