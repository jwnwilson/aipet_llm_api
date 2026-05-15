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
