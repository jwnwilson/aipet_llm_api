#!/usr/bin/env python3
"""Post-deploy smoke test — validates the live API endpoints."""

from __future__ import annotations

import os
import subprocess
import sys

import httpx


def require_env(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        print(f"ERROR: {name} environment variable is required", file=sys.stderr)
        sys.exit(1)
    return val


def check(label: str, resp: httpx.Response, expected_status: int = 200) -> dict:
    print(f"-- {label}...")
    if resp.status_code != expected_status:
        print(f"ERROR: expected HTTP {expected_status}, got {resp.status_code}", file=sys.stderr)
        print(resp.text, file=sys.stderr)
        sys.exit(1)
    return resp.json()


def main() -> None:
    print("=== Smoke Tests ===\n")

    api_url = require_env("API_URL").rstrip("/")
    auth0_domain = require_env("AUTH0_DOMAIN")
    auth0_client_id = require_env("AUTH0_MGMT_CLIENT_ID")
    auth0_client_secret = require_env("AUTH0_MGMT_CLIENT_SECRET")
    auth0_audience = require_env("AUTH0_AUDIENCE")

    client = httpx.Client(timeout=30)

    # 1. Authenticate via Auth0 M2M client credentials
    token_url = f"https://{auth0_domain}/oauth/token"
    client_id_hint = auth0_client_id[:6] + "..." if len(auth0_client_id) > 6 else auth0_client_id
    print(f"-- Authenticating via Auth0...")
    print(f"   token_url : {token_url}")
    print(f"   client_id : {client_id_hint}")
    print(f"   audience  : {auth0_audience}")
    token_resp = client.post(
        token_url,
        json={
            "grant_type": "client_credentials",
            "client_id": auth0_client_id,
            "client_secret": auth0_client_secret,
            "audience": auth0_audience,
        },
    )
    print(f"   status    : {token_resp.status_code}")
    if token_resp.status_code != 200:
        print(f"ERROR: Auth0 token exchange failed ({token_resp.status_code})", file=sys.stderr)
        print(f"   response  : {token_resp.text}", file=sys.stderr)
        sys.exit(1)
    access_token = token_resp.json()["access_token"]
    auth_headers = {"Authorization": f"Bearer {access_token}"}
    print("OK — token acquired\n")

    # 2. Health check (no auth required)
    health = check("GET /health", client.get(f"{api_url}/health"))
    print(f"OK — status={health.get('status')}\n")

    # 3. Model listing
    models = check("GET /api/models", client.get(f"{api_url}/api/models", headers=auth_headers))
    print(f"OK — {len(models)} model(s) returned\n")

    # 4. Run listing
    runs = check("GET /api/runs", client.get(f"{api_url}/api/runs", headers=auth_headers))
    print(f"OK — {len(runs)} run(s) returned\n")

    # 5. Inference — minimal scene with a bowl so EAT is a valid candidate
    infer_payload = {
        "scene": {
            "objects": [{"id": "bowl1", "type": "bowl", "distance": 1.5}],
            "tick": 1,
        },
        "pet_stats": {
            "hunger": 0.8,
            "boredom": 0.3,
            "social": 0.2,
            "toilet": 0.1,
            "tiredness": 0.4,
        },
    }
    infer = check("POST /infer", client.post(f"{api_url}/infer", json=infer_payload, headers=auth_headers))
    print(f"OK — action={infer['action']}\n")

    # 6. Database tables via kubectl
    print("-- Checking database tables...")
    result = subprocess.run(
        [
            "kubectl", "exec", "aipet-db-0", "--",
            "psql", "-U", "aipet", "-d", "aipet", "-t", "-c",
            "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename;",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: kubectl exec failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    tables = result.stdout
    print(tables)
    for table in ("alembic_version", "training_models", "training_runs"):
        if table not in tables:
            print(f"ERROR: expected table '{table}' not found in database", file=sys.stderr)
            sys.exit(1)
    print("OK — all required tables present\n")

    print("=== Smoke tests passed ===")


if __name__ == "__main__":
    main()
