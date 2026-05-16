"""Admin endpoints for managing user access via Auth0 roles."""
from __future__ import annotations

import os
from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from adapters.auth.auth0_management import (
    assign_role,
    list_auth0_users,
    list_users_with_role,
    revoke_role,
)
from domain.models import UserContext
from interactors.api.auth import require_admin

router = APIRouter(prefix="/api/admin", tags=["admin"])

_ROLE_USER = "user"


def _mgmt_creds() -> tuple[str, str, str]:
    return (
        os.environ.get("AUTH0_DOMAIN", ""),
        os.environ.get("AUTH0_MGMT_CLIENT_ID", ""),
        os.environ.get("AUTH0_MGMT_CLIENT_SECRET", ""),
    )


class ApproveUserRequest(BaseModel):
    user_id: str
    email: str | None = None


@router.post("/users", status_code=201, dependencies=[Depends(require_admin)])
def approve_user(payload: ApproveUserRequest) -> dict:
    domain, client_id, client_secret = _mgmt_creds()
    assign_role(domain, client_id, client_secret, payload.user_id, _ROLE_USER)
    return {"approved": payload.user_id}


@router.get("/users", dependencies=[Depends(require_admin)])
def list_users(
    status: Literal["approved", "pending"] = Query(default="approved"),
) -> list[UserContext]:
    domain, client_id, client_secret = _mgmt_creds()
    if status == "approved":
        users = list_users_with_role(domain, client_id, client_secret, _ROLE_USER)
        return [
            UserContext(user_id=u["user_id"], email=u.get("email"), status="approved")
            for u in users
        ]
    all_users = list_auth0_users(domain, client_id, client_secret)
    approved_ids = {u["user_id"] for u in list_users_with_role(domain, client_id, client_secret, _ROLE_USER)}
    return [
        UserContext(user_id=u["user_id"], email=u.get("email"), status="pending")
        for u in all_users
        if u["user_id"] not in approved_ids
    ]


@router.delete("/users/{user_id}", status_code=204, dependencies=[Depends(require_admin)])
def revoke_user(user_id: str) -> None:
    domain, client_id, client_secret = _mgmt_creds()
    revoke_role(domain, client_id, client_secret, user_id, _ROLE_USER)
