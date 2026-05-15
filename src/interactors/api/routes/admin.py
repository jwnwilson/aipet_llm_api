"""Admin endpoints for managing the approved-users allowlist."""
from __future__ import annotations

import os
from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from domain.models import UserContext
from domain.ports import UserStorePort
from interactors.api.auth import require_auth
from interactors.api.deps import get_user_store

router = APIRouter(prefix="/api/admin", tags=["admin"])


class ApproveUserRequest(BaseModel):
    user_id: str
    email: str | None = None


@router.post("/users", status_code=201, dependencies=[Depends(require_auth)])
def approve_user(
    payload: ApproveUserRequest,
    user_store: UserStorePort = Depends(get_user_store),
) -> dict:
    user_store.approve(payload.user_id, payload.email)
    return {"approved": payload.user_id}


@router.get("/users", dependencies=[Depends(require_auth)])
def list_users(
    status: Literal["approved", "pending"] = Query(default="approved"),
    user_store: UserStorePort = Depends(get_user_store),
) -> list[UserContext]:
    if status == "pending":
        from adapters.auth.auth0_management import list_auth0_users

        domain = os.environ.get("AUTH0_DOMAIN", "")
        client_id = os.environ.get("AUTH0_CLIENT_ID", "")
        client_secret = os.environ.get("AUTH0_CLIENT_SECRET", "")
        auth0_users = list_auth0_users(domain, client_id, client_secret)
        approved_ids = {u.user_id for u in user_store.list_approved()}
        return [
            UserContext(user_id=u["user_id"], email=u.get("email"), status="pending")
            for u in auth0_users
            if u["user_id"] not in approved_ids
        ]
    return user_store.list_approved()


@router.delete("/users/{user_id}", status_code=204, dependencies=[Depends(require_auth)])
def revoke_user(
    user_id: str,
    user_store: UserStorePort = Depends(get_user_store),
) -> None:
    user_store.revoke(user_id)
