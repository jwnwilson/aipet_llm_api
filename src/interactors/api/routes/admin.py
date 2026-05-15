"""Admin endpoints for managing the approved-users allowlist."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from domain.models import UserContext
from domain.ports import UserStorePort
from interactors.api.deps import get_user_store

router = APIRouter(prefix="/api/admin", tags=["admin"])


class ApproveUserRequest(BaseModel):
    user_id: str
    email: str | None = None


def _require_admin(x_admin_secret: str | None = Header(default=None)) -> None:
    secret = os.environ.get("ADMIN_SECRET", "")
    if not secret or x_admin_secret != secret:
        raise HTTPException(status_code=401, detail="Invalid or missing admin secret")


@router.post("/users", status_code=201, dependencies=[Depends(_require_admin)])
def approve_user(
    payload: ApproveUserRequest,
    user_store: UserStorePort = Depends(get_user_store),
) -> dict:
    user_store.approve(payload.user_id, payload.email)
    return {"approved": payload.user_id}


@router.get("/users", dependencies=[Depends(_require_admin)])
def list_approved_users(
    user_store: UserStorePort = Depends(get_user_store),
) -> list[UserContext]:
    return user_store.list_approved()


@router.delete("/users/{user_id}", status_code=204, dependencies=[Depends(_require_admin)])
def revoke_user(
    user_id: str,
    user_store: UserStorePort = Depends(get_user_store),
) -> None:
    user_store.revoke(user_id)