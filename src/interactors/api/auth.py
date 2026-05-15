"""FastAPI dependencies for Auth0 JWT authentication and user approval."""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException

from domain.models import UserContext
from domain.ports import AuthPort, UserStorePort
from interactors.api.deps import get_auth, get_user_store

_WWW_AUTH = {"WWW-Authenticate": "Bearer"}


def require_auth(
    authorization: str | None = Header(default=None),
    auth_port: AuthPort = Depends(get_auth),
) -> None:
    if authorization is None:
        raise HTTPException(status_code=401, detail="Not authenticated", headers=_WWW_AUTH)
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Not authenticated", headers=_WWW_AUTH)
    if auth_port.authenticate(token) is None:
        raise HTTPException(status_code=401, detail="Invalid token", headers=_WWW_AUTH)


def get_current_user(
    authorization: str | None = Header(default=None),
    auth_port: AuthPort = Depends(get_auth),
) -> UserContext:
    if authorization is None:
        raise HTTPException(status_code=401, detail="Not authenticated", headers=_WWW_AUTH)
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Not authenticated", headers=_WWW_AUTH)
    user = auth_port.authenticate(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token", headers=_WWW_AUTH)
    return user


def require_approved(
    user: UserContext = Depends(get_current_user),
    user_store: UserStorePort = Depends(get_user_store),
) -> UserContext:
    if not user_store.is_approved(user.user_id):
        raise HTTPException(
            status_code=403,
            detail="Access not approved. Contact an administrator.",
        )
    return user