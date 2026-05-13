"""FastAPI dependency for Auth0 JWT authentication."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException

from domain.ports import AuthPort
from interactors.api.deps import get_auth


def require_auth(
    authorization: str | None = Header(default=None),
    auth_port: AuthPort = Depends(get_auth),
) -> None:
    if authorization is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Not authenticated")
    if auth_port.authenticate(token) is None:
        raise HTTPException(status_code=401, detail="Invalid token")
