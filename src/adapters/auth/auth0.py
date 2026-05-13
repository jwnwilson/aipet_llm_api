"""Auth0 JWT validation adapter."""

from __future__ import annotations

import logging

import jwt

from domain.models import UserContext
from domain.ports import AuthPort

log = logging.getLogger(__name__)


class Auth0Adapter(AuthPort):
    def __init__(self, domain: str, audience: str) -> None:
        self._audience = audience
        self._issuer = f"https://{domain}/"
        self._jwks_client = jwt.PyJWKClient(
            f"https://{domain}/.well-known/jwks.json",
            cache_keys=True,
        )

    def authenticate(self, token: str) -> UserContext | None:
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self._audience,
                issuer=self._issuer,
            )
            sub = payload.get("sub")
            if sub is None:
                log.debug("JWT missing sub claim")
                return None
            return UserContext(
                user_id=sub,
                email=payload.get("email"),
            )
        except jwt.InvalidTokenError as exc:
            log.debug("JWT validation failed: %s", type(exc).__name__)
            return None
        except Exception:
            log.warning("Unexpected error validating JWT", exc_info=True)
            return None
