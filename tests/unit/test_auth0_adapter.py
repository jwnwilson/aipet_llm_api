"""Unit tests for Auth0Adapter JWT validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from adapters.auth.auth0 import Auth0Adapter
from domain.models import UserContext

DOMAIN = "test.auth0.com"
AUDIENCE = "https://api.aipet.test"

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PUBLIC_KEY = _PRIVATE_KEY.public_key()


def _make_token(
    sub: str = "auth0|abc123",
    email: str | None = "user@example.com",
    expired: bool = False,
    audience: str = AUDIENCE,
) -> str:
    now = datetime.now(timezone.utc)
    exp = now - timedelta(hours=1) if expired else now + timedelta(hours=1)
    claims: dict = {
        "sub": sub,
        "iss": f"https://{DOMAIN}/",
        "aud": audience,
        "iat": now,
        "exp": exp,
    }
    if email is not None:
        claims["email"] = email
    return jwt.encode(claims, _PRIVATE_KEY, algorithm="RS256")


@pytest.fixture
def adapter():
    mock_signing_key = MagicMock()
    mock_signing_key.key = _PUBLIC_KEY
    mock_jwks_client = MagicMock()
    mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
    with patch("adapters.auth.auth0.jwt.PyJWKClient", return_value=mock_jwks_client):
        yield Auth0Adapter(domain=DOMAIN, audience=AUDIENCE)


class TestAuthenticate:
    def test_valid_token_returns_user_context(self, adapter):
        result = adapter.authenticate(_make_token())
        assert isinstance(result, UserContext)
        assert result.user_id == "auth0|abc123"

    def test_includes_email_when_present(self, adapter):
        result = adapter.authenticate(_make_token(email="user@example.com"))
        assert result is not None
        assert result.email == "user@example.com"

    def test_email_is_none_when_absent(self, adapter):
        result = adapter.authenticate(_make_token(email=None))
        assert result is not None
        assert result.email is None

    def test_expired_token_returns_none(self, adapter):
        assert adapter.authenticate(_make_token(expired=True)) is None

    def test_wrong_audience_returns_none(self, adapter):
        assert adapter.authenticate(_make_token(audience="https://other-api.example.com")) is None

    def test_tampered_token_returns_none(self, adapter):
        token = _make_token()
        tampered = token[:-10] + "x" * 10
        assert adapter.authenticate(tampered) is None

    def test_garbage_string_returns_none(self, adapter):
        assert adapter.authenticate("not.a.jwt") is None

    def test_token_without_sub_returns_none(self, adapter):
        now = datetime.now(timezone.utc)
        claims = {
            "iss": f"https://{DOMAIN}/",
            "aud": AUDIENCE,
            "iat": now,
            "exp": now + timedelta(hours=1),
            # no "sub"
        }
        no_sub_token = jwt.encode(claims, _PRIVATE_KEY, algorithm="RS256")
        assert adapter.authenticate(no_sub_token) is None
