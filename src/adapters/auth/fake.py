"""Fake auth adapter for local development — accepts any bearer token."""

from __future__ import annotations

from domain.models import UserContext
from domain.ports import AuthPort


class FakeAuthAdapter(AuthPort):
    def authenticate(self, token: str) -> UserContext | None:
        if not token.strip():
            return None
        return UserContext(user_id="dev-user", email="dev@localhost")
