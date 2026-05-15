"""Unit tests for SQLAlchemyUserStore."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from adapters.database import Base, init_db
from adapters.database.user_store import SQLAlchemyUserStore


@pytest.fixture
def store() -> SQLAlchemyUserStore:
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    return SQLAlchemyUserStore(engine)


class TestIsApproved:
    def test_unknown_user_is_not_approved(self, store: SQLAlchemyUserStore) -> None:
        assert store.is_approved("auth0|unknown") is False

    def test_approved_user_is_approved(self, store: SQLAlchemyUserStore) -> None:
        store.approve("auth0|abc", "user@example.com")
        assert store.is_approved("auth0|abc") is True

    def test_revoked_user_is_not_approved(self, store: SQLAlchemyUserStore) -> None:
        store.approve("auth0|abc")
        store.revoke("auth0|abc")
        assert store.is_approved("auth0|abc") is False


class TestApprove:
    def test_approve_creates_record(self, store: SQLAlchemyUserStore) -> None:
        store.approve("auth0|abc", "user@example.com")
        users = store.list_approved()
        assert len(users) == 1
        assert users[0].user_id == "auth0|abc"
        assert users[0].email == "user@example.com"

    def test_approve_is_idempotent(self, store: SQLAlchemyUserStore) -> None:
        store.approve("auth0|abc")
        store.approve("auth0|abc")
        assert len(store.list_approved()) == 1

    def test_approve_without_email(self, store: SQLAlchemyUserStore) -> None:
        store.approve("auth0|abc")
        users = store.list_approved()
        assert users[0].email is None


class TestRevoke:
    def test_revoke_removes_user(self, store: SQLAlchemyUserStore) -> None:
        store.approve("auth0|abc")
        store.revoke("auth0|abc")
        assert store.list_approved() == []

    def test_revoke_unknown_user_is_noop(self, store: SQLAlchemyUserStore) -> None:
        store.revoke("auth0|nonexistent")  # must not raise