"""Direct integration tests for CRUDRepository — real SQLite in-memory, no mocking."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel
from sqlalchemy import DateTime, Integer, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

from infrastructure.database.crud import CRUDRepository

# ---------------------------------------------------------------------------
# Minimal test schema — isolated from the application's Base
# ---------------------------------------------------------------------------


class _TestBase(DeclarativeBase):
    pass


class _WidgetRow(_TestBase):
    __tablename__ = "widgets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WidgetConfig(BaseModel):
    label: str
    count: int = 0


class Widget(BaseModel):
    id: str
    label: str
    count: int
    created_at: datetime
    updated_at: datetime


def _to_widget(row: _WidgetRow) -> Widget:
    return Widget(
        id=row.id,
        label=row.label,
        count=row.count,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> Engine:
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    _TestBase.metadata.create_all(eng)
    return eng


@pytest.fixture
def repo(engine: Engine) -> CRUDRepository[_WidgetRow, Widget, WidgetConfig]:
    return CRUDRepository(
        engine=engine,
        row_class=_WidgetRow,
        to_domain=_to_widget,
        order_by=_WidgetRow.created_at.desc(),
    )


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


class TestList:
    def test_empty_returns_empty_list(self, repo: CRUDRepository) -> None:
        assert repo.list() == []

    def test_returns_all_items(self, repo: CRUDRepository) -> None:
        repo.create(WidgetConfig(label="a"))
        repo.create(WidgetConfig(label="b"))
        assert len(repo.list()) == 2

    def test_ordered_newest_first(self, repo: CRUDRepository) -> None:
        repo.create(WidgetConfig(label="first"))
        time.sleep(0.01)
        repo.create(WidgetConfig(label="second"))
        items = repo.list()
        assert items[0].label == "second"
        assert items[1].label == "first"


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


class TestGet:
    def test_unknown_id_returns_none(self, repo: CRUDRepository) -> None:
        assert repo.get("does-not-exist") is None

    def test_known_id_returns_correct_item(self, repo: CRUDRepository) -> None:
        created = repo.create(WidgetConfig(label="hello", count=7))
        fetched = repo.get(created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.label == "hello"
        assert fetched.count == 7


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


class TestCreate:
    def test_returns_item_with_auto_uuid(self, repo: CRUDRepository) -> None:
        w = repo.create(WidgetConfig(label="x"))
        assert len(w.id) == 36  # xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

    def test_each_create_gets_unique_id(self, repo: CRUDRepository) -> None:
        ids = {repo.create(WidgetConfig(label=str(i))).id for i in range(5)}
        assert len(ids) == 5

    def test_config_fields_are_persisted(self, repo: CRUDRepository) -> None:
        w = repo.create(WidgetConfig(label="stored", count=42))
        assert w.label == "stored"
        assert w.count == 42

    def test_timestamps_are_set_within_call_window(self, repo: CRUDRepository) -> None:
        # SQLite returns timezone-naive datetimes; strip tz before comparing
        def _naive(dt: datetime) -> datetime:
            return dt.replace(tzinfo=None)

        before = _naive(datetime.now(timezone.utc))
        w = repo.create(WidgetConfig(label="ts"))
        after = _naive(datetime.now(timezone.utc))
        assert before <= _naive(w.created_at) <= after
        assert before <= _naive(w.updated_at) <= after

    def test_created_at_equals_updated_at_on_fresh_row(self, repo: CRUDRepository) -> None:
        w = repo.create(WidgetConfig(label="eq"))
        assert w.created_at == w.updated_at

    def test_item_is_retrievable_after_create(self, repo: CRUDRepository) -> None:
        w = repo.create(WidgetConfig(label="persist"))
        assert repo.get(w.id) is not None


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_unknown_id_returns_none(self, repo: CRUDRepository) -> None:
        assert repo.update("does-not-exist", WidgetConfig(label="x")) is None

    def test_config_fields_are_updated(self, repo: CRUDRepository) -> None:
        w = repo.create(WidgetConfig(label="old", count=1))
        updated = repo.update(w.id, WidgetConfig(label="new", count=99))
        assert updated is not None
        assert updated.label == "new"
        assert updated.count == 99

    def test_updated_at_advances_after_update(self, repo: CRUDRepository) -> None:
        w = repo.create(WidgetConfig(label="x"))
        time.sleep(0.01)
        updated = repo.update(w.id, WidgetConfig(label="y"))
        assert updated.updated_at > w.updated_at

    def test_created_at_is_unchanged_after_update(self, repo: CRUDRepository) -> None:
        w = repo.create(WidgetConfig(label="x"))
        updated = repo.update(w.id, WidgetConfig(label="y"))
        assert updated.created_at == w.created_at

    def test_change_is_persisted(self, repo: CRUDRepository) -> None:
        w = repo.create(WidgetConfig(label="orig"))
        repo.update(w.id, WidgetConfig(label="changed"))
        assert repo.get(w.id).label == "changed"

    def test_other_items_unaffected(self, repo: CRUDRepository) -> None:
        a = repo.create(WidgetConfig(label="a"))
        b = repo.create(WidgetConfig(label="b"))
        repo.update(a.id, WidgetConfig(label="a-updated"))
        assert repo.get(b.id).label == "b"


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


class TestDelete:
    def test_unknown_id_returns_false(self, repo: CRUDRepository) -> None:
        assert repo.delete("does-not-exist") is False

    def test_known_id_returns_true(self, repo: CRUDRepository) -> None:
        w = repo.create(WidgetConfig(label="del"))
        assert repo.delete(w.id) is True

    def test_deleted_item_not_retrievable(self, repo: CRUDRepository) -> None:
        w = repo.create(WidgetConfig(label="del"))
        repo.delete(w.id)
        assert repo.get(w.id) is None

    def test_deleted_item_absent_from_list(self, repo: CRUDRepository) -> None:
        w = repo.create(WidgetConfig(label="del"))
        repo.delete(w.id)
        assert all(x.id != w.id for x in repo.list())

    def test_second_delete_returns_false(self, repo: CRUDRepository) -> None:
        w = repo.create(WidgetConfig(label="del"))
        repo.delete(w.id)
        assert repo.delete(w.id) is False

    def test_other_items_unaffected(self, repo: CRUDRepository) -> None:
        a = repo.create(WidgetConfig(label="a"))
        b = repo.create(WidgetConfig(label="b"))
        repo.delete(a.id)
        assert repo.get(b.id) is not None
        assert len(repo.list()) == 1
