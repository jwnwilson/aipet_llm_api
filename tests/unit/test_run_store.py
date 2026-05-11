"""Unit tests for SQLAlchemyRunStore."""

from __future__ import annotations

import time

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from domain.models import RunConfig, RunStatus
from infrastructure.database import Base, init_db
from infrastructure.models.run_store import SQLAlchemyRunStore


@pytest.fixture()
def store() -> SQLAlchemyRunStore:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    return SQLAlchemyRunStore(engine)


def _config(model_id: str = "model-1", workflow_id: str = "wf-1") -> RunConfig:
    return RunConfig(model_id=model_id, workflow_id=workflow_id)


class TestCreate:
    def test_returns_run_record_with_pending_status(self, store):
        run = store.create(_config())
        assert run.status == RunStatus.PENDING

    def test_auto_generates_uuid(self, store):
        run = store.create(_config())
        assert run.id and len(run.id) == 36

    def test_persists_model_and_workflow_ids(self, store):
        run = store.create(_config(model_id="m1", workflow_id="wf-abc"))
        assert run.model_id == "m1"
        assert run.workflow_id == "wf-abc"

    def test_eval_valid_pct_is_none_initially(self, store):
        run = store.create(_config())
        assert run.eval_valid_pct is None

    def test_sets_timestamps(self, store):
        run = store.create(_config())
        assert run.created_at is not None
        assert run.updated_at is not None


class TestGet:
    def test_returns_run_by_id(self, store):
        created = store.create(_config())
        fetched = store.get(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    def test_returns_none_for_unknown_id(self, store):
        assert store.get("nonexistent-id") is None


class TestList:
    def test_returns_all_runs(self, store):
        store.create(_config(model_id="m1"))
        store.create(_config(model_id="m2"))
        runs = store.list()
        assert len(runs) == 2

    def test_returns_empty_list_when_no_runs(self, store):
        assert store.list() == []

    def test_filters_by_model_id(self, store):
        store.create(_config(model_id="m1"))
        store.create(_config(model_id="m1"))
        store.create(_config(model_id="m2"))
        runs = store.list(model_id="m1")
        assert len(runs) == 2
        assert all(r.model_id == "m1" for r in runs)

    def test_filtered_list_excludes_other_models(self, store):
        store.create(_config(model_id="m1"))
        store.create(_config(model_id="m2"))
        runs = store.list(model_id="m2")
        assert len(runs) == 1

    def test_ordered_newest_first(self, store):
        r1 = store.create(_config(model_id="m1", workflow_id="wf-1"))
        time.sleep(0.01)
        r2 = store.create(_config(model_id="m1", workflow_id="wf-2"))
        runs = store.list()
        assert runs[0].id == r2.id
        assert runs[1].id == r1.id


class TestUpdateStatus:
    def test_changes_status_to_completed(self, store):
        run = store.create(_config())
        updated = store.update_status(run.id, RunStatus.COMPLETED)
        assert updated is not None
        assert updated.status == RunStatus.COMPLETED

    def test_changes_status_to_failed(self, store):
        run = store.create(_config())
        updated = store.update_status(run.id, RunStatus.FAILED)
        assert updated is not None
        assert updated.status == RunStatus.FAILED

    def test_updates_updated_at(self, store):
        run = store.create(_config())
        original_ts = run.updated_at
        time.sleep(0.01)
        updated = store.update_status(run.id, RunStatus.COMPLETED)
        assert updated.updated_at > original_ts

    def test_returns_none_for_unknown_id(self, store):
        assert store.update_status("no-such-id", RunStatus.COMPLETED) is None


class TestUpdateEval:
    def test_sets_eval_valid_pct(self, store):
        run = store.create(_config())
        updated = store.update_eval(run.id, 0.97)
        assert updated is not None
        assert abs(updated.eval_valid_pct - 0.97) < 1e-6

    def test_updates_updated_at(self, store):
        run = store.create(_config())
        original_ts = run.updated_at
        time.sleep(0.01)
        updated = store.update_eval(run.id, 0.95)
        assert updated.updated_at > original_ts

    def test_returns_none_for_unknown_id(self, store):
        assert store.update_eval("no-such-id", 0.9) is None
