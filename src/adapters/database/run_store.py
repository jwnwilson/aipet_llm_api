"""SQLAlchemy implementation of RunStorePort."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Float, String, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from domain.models import RunConfig, RunRecord, RunStatus
from domain.ports import RunStorePort
from adapters.database import Base


class _RunRow(Base):
    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(36), nullable=False)
    workflow_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    eval_valid_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    updated_at: Mapped[datetime] = mapped_column(nullable=False)


def _row_to_domain(row: _RunRow) -> RunRecord:
    return RunRecord(
        id=row.id,
        model_id=row.model_id,
        workflow_id=row.workflow_id,
        status=RunStatus(row.status),
        eval_valid_pct=row.eval_valid_pct,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SQLAlchemyRunStore(RunStorePort):
    """RunStorePort backed by a SQLAlchemy-managed relational database."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def create(self, config: RunConfig) -> RunRecord:
        now = datetime.now(timezone.utc)
        row = _RunRow(
            id=str(uuid.uuid4()),
            model_id=config.model_id,
            workflow_id=config.workflow_id,
            status=RunStatus.PENDING.value,
            eval_valid_pct=None,
            created_at=now,
            updated_at=now,
        )
        with Session(self._engine) as db:
            db.add(row)
            db.commit()
            db.refresh(row)
            return _row_to_domain(row)

    def get(self, id: str) -> RunRecord | None:
        with Session(self._engine) as db:
            row = db.get(_RunRow, id)
            return _row_to_domain(row) if row else None

    def list(self, model_id: str | None = None) -> list[RunRecord]:  # type: ignore[override]
        with Session(self._engine) as db:
            stmt = select(_RunRow)
            if model_id is not None:
                stmt = stmt.where(_RunRow.model_id == model_id)
            stmt = stmt.order_by(_RunRow.created_at.desc())
            rows = db.scalars(stmt).all()
            return [_row_to_domain(r) for r in rows]

    def update(self, id: str, config: RunConfig) -> RunRecord | None:
        with Session(self._engine) as db:
            row = db.get(_RunRow, id)
            if row is None:
                return None
            row.model_id = config.model_id
            row.workflow_id = config.workflow_id
            row.updated_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(row)
            return _row_to_domain(row)

    def delete(self, id: str) -> bool:
        with Session(self._engine) as db:
            row = db.get(_RunRow, id)
            if row is None:
                return False
            db.delete(row)
            db.commit()
            return True

    def update_status(self, run_id: str, status: RunStatus) -> RunRecord | None:
        with Session(self._engine) as db:
            row = db.get(_RunRow, run_id)
            if row is None:
                return None
            row.status = status.value
            row.updated_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(row)
            return _row_to_domain(row)

    def update_eval(self, run_id: str, valid_pct: float) -> RunRecord | None:
        with Session(self._engine) as db:
            row = db.get(_RunRow, run_id)
            if row is None:
                return None
            row.eval_valid_pct = valid_pct
            row.updated_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(row)
            return _row_to_domain(row)
