"""Generic SQLAlchemy CRUD repository."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Generic, Type, TypeVar

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

TRow = TypeVar("TRow")
TDomain = TypeVar("TDomain")
TConfig = TypeVar("TConfig")


class CRUDRepository(Generic[TRow, TDomain, TConfig]):
    """Generic CRUD repository backed by SQLAlchemy.

    Convention: every ORM model used here must have `id`, `created_at`,
    and `updated_at` columns. Config objects must be Pydantic BaseModels.
    """

    def __init__(
        self,
        engine: Engine,
        row_class: Type[TRow],
        to_domain: Callable[[TRow], TDomain],
        order_by: Any | None = None,
    ) -> None:
        self._engine = engine
        self._row_class = row_class
        self._to_domain = to_domain
        self._order_by = order_by

    def list(self) -> list[TDomain]:
        with Session(self._engine) as db:
            stmt = select(self._row_class)
            if self._order_by is not None:
                stmt = stmt.order_by(self._order_by)
            rows = db.scalars(stmt).all()
            return [self._to_domain(r) for r in rows]

    def get(self, id: str) -> TDomain | None:
        with Session(self._engine) as db:
            row = db.get(self._row_class, id)
            return self._to_domain(row) if row else None

    def create(self, config: TConfig) -> TDomain:
        now = datetime.now(timezone.utc)
        row = self._row_class(
            id=str(uuid.uuid4()),
            created_at=now,
            updated_at=now,
            **config.model_dump(),  # type: ignore[union-attr]
        )
        with Session(self._engine) as db:
            db.add(row)
            db.commit()
            db.refresh(row)
            return self._to_domain(row)

    def update(self, id: str, config: TConfig) -> TDomain | None:
        with Session(self._engine) as db:
            row = db.get(self._row_class, id)
            if row is None:
                return None
            for field, value in config.model_dump().items():  # type: ignore[union-attr]
                setattr(row, field, value)
            setattr(row, "updated_at", datetime.now(timezone.utc))
            db.commit()
            db.refresh(row)
            return self._to_domain(row)

    def delete(self, id: str) -> bool:
        with Session(self._engine) as db:
            row = db.get(self._row_class, id)
            if row is None:
                return False
            db.delete(row)
            db.commit()
            return True
