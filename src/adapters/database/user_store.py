"""SQLAlchemy implementation of UserStorePort."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, String, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from adapters.database import Base
from domain.models import UserContext
from domain.ports import UserStorePort


class _ApprovedUserRow(Base):
    __tablename__ = "approved_users"

    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    approved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SQLAlchemyUserStore(UserStorePort):
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def is_approved(self, user_id: str) -> bool:
        with Session(self._engine) as db:
            return db.get(_ApprovedUserRow, user_id) is not None

    def approve(self, user_id: str, email: str | None = None) -> None:
        with Session(self._engine) as db:
            existing = db.get(_ApprovedUserRow, user_id)
            if existing is None:
                db.add(_ApprovedUserRow(
                    user_id=user_id,
                    email=email,
                    approved_at=datetime.now(timezone.utc),
                ))
            elif email is not None:
                existing.email = email
            db.commit()

    def list_approved(self) -> list[UserContext]:
        with Session(self._engine) as db:
            rows = db.scalars(select(_ApprovedUserRow)).all()
            return [UserContext(user_id=r.user_id, email=r.email) for r in rows]

    def revoke(self, user_id: str) -> None:
        with Session(self._engine) as db:
            row = db.get(_ApprovedUserRow, user_id)
            if row is not None:
                db.delete(row)
                db.commit()