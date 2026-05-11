"""SQLAlchemy implementation of ModelStorePort."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, mapped_column, Mapped

from domain.models import TrainingModel, TrainingModelConfig
from domain.ports import ModelStorePort
from infrastructure.database import Base


class _TrainingModelRow(Base):
    __tablename__ = "training_models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    base_model: Mapped[str] = mapped_column(String(255), nullable=False)
    train_data: Mapped[str] = mapped_column(String(512), nullable=False)
    eval_data: Mapped[str] = mapped_column(String(512), nullable=False)
    epochs: Mapped[int] = mapped_column(Integer, nullable=False)
    patience: Mapped[int] = mapped_column(Integer, nullable=False)
    warmup_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    remote_backend: Mapped[str] = mapped_column(String(64), nullable=False)
    skip_generate: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def _row_to_domain(row: _TrainingModelRow) -> TrainingModel:
    return TrainingModel(
        id=row.id,
        name=row.name,
        description=row.description,
        base_model=row.base_model,
        train_data=row.train_data,
        eval_data=row.eval_data,
        epochs=row.epochs,
        patience=row.patience,
        warmup_ratio=row.warmup_ratio,
        remote_backend=row.remote_backend,
        skip_generate=row.skip_generate,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SQLAlchemyModelStore(ModelStorePort):
    """ModelStorePort backed by a SQLAlchemy-managed relational database."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def list(self) -> list[TrainingModel]:
        with Session(self._engine) as db:
            rows = db.scalars(
                select(_TrainingModelRow).order_by(_TrainingModelRow.created_at.desc())
            ).all()
            return [_row_to_domain(r) for r in rows]

    def get(self, id: str) -> TrainingModel | None:
        with Session(self._engine) as db:
            row = db.get(_TrainingModelRow, id)
            return _row_to_domain(row) if row else None

    def create(self, config: TrainingModelConfig) -> TrainingModel:
        now = datetime.now(timezone.utc)
        row = _TrainingModelRow(
            id=str(uuid.uuid4()),
            name=config.name,
            description=config.description,
            base_model=config.base_model,
            train_data=config.train_data,
            eval_data=config.eval_data,
            epochs=config.epochs,
            patience=config.patience,
            warmup_ratio=config.warmup_ratio,
            remote_backend=config.remote_backend,
            skip_generate=config.skip_generate,
            created_at=now,
            updated_at=now,
        )
        with Session(self._engine) as db:
            db.add(row)
            db.commit()
            db.refresh(row)
            return _row_to_domain(row)

    def update(self, id: str, config: TrainingModelConfig) -> TrainingModel | None:
        with Session(self._engine) as db:
            row = db.get(_TrainingModelRow, id)
            if row is None:
                return None
            row.name = config.name
            row.description = config.description
            row.base_model = config.base_model
            row.train_data = config.train_data
            row.eval_data = config.eval_data
            row.epochs = config.epochs
            row.patience = config.patience
            row.warmup_ratio = config.warmup_ratio
            row.remote_backend = config.remote_backend
            row.skip_generate = config.skip_generate
            row.updated_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(row)
            return _row_to_domain(row)

    def delete(self, id: str) -> bool:
        with Session(self._engine) as db:
            row = db.get(_TrainingModelRow, id)
            if row is None:
                return False
            db.delete(row)
            db.commit()
            return True
