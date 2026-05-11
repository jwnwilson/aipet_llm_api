"""SQLAlchemy implementation of ModelStorePort."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, mapped_column

from domain.models import TrainingModel, TrainingModelConfig
from domain.ports import ModelStorePort
from infrastructure.database import Base
from infrastructure.database.crud import CRUDRepository


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
        self._crud: CRUDRepository[_TrainingModelRow, TrainingModel, TrainingModelConfig] = CRUDRepository(
            engine=engine,
            row_class=_TrainingModelRow,
            to_domain=_row_to_domain,
            order_by=_TrainingModelRow.created_at.desc(),
        )

    def list(self) -> list[TrainingModel]:
        return self._crud.list()

    def get(self, id: str) -> TrainingModel | None:
        return self._crud.get(id)

    def create(self, config: TrainingModelConfig) -> TrainingModel:
        return self._crud.create(config)

    def update(self, id: str, config: TrainingModelConfig) -> TrainingModel | None:
        return self._crud.update(id, config)

    def delete(self, id: str) -> bool:
        return self._crud.delete(id)
