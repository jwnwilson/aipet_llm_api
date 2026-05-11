"""Alembic environment — connects to the database and runs migrations."""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Add src/ to path so domain/adapters imports resolve regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from adapters.database.engine import Base
import adapters.database.model_store  # noqa: F401 — registers _TrainingModelRow with Base
import adapters.database.run_store    # noqa: F401 — registers _RunRow with Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        config.get_main_option("sqlalchemy.url", "sqlite:///data/aipet.db"),
    )


def run_migrations_offline() -> None:
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        {"sqlalchemy.url": _get_url()},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
