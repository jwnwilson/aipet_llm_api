"""SQLAlchemy engine, declarative base, and FastAPI session dependency."""

from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

_DEFAULT_DB = "sqlite:///data/aipet.db"


class Base(DeclarativeBase):
    pass


def make_engine(url: str | None = None) -> Engine:
    url = url or os.getenv("DATABASE_URL", _DEFAULT_DB)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def init_db(engine: Engine) -> None:
    """Initialise the module-level engine and create all tables."""
    global _engine, _SessionLocal
    _engine = engine
    _SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session per request."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
