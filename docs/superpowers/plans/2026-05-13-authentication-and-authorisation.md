# Authentication & Authorisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add API key authentication and configurable CORS to protect all endpoints, with a key management CLI and env-var-based key seeding.

**Architecture:** Three-tier extension of the existing pattern — domain model + port (`ApiKey`, `ApiKeyPort`), SQLAlchemy adapter (`SQLAlchemyApiKeyStore`), and a FastAPI dependency (`require_api_key`) applied per-route on `/infer` and at router level on `/api/models` and `/api/runs`. `GET /health` stays unauthenticated. Keys are seeded from `API_KEYS` env var on startup; CORS origins come from `CORS_ORIGINS` env var (wildcarded in `APP_ENV=development`).

**Tech Stack:** FastAPI `Depends`, `hashlib` sha256, SQLAlchemy ORM, Alembic migration, `argparse` CLI

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/domain/models.py` | Add `ApiKey` Pydantic model |
| Modify | `src/domain/ports.py` | Add `ApiKeyPort` abstract interface |
| Create | `src/adapters/database/alembic/versions/0004_add_api_keys.py` | DB migration — `api_keys` table |
| Create | `src/adapters/database/api_key_store.py` | `SQLAlchemyApiKeyStore` implementing `ApiKeyPort` |
| Create | `src/interactors/api/auth.py` | `require_api_key` FastAPI dependency |
| Modify | `src/interactors/api/deps.py` | Add `configure_api_key_store` / `get_api_key_store` |
| Modify | `src/interactors/api/app.py` | Init store + seed keys + update CORS |
| Modify | `src/interactors/api/routes/inference.py` | Add auth to `/infer` route only |
| Modify | `src/interactors/api/routes/models.py` | Add auth at router level |
| Modify | `src/interactors/api/routes/runs.py` | Add auth at router level |
| Create | `src/interactors/cli/manage_keys.py` | `create` / `list` / `revoke` subcommands |
| Create | `tests/unit/test_api_key_store.py` | Unit tests for the store (in-memory SQLite) |
| Create | `tests/unit/test_auth_dependency.py` | Unit tests for `require_api_key` |
| Create | `tests/integration/test_auth.py` | Integration tests — full request cycle |
| Create | `tests/cli/test_manage_keys.py` | CLI tests |

---

## Task 1: ApiKey domain model and ApiKeyPort

**Files:**
- Modify: `src/domain/models.py`
- Modify: `src/domain/ports.py`

Pure type definitions — no I/O, no tests needed.

- [ ] **Step 1: Add `ApiKey` to `src/domain/models.py`**

Append after the `RunRecord` class (line ~99):

```python
class ApiKey(BaseModel):
    key_hash: str
    label: str
    created_at: datetime
    is_active: bool
```

- [ ] **Step 2: Add `ApiKeyPort` to `src/domain/ports.py`**

Add the import at the top with the other model imports:
```python
from domain.models import (
    ApiKey,
    InferenceRequest,
    # ... existing imports ...
)
```

Append after the `RunStorePort` class:

```python
class ApiKeyPort(ABC):
    """Abstract interface for storing and validating API keys."""

    @abstractmethod
    def lookup(self, key_hash: str) -> "ApiKey | None":
        """Return the ApiKey whose hash matches, or None."""

    @abstractmethod
    def create(self, label: str, key_hash: str) -> "ApiKey":
        """Store a new key and return it."""

    @abstractmethod
    def list_keys(self) -> "list[ApiKey]":
        """Return all stored keys."""

    @abstractmethod
    def revoke(self, label: str) -> bool:
        """Set is_active=False for the key with this label. Return True if found."""
```

- [ ] **Step 3: Commit**

```bash
git add src/domain/models.py src/domain/ports.py
git commit -m "feat: add ApiKey domain model and ApiKeyPort interface"
```

---

## Task 2: Alembic migration for api_keys table

**Files:**
- Create: `src/adapters/database/alembic/versions/0004_add_api_keys.py`

- [ ] **Step 1: Write the migration**

```python
"""Add api_keys table.

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-13
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("key_hash", sa.String(64), nullable=False),
        sa.Column("label", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("key_hash"),
        sa.UniqueConstraint("label"),
    )


def downgrade() -> None:
    op.drop_table("api_keys")
```

- [ ] **Step 2: Verify the migration runs**

```bash
DATABASE_URL=sqlite:///data/test_migration.db uv run alembic upgrade head
```

Expected: `Running upgrade 0003 -> 0004, Add api_keys table`

```bash
rm data/test_migration.db
```

- [ ] **Step 3: Commit**

```bash
git add src/adapters/database/alembic/versions/0004_add_api_keys.py
git commit -m "feat: add api_keys table migration"
```

---

## Task 3: SQLAlchemyApiKeyStore adapter

**Files:**
- Create: `src/adapters/database/api_key_store.py`
- Create: `tests/unit/test_api_key_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_api_key_store.py`:

```python
"""Unit tests for SQLAlchemyApiKeyStore."""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from adapters.database import Base, init_db
from adapters.database.api_key_store import SQLAlchemyApiKeyStore


@pytest.fixture()
def store() -> SQLAlchemyApiKeyStore:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    return SQLAlchemyApiKeyStore(engine)


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


class TestCreate:
    def test_returns_api_key_with_label(self, store):
        key = store.create(label="react-app", key_hash=_hash("secret"))
        assert key.label == "react-app"

    def test_is_active_by_default(self, store):
        key = store.create(label="react-app", key_hash=_hash("secret"))
        assert key.is_active is True

    def test_stores_the_hash(self, store):
        h = _hash("my-secret")
        key = store.create(label="react-app", key_hash=h)
        assert key.key_hash == h

    def test_sets_created_at(self, store):
        key = store.create(label="react-app", key_hash=_hash("secret"))
        assert key.created_at is not None


class TestLookup:
    def test_returns_key_for_matching_hash(self, store):
        h = _hash("mysecret")
        store.create(label="app", key_hash=h)
        found = store.lookup(h)
        assert found is not None
        assert found.label == "app"

    def test_returns_none_for_unknown_hash(self, store):
        assert store.lookup(_hash("nonexistent")) is None

    def test_returns_none_after_revoke(self, store):
        h = _hash("revokeme")
        store.create(label="old-key", key_hash=h)
        store.revoke("old-key")
        found = store.lookup(h)
        assert found is not None
        assert found.is_active is False


class TestListKeys:
    def test_returns_empty_list_initially(self, store):
        assert store.list_keys() == []

    def test_returns_all_keys(self, store):
        store.create(label="key-a", key_hash=_hash("a"))
        store.create(label="key-b", key_hash=_hash("b"))
        keys = store.list_keys()
        assert len(keys) == 2

    def test_includes_revoked_keys(self, store):
        store.create(label="active-key", key_hash=_hash("x"))
        store.create(label="dead-key", key_hash=_hash("y"))
        store.revoke("dead-key")
        keys = store.list_keys()
        assert len(keys) == 2


class TestRevoke:
    def test_sets_is_active_false(self, store):
        h = _hash("killme")
        store.create(label="old", key_hash=h)
        result = store.revoke("old")
        assert result is True
        found = store.lookup(h)
        assert found is not None
        assert found.is_active is False

    def test_returns_false_for_unknown_label(self, store):
        assert store.revoke("no-such-label") is False

    def test_revoke_only_affects_target_label(self, store):
        store.create(label="keep", key_hash=_hash("k1"))
        store.create(label="remove", key_hash=_hash("k2"))
        store.revoke("remove")
        keep = store.lookup(_hash("k1"))
        assert keep is not None
        assert keep.is_active is True
```

- [ ] **Step 2: Run to verify tests fail**

```bash
uv run pytest tests/unit/test_api_key_store.py -v
```

Expected: `ModuleNotFoundError: No module named 'adapters.database.api_key_store'`

- [ ] **Step 3: Implement `SQLAlchemyApiKeyStore`**

Create `src/adapters/database/api_key_store.py`:

```python
"""SQLAlchemy implementation of ApiKeyPort."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, String, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from adapters.database import Base
from domain.models import ApiKey
from domain.ports import ApiKeyPort


class _ApiKeyRow(Base):
    __tablename__ = "api_keys"

    key_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


def _row_to_domain(row: _ApiKeyRow) -> ApiKey:
    return ApiKey(
        key_hash=row.key_hash,
        label=row.label,
        created_at=row.created_at,
        is_active=row.is_active,
    )


class SQLAlchemyApiKeyStore(ApiKeyPort):
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def lookup(self, key_hash: str) -> ApiKey | None:
        with Session(self._engine) as db:
            row = db.get(_ApiKeyRow, key_hash)
            return _row_to_domain(row) if row else None

    def create(self, label: str, key_hash: str) -> ApiKey:
        now = datetime.now(timezone.utc)
        row = _ApiKeyRow(key_hash=key_hash, label=label, created_at=now, is_active=True)
        with Session(self._engine) as db:
            db.add(row)
            db.commit()
            db.refresh(row)
            return _row_to_domain(row)

    def list_keys(self) -> list[ApiKey]:
        with Session(self._engine) as db:
            rows = db.scalars(select(_ApiKeyRow)).all()
            return [_row_to_domain(r) for r in rows]

    def revoke(self, label: str) -> bool:
        with Session(self._engine) as db:
            row = db.scalars(
                select(_ApiKeyRow).where(_ApiKeyRow.label == label)
            ).first()
            if row is None:
                return False
            row.is_active = False
            db.commit()
            return True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_api_key_store.py -v
```

Expected: all green, no failures.

- [ ] **Step 5: Commit**

```bash
git add src/adapters/database/api_key_store.py tests/unit/test_api_key_store.py
git commit -m "feat: add SQLAlchemyApiKeyStore with full unit test coverage"
```

---

## Task 4: `require_api_key` FastAPI dependency

**Files:**
- Create: `src/interactors/api/auth.py`
- Create: `tests/unit/test_auth_dependency.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_auth_dependency.py`:

```python
"""Unit tests for the require_api_key dependency."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from domain.models import ApiKey
from domain.ports import ApiKeyPort
from interactors.api.deps import configure_api_key_store


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


class FakeApiKeyStore(ApiKeyPort):
    def __init__(self, keys: dict[str, ApiKey]) -> None:
        self._keys = keys

    def lookup(self, key_hash: str) -> ApiKey | None:
        return self._keys.get(key_hash)

    def create(self, label: str, key_hash: str) -> ApiKey:
        raise NotImplementedError

    def list_keys(self) -> list[ApiKey]:
        raise NotImplementedError

    def revoke(self, label: str) -> bool:
        raise NotImplementedError


def _make_active_key(label: str = "test") -> ApiKey:
    return ApiKey(
        key_hash=_hash("valid-key"),
        label=label,
        created_at=datetime.now(timezone.utc),
        is_active=True,
    )


def _make_revoked_key(label: str = "dead") -> ApiKey:
    return ApiKey(
        key_hash=_hash("revoked-key"),
        label=label,
        created_at=datetime.now(timezone.utc),
        is_active=False,
    )


@pytest.fixture()
def app_with_auth():
    from fastapi import Depends
    from interactors.api.auth import require_api_key

    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_api_key)])
    def protected():
        return {"ok": True}

    return app


@pytest.fixture()
def client_with_active_key(app_with_auth):
    store = FakeApiKeyStore({_hash("valid-key"): _make_active_key()})
    configure_api_key_store(store)
    return TestClient(app_with_auth, raise_server_exceptions=False)


@pytest.fixture()
def client_with_revoked_key(app_with_auth):
    store = FakeApiKeyStore({_hash("revoked-key"): _make_revoked_key()})
    configure_api_key_store(store)
    return TestClient(app_with_auth, raise_server_exceptions=False)


@pytest.fixture()
def client_with_empty_store(app_with_auth):
    store = FakeApiKeyStore({})
    configure_api_key_store(store)
    return TestClient(app_with_auth, raise_server_exceptions=False)


class TestRequireApiKey:
    def test_missing_header_returns_401(self, client_with_active_key):
        response = client_with_active_key.get("/protected")
        assert response.status_code == 401

    def test_wrong_key_returns_401(self, client_with_empty_store):
        response = client_with_empty_store.get(
            "/protected", headers={"X-Api-Key": "wrong-key"}
        )
        assert response.status_code == 401

    def test_valid_key_returns_200(self, client_with_active_key):
        response = client_with_active_key.get(
            "/protected", headers={"X-Api-Key": "valid-key"}
        )
        assert response.status_code == 200

    def test_revoked_key_returns_403(self, client_with_revoked_key):
        response = client_with_revoked_key.get(
            "/protected", headers={"X-Api-Key": "revoked-key"}
        )
        assert response.status_code == 403

    def test_valid_key_response_body(self, client_with_active_key):
        response = client_with_active_key.get(
            "/protected", headers={"X-Api-Key": "valid-key"}
        )
        assert response.json() == {"ok": True}
```

- [ ] **Step 2: Run to verify tests fail**

```bash
uv run pytest tests/unit/test_auth_dependency.py -v
```

Expected: `ModuleNotFoundError: No module named 'interactors.api.auth'`

- [ ] **Step 3: Add `configure_api_key_store` / `get_api_key_store` to `deps.py`**

Append to `src/interactors/api/deps.py`:

```python
# ---------------------------------------------------------------------------
# API key store
# ---------------------------------------------------------------------------

from domain.ports import ApiKeyPort as _ApiKeyPort

_api_key_store: _ApiKeyPort | None = None


def get_api_key_store() -> _ApiKeyPort:
    if _api_key_store is None:
        raise RuntimeError("ApiKeyPort has not been configured.")
    return _api_key_store


def configure_api_key_store(store: _ApiKeyPort) -> None:
    global _api_key_store
    _api_key_store = store
```

- [ ] **Step 4: Create `src/interactors/api/auth.py`**

```python
"""FastAPI dependency for API key authentication."""

from __future__ import annotations

import hashlib

from fastapi import Depends, Header, HTTPException

from domain.ports import ApiKeyPort
from interactors.api.deps import get_api_key_store


def require_api_key(
    x_api_key: str | None = Header(default=None),
    store: ApiKeyPort = Depends(get_api_key_store),
) -> None:
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key required")
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    api_key = store.lookup(key_hash)
    if api_key is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not api_key.is_active:
        raise HTTPException(status_code=403, detail="API key revoked")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_auth_dependency.py -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/interactors/api/deps.py src/interactors/api/auth.py tests/unit/test_auth_dependency.py
git commit -m "feat: add require_api_key dependency and ApiKeyPort wiring in deps"
```

---

## Task 5: Wire auth into app.py (store init + key seeding + CORS)

**Files:**
- Modify: `src/interactors/api/app.py`

- [ ] **Step 1: Update the lifespan function in `src/interactors/api/app.py`**

Add `configure_api_key_store` to the lifespan imports and init the store + seed keys. Replace the existing `lifespan` function with:

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    import hashlib

    from adapters.database import init_db, make_engine
    from adapters.database.api_key_store import SQLAlchemyApiKeyStore
    from adapters.database.model_store import SQLAlchemyModelStore
    from adapters.database.run_store import SQLAlchemyRunStore
    from adapters.inference import LlamaCppInferenceAdapter
    from interactors.api.deps import (
        clear_adapter,
        configure,
        configure_api_key_store,
        configure_model_store,
        configure_run_store,
    )
    from interactors.temporal.activities import (
        configure_run_store as configure_activity_run_store,
        configure_storage,
    )

    engine = make_engine()
    init_db(engine)

    store = SQLAlchemyModelStore(engine)
    configure_model_store(store)

    run_store = SQLAlchemyRunStore(engine)
    configure_run_store(run_store)
    configure_activity_run_store(run_store)

    storage = _make_storage_adapter()
    configure_storage(storage)

    api_key_store = SQLAlchemyApiKeyStore(engine)
    configure_api_key_store(api_key_store)

    raw_keys = os.getenv("API_KEYS", "")
    for i, raw_key in enumerate(k.strip() for k in raw_keys.split(",") if k.strip()):
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        if api_key_store.lookup(key_hash) is None:
            api_key_store.create(label=f"env-key-{i}", key_hash=key_hash)
            logger.info("Seeded API key env-key-%d from API_KEYS", i)

    active = store.active()
    if active and active.gguf_path:
        local_path = Path("models/cache") / active.id / "model.gguf"
        try:
            storage.download(active.gguf_path, local_path)
            model_path = str(local_path)
            logger.info("Loading active model %s from storage key %s", active.id, active.gguf_path)
        except Exception:
            logger.warning(
                "Could not load active model %s from storage; falling back to MODEL_PATH",
                active.id,
                exc_info=True,
            )
            model_path = os.getenv("MODEL_PATH", "models/aipet.gguf")
    else:
        model_path = os.getenv("MODEL_PATH", "models/aipet.gguf")

    configure(LlamaCppInferenceAdapter(model_path=model_path))

    try:
        yield
    finally:
        clear_adapter()
```

- [ ] **Step 2: Update CORS configuration in `src/interactors/api/app.py`**

Replace the existing `app.add_middleware(CORSMiddleware, ...)` block with:

```python
_cors_raw = os.getenv("CORS_ORIGINS", "")
if os.getenv("APP_ENV") == "development":
    _cors_origins: list[str] = ["*"]
elif _cors_raw:
    _cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
else:
    _cors_origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["X-Api-Key", "Authorization", "Content-Type"],
)
```

- [ ] **Step 3: Commit**

```bash
git add src/interactors/api/app.py
git commit -m "feat: init ApiKeyStore in lifespan, seed from API_KEYS env var, scope CORS via CORS_ORIGINS"
```

---

## Task 6: Apply auth to routes + integration tests

**Files:**
- Modify: `src/interactors/api/routes/inference.py`
- Modify: `src/interactors/api/routes/models.py`
- Modify: `src/interactors/api/routes/runs.py`
- Create: `tests/integration/test_auth.py`

- [ ] **Step 1: Write the failing integration tests**

Create `tests/integration/test_auth.py`:

```python
"""Integration tests — API key auth applied to all routes except GET /health."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest
import pytest_asyncio
import httpx
from httpx import ASGITransport

from domain.actions import Action
from domain.models import ApiKey, InferenceRequest, InferenceResponse
from domain.ports import ApiKeyPort, InferencePort
from interactors.api.app import app
from interactors.api.deps import configure, configure_api_key_store


VALID_RAW_KEY = "test-secret-key-abc123"
REVOKED_RAW_KEY = "revoked-key-xyz789"


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


class FakeInferenceAdapter(InferencePort):
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(action=Action.IDLE)


class FakeApiKeyStore(ApiKeyPort):
    def __init__(self) -> None:
        self._keys: dict[str, ApiKey] = {
            _hash(VALID_RAW_KEY): ApiKey(
                key_hash=_hash(VALID_RAW_KEY),
                label="valid",
                created_at=datetime.now(timezone.utc),
                is_active=True,
            ),
            _hash(REVOKED_RAW_KEY): ApiKey(
                key_hash=_hash(REVOKED_RAW_KEY),
                label="revoked",
                created_at=datetime.now(timezone.utc),
                is_active=False,
            ),
        }

    def lookup(self, key_hash: str) -> ApiKey | None:
        return self._keys.get(key_hash)

    def create(self, label: str, key_hash: str) -> ApiKey:
        raise NotImplementedError

    def list_keys(self) -> list[ApiKey]:
        return list(self._keys.values())

    def revoke(self, label: str) -> bool:
        raise NotImplementedError


VALID_INFER_PAYLOAD = {
    "scene": {"objects": [], "tick": 1},
    "pet_stats": {
        "hunger": 0.5,
        "boredom": 0.3,
        "social": 0.2,
        "toilet": 0.1,
        "tiredness": 0.4,
    },
}


@pytest_asyncio.fixture(autouse=True)
async def setup_stores():
    configure(FakeInferenceAdapter())
    configure_api_key_store(FakeApiKeyStore())


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


class TestHealthIsPublic:
    @pytest.mark.asyncio
    async def test_health_without_key_returns_200(self, client):
        response = await client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_with_key_returns_200(self, client):
        response = await client.get("/health", headers={"X-Api-Key": VALID_RAW_KEY})
        assert response.status_code == 200


class TestInferRequiresAuth:
    @pytest.mark.asyncio
    async def test_no_key_returns_401(self, client):
        response = await client.post("/infer", json=VALID_INFER_PAYLOAD)
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_key_returns_401(self, client):
        response = await client.post(
            "/infer", json=VALID_INFER_PAYLOAD, headers={"X-Api-Key": "bad-key"}
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_revoked_key_returns_403(self, client):
        response = await client.post(
            "/infer", json=VALID_INFER_PAYLOAD, headers={"X-Api-Key": REVOKED_RAW_KEY}
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_valid_key_returns_200(self, client):
        response = await client.post(
            "/infer", json=VALID_INFER_PAYLOAD, headers={"X-Api-Key": VALID_RAW_KEY}
        )
        assert response.status_code == 200


class TestModelsRequiresAuth:
    @pytest.mark.asyncio
    async def test_no_key_returns_401(self, client):
        response = await client.get("/api/models")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_key_returns_200(self, client):
        response = await client.get(
            "/api/models", headers={"X-Api-Key": VALID_RAW_KEY}
        )
        assert response.status_code == 200


class TestRunsRequiresAuth:
    @pytest.mark.asyncio
    async def test_no_key_returns_401(self, client):
        response = await client.get("/api/runs")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_key_returns_200(self, client):
        response = await client.get(
            "/api/runs", headers={"X-Api-Key": VALID_RAW_KEY}
        )
        assert response.status_code == 200
```

- [ ] **Step 2: Run to verify tests fail**

```bash
uv run pytest tests/integration/test_auth.py -v
```

Expected: `TestInferRequiresAuth::test_no_key_returns_401` returns 200 (auth not applied yet), so tests fail.

- [ ] **Step 3: Apply auth to `/infer` in `src/interactors/api/routes/inference.py`**

Add the import at the top:
```python
from fastapi import APIRouter, Depends, HTTPException
from interactors.api.auth import require_api_key
```

Change the `/infer` route decorator:
```python
@router.post("/infer", response_model=InferenceResponse, dependencies=[Depends(require_api_key)])
def infer(
    request: InferenceRequest,
    adapter: InferencePort = Depends(get_adapter),
) -> InferenceResponse:
    try:
        return adapter.infer(request)
    except Exception:
        logger.exception("Unexpected error during inference")
        raise HTTPException(status_code=500, detail={"error": "inference_failed"})
```

(`/health` route stays unchanged.)

- [ ] **Step 4: Apply auth at router level in `src/interactors/api/routes/models.py`**

Change the router declaration at the top of the file:
```python
from interactors.api.auth import require_api_key
from fastapi import Depends

router = APIRouter(
    prefix="/api/models",
    tags=["models"],
    dependencies=[Depends(require_api_key)],
)
```

- [ ] **Step 5: Apply auth at router level in `src/interactors/api/routes/runs.py`**

Change the router declaration at the top of the file:
```python
from interactors.api.auth import require_api_key
from fastapi import Depends

router = APIRouter(
    prefix="/api/runs",
    tags=["runs"],
    dependencies=[Depends(require_api_key)],
)
```

- [ ] **Step 6: Run integration tests to verify they pass**

```bash
uv run pytest tests/integration/test_auth.py -v
```

Expected: all green.

- [ ] **Step 7: Verify existing tests still pass**

```bash
uv run pytest tests/unit/ tests/integration/test_api.py -v
```

Expected: all green. (Existing `/infer` tests in `test_api.py` don't send an API key, so they will now return 401 — you must update that fixture to configure a fake store too. See the note below.)

> **Note:** If `tests/integration/test_api.py` fails because it no longer has an api key store configured, add this fixture to that file or its conftest — it makes the store return a fixed valid key for all tests in that file:
>
> ```python
> @pytest_asyncio.fixture(autouse=True)
> async def setup_auth():
>     from datetime import datetime, timezone
>     import hashlib
>     from domain.models import ApiKey
>     from domain.ports import ApiKeyPort
>     from interactors.api.deps import configure_api_key_store
>
>     RAW = "test-key"
>
>     class PermissiveStore(ApiKeyPort):
>         def lookup(self, key_hash: str) -> ApiKey | None:
>             return ApiKey(key_hash=key_hash, label="test", created_at=datetime.now(timezone.utc), is_active=True)
>         def create(self, label, key_hash): raise NotImplementedError
>         def list_keys(self): return []
>         def revoke(self, label): raise NotImplementedError
>
>     configure_api_key_store(PermissiveStore())
> ```
>
> Also add `headers={"X-Api-Key": "test-key"}` to every request in `test_api.py` that hits a protected endpoint.

- [ ] **Step 8: Commit**

```bash
git add \
  src/interactors/api/routes/inference.py \
  src/interactors/api/routes/models.py \
  src/interactors/api/routes/runs.py \
  tests/integration/test_auth.py
git commit -m "feat: apply require_api_key to /infer, /api/models, /api/runs; add auth integration tests"
```

---

## Task 7: Key management CLI

**Files:**
- Create: `src/interactors/cli/manage_keys.py`
- Create: `tests/cli/test_manage_keys.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/cli/test_manage_keys.py`:

```python
"""CLI tests for manage_keys.py."""

from __future__ import annotations

import hashlib
import sys
from io import StringIO
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from adapters.database import Base, init_db
from adapters.database.api_key_store import SQLAlchemyApiKeyStore


def _make_store():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    return SQLAlchemyApiKeyStore(engine)


def _run_cli(args: list[str], store: SQLAlchemyApiKeyStore) -> tuple[str, int]:
    """Run manage_keys main() with the given args and fake store. Returns (stdout, exit_code)."""
    from interactors.cli import manage_keys

    buf = StringIO()
    exit_code = 0
    with (
        patch.object(sys, "argv", ["manage_keys"] + args),
        patch("interactors.cli.manage_keys._make_store", return_value=store),
        patch("sys.stdout", buf),
    ):
        try:
            manage_keys.main()
        except SystemExit as e:
            exit_code = e.code if isinstance(e.code, int) else 1
    return buf.getvalue(), exit_code


class TestCreate:
    def test_create_prints_raw_key(self):
        store = _make_store()
        out, code = _run_cli(["create", "--label", "myapp"], store)
        assert code == 0
        assert "myapp" in out
        assert "Raw key" in out

    def test_create_stores_key_in_db(self):
        store = _make_store()
        _run_cli(["create", "--label", "myapp"], store)
        keys = store.list_keys()
        assert len(keys) == 1
        assert keys[0].label == "myapp"
        assert keys[0].is_active is True

    def test_create_key_is_lookupable_by_hash(self):
        import re
        store = _make_store()
        out, _ = _run_cli(["create", "--label", "myapp"], store)
        # Extract the raw key printed on the last line
        match = re.search(r"Raw key.*?:\s*(\S+)", out)
        assert match, f"Raw key not found in output: {out}"
        raw_key = match.group(1)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        found = store.lookup(key_hash)
        assert found is not None
        assert found.label == "myapp"


class TestList:
    def test_list_empty_store(self):
        store = _make_store()
        out, code = _run_cli(["list"], store)
        assert code == 0

    def test_list_shows_labels(self):
        store = _make_store()
        store.create(label="alpha", key_hash=hashlib.sha256(b"a").hexdigest())
        store.create(label="beta", key_hash=hashlib.sha256(b"b").hexdigest())
        out, code = _run_cli(["list"], store)
        assert code == 0
        assert "alpha" in out
        assert "beta" in out

    def test_list_shows_revoked_status(self):
        store = _make_store()
        store.create(label="dead", key_hash=hashlib.sha256(b"d").hexdigest())
        store.revoke("dead")
        out, _ = _run_cli(["list"], store)
        assert "revoked" in out


class TestRevoke:
    def test_revoke_existing_label_exits_0(self):
        store = _make_store()
        store.create(label="todelete", key_hash=hashlib.sha256(b"t").hexdigest())
        _, code = _run_cli(["revoke", "--label", "todelete"], store)
        assert code == 0

    def test_revoke_deactivates_the_key(self):
        store = _make_store()
        h = hashlib.sha256(b"k").hexdigest()
        store.create(label="k", key_hash=h)
        _run_cli(["revoke", "--label", "k"], store)
        found = store.lookup(h)
        assert found is not None
        assert found.is_active is False

    def test_revoke_unknown_label_exits_nonzero(self):
        store = _make_store()
        _, code = _run_cli(["revoke", "--label", "nope"], store)
        assert code != 0
```

- [ ] **Step 2: Run to verify tests fail**

```bash
uv run pytest tests/cli/test_manage_keys.py -v
```

Expected: `ModuleNotFoundError: No module named 'interactors.cli.manage_keys'`

- [ ] **Step 3: Implement `src/interactors/cli/manage_keys.py`**

```python
"""Key management CLI — create, list, and revoke API keys."""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import sys


def _make_store():
    from adapters.database.api_key_store import SQLAlchemyApiKeyStore
    from adapters.database.engine import make_engine, run_migrations

    url = os.environ.get("DATABASE_URL", "sqlite:///data/aipet.db")
    engine = make_engine(url)
    run_migrations(engine)
    return SQLAlchemyApiKeyStore(engine)


def _cmd_create(args: argparse.Namespace, store) -> None:
    raw_key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    store.create(label=args.label, key_hash=key_hash)
    print(f"Created key with label '{args.label}'")
    print(f"Raw key (store this securely — shown only once): {raw_key}")


def _cmd_list(args: argparse.Namespace, store) -> None:
    keys = store.list_keys()
    if not keys:
        print("No API keys found.")
        return
    for k in keys:
        status = "active" if k.is_active else "revoked"
        print(f"  {k.label:<30s}  {status:<8s}  {k.created_at.isoformat()}")


def _cmd_revoke(args: argparse.Namespace, store) -> None:
    found = store.revoke(label=args.label)
    if found:
        print(f"Revoked key '{args.label}'")
    else:
        print(f"No key found with label '{args.label}'")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage API keys")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_p = subparsers.add_parser("create", help="Issue a new API key")
    create_p.add_argument("--label", required=True, help="Human-readable name for this key")

    subparsers.add_parser("list", help="List all API keys")

    revoke_p = subparsers.add_parser("revoke", help="Revoke an API key by label")
    revoke_p.add_argument("--label", required=True, help="Label of the key to revoke")

    args = parser.parse_args()
    store = _make_store()

    if args.command == "create":
        _cmd_create(args, store)
    elif args.command == "list":
        _cmd_list(args, store)
    elif args.command == "revoke":
        _cmd_revoke(args, store)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/cli/test_manage_keys.py -v
```

Expected: all green.

- [ ] **Step 5: Run the full test suite**

```bash
uv run pytest tests/unit/ tests/integration/ tests/cli/ -v --ignore=tests/integration/test_real_inference.py --ignore=tests/integration/test_model_quality.py
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/interactors/cli/manage_keys.py tests/cli/test_manage_keys.py
git commit -m "feat: add manage_keys CLI for create/list/revoke API keys"
```

---

## Self-Review Checklist

- **Spec coverage:**
  - [x] API key authentication on protected routes (`require_api_key`)
  - [x] `GET /health` unauthenticated
  - [x] HTTP 401 for missing/unknown key
  - [x] HTTP 403 for revoked key
  - [x] CORS via `CORS_ORIGINS` env var (wildcard in development)
  - [x] Keys seeded from `API_KEYS` env var on startup
  - [x] Key management CLI (`create` / `list` / `revoke`)
  - [x] Keys stored hashed (sha256) — plaintext never persisted
  - [x] Integration tests covering all auth scenarios

- **Type consistency:** `ApiKeyPort` defined in Task 1, used identically in Tasks 3, 4, and 7. `ApiKey` model matches the ORM `_ApiKeyRow` fields in Task 3.

- **Existing test compatibility:** Existing `test_api.py` tests will break once auth is applied (Task 6). The note in Task 6, Step 7 explains exactly how to fix them.

---

## Environment Variables Reference

| Variable | Required | Example | Purpose |
|---|---|---|---|
| `API_KEYS` | Yes (prod) | `key1,key2` | Comma-separated raw API keys seeded on startup |
| `CORS_ORIGINS` | Yes (prod) | `https://app.example.com` | Allowed React app origins (comma-separated) |
| `APP_ENV` | No | `development` | Set to `development` to allow wildcard CORS locally |

Example local dev startup:
```bash
APP_ENV=development API_KEYS=mylocalsecret uv run uvicorn interactors.api.app:app --reload
```

Example API call:
```bash
curl -H "X-Api-Key: mylocalsecret" http://localhost:8000/api/models
```
