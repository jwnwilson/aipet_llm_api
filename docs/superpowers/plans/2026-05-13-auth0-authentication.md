# Auth0 Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Protect all API endpoints with Auth0 JWT validation using the adapter pattern, with a built-in login flow for isolated testing.

**Architecture:** The API acts as an OAuth2 resource server — it validates RS256 JWTs issued by Auth0 via JWKS. An `Auth0Adapter` implements the domain `AuthPort` and is wired into FastAPI via a `require_auth` dependency applied at router level. A `/auth/login` → `/auth/callback` flow lets developers get tokens without a separate client app.

**Tech Stack:** `PyJWT[cryptography]>=2.8` (JWT validation + JWKS), `httpx` (Auth0 token exchange), FastAPI `Depends`, existing SQLAlchemy/Alembic stack unchanged.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/domain/models.py` | Add `UserContext` Pydantic model |
| Modify | `src/domain/ports.py` | Add `AuthPort` abstract interface |
| Create | `src/adapters/auth/__init__.py` | Package marker |
| Create | `src/adapters/auth/auth0.py` | `Auth0Adapter` — JWKS-based RS256 JWT validation |
| Modify | `src/interactors/api/deps.py` | Add `configure_auth` / `get_auth` singleton |
| Create | `src/interactors/api/auth.py` | `require_auth` FastAPI dependency |
| Modify | `src/interactors/api/routes/inference.py` | Add `require_auth` to `/infer` |
| Modify | `src/interactors/api/routes/models.py` | Add `require_auth` at router level |
| Modify | `src/interactors/api/routes/runs.py` | Add `require_auth` at router level |
| Create | `src/interactors/api/routes/login.py` | `GET /auth/login` + `GET /auth/callback` |
| Modify | `src/interactors/api/app.py` | Wire `Auth0Adapter`, tighten CORS, include login router |
| Modify | `pyproject.toml` | Add `PyJWT[cryptography]`, `httpx` to main deps |
| Create | `tests/unit/test_auth0_adapter.py` | Unit tests for JWT validation |
| Create | `tests/unit/test_auth_dependency.py` | Unit tests for `require_auth` |
| Modify | `tests/integration/conftest.py` | Add `_auth_bypass` autouse fixture |
| Create | `tests/integration/test_auth.py` | Auth enforcement integration tests |

---

## Task 1: Domain types — `UserContext` and `AuthPort`

**Files:**
- Modify: `src/domain/models.py`
- Modify: `src/domain/ports.py`

Pure type definitions — no I/O, no tests needed.

- [ ] **Step 1: Add `UserContext` to `src/domain/models.py`**

Append after the last class in the file (after `RunRecord`, before the trailing blank line):

```python
class UserContext(BaseModel):
    user_id: str
    email: str | None = None
```

- [ ] **Step 2: Add `UserContext` import and `AuthPort` to `src/domain/ports.py`**

Add `UserContext` to the existing import block at the top of `ports.py`. The import currently reads:

```python
from domain.models import (
    InferenceRequest,
    InferenceResponse,
    RemoteTrainConfig,
    RunConfig,
    RunRecord,
    RunStatus,
    TrainingModel,
    TrainingModelConfig,
)
```

Change it to:

```python
from domain.models import (
    InferenceRequest,
    InferenceResponse,
    RemoteTrainConfig,
    RunConfig,
    RunRecord,
    RunStatus,
    TrainingModel,
    TrainingModelConfig,
    UserContext,
)
```

Then append after the last class in the file (`RunStorePort`):

```python
class AuthPort(ABC):
    """Abstract interface for validating bearer tokens."""

    @abstractmethod
    def authenticate(self, token: str) -> UserContext | None:
        """Validate the JWT and return a UserContext, or None if invalid/expired."""
```

- [ ] **Step 3: Commit**

```bash
git add src/domain/models.py src/domain/ports.py
git commit -m "feat: add UserContext model and AuthPort interface"
```

---

## Task 2: `Auth0Adapter` and unit tests

**Files:**
- Modify: `pyproject.toml`
- Create: `src/adapters/auth/__init__.py`
- Create: `src/adapters/auth/auth0.py`
- Create: `tests/unit/test_auth0_adapter.py`

- [ ] **Step 1: Add dependencies to `pyproject.toml`**

In `pyproject.toml`, add `PyJWT[cryptography]` and `httpx` to the main `dependencies` list, and remove `httpx` from the `dev` extras (it's now a main dep):

```toml
dependencies = [
    "fastapi>=0.136.1",
    "httpx>=0.28.1",
    "llama-cpp-python>=0.3.22",
    "pydantic>=2.13.3",
    "PyJWT[cryptography]>=2.8",
    "temporalio>=1.27.0",
    "uvicorn>=0.46.0",
    "datasets>=4.8.5",
    "transformers>=5.7.0",
    "torch>=2.0",
    "accelerate>=1.1.0",
    "bitsandbytes>=0.43.0",
    "peft>=0.9.0",
    "sentencepiece>=0.2.0",
    "kaggle>=1.6",
    "boto3>=1.35",
    "runpod>=1.7",
    "vastai>=1.0",
    "sqlalchemy>=2.0.49",
    "alembic>=1.18.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=9.0.3",
    "pytest-asyncio>=1.3.0",
]
```

Install:

```bash
uv sync
```

Expected: resolves without errors, `PyJWT` and `cryptography` appear in the environment.

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_auth0_adapter.py`:

```python
"""Unit tests for Auth0Adapter JWT validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from adapters.auth.auth0 import Auth0Adapter
from domain.models import UserContext

DOMAIN = "test.auth0.com"
AUDIENCE = "https://api.aipet.test"

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PUBLIC_KEY = _PRIVATE_KEY.public_key()


def _make_token(
    sub: str = "auth0|abc123",
    email: str | None = "user@example.com",
    expired: bool = False,
    audience: str = AUDIENCE,
) -> str:
    now = datetime.now(timezone.utc)
    exp = now - timedelta(hours=1) if expired else now + timedelta(hours=1)
    claims: dict = {
        "sub": sub,
        "iss": f"https://{DOMAIN}/",
        "aud": audience,
        "iat": now,
        "exp": exp,
    }
    if email is not None:
        claims["email"] = email
    return jwt.encode(claims, _PRIVATE_KEY, algorithm="RS256")


@pytest.fixture
def adapter():
    mock_signing_key = MagicMock()
    mock_signing_key.key = _PUBLIC_KEY
    mock_jwks_client = MagicMock()
    mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key
    with patch("adapters.auth.auth0.jwt.PyJWKClient", return_value=mock_jwks_client):
        yield Auth0Adapter(domain=DOMAIN, audience=AUDIENCE)


class TestAuthenticate:
    def test_valid_token_returns_user_context(self, adapter):
        result = adapter.authenticate(_make_token())
        assert isinstance(result, UserContext)
        assert result.user_id == "auth0|abc123"

    def test_includes_email_when_present(self, adapter):
        result = adapter.authenticate(_make_token(email="user@example.com"))
        assert result is not None
        assert result.email == "user@example.com"

    def test_email_is_none_when_absent(self, adapter):
        result = adapter.authenticate(_make_token(email=None))
        assert result is not None
        assert result.email is None

    def test_expired_token_returns_none(self, adapter):
        assert adapter.authenticate(_make_token(expired=True)) is None

    def test_wrong_audience_returns_none(self, adapter):
        assert adapter.authenticate(_make_token(audience="https://other-api.example.com")) is None

    def test_tampered_token_returns_none(self, adapter):
        token = _make_token()
        tampered = token[:-10] + "x" * 10
        assert adapter.authenticate(tampered) is None

    def test_garbage_string_returns_none(self, adapter):
        assert adapter.authenticate("not.a.jwt") is None
```

- [ ] **Step 3: Run to verify tests fail**

```bash
uv run pytest tests/unit/test_auth0_adapter.py -v
```

Expected: `ModuleNotFoundError: No module named 'adapters.auth'`

- [ ] **Step 4: Create `src/adapters/auth/__init__.py`**

```python
```

(Empty file — just makes `adapters.auth` a package.)

- [ ] **Step 5: Create `src/adapters/auth/auth0.py`**

```python
"""Auth0 JWT validation adapter."""

from __future__ import annotations

import logging

import jwt

from domain.models import UserContext
from domain.ports import AuthPort

logger = logging.getLogger(__name__)


class Auth0Adapter(AuthPort):
    def __init__(self, domain: str, audience: str) -> None:
        self._audience = audience
        self._issuer = f"https://{domain}/"
        self._jwks_client = jwt.PyJWKClient(
            f"https://{domain}/.well-known/jwks.json",
            cache_keys=True,
        )

    def authenticate(self, token: str) -> UserContext | None:
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self._audience,
                issuer=self._issuer,
            )
            return UserContext(
                user_id=payload["sub"],
                email=payload.get("email"),
            )
        except jwt.InvalidTokenError as exc:
            logger.debug("JWT validation failed: %s", type(exc).__name__)
            return None
        except Exception:
            logger.warning("Unexpected error validating JWT", exc_info=True)
            return None
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_auth0_adapter.py -v
```

Expected: all 8 tests green.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/adapters/auth/ tests/unit/test_auth0_adapter.py
git commit -m "feat: add Auth0Adapter with JWKS-based RS256 JWT validation"
```

---

## Task 3: `require_auth` FastAPI dependency

**Files:**
- Modify: `src/interactors/api/deps.py`
- Create: `src/interactors/api/auth.py`
- Create: `tests/unit/test_auth_dependency.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_auth_dependency.py`:

```python
"""Unit tests for the require_auth FastAPI dependency."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from domain.models import UserContext
from domain.ports import AuthPort
from interactors.api.deps import configure_auth


class _StubAuthPort(AuthPort):
    def __init__(self, result: UserContext | None) -> None:
        self._result = result

    def authenticate(self, token: str) -> UserContext | None:
        return self._result


_VALID_USER = UserContext(user_id="u1", email="u@example.com")


def _make_client(auth_port: AuthPort) -> TestClient:
    from interactors.api.auth import require_auth

    configure_auth(auth_port)
    test_app = FastAPI()

    @test_app.get("/protected", dependencies=[Depends(require_auth)])
    def protected() -> dict:
        return {"ok": True}

    return TestClient(test_app, raise_server_exceptions=False)


class TestRequireAuth:
    def test_missing_header_returns_401(self):
        client = _make_client(_StubAuthPort(_VALID_USER))
        assert client.get("/protected").status_code == 401

    def test_non_bearer_scheme_returns_401(self):
        client = _make_client(_StubAuthPort(_VALID_USER))
        assert client.get("/protected", headers={"Authorization": "ApiKey abc"}).status_code == 401

    def test_bearer_with_no_token_returns_401(self):
        client = _make_client(_StubAuthPort(_VALID_USER))
        assert client.get("/protected", headers={"Authorization": "Bearer "}).status_code == 401

    def test_invalid_token_returns_401(self):
        client = _make_client(_StubAuthPort(None))
        assert client.get("/protected", headers={"Authorization": "Bearer bad"}).status_code == 401

    def test_valid_token_returns_200(self):
        client = _make_client(_StubAuthPort(_VALID_USER))
        assert client.get("/protected", headers={"Authorization": "Bearer valid"}).status_code == 200

    def test_valid_token_response_body(self):
        client = _make_client(_StubAuthPort(_VALID_USER))
        response = client.get("/protected", headers={"Authorization": "Bearer valid"})
        assert response.json() == {"ok": True}
```

- [ ] **Step 2: Run to verify tests fail**

```bash
uv run pytest tests/unit/test_auth_dependency.py -v
```

Expected: `ModuleNotFoundError: No module named 'interactors.api.auth'`

- [ ] **Step 3: Add `configure_auth` / `get_auth` to `src/interactors/api/deps.py`**

Append to the end of `src/interactors/api/deps.py`:

```python
# ---------------------------------------------------------------------------
# Auth port
# ---------------------------------------------------------------------------

from domain.ports import AuthPort as _AuthPort

_auth_port: _AuthPort | None = None


def get_auth() -> _AuthPort:
    if _auth_port is None:
        raise RuntimeError("AuthPort has not been configured.")
    return _auth_port


def configure_auth(port: _AuthPort) -> None:
    global _auth_port
    _auth_port = port
```

- [ ] **Step 4: Create `src/interactors/api/auth.py`**

```python
"""FastAPI dependency for Auth0 JWT authentication."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException

from domain.ports import AuthPort
from interactors.api.deps import get_auth


def require_auth(
    authorization: str | None = Header(default=None),
    auth_port: AuthPort = Depends(get_auth),
) -> None:
    if authorization is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Not authenticated")
    if auth_port.authenticate(token) is None:
        raise HTTPException(status_code=401, detail="Invalid token")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_auth_dependency.py -v
```

Expected: all 6 tests green.

- [ ] **Step 6: Commit**

```bash
git add src/interactors/api/deps.py src/interactors/api/auth.py tests/unit/test_auth_dependency.py
git commit -m "feat: add require_auth dependency and configure_auth wiring"
```

---

## Task 4: Apply auth to routes and fix integration tests

**Files:**
- Modify: `src/interactors/api/routes/inference.py`
- Modify: `src/interactors/api/routes/models.py`
- Modify: `src/interactors/api/routes/runs.py`
- Modify: `tests/integration/conftest.py`
- Create: `tests/integration/test_auth.py`

- [ ] **Step 1: Write the failing auth integration tests**

Create `tests/integration/test_auth.py`:

```python
"""Integration tests — auth enforced on all routes except GET /health."""

from __future__ import annotations

import pytest
import pytest_asyncio
import httpx
from httpx import ASGITransport

from domain.actions import Action
from domain.models import InferenceRequest, InferenceResponse, UserContext
from domain.ports import AuthPort, InferencePort
from interactors.api.app import app
from interactors.api.deps import configure, configure_auth

VALID_TOKEN = "valid-test-token"

VALID_PAYLOAD = {
    "scene": {"objects": [], "tick": 1},
    "pet_stats": {
        "hunger": 0.5,
        "boredom": 0.3,
        "social": 0.2,
        "toilet": 0.1,
        "tiredness": 0.4,
    },
}


class _FakeInferenceAdapter(InferencePort):
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(action=Action.IDLE)


class _FakeAuthAdapter(AuthPort):
    def authenticate(self, token: str) -> UserContext | None:
        if token == VALID_TOKEN:
            return UserContext(user_id="u1", email="u@example.com")
        return None


@pytest.fixture(autouse=True)
def _auth_bypass():
    # Override the conftest _auth_bypass: remove dependency override and
    # use a real (fake) AuthAdapter so auth is actually enforced.
    from interactors.api.auth import require_auth
    app.dependency_overrides.pop(require_auth, None)
    configure_auth(_FakeAuthAdapter())
    yield
    app.dependency_overrides[require_auth] = lambda: None


@pytest_asyncio.fixture
async def client():
    configure(_FakeInferenceAdapter())
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


VALID_HEADERS = {"Authorization": f"Bearer {VALID_TOKEN}"}


class TestHealthIsPublic:
    @pytest.mark.asyncio
    async def test_no_auth_returns_200(self, client):
        assert (await client.get("/health")).status_code == 200

    @pytest.mark.asyncio
    async def test_with_valid_auth_returns_200(self, client):
        assert (await client.get("/health", headers=VALID_HEADERS)).status_code == 200


class TestInferRequiresAuth:
    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client):
        assert (await client.post("/infer", json=VALID_PAYLOAD)).status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_token_returns_401(self, client):
        resp = await client.post(
            "/infer", json=VALID_PAYLOAD, headers={"Authorization": "Bearer wrong"}
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_token_returns_200(self, client):
        resp = await client.post("/infer", json=VALID_PAYLOAD, headers=VALID_HEADERS)
        assert resp.status_code == 200


class TestModelsRequiresAuth:
    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client):
        assert (await client.get("/api/models")).status_code == 401

    @pytest.mark.asyncio
    async def test_no_auth_on_post_returns_401(self, client):
        assert (await client.post("/api/models", json={})).status_code == 401


class TestRunsRequiresAuth:
    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client):
        assert (await client.get("/api/runs")).status_code == 401

    @pytest.mark.asyncio
    async def test_no_auth_on_get_by_id_returns_401(self, client):
        assert (await client.get("/api/runs/some-id")).status_code == 401
```

- [ ] **Step 2: Run to verify tests fail (401 not yet enforced)**

```bash
uv run pytest tests/integration/test_auth.py -v
```

Expected: `TestInferRequiresAuth::test_no_auth_returns_401` fails with 200 (auth not applied yet).

- [ ] **Step 3: Apply auth to `src/interactors/api/routes/inference.py`**

Replace the file content with:

```python
"""Inference and health endpoints."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException

from domain.models import InferenceRequest, InferenceResponse
from domain.ports import InferencePort
from interactors.api.auth import require_auth
from interactors.api.deps import get_adapter

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/infer", response_model=InferenceResponse, dependencies=[Depends(require_auth)])
def infer(
    request: InferenceRequest,
    adapter: InferencePort = Depends(get_adapter),
) -> InferenceResponse:
    try:
        return adapter.infer(request)
    except Exception:
        logger.exception("Unexpected error during inference")
        raise HTTPException(status_code=500, detail={"error": "inference_failed"})


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": os.getenv("MODEL_PATH", "models/aipet.gguf"),
    }
```

- [ ] **Step 4: Apply auth at router level in `src/interactors/api/routes/models.py`**

Replace the router declaration (line 16) with:

```python
from interactors.api.auth import require_auth

router = APIRouter(
    prefix="/api/models",
    tags=["models"],
    dependencies=[Depends(require_auth)],
)
```

The rest of the file is unchanged.

- [ ] **Step 5: Apply auth at router level in `src/interactors/api/routes/runs.py`**

Replace the router declaration (line 19) with:

```python
from interactors.api.auth import require_auth

router = APIRouter(
    prefix="/api/runs",
    tags=["runs"],
    dependencies=[Depends(require_auth)],
)
```

The rest of the file is unchanged.

- [ ] **Step 6: Add `_auth_bypass` autouse fixture to `tests/integration/conftest.py`**

Append to the end of `tests/integration/conftest.py`:

```python
@pytest.fixture(autouse=True)
def _auth_bypass():
    """Bypass require_auth for all integration tests by default.

    test_auth.py overrides this fixture with the same name to test real auth enforcement.
    """
    from interactors.api.app import app
    from interactors.api.auth import require_auth

    app.dependency_overrides[require_auth] = lambda: None
    yield
    app.dependency_overrides.pop(require_auth, None)
```

Also add the missing `import pytest` if it is not already at the top — check the file; it already has `import pytest`.

- [ ] **Step 7: Run the new auth tests to verify they pass**

```bash
uv run pytest tests/integration/test_auth.py -v
```

Expected: all 9 tests green.

- [ ] **Step 8: Verify existing integration tests still pass**

```bash
uv run pytest tests/integration/test_api.py tests/integration/test_run_api.py tests/integration/test_training_api.py -v
```

Expected: all green (auth bypassed by conftest fixture).

- [ ] **Step 9: Commit**

```bash
git add \
  src/interactors/api/routes/inference.py \
  src/interactors/api/routes/models.py \
  src/interactors/api/routes/runs.py \
  tests/integration/conftest.py \
  tests/integration/test_auth.py
git commit -m "feat: apply require_auth to /infer, /api/models, /api/runs; add auth integration tests"
```

---

## Task 5: Login flow for isolated testing

**Files:**
- Create: `src/interactors/api/routes/login.py`

No automated tests — this route is thin glue between FastAPI and Auth0's OAuth2 endpoint. Manual verification is covered in the wiring task.

- [ ] **Step 1: Create `src/interactors/api/routes/login.py`**

```python
"""Auth0 OAuth2 authorisation-code login — for development and isolated testing."""

from __future__ import annotations

import os
import urllib.parse

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, RedirectResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/login")
def login() -> RedirectResponse:
    params = {
        "response_type": "code",
        "client_id": os.environ["AUTH0_CLIENT_ID"],
        "redirect_uri": os.environ["AUTH0_CALLBACK_URL"],
        "audience": os.environ["AUTH0_AUDIENCE"],
        "scope": "openid email",
    }
    url = (
        f"https://{os.environ['AUTH0_DOMAIN']}/authorize?"
        + urllib.parse.urlencode(params)
    )
    return RedirectResponse(url=url)


@router.get("/callback")
def callback(code: str) -> PlainTextResponse:
    domain = os.environ["AUTH0_DOMAIN"]
    resp = httpx.post(
        f"https://{domain}/oauth/token",
        json={
            "grant_type": "authorization_code",
            "client_id": os.environ["AUTH0_CLIENT_ID"],
            "client_secret": os.environ["AUTH0_CLIENT_SECRET"],
            "code": code,
            "redirect_uri": os.environ["AUTH0_CALLBACK_URL"],
        },
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Auth0 token exchange failed")
    access_token = resp.json().get("access_token", "")
    return PlainTextResponse(
        f"Access token (copy this for API calls):\n\n{access_token}\n\n"
        "Use with:  Authorization: Bearer <token>"
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/interactors/api/routes/login.py
git commit -m "feat: add /auth/login and /auth/callback for isolated testing"
```

---

## Task 6: Wire `Auth0Adapter` and tighten CORS in `app.py`

**Files:**
- Modify: `src/interactors/api/app.py`

- [ ] **Step 1: Replace `src/interactors/api/app.py` with the updated version**

```python
"""FastAPI application factory for the aipet inference service."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


def _make_storage_adapter():
    if os.getenv("AWS_S3_BUCKET"):
        from adapters.storage.s3 import S3StorageAdapter
        return S3StorageAdapter()
    from adapters.storage.local import LocalStorageAdapter
    return LocalStorageAdapter()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from adapters.auth.auth0 import Auth0Adapter
    from adapters.database import init_db, make_engine
    from adapters.database.model_store import SQLAlchemyModelStore
    from adapters.database.run_store import SQLAlchemyRunStore
    from adapters.inference import LlamaCppInferenceAdapter
    from interactors.api.deps import (
        clear_adapter,
        configure,
        configure_auth,
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

    auth0_domain = os.environ.get("AUTH0_DOMAIN", "")
    auth0_audience = os.environ.get("AUTH0_AUDIENCE", "")
    if auth0_domain and auth0_audience:
        configure_auth(Auth0Adapter(domain=auth0_domain, audience=auth0_audience))
    else:
        logger.warning(
            "AUTH0_DOMAIN or AUTH0_AUDIENCE not set — "
            "protected endpoints will return 500 until configured"
        )

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


from interactors.api.routes.inference import router as inference_router  # noqa: E402
from interactors.api.routes.login import router as login_router  # noqa: E402
from interactors.api.routes.models import router as models_router  # noqa: E402
from interactors.api.routes.runs import router as runs_router  # noqa: E402

app = FastAPI(title="aipet-llm inference service", lifespan=lifespan)

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
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(inference_router)
app.include_router(models_router)
app.include_router(runs_router)
app.include_router(login_router)
```

- [ ] **Step 2: Run the full test suite**

```bash
uv run pytest tests/unit/ tests/integration/test_api.py tests/integration/test_auth.py tests/integration/test_run_api.py tests/integration/test_training_api.py -v
```

Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add src/interactors/api/app.py
git commit -m "feat: wire Auth0Adapter in lifespan, scope CORS via CORS_ORIGINS env var"
```

---

## Self-Review

**Spec coverage:**
- [x] `AuthPort` and `UserContext` in domain layer (Task 1)
- [x] `Auth0Adapter` validates RS256 JWT via JWKS with in-memory key cache (Task 2)
- [x] `require_auth` dependency — 401 on missing/invalid header (Task 3)
- [x] `/infer` protected (Task 4)
- [x] `/api/models` protected at router level (Task 4)
- [x] `/api/runs` protected at router level (Task 4)
- [x] `GET /health` unauthenticated (Task 4 — not changed)
- [x] Login flow `/auth/login` → `/auth/callback` for isolated testing (Task 5)
- [x] `CORS_ORIGINS` env var replaces wildcard; `APP_ENV=development` restores wildcard (Task 6)
- [x] `AUTH0_DOMAIN`, `AUTH0_AUDIENCE`, `AUTH0_CLIENT_ID`, `AUTH0_CLIENT_SECRET`, `AUTH0_CALLBACK_URL` env vars (Task 6)
- [x] Unit tests for adapter and dependency with no live Auth0 tenant required (Tasks 2, 3)
- [x] Integration tests confirm 401 enforcement + existing tests unbroken (Task 4)

**Type consistency:** `AuthPort.authenticate` returns `UserContext | None` — used identically in `Auth0Adapter`, `_StubAuthPort`, `_FakeAuthAdapter`. `configure_auth` / `get_auth` in `deps.py` use `AuthPort` from the same import.

**Placeholder scan:** No TBDs, no vague steps, all code shown in full.

---

## Auth0 Setup Reference (one-time manual step)

Before running in production:

1. Log in to [auth0.com](https://auth0.com) → create a **free tenant**
2. **APIs** → Create API: set identifier (this is `AUTH0_AUDIENCE`)
3. **Applications** → Create Application → **Regular Web Application**: copy Client ID and Client Secret
4. In the application settings, add your callback URL to **Allowed Callback URLs**
5. Set env vars: `AUTH0_DOMAIN`, `AUTH0_AUDIENCE`, `AUTH0_CLIENT_ID`, `AUTH0_CLIENT_SECRET`, `AUTH0_CALLBACK_URL`

Local dev startup:
```bash
APP_ENV=development \
AUTH0_DOMAIN=your-tenant.auth0.com \
AUTH0_AUDIENCE=https://api.aipet.example.com \
AUTH0_CLIENT_ID=abc123 \
AUTH0_CLIENT_SECRET=secret \
AUTH0_CALLBACK_URL=http://localhost:8000/auth/callback \
uv run uvicorn interactors.api.app:app --reload
```

Then navigate to `http://localhost:8000/auth/login` to get a token.
