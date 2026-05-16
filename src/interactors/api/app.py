"""FastAPI application factory for the aipet inference service."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger(__name__)


def _make_storage_adapter():
    if os.getenv("AWS_S3_BUCKET"):
        from adapters.storage.s3 import S3StorageAdapter
        return S3StorageAdapter()
    from adapters.storage.local import LocalStorageAdapter
    return LocalStorageAdapter()


def _resolve_model_path(storage) -> str:
    """Return a local model path, downloading from S3 via MODEL_S3_KEY if configured."""
    s3_key = os.getenv("MODEL_S3_KEY")
    if s3_key:
        local_path = Path("models/cache/default/model.gguf")
        try:
            storage.download(s3_key, local_path)
            log.info("Downloaded model from MODEL_S3_KEY=%s to %s", s3_key, local_path)
            return str(local_path)
        except Exception:
            log.warning("Could not download model from MODEL_S3_KEY=%s", s3_key, exc_info=True)
    return os.getenv("MODEL_PATH", "models/aipet.gguf")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from adapters.auth.auth0 import Auth0Adapter
    from adapters.database import init_db, make_engine
    from adapters.database.model_store import SQLAlchemyModelStore
    from adapters.database.run_store import SQLAlchemyRunStore
    from adapters.inference import LlamaCppInferenceAdapter
    from interactors.api.deps import (
        clear_adapter,
        clear_auth,
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
    elif os.getenv("APP_ENV") == "development":
        from adapters.auth.fake import FakeAuthAdapter
        log.warning("AUTH0 not configured — using FakeAuthAdapter for local development")
        configure_auth(FakeAuthAdapter())
    else:
        log.warning(
            "AUTH0_DOMAIN or AUTH0_AUDIENCE not set — "
            "protected endpoints will return 500 until configured"
        )

    active = store.active()
    if active and active.gguf_path:
        local_path = Path("models/cache") / active.id / "model.gguf"
        try:
            storage.download(active.gguf_path, local_path)
            model_path = str(local_path)
            log.info("Loading active model %s from storage key %s", active.id, active.gguf_path)
        except Exception:
            log.warning(
                "Could not load active model %s from storage; falling back",
                active.id,
                exc_info=True,
            )
            model_path = _resolve_model_path(storage)
    else:
        model_path = _resolve_model_path(storage)

    adapter = LlamaCppInferenceAdapter(model_path=model_path)
    try:
        adapter.load()
        log.info("Model loaded into memory: %s", model_path)
    except Exception as exc:
        log.warning("Could not pre-load model — will load on first request: %s", exc)
    configure(adapter)

    try:
        yield
    finally:
        clear_adapter()
        clear_auth()


from interactors.api.routes.admin import router as admin_router  # noqa: E402
from interactors.api.routes.inference import router as inference_router  # noqa: E402
from interactors.api.routes.login import router as login_router  # noqa: E402
from interactors.api.routes.models import router as models_router  # noqa: E402
from interactors.api.routes.runs import router as runs_router  # noqa: E402

_auth0_audience = os.getenv("AUTH0_AUDIENCE", "")
_auth0_client_id = os.getenv("AUTH0_CLIENT_ID", "")

app = FastAPI(
    title="aipet-llm inference service",
    lifespan=lifespan,
    swagger_ui_oauth2_redirect_url="/docs/oauth2-redirect",
    swagger_ui_init_oauth={
        "clientId": _auth0_client_id,
        "additionalQueryStringParams": {"audience": _auth0_audience},
        "usePkceWithAuthorizationCodeGrant": True,
        "scopes": "openid profile email",
    },
)

_cors_raw = os.getenv("CORS_ORIGINS", "")
if os.getenv("APP_ENV") == "development":
    _cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"]
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

app.include_router(admin_router)
app.include_router(inference_router)
app.include_router(models_router)
app.include_router(runs_router)
app.include_router(login_router)

