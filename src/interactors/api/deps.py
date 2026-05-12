"""Shared FastAPI dependencies — all adapter and store singletons."""

from __future__ import annotations

from domain.ports import InferencePort, ModelStorePort, RunStorePort

# ---------------------------------------------------------------------------
# Inference adapter
# ---------------------------------------------------------------------------

_adapter: InferencePort | None = None


def get_adapter() -> InferencePort:
    if _adapter is None:
        raise RuntimeError("InferencePort adapter has not been configured.")
    return _adapter


def configure(adapter: InferencePort) -> None:
    global _adapter
    _adapter = adapter


def clear_adapter() -> None:
    global _adapter
    _adapter = None


# ---------------------------------------------------------------------------
# Model store
# ---------------------------------------------------------------------------

_model_store: ModelStorePort | None = None


def get_model_store() -> ModelStorePort:
    if _model_store is None:
        raise RuntimeError("ModelStorePort has not been configured.")
    return _model_store


def configure_model_store(store: ModelStorePort) -> None:
    global _model_store
    _model_store = store


# ---------------------------------------------------------------------------
# Run store
# ---------------------------------------------------------------------------

_run_store: RunStorePort | None = None


def get_run_store() -> RunStorePort:
    if _run_store is None:
        raise RuntimeError("RunStorePort has not been configured.")
    return _run_store


def configure_run_store(store: RunStorePort) -> None:
    global _run_store
    _run_store = store
