from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Literal, TypeVar

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

TDomain = TypeVar("TDomain")
TConfig = TypeVar("TConfig")


class StoragePort(ABC):
    """Abstract interface for storing and retrieving model artifact files (GGUFs, checkpoints).

    Keys are relative paths such as ``workflow/{run_id}/model.gguf``.  Backends map
    these to their own namespace (local filesystem prefix, S3 key prefix, etc.).
    """

    @abstractmethod
    def upload(self, local_path: Path, key: str) -> None:
        """Copy a local file into storage under ``key``."""

    @abstractmethod
    def download(self, key: str, dest: Path) -> None:
        """Fetch the artifact at ``key`` to ``dest`` (creates parent dirs).

        Must be a no-op when the source and destination resolve to the same path
        (i.e. local storage where the file is already in place).
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if ``key`` exists in storage."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key`` from storage (silent no-op if already absent)."""


class InferencePort(ABC):
    """Abstract interface the domain layer expects from any LLM inference backend.

    Contract:
    - ``infer`` must always return a valid ``InferenceResponse``.
    - ``infer`` must never raise on recoverable LLM errors; instead return
      ``InferenceResponse(action=Action.IDLE)`` so the pet remains in a safe,
      neutral state while the problem is handled upstream.
    """

    @abstractmethod
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference for the given request and return a structured response.

        Args:
            request: The scene and pet-stat context to reason over.

        Returns:
            A valid ``InferenceResponse``.  On recoverable errors the
            implementation must return ``InferenceResponse(action=Action.IDLE)``
            rather than raising an exception.
        """


class RemoteTrainingPort(ABC):
    """Abstract interface for offloading fine-tuning to a remote compute backend.

    Implementations live in ``src/adapters/`` — never in the domain layer.

    Contract:
    - ``submit`` must start the remote job and return an opaque ``run_id``.
    - ``status`` must be non-blocking (poll, don't wait).
    - ``download`` must be called only after ``status`` returns ``"done"``; it
      fetches the checkpoint into ``dest`` and returns the local path as a string.
    """

    @abstractmethod
    def submit(self, config: RemoteTrainConfig) -> str:
        """Upload data + code and start the remote training job.

        Returns:
            An opaque ``run_id`` string used by ``status`` and ``download``.
        """

    @abstractmethod
    def status(self, run_id: str) -> Literal["pending", "running", "done", "failed"]:
        """Poll the current state of the remote job without blocking."""

    @abstractmethod
    def download(self, run_id: str, dest: Path) -> str:
        """Fetch the trained checkpoint into ``dest`` and return the local path."""

    def logs(self, run_id: str) -> str:  # noqa: ARG002
        """Return recent log output for the running job (best-effort, may be empty)."""
        return ""

    def eval(self, run_id: str, eval_data: str) -> tuple[float, bool]:  # noqa: ARG002
        """Run evaluation on the remote machine and return ``(valid_pct, passed)``.

        Raises ``NotImplementedError`` if the backend does not support remote
        evaluation (e.g. Kaggle batch kernels).  ``evaluate_activity`` catches
        this and raises an ``ApplicationError`` with a descriptive message.
        """
        raise NotImplementedError


class StorePort(ABC, Generic[TDomain, TConfig]):
    """Generic CRUD base for any domain entity store."""

    @abstractmethod
    def list(self) -> list[TDomain]:
        """Return all stored entities."""

    @abstractmethod
    def get(self, id: str) -> TDomain | None:
        """Return the entity with the given id, or None if not found."""

    @abstractmethod
    def create(self, config: TConfig) -> TDomain:
        """Persist a new entity and return it with id and timestamps."""

    @abstractmethod
    def update(self, id: str, config: TConfig) -> TDomain | None:
        """Update an existing entity; return updated entity or None if not found."""

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete an entity by id; return True if deleted, False if not found."""


class ModelStorePort(StorePort["TrainingModel", "TrainingModelConfig"]):
    """Abstract interface for persisting training model configurations."""

    @abstractmethod
    def activate(self, id: str) -> TrainingModel | None:
        """Set ``is_active=True`` for this model, ``False`` for all others.

        Returns the updated model, or ``None`` if ``id`` is not found.
        """

    @abstractmethod
    def active(self) -> TrainingModel | None:
        """Return the currently active model, or ``None`` if none is set."""


class RunStorePort(StorePort["RunRecord", "RunConfig"]):
    """Abstract interface for persisting training run records."""

    @abstractmethod
    def list(self, model_id: str | None = None) -> list[RunRecord]:  # type: ignore[override]
        """Return all runs, optionally filtered by model_id."""

    @abstractmethod
    def update_status(self, run_id: str, status: RunStatus) -> RunRecord | None:
        """Set the run status; return updated record or None if not found."""

    @abstractmethod
    def update_eval(self, run_id: str, valid_pct: float) -> RunRecord | None:
        """Persist the eval result; return updated record or None if not found."""
