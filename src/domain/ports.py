from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from domain.models import InferenceRequest, InferenceResponse, RemoteTrainConfig, TrainingModel, TrainingModelConfig


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


class ModelStorePort(ABC):
    """Abstract interface for persisting training model configurations."""

    @abstractmethod
    def list(self) -> list[TrainingModel]:
        """Return all stored training models."""

    @abstractmethod
    def get(self, id: str) -> TrainingModel | None:
        """Return the model with the given id, or None if not found."""

    @abstractmethod
    def create(self, config: TrainingModelConfig) -> TrainingModel:
        """Persist a new training model and return it with id and timestamps."""

    @abstractmethod
    def update(self, id: str, config: TrainingModelConfig) -> TrainingModel | None:
        """Update an existing model; return updated model or None if not found."""

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete a model by id; return True if deleted, False if not found."""
