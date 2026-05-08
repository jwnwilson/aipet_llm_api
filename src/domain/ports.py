from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from domain.models import InferenceRequest, InferenceResponse, RemoteTrainConfig


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
