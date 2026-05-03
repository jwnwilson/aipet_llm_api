from abc import ABC, abstractmethod

from src.domain.models import InferenceRequest, InferenceResponse


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
