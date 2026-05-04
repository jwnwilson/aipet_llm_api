"""LlamaCpp-backed inference adapter implementing InferencePort."""

from __future__ import annotations

import logging
from typing import Any

import llama_cpp

from domain.actions import Action
from domain.models import InferenceRequest, InferenceResponse
from domain.ports import InferencePort
from infrastructure.prompt import build_prompt, parse_response

logger = logging.getLogger(__name__)

# Actions that require a target object and the scene types they must come from.
_ACTION_TARGET_TYPES: dict[Action, set[str]] = {
    Action.EAT: {"bowl"},
    Action.DRINK: {"bowl"},
    Action.PLAY: {"toy"},
    Action.FETCH: {"toy"},
    Action.SLEEP: {"bed"},
    Action.SOCIAL: {"player", "pet"},
    Action.FOLLOW: {"player", "pet"},
}


class LlamaCppInferenceAdapter(InferencePort):
    """InferencePort implementation backed by a GGUF-quantised model via llama-cpp-python.

    The model is loaded lazily on the first call to ``infer`` so that
    construction is cheap and test set-up does not require a real model file.
    """

    def __init__(self, model_path: str, context_size: int = 2048) -> None:
        self._model_path = model_path
        self._context_size = context_size
        self._llm: llama_cpp.Llama | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> llama_cpp.Llama:
        """Instantiate and return the Llama model (called once, lazily)."""
        logger.info("Loading GGUF model from %s", self._model_path)
        return llama_cpp.Llama(
            model_path=self._model_path,
            n_ctx=self._context_size,
            verbose=False,
        )

    def _get_llm(self) -> llama_cpp.Llama:
        if self._llm is None:
            self._llm = self._load_model()
        return self._llm

    def _ensure_target(self, response: InferenceResponse, request: InferenceRequest) -> InferenceResponse:
        """If action requires a target but model omitted it, pick the closest valid scene object."""
        if response.target_object_id is not None:
            return response
        required_types = _ACTION_TARGET_TYPES.get(response.action)
        if not required_types:
            return response
        candidates = [o for o in request.scene.objects if o.type in required_types]
        if not candidates:
            return response
        closest = min(candidates, key=lambda o: o.distance)
        return InferenceResponse(
            action=response.action,
            target_object_id=closest.id,
            confidence=response.confidence,
        )

    # ------------------------------------------------------------------
    # InferencePort implementation
    # ------------------------------------------------------------------

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference and return a structured response.

        On any recoverable error (model loading failure, malformed output, etc.)
        a warning is logged and ``InferenceResponse(action=Action.IDLE)`` is
        returned so the pet remains in a safe, neutral state.
        """
        fallback = InferenceResponse(action=Action.IDLE, target_object_id=None)

        try:
            prompt = build_prompt(request)
            llm = self._get_llm()
            completion: Any = llm(
                prompt,
                max_tokens=256,
                temperature=0.1,
                stop=["\n", "```"],
            )
            raw_text: str = completion["choices"][0]["text"]
            response = parse_response(raw_text)
            return self._ensure_target(response, request)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Inference failed, returning IDLE fallback. Reason: %s",
                exc,
                exc_info=True,
            )
            return fallback
