"""LlamaCpp-backed inference adapter implementing InferencePort."""

from __future__ import annotations

import logging
from typing import Any

import llama_cpp

from src.domain.actions import Action
from src.domain.models import InferenceRequest, InferenceResponse
from src.domain.ports import InferencePort
from src.infrastructure.prompt import build_prompt, parse_response

logger = logging.getLogger(__name__)


class LlamaCppInferenceAdapter(InferencePort):
    """InferencePort implementation backed by a GGUF-quantised model via llama-cpp-python.

    The model is loaded lazily on the first call to ``infer`` so that
    construction is cheap and test set-up does not require a real model file.
    """

    def __init__(self, model_path: str, context_size: int = 512) -> None:
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
                max_tokens=128,
                temperature=0.1,
                stop=[],
            )
            raw_text: str = completion["choices"][0]["text"]
            return parse_response(raw_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Inference failed, returning IDLE fallback. Reason: %s",
                exc,
                exc_info=True,
            )
            return fallback
