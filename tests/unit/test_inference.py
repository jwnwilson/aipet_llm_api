"""Unit tests for LlamaCppInferenceAdapter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.domain.actions import Action
from src.domain.models import (
    InferenceRequest,
    InferenceResponse,
    PetStats,
    SceneData,
)
from src.infrastructure.inference import LlamaCppInferenceAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_COMPLETION = {
    "choices": [{"text": json.dumps({"action": "IDLE", "target_object_id": None})}]
}


@pytest.fixture()
def inference_request() -> InferenceRequest:
    scene = SceneData(objects=[], tick=0)
    stats = PetStats(
        hunger=0.0,
        boredom=0.0,
        social=0.0,
        toilet=0.0,
        tiredness=0.0,
    )
    return InferenceRequest(scene=scene, pet_stats=stats)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter() -> LlamaCppInferenceAdapter:
    """Return an adapter pointed at a fake model path."""
    return LlamaCppInferenceAdapter(model_path="/fake/model.gguf", context_size=512)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLlamaCppInferenceAdapter:
    def test_returns_idle_on_valid_mock_response(
        self, inference_request: InferenceRequest
    ) -> None:
        """Adapter returns Action.IDLE when the LLM mock returns a valid JSON payload."""
        mock_llm_instance = MagicMock(return_value=FAKE_COMPLETION)

        with patch("llama_cpp.Llama", return_value=mock_llm_instance) as mock_llama_cls:
            with patch(
                "src.infrastructure.inference.build_prompt",
                return_value="<prompt>",
            ):
                with patch(
                    "src.infrastructure.inference.parse_response",
                    return_value=InferenceResponse(
                        action=Action.IDLE, target_object_id=None
                    ),
                ):
                    adapter = _make_adapter()
                    response = adapter.infer(inference_request)

        assert isinstance(response, InferenceResponse)
        assert response.action == Action.IDLE

    def test_returns_idle_when_llm_raises(
        self, inference_request: InferenceRequest
    ) -> None:
        """Adapter swallows exceptions and returns IDLE instead of propagating."""
        with patch(
            "src.infrastructure.inference.build_prompt", return_value="<prompt>"
        ):
            with patch("llama_cpp.Llama", side_effect=RuntimeError("model not found")):
                adapter = _make_adapter()
                response = adapter.infer(inference_request)

        assert isinstance(response, InferenceResponse)
        assert response.action == Action.IDLE
        assert response.target_object_id is None

    def test_returns_idle_when_parse_response_raises(
        self, inference_request: InferenceRequest
    ) -> None:
        """Adapter swallows parse errors and returns IDLE."""
        mock_llm_instance = MagicMock(return_value=FAKE_COMPLETION)

        with patch("llama_cpp.Llama", return_value=mock_llm_instance):
            with patch(
                "src.infrastructure.inference.build_prompt", return_value="<prompt>"
            ):
                with patch(
                    "src.infrastructure.inference.parse_response",
                    side_effect=ValueError("bad JSON"),
                ):
                    adapter = _make_adapter()
                    response = adapter.infer(inference_request)

        assert response.action == Action.IDLE

    def test_model_loaded_lazily(self, inference_request: InferenceRequest) -> None:
        """The Llama constructor is not called until infer() is first invoked."""
        with patch("llama_cpp.Llama", return_value=MagicMock(return_value=FAKE_COMPLETION)) as mock_llama_cls:
            adapter = _make_adapter()
            mock_llama_cls.assert_not_called()  # not yet loaded

            with patch(
                "src.infrastructure.inference.build_prompt", return_value="<prompt>"
            ):
                with patch(
                    "src.infrastructure.inference.parse_response",
                    return_value=InferenceResponse(
                        action=Action.IDLE, target_object_id=None
                    ),
                ):
                    adapter.infer(inference_request)

            mock_llama_cls.assert_called_once()

    def test_model_loaded_only_once_across_multiple_calls(
        self, inference_request: InferenceRequest
    ) -> None:
        """The model is instantiated exactly once even across repeated infer() calls."""
        mock_llm_instance = MagicMock(return_value=FAKE_COMPLETION)

        with patch("llama_cpp.Llama", return_value=mock_llm_instance) as mock_llama_cls:
            with patch(
                "src.infrastructure.inference.build_prompt", return_value="<prompt>"
            ):
                with patch(
                    "src.infrastructure.inference.parse_response",
                    return_value=InferenceResponse(
                        action=Action.IDLE, target_object_id=None
                    ),
                ):
                    adapter = _make_adapter()
                    adapter.infer(inference_request)
                    adapter.infer(inference_request)
                    adapter.infer(inference_request)

        mock_llama_cls.assert_called_once()
