"""Unit tests for LlamaCppInferenceAdapter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from domain.actions import Action
from domain.models import (
    InferenceRequest,
    InferenceResponse,
    PetStats,
    SceneData,
    SceneObject,
)
from adapters.inference import LlamaCppInferenceAdapter

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
                "adapters.inference.build_prompt",
                return_value="<prompt>",
            ):
                with patch(
                    "adapters.inference.parse_response",
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
            "adapters.inference.build_prompt", return_value="<prompt>"
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
                "adapters.inference.build_prompt", return_value="<prompt>"
            ):
                with patch(
                    "adapters.inference.parse_response",
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
                "adapters.inference.build_prompt", return_value="<prompt>"
            ):
                with patch(
                    "adapters.inference.parse_response",
                    return_value=InferenceResponse(
                        action=Action.IDLE, target_object_id=None
                    ),
                ):
                    adapter.infer(inference_request)

            mock_llama_cls.assert_called_once()

    def test_full_inference_path_text_extracted_and_parsed(
        self, inference_request: InferenceRequest
    ) -> None:
        """End-to-end: adapter extracts text from completion dict and passes it to parse_response."""
        raw_json = json.dumps({"action": "IDLE"})
        completion = {"choices": [{"text": raw_json}]}
        mock_llm_instance = MagicMock(return_value=completion)

        with patch("llama_cpp.Llama", return_value=mock_llm_instance):
            adapter = _make_adapter()
            response = adapter.infer(inference_request)

        # parse_response ran for real on the LLM output — not short-circuited by a mock.
        assert response.action == Action.IDLE
        mock_llm_instance.assert_called_once()
        called_prompt = mock_llm_instance.call_args[0][0]
        assert isinstance(called_prompt, str) and len(called_prompt) > 0

    def test_model_loaded_only_once_across_multiple_calls(
        self, inference_request: InferenceRequest
    ) -> None:
        """The model is instantiated exactly once even across repeated infer() calls."""
        mock_llm_instance = MagicMock(return_value=FAKE_COMPLETION)

        with patch("llama_cpp.Llama", return_value=mock_llm_instance) as mock_llama_cls:
            with patch(
                "adapters.inference.build_prompt", return_value="<prompt>"
            ):
                with patch(
                    "adapters.inference.parse_response",
                    return_value=InferenceResponse(
                        action=Action.IDLE, target_object_id=None
                    ),
                ):
                    adapter = _make_adapter()
                    adapter.infer(inference_request)
                    adapter.infer(inference_request)
                    adapter.infer(inference_request)

        mock_llama_cls.assert_called_once()


# ---------------------------------------------------------------------------
# _ensure_target tests
# ---------------------------------------------------------------------------


def _make_scene(*objects: SceneObject, tick: int = 0) -> SceneData:
    return SceneData(objects=list(objects), tick=tick)


def _make_full_request(*objects: SceneObject) -> InferenceRequest:
    stats = PetStats(hunger=0.9, boredom=0.1, social=0.1, toilet=0.1, tiredness=0.1)
    return InferenceRequest(scene=_make_scene(*objects), pet_stats=stats)


class TestEnsureTarget:
    def setup_method(self):
        self.adapter = LlamaCppInferenceAdapter(model_path="/fake/model.gguf")

    def _ensure(self, response: InferenceResponse, *objects: SceneObject) -> InferenceResponse:
        return self.adapter._ensure_target(response, _make_full_request(*objects))

    def test_picks_closest_of_multiple_valid_objects(self):
        far_bowl = SceneObject(id="bowl_far", type="bowl", distance=20.0)
        close_bowl = SceneObject(id="bowl_close", type="bowl", distance=3.0)
        mid_bowl = SceneObject(id="bowl_mid", type="bowl", distance=10.0)
        response = InferenceResponse(action=Action.EAT, target_object_id=None)

        result = self._ensure(response, far_bowl, close_bowl, mid_bowl)

        assert result.target_object_id == "bowl_close"

    def test_does_not_override_target_model_already_provided(self):
        bowl_a = SceneObject(id="bowl_a", type="bowl", distance=1.0)
        bowl_b = SceneObject(id="bowl_b", type="bowl", distance=5.0)
        response = InferenceResponse(action=Action.EAT, target_object_id="bowl_b")

        result = self._ensure(response, bowl_a, bowl_b)

        assert result.target_object_id == "bowl_b"

    def test_ignores_wrong_type_objects(self):
        close_toy = SceneObject(id="toy_1", type="toy", distance=1.0)
        far_bowl = SceneObject(id="bowl_1", type="bowl", distance=15.0)
        response = InferenceResponse(action=Action.EAT, target_object_id=None)

        result = self._ensure(response, close_toy, far_bowl)

        assert result.target_object_id == "bowl_1"

    def test_no_target_required_actions_are_unchanged(self):
        bowl = SceneObject(id="bowl_1", type="bowl", distance=1.0)
        for action in (Action.IDLE, Action.EXPLORE, Action.TOILET):
            response = InferenceResponse(action=action, target_object_id=None)
            result = self._ensure(response, bowl)
            assert result.target_object_id is None, f"{action} should never have a target"

    def test_returns_unchanged_when_no_valid_object_in_scene(self):
        toy = SceneObject(id="toy_1", type="toy", distance=1.0)
        response = InferenceResponse(action=Action.EAT, target_object_id=None)

        result = self._ensure(response, toy)

        assert result.target_object_id is None

    def test_social_picks_closest_player_or_pet(self):
        far_player = SceneObject(id="player_far", type="player", distance=30.0)
        close_pet = SceneObject(id="pet_close", type="pet", distance=5.0)
        response = InferenceResponse(action=Action.SOCIAL, target_object_id=None)

        result = self._ensure(response, far_player, close_pet)

        assert result.target_object_id == "pet_close"

    def test_replaces_invalid_target_type_with_closest_valid(self):
        toy = SceneObject(id="toy_1", type="toy", distance=1.0)
        far_bowl = SceneObject(id="bowl_far", type="bowl", distance=20.0)
        close_bowl = SceneObject(id="bowl_close", type="bowl", distance=5.0)
        # model hallucinated a toy id for an EAT action
        response = InferenceResponse(action=Action.EAT, target_object_id="toy_1")

        result = self._ensure(response, toy, far_bowl, close_bowl)

        assert result.target_object_id == "bowl_close"

    def test_replaces_nonexistent_target_id_with_closest_valid(self):
        bowl = SceneObject(id="bowl_real", type="bowl", distance=8.0)
        # model returned an id that does not exist in the scene
        response = InferenceResponse(action=Action.EAT, target_object_id="bowl_ghost")

        result = self._ensure(response, bowl)

        assert result.target_object_id == "bowl_real"

    def test_preserves_confidence_when_replacing_invalid_target(self):
        bowl = SceneObject(id="bowl_1", type="bowl", distance=5.0)
        response = InferenceResponse(action=Action.EAT, target_object_id="bad_id", confidence=0.85)

        result = self._ensure(response, bowl)

        assert result.target_object_id == "bowl_1"
        assert result.confidence == 0.85
