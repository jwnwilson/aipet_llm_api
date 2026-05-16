"""Unit tests for LlamaCppInferenceAdapter.

Design notes:
- We avoid patching `build_prompt` and `parse_response` imports because they are
  pure functions with no I/O; letting them run for real makes tests more meaningful.
- We still patch `llama_cpp.Llama` because that IS the external I/O boundary — the
  adapter's whole purpose is to wrap it.  We patch at the correct location
  (`llama_cpp.Llama`) rather than at an import alias.
- Assertions check return values (InferenceResponse fields) and adapter state
  (_llm attribute) rather than mock call counts wherever possible.
"""

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
# Helpers
# ---------------------------------------------------------------------------

IDLE_COMPLETION = {
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


def _make_adapter() -> LlamaCppInferenceAdapter:
    """Return an adapter pointed at a fake model path."""
    return LlamaCppInferenceAdapter(model_path="/fake/model.gguf", context_size=512)


def _mock_llm(completion: dict = IDLE_COMPLETION) -> MagicMock:
    """Return a callable mock that returns the given completion dict."""
    return MagicMock(return_value=completion)


# ---------------------------------------------------------------------------
# Tests: infer() return values
# ---------------------------------------------------------------------------


class TestInferReturnValues:
    def test_valid_llm_response_returns_inference_response(
        self, inference_request: InferenceRequest
    ) -> None:
        """infer() returns InferenceResponse when the LLM emits valid JSON."""
        with patch("llama_cpp.Llama", return_value=_mock_llm()):
            adapter = _make_adapter()
            result = adapter.infer(inference_request)

        assert isinstance(result, InferenceResponse)
        assert result.action == Action.IDLE

    def test_returns_idle_when_llm_raises_on_load(
        self, inference_request: InferenceRequest
    ) -> None:
        """When the model file cannot be loaded, infer() falls back to IDLE."""
        with patch("llama_cpp.Llama", side_effect=RuntimeError("model not found")):
            adapter = _make_adapter()
            result = adapter.infer(inference_request)

        assert result.action == Action.IDLE
        assert result.target_object_id is None

    def test_returns_idle_when_llm_raises_on_call(
        self, inference_request: InferenceRequest
    ) -> None:
        """When calling the LLM raises, infer() falls back to IDLE."""
        mock_llm = MagicMock(side_effect=RuntimeError("OOM"))
        with patch("llama_cpp.Llama", return_value=mock_llm):
            adapter = _make_adapter()
            result = adapter.infer(inference_request)

        assert result.action == Action.IDLE
        assert result.target_object_id is None

    def test_returns_idle_when_llm_emits_invalid_json(
        self, inference_request: InferenceRequest
    ) -> None:
        """When the LLM outputs unparseable text, infer() falls back to IDLE."""
        bad_completion = {"choices": [{"text": "I cannot decide right now."}]}
        with patch("llama_cpp.Llama", return_value=_mock_llm(bad_completion)):
            adapter = _make_adapter()
            result = adapter.infer(inference_request)

        assert result.action == Action.IDLE

    def test_full_inference_path_parses_real_llm_output(
        self, inference_request: InferenceRequest
    ) -> None:
        """End-to-end: adapter extracts text from completion dict and parses it."""
        raw_json = json.dumps({"action": "EXPLORE"})
        completion = {"choices": [{"text": raw_json}]}
        with patch("llama_cpp.Llama", return_value=_mock_llm(completion)):
            adapter = _make_adapter()
            result = adapter.infer(inference_request)

        assert result.action == Action.EXPLORE

    def test_non_idle_action_is_returned_verbatim(
        self, inference_request: InferenceRequest
    ) -> None:
        """infer() must not silently replace a valid non-IDLE response with IDLE."""
        eat_completion = {"choices": [{"text": json.dumps({"action": "TOILET"})}]}
        with patch("llama_cpp.Llama", return_value=_mock_llm(eat_completion)):
            adapter = _make_adapter()
            result = adapter.infer(inference_request)

        assert result.action == Action.TOILET


# ---------------------------------------------------------------------------
# Tests: lazy model loading — verified via observable _llm state
# ---------------------------------------------------------------------------


class TestLazyModelLoading:
    def test_model_not_loaded_at_construction(self) -> None:
        """The Llama model must not be instantiated during __init__."""
        with patch("llama_cpp.Llama") as mock_llama_cls:
            adapter = _make_adapter()
            # Observable state: _llm should still be None
            assert adapter._llm is None
            mock_llama_cls.assert_not_called()

    def test_model_loaded_after_first_infer(
        self, inference_request: InferenceRequest
    ) -> None:
        """After the first infer() call, _llm must be non-None (model was loaded)."""
        with patch("llama_cpp.Llama", return_value=_mock_llm()):
            adapter = _make_adapter()
            adapter.infer(inference_request)
            assert adapter._llm is not None

    def test_model_not_reloaded_on_subsequent_calls(
        self, inference_request: InferenceRequest
    ) -> None:
        """After three infer() calls the _llm instance must be the same object."""
        with patch("llama_cpp.Llama", return_value=_mock_llm()) as mock_llama_cls:
            adapter = _make_adapter()
            adapter.infer(inference_request)
            first_llm = adapter._llm
            adapter.infer(inference_request)
            adapter.infer(inference_request)
            second_llm = adapter._llm

        # Same object — not re-created
        assert first_llm is second_llm
        # Llama class constructor called exactly once
        mock_llama_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: lifecycle methods (load / release)
# ---------------------------------------------------------------------------


class TestLifecycleMethods:
    def test_load_eagerly_populates_llm(self) -> None:
        """load() must populate _llm without requiring an infer() call."""
        with patch("llama_cpp.Llama", return_value=MagicMock()) as mock_cls:
            adapter = _make_adapter()
            assert adapter._llm is None
            adapter.load()
            assert adapter._llm is not None
            mock_cls.assert_called_once()

    def test_load_is_noop_when_already_loaded(self) -> None:
        """Calling load() a second time must not re-instantiate the model."""
        with patch("llama_cpp.Llama", return_value=MagicMock()) as mock_cls:
            adapter = _make_adapter()
            adapter.load()
            first_llm = adapter._llm
            adapter.load()
            assert adapter._llm is first_llm
            mock_cls.assert_called_once()

    def test_release_clears_llm_reference(self) -> None:
        """release() must set _llm to None so the model can be garbage-collected."""
        with patch("llama_cpp.Llama", return_value=MagicMock()):
            adapter = _make_adapter()
            adapter.load()
            assert adapter._llm is not None
            adapter.release()
            assert adapter._llm is None

    def test_reload_after_release_creates_new_instance(self) -> None:
        """load() after release() must load the model again."""
        with patch("llama_cpp.Llama", return_value=MagicMock()) as mock_cls:
            adapter = _make_adapter()
            adapter.load()
            adapter.release()
            adapter.load()
            assert adapter._llm is not None
            assert mock_cls.call_count == 2


# ---------------------------------------------------------------------------
# _ensure_target tests (domain rule: adapter must provide a valid target)
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
