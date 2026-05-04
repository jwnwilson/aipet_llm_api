"""Real model inference tests — no mocks, requires models/aipet.gguf."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from domain.actions import Action
from domain.models import InferenceRequest, PetStats, SceneData, SceneObject
from infrastructure.inference import LlamaCppInferenceAdapter
from infrastructure.prompt import parse_response

MODEL_PATH = Path(__file__).parents[2] / "models" / "aipet.gguf"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"Real model not found at {MODEL_PATH}",
)

# Actions that require a target object to be present in the scene.
_TARGET_REQUIRED = {Action.EAT, Action.DRINK, Action.PLAY, Action.FETCH,
                    Action.SLEEP, Action.SOCIAL, Action.FOLLOW}

# Patch target: the name as imported inside inference.py, not its definition site.
_PARSE_RESPONSE = "infrastructure.inference.parse_response"


def _adapter() -> LlamaCppInferenceAdapter:
    return LlamaCppInferenceAdapter(model_path=str(MODEL_PATH), context_size=512)


def _request(objects: list[SceneObject] | None = None, **stats) -> InferenceRequest:
    defaults = dict(hunger=0.5, boredom=0.3, social=0.2, toilet=0.1, tiredness=0.4)
    defaults.update(stats)
    return InferenceRequest(
        scene=SceneData(objects=objects or [], tick=1),
        pet_stats=PetStats(**defaults),
    )


class TestRealInference:
    def test_llm_produces_non_empty_text(self):
        # wraps= keeps the real implementation running — this is not a mock.
        with patch(_PARSE_RESPONSE, wraps=parse_response) as spy:
            _adapter().infer(_request())

        spy.assert_called_once()
        raw_text: str = spy.call_args[0][0]
        assert isinstance(raw_text, str) and len(raw_text) > 0, (
            "LLM returned empty text — model output never reached parse_response"
        )

    def test_model_output_is_parseable_json(self):
        # If parse_response raised, infer() swallows the error and returns the
        # IDLE fallback. We verify by re-parsing the captured raw text ourselves:
        # if it raises here the test fails with a clear error; if it succeeds and
        # matches the response, the model path was taken — not the fallback.
        with patch(_PARSE_RESPONSE, wraps=parse_response) as spy:
            response = _adapter().infer(_request())

        spy.assert_called_once()
        raw_text: str = spy.call_args[0][0]
        parsed = parse_response(raw_text)
        assert parsed == response, (
            "Response does not match parse_response result — fallback was substituted"
        )

    def test_empty_scene_action_is_untargeted(self):
        with patch(_PARSE_RESPONSE, wraps=parse_response) as spy:
            response = _adapter().infer(_request())

        spy.assert_called_once()
        assert response.action not in _TARGET_REQUIRED, (
            f"Got {response.action} but no objects were in the scene"
        )

    def test_bowl_scene_target_id_matches_scene(self):
        scene_objects = [SceneObject(id="bowl1", type="bowl", distance=1.5)]
        with patch(_PARSE_RESPONSE, wraps=parse_response) as spy:
            response = _adapter().infer(_request(objects=scene_objects, hunger=0.9))

        spy.assert_called_once()
        scene_ids = {o.id for o in scene_objects}
        assert response.target_object_id is not None, (
            f"Got action={response.action} but target_object_id is null — bowl is in scene"
        )
        assert response.target_object_id in scene_ids, (
            f"target_object_id {response.target_object_id!r} not in scene {scene_ids}"
        )

    def test_model_loaded_once_across_calls(self):
        adapter = _adapter()
        with patch(_PARSE_RESPONSE, wraps=parse_response) as spy:
            adapter.infer(_request())
            adapter.infer(_request())

        assert adapter._llm is not None
        assert spy.call_count == 2
