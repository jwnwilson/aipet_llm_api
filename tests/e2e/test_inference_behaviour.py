"""Behavioural inference tests — require a real model, not run in the standard integration suite.

Run locally with:
    AIPET_TEST_MODEL_PATH=models/test_aipet.gguf uv run pytest tests/e2e/test_inference_behaviour.py -v
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from adapters.inference import LlamaCppInferenceAdapter
from adapters.prompt import parse_response
from domain.actions import Action
from domain.models import InferenceRequest, PetStats, SceneData, SceneObject

_DEFAULT_MODEL_PATH = Path(__file__).parents[2] / "models" / "test_aipet.gguf"
MODEL_PATH = Path(os.environ.get("AIPET_TEST_MODEL_PATH", str(_DEFAULT_MODEL_PATH)))

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"Real model not found at {MODEL_PATH}",
)

_TARGET_REQUIRED = {Action.EAT, Action.DRINK, Action.PLAY, Action.FETCH,
                    Action.SLEEP, Action.SOCIAL, Action.FOLLOW}

_PARSE_RESPONSE = "adapters.inference.parse_response"


@pytest.fixture(scope="module")
def adapter() -> LlamaCppInferenceAdapter:
    return LlamaCppInferenceAdapter(model_path=str(MODEL_PATH), context_size=2048)


def _request(objects: list[SceneObject] | None = None, **stats) -> InferenceRequest:
    defaults = dict(hunger=0.5, boredom=0.3, social=0.2, toilet=0.1, tiredness=0.4)
    defaults.update(stats)
    return InferenceRequest(
        scene=SceneData(objects=objects or [], tick=1),
        pet_stats=PetStats(**defaults),
    )


class TestRealInferenceOutputs:
    def test_llm_produces_non_empty_text(self, adapter):
        with patch(_PARSE_RESPONSE, wraps=parse_response) as spy:
            adapter.infer(_request())

        spy.assert_called_once()
        raw_text: str = spy.call_args[0][0]
        assert isinstance(raw_text, str) and len(raw_text) > 0, (
            "LLM returned empty text — model output never reached parse_response"
        )

    def test_empty_scene_action_is_untargeted(self, adapter):
        response = adapter.infer(_request())
        assert response.action not in _TARGET_REQUIRED, (
            f"Got {response.action} but no objects were in the scene"
        )

    def test_bowl_scene_target_id_matches_scene(self, adapter):
        scene_objects = [SceneObject(id="bowl1", type="bowl", distance=1.5)]
        response = adapter.infer(_request(objects=scene_objects, hunger=0.9))

        scene_ids = {o.id for o in scene_objects}
        assert response.target_object_id is not None, (
            f"Got action={response.action} but target_object_id is null — bowl is in scene"
        )
        assert response.target_object_id in scene_ids, (
            f"target_object_id {response.target_object_id!r} not in scene {scene_ids}"
        )

    def test_model_loaded_once_across_calls(self, adapter):
        with patch(_PARSE_RESPONSE, wraps=parse_response) as spy:
            adapter.infer(_request())
            adapter.infer(_request())

        assert adapter._llm is not None
        assert spy.call_count == 2


class TestStatPrioritization:
    """Verify the model picks an action that satisfies the pet's most urgent need.

    A stat value of 0.95 means that need is critical; all others are kept at 0.1
    so there is no ambiguity about which drive should win. The scene always
    contains the object required by the dominant stat plus one unrelated distractor
    so the model must discriminate rather than default to the first available action.
    """

    def test_high_hunger_triggers_eat_or_drink(self, adapter):
        objects = [
            SceneObject(id="bowl1", type="bowl", distance=2.0),
            SceneObject(id="toy1", type="toy", distance=3.0),
        ]
        response = adapter.infer(
            _request(objects=objects, hunger=0.95, boredom=0.1, social=0.1, toilet=0.1, tiredness=0.1)
        )
        assert response.action in {Action.EAT, Action.DRINK}, (
            f"Expected EAT or DRINK for hunger=0.95, got {response.action}"
        )

    def test_high_toilet_triggers_toilet(self, adapter):
        objects = [
            SceneObject(id="toy1", type="toy", distance=2.0),
            SceneObject(id="bed1", type="bed", distance=3.0),
        ]
        response = adapter.infer(
            _request(objects=objects, hunger=0.1, boredom=0.1, social=0.1, toilet=0.95, tiredness=0.1)
        )
        assert response.action == Action.TOILET, (
            f"Expected TOILET for toilet=0.95, got {response.action}"
        )

    def test_high_tiredness_triggers_sleep(self, adapter):
        objects = [
            SceneObject(id="bed1", type="bed", distance=2.0),
            SceneObject(id="bowl1", type="bowl", distance=3.0),
        ]
        response = adapter.infer(
            _request(objects=objects, hunger=0.1, boredom=0.1, social=0.1, toilet=0.1, tiredness=0.95)
        )
        assert response.action == Action.SLEEP, (
            f"Expected SLEEP for tiredness=0.95, got {response.action}"
        )

    def test_high_social_triggers_social_or_follow(self, adapter):
        objects = [
            SceneObject(id="player1", type="player", distance=2.0),
            SceneObject(id="toy1", type="toy", distance=3.0),
        ]
        response = adapter.infer(
            _request(objects=objects, hunger=0.1, boredom=0.1, social=0.95, toilet=0.1, tiredness=0.1)
        )
        assert response.action in {Action.SOCIAL, Action.FOLLOW}, (
            f"Expected SOCIAL or FOLLOW for social=0.95, got {response.action}"
        )

    def test_high_boredom_triggers_play_or_fetch(self, adapter):
        objects = [
            SceneObject(id="toy1", type="toy", distance=2.0),
            SceneObject(id="bed1", type="bed", distance=3.0),
        ]
        response = adapter.infer(
            _request(objects=objects, hunger=0.1, boredom=0.95, social=0.1, toilet=0.1, tiredness=0.1)
        )
        assert response.action in {Action.PLAY, Action.FETCH}, (
            f"Expected PLAY or FETCH for boredom=0.95, got {response.action}"
        )
