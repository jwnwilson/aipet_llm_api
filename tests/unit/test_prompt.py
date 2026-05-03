"""Unit tests for src/infrastructure/prompt.py."""

import pytest

from src.domain.actions import Action
from src.domain.models import InferenceRequest, InferenceResponse, PetStats, SceneData, SceneObject
from src.infrastructure.prompt import build_prompt, parse_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(objects: list[SceneObject] | None = None, tick: int = 0) -> InferenceRequest:
    scene = SceneData(objects=objects or [], tick=tick)
    stats = PetStats(hunger=0.5, boredom=0.3, social=0.2, toilet=0.1, tiredness=0.4)
    return InferenceRequest(scene=scene, pet_stats=stats)


def _bowl_object() -> SceneObject:
    return SceneObject(id="bowl_1", type="bowl", distance=2.0)


def _toy_object() -> SceneObject:
    return SceneObject(id="toy_1", type="toy", distance=3.0)


def _bed_object() -> SceneObject:
    return SceneObject(id="bed_1", type="bed", distance=1.5)


def _player_object() -> SceneObject:
    return SceneObject(id="player_1", type="player", distance=4.0)


# ---------------------------------------------------------------------------
# build_prompt tests
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_bowl_in_scene_includes_eat_and_drink(self):
        request = _make_request(objects=[_bowl_object()])
        prompt = build_prompt(request)
        assert "EAT" in prompt
        assert "DRINK" in prompt

    def test_bowl_in_scene_excludes_sleep(self):
        request = _make_request(objects=[_bowl_object()])
        prompt = build_prompt(request)
        # Extract only the "Available actions" line to avoid matching schema enum values.
        actions_line = next(l for l in prompt.splitlines() if l.startswith("Available actions:"))
        assert "SLEEP" not in actions_line

    def test_empty_scene_only_toilet_idle_explore(self):
        request = _make_request(objects=[])
        prompt = build_prompt(request)
        # Check only the "Available actions" line to avoid matching schema enum values.
        actions_line = next(l for l in prompt.splitlines() if l.startswith("Available actions:"))
        for action in [Action.TOILET, Action.IDLE, Action.EXPLORE]:
            assert action.value in actions_line
        for action in [Action.EAT, Action.DRINK, Action.PLAY, Action.FETCH,
                       Action.SLEEP, Action.SOCIAL, Action.FOLLOW]:
            assert action.value not in actions_line

    def test_toy_in_scene_includes_play_and_fetch(self):
        request = _make_request(objects=[_toy_object()])
        prompt = build_prompt(request)
        assert "PLAY" in prompt
        assert "FETCH" in prompt

    def test_bed_in_scene_includes_sleep(self):
        request = _make_request(objects=[_bed_object()])
        prompt = build_prompt(request)
        assert "SLEEP" in prompt

    def test_player_in_scene_includes_social_and_follow(self):
        request = _make_request(objects=[_player_object()])
        prompt = build_prompt(request)
        assert "SOCIAL" in prompt
        assert "FOLLOW" in prompt

    def test_prompt_under_1200_chars(self):
        # Use a scene with all object types to maximise prompt length.
        objects = [_bowl_object(), _toy_object(), _bed_object(), _player_object()]
        request = _make_request(objects=objects)
        prompt = build_prompt(request)
        assert len(prompt) < 1200, f"Prompt too long: {len(prompt)} chars"

    def test_prompt_contains_schema(self):
        request = _make_request()
        prompt = build_prompt(request)
        assert "action" in prompt  # schema field name should appear

    def test_prompt_instructs_json_only(self):
        request = _make_request()
        prompt = build_prompt(request)
        assert "JSON" in prompt


# ---------------------------------------------------------------------------
# parse_response tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_valid_json_returns_inference_response(self):
        raw = '{"action": "IDLE"}'
        result = parse_response(raw)
        assert isinstance(result, InferenceResponse)
        assert result.action == Action.IDLE

    def test_valid_json_with_optional_fields(self):
        raw = '{"action": "EAT", "target_object_id": "bowl_1", "confidence": 0.9}'
        result = parse_response(raw)
        assert result.action == Action.EAT
        assert result.target_object_id == "bowl_1"
        assert result.confidence == pytest.approx(0.9)

    def test_json_with_surrounding_text(self):
        raw = 'Sure! Here is the JSON: {"action": "EXPLORE"} Hope that helps!'
        result = parse_response(raw)
        assert result.action == Action.EXPLORE

    def test_json_prefixed_with_prose(self):
        raw = "Based on the pet's stats, I recommend: {\"action\": \"SLEEP\"}"
        result = parse_response(raw)
        assert result.action == Action.SLEEP

    def test_malformed_json_raises_value_error(self):
        raw = "{action: IDLE}"  # missing quotes — invalid JSON
        with pytest.raises(ValueError, match="(?i)(json|found)"):
            parse_response(raw)

    def test_missing_required_field_raises_value_error(self):
        raw = '{"target_object_id": "bowl_1"}'  # missing "action"
        with pytest.raises(ValueError):
            parse_response(raw)

    def test_no_json_object_raises_value_error(self):
        raw = "I cannot determine the action right now."
        with pytest.raises(ValueError, match="No JSON object found"):
            parse_response(raw)

    def test_invalid_action_value_raises_value_error(self):
        raw = '{"action": "DANCE"}'  # not a valid Action enum value
        with pytest.raises(ValueError):
            parse_response(raw)
