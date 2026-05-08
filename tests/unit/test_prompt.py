"""Unit tests for src/infrastructure/prompt.py."""

import pytest

from domain.actions import Action
from domain.models import InferenceRequest, InferenceResponse, PetStats, SceneData, SceneObject
from infrastructure.prompt import build_prompt, parse_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    objects: list[SceneObject] | None = None,
    tick: int = 0,
    hunger: float = 0.5,
    boredom: float = 0.3,
    social: float = 0.2,
    toilet: float = 0.1,
    tiredness: float = 0.4,
) -> InferenceRequest:
    scene = SceneData(objects=objects or [], tick=tick)
    stats = PetStats(hunger=hunger, boredom=boredom, social=social, toilet=toilet, tiredness=tiredness)
    return InferenceRequest(scene=scene, pet_stats=stats)


def _bowl_object(distance: float = 2.0) -> SceneObject:
    return SceneObject(id="bowl_1", type="bowl", distance=distance)


def _toy_object(distance: float = 3.0) -> SceneObject:
    return SceneObject(id="toy_1", type="toy", distance=distance)


def _bed_object(distance: float = 1.5) -> SceneObject:
    return SceneObject(id="bed_1", type="bed", distance=distance)


def _player_object(distance: float = 4.0) -> SceneObject:
    return SceneObject(id="player_1", type="player", distance=distance)


# ---------------------------------------------------------------------------
# build_prompt — available actions filtering
# ---------------------------------------------------------------------------


class TestAvailableActions:
    def test_bowl_in_scene_includes_eat_and_drink(self):
        request = _make_request(objects=[_bowl_object()])
        prompt = build_prompt(request)
        assert "EAT" in prompt
        assert "DRINK" in prompt

    def test_bowl_in_scene_excludes_sleep(self):
        request = _make_request(objects=[_bowl_object()])
        prompt = build_prompt(request)
        actions_line = next(l for l in prompt.splitlines() if l.startswith("Available actions:"))
        assert "SLEEP" not in actions_line

    def test_empty_scene_only_toilet_idle_explore(self):
        request = _make_request(objects=[])
        prompt = build_prompt(request)
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


# ---------------------------------------------------------------------------
# build_prompt — sorted stats
# ---------------------------------------------------------------------------


class TestSortedStats:
    def test_stats_line_starts_with_highest_first_label(self):
        request = _make_request(hunger=0.9, tiredness=0.1, boredom=0.2, social=0.05, toilet=0.3)
        prompt = build_prompt(request)
        stats_line = next(l for l in prompt.splitlines() if l.startswith("Stats"))
        assert "hunger=0.90 (highest)" in stats_line

    def test_highest_stat_appears_first_in_stats_line(self):
        # tiredness is highest
        request = _make_request(hunger=0.3, tiredness=0.8, boredom=0.2, social=0.1, toilet=0.4)
        prompt = build_prompt(request)
        stats_line = next(l for l in prompt.splitlines() if l.startswith("Stats"))
        parts = [p.strip() for p in stats_line.split(",")]
        first_stat_name = parts[0].split("=")[0].split(")")[0].split("(")[0].strip().split()[-1]
        assert "tiredness" in parts[0], f"Expected tiredness first, got: {stats_line}"

    def test_stats_in_descending_order(self):
        request = _make_request(hunger=0.2, tiredness=0.9, boredom=0.5, social=0.1, toilet=0.7)
        prompt = build_prompt(request)
        stats_line = next(l for l in prompt.splitlines() if l.startswith("Stats"))
        # Extract the float values in the order they appear
        import re
        values = [float(m) for m in re.findall(r"=(\d+\.\d+)", stats_line)]
        assert values == sorted(values, reverse=True), (
            f"Stats not in descending order: {values}"
        )

    def test_highest_label_appears_exactly_once(self):
        request = _make_request()
        prompt = build_prompt(request)
        assert prompt.count("(highest)") == 1


# ---------------------------------------------------------------------------
# build_prompt — explicit rule line
# ---------------------------------------------------------------------------


class TestRuleLine:
    def test_rule_line_present(self):
        request = _make_request()
        prompt = build_prompt(request)
        assert "Rule:" in prompt

    def test_rule_mentions_highest_stat(self):
        request = _make_request()
        prompt = build_prompt(request)
        rule_line = next(l for l in prompt.splitlines() if "Rule:" in l)
        assert "highest stat" in rule_line

    def test_rule_mentions_closest(self):
        request = _make_request()
        prompt = build_prompt(request)
        rule_line = next(l for l in prompt.splitlines() if "Rule:" in l)
        assert "closest" in rule_line


# ---------------------------------------------------------------------------
# build_prompt — sorted scene objects
# ---------------------------------------------------------------------------


class TestSortedScene:
    def test_objects_sorted_nearest_first(self):
        objects = [
            SceneObject(id="far", type="bowl", distance=20.0),
            SceneObject(id="close", type="toy", distance=1.5),
            SceneObject(id="mid", type="bed", distance=8.0),
        ]
        request = _make_request(objects=objects)
        prompt = build_prompt(request)
        scene_line = next(l for l in prompt.splitlines() if l.startswith("Scene"))
        # "close" should appear before "mid" which should appear before "far"
        pos_close = scene_line.index("close")
        pos_mid = scene_line.index("mid")
        pos_far = scene_line.index("far")
        assert pos_close < pos_mid < pos_far, (
            f"Objects not sorted nearest-first: {scene_line}"
        )

    def test_scene_line_label_nearest_first(self):
        request = _make_request(objects=[_bowl_object()])
        prompt = build_prompt(request)
        scene_line = next(l for l in prompt.splitlines() if l.startswith("Scene"))
        assert "nearest first" in scene_line

    def test_empty_scene_shows_empty(self):
        request = _make_request(objects=[])
        prompt = build_prompt(request)
        assert "empty" in prompt


# ---------------------------------------------------------------------------
# build_prompt — length constraint
# ---------------------------------------------------------------------------


class TestPromptLength:
    def test_prompt_under_1200_chars(self):
        objects = [_bowl_object(), _toy_object(), _bed_object(), _player_object()]
        request = _make_request(objects=objects)
        prompt = build_prompt(request)
        assert len(prompt) < 1200, f"Prompt too long: {len(prompt)} chars"

    def test_prompt_instructs_json_only(self):
        request = _make_request()
        prompt = build_prompt(request)
        assert "JSON" in prompt

    def test_prompt_does_not_contain_schema(self):
        request = _make_request()
        prompt = build_prompt(request)
        assert "Schema:" not in prompt
        assert "$defs" not in prompt


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

    # Truncated-response fallback (grammar loop hit max_tokens before closing })
    def test_truncated_json_with_repeated_fields_recovers_action(self):
        raw = '{"action":"EAT","target_object_id":"bowl_1","confidence":0.90,"confidence":0.90,"confidence":0.9'
        result = parse_response(raw)
        assert result.action == Action.EAT
        assert result.target_object_id == "bowl_1"

    def test_truncated_json_without_target_recovers_action(self):
        raw = '{"action":"SLEEP","confidence":0.85,"confidence":0.85,"confidence":0.85,"confidence":0.8'
        result = parse_response(raw)
        assert result.action == Action.SLEEP
        assert result.target_object_id is None

    def test_truncated_json_with_no_action_raises(self):
        raw = '{"target_object_id":"bowl","confidence":0.90,"confidence":0.90,"confidence":0.9'
        with pytest.raises(ValueError):
            parse_response(raw)
