"""Unit tests for domain/train/dataset.py — focusing on target object selection."""

from __future__ import annotations

from domain.actions import Action
from domain.models import InferenceRequest, PetStats, SceneData, SceneObject
from domain.train.dataset import label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _request(
    *objects: SceneObject,
    dominant_stat: str,
    dominant_value: float = 0.9,
    tick: int = 0,
) -> InferenceRequest:
    low = 0.1
    stat_values = {s: low for s in ("hunger", "tiredness", "boredom", "social", "toilet")}
    stat_values[dominant_stat] = dominant_value
    return InferenceRequest(
        scene=SceneData(objects=list(objects), tick=tick),
        pet_stats=PetStats(**stat_values),
    )


def _bowl(id: str, distance: float) -> SceneObject:
    return SceneObject(id=id, type="bowl", distance=distance)


def _toy(id: str, distance: float) -> SceneObject:
    return SceneObject(id=id, type="toy", distance=distance)


def _bed(id: str, distance: float) -> SceneObject:
    return SceneObject(id=id, type="bed", distance=distance)


def _player(id: str, distance: float) -> SceneObject:
    return SceneObject(id=id, type="player", distance=distance)


def _pet(id: str, distance: float) -> SceneObject:
    return SceneObject(id=id, type="pet", distance=distance)


# ---------------------------------------------------------------------------
# label() — target object selection
# ---------------------------------------------------------------------------


class TestLabelTargetSelection:
    def test_picks_closest_bowl_when_hungry(self):
        request = _request(
            _bowl("bowl_far", 20.0),
            _bowl("bowl_close", 3.0),
            _bowl("bowl_mid", 10.0),
            dominant_stat="hunger",
            tick=0,  # tick=0 → EAT
        )
        result = label(request)
        assert result.action == Action.EAT
        assert result.target_object_id == "bowl_close"

    def test_picks_closest_toy_when_bored(self):
        request = _request(
            _toy("toy_far", 25.0),
            _toy("toy_close", 2.0),
            dominant_stat="boredom",
            tick=0,  # tick=0 → PLAY
        )
        result = label(request)
        assert result.action == Action.PLAY
        assert result.target_object_id == "toy_close"

    def test_picks_closest_bed_when_tired(self):
        request = _request(
            _bed("bed_far", 40.0),
            _bed("bed_close", 5.0),
            dominant_stat="tiredness",
        )
        result = label(request)
        assert result.action == Action.SLEEP
        assert result.target_object_id == "bed_close"

    def test_picks_closest_social_target_across_player_and_pet(self):
        # pet is closer than player — should pick pet regardless of type order
        request = _request(
            _player("player_1", 30.0),
            _pet("pet_1", 4.0),
            dominant_stat="social",
            tick=0,  # tick=0 → SOCIAL
        )
        result = label(request)
        assert result.action == Action.SOCIAL
        assert result.target_object_id == "pet_1"

    def test_ignores_wrong_type_objects_for_action(self):
        # hungry pet — scene has a close toy and a far bowl; must pick the bowl
        request = _request(
            _toy("toy_close", 1.0),
            _bowl("bowl_far", 30.0),
            dominant_stat="hunger",
            tick=0,
        )
        result = label(request)
        assert result.action == Action.EAT
        assert result.target_object_id == "bowl_far"

    def test_target_id_changes_with_different_scenes(self):
        req_a = _request(_bowl("bowl_a", 5.0), _bowl("bowl_b", 15.0), dominant_stat="hunger", tick=0)
        req_b = _request(_bowl("bowl_a", 15.0), _bowl("bowl_b", 5.0), dominant_stat="hunger", tick=0)

        assert label(req_a).target_object_id == "bowl_a"
        assert label(req_b).target_object_id == "bowl_b"

    def test_falls_back_to_idle_when_required_object_absent(self):
        request = _request(_toy("toy_1", 5.0), dominant_stat="hunger", tick=0)
        result = label(request)
        assert result.action == Action.IDLE
        assert result.target_object_id is None

    def test_falls_back_when_all_stats_low(self):
        request = _request(_bowl("bowl_1", 5.0), dominant_stat="hunger", dominant_value=0.3, tick=0)
        result = label(request)
        assert result.action in (Action.IDLE, Action.EXPLORE)
        assert result.target_object_id is None

    def test_toilet_never_needs_target(self):
        request = _request(_bowl("bowl_1", 1.0), dominant_stat="toilet")
        result = label(request)
        assert result.action == Action.TOILET
        assert result.target_object_id is None
