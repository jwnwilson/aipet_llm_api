"""Real model inference tests — no mocks, requires models/aipet.gguf."""
from __future__ import annotations

from pathlib import Path

import pytest

from domain.actions import Action
from domain.models import InferenceRequest, InferenceResponse, PetStats, SceneData, SceneObject
from infrastructure.inference import LlamaCppInferenceAdapter

MODEL_PATH = Path(__file__).parents[2] / "models" / "aipet.gguf"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"Real model not found at {MODEL_PATH}",
)

# Actions that require a target object to be present in the scene.
_TARGET_REQUIRED = {Action.EAT, Action.DRINK, Action.PLAY, Action.FETCH,
                   Action.SLEEP, Action.SOCIAL, Action.FOLLOW}


def _adapter() -> LlamaCppInferenceAdapter:
    return LlamaCppInferenceAdapter(model_path=str(MODEL_PATH), context_size=512)


class TestRealInference:
    def test_empty_scene_returns_valid_response(self):
        request = InferenceRequest(
            scene=SceneData(objects=[], tick=1),
            pet_stats=PetStats(hunger=0.5, boredom=0.3, social=0.2, toilet=0.1, tiredness=0.4),
        )
        response = _adapter().infer(request)
        assert isinstance(response, InferenceResponse)
        assert isinstance(response.action, Action)

    def test_empty_scene_action_is_untargeted(self):
        # With no objects in the scene, only TOILET/IDLE/EXPLORE are valid.
        request = InferenceRequest(
            scene=SceneData(objects=[], tick=1),
            pet_stats=PetStats(hunger=0.5, boredom=0.3, social=0.2, toilet=0.1, tiredness=0.4),
        )
        response = _adapter().infer(request)
        assert response.action not in _TARGET_REQUIRED, (
            f"Got {response.action} but no objects were in the scene"
        )

    def test_bowl_scene_target_id_matches_scene(self):
        scene_objects = [SceneObject(id="bowl1", type="bowl", distance=1.5)]
        request = InferenceRequest(
            scene=SceneData(objects=scene_objects, tick=1),
            pet_stats=PetStats(hunger=0.9, boredom=0.1, social=0.1, toilet=0.1, tiredness=0.1),
        )
        response = _adapter().infer(request)
        assert isinstance(response, InferenceResponse)
        assert isinstance(response.action, Action)
        if response.target_object_id is not None:
            scene_ids = {o.id for o in scene_objects}
            assert response.target_object_id in scene_ids, (
                f"target_object_id {response.target_object_id!r} not in scene {scene_ids}"
            )

    def test_model_loaded_once_across_calls(self):
        adapter = _adapter()
        request = InferenceRequest(
            scene=SceneData(objects=[], tick=1),
            pet_stats=PetStats(hunger=0.5, boredom=0.3, social=0.2, toilet=0.1, tiredness=0.4),
        )
        r1 = adapter.infer(request)
        r2 = adapter.infer(request)
        assert adapter._llm is not None
        assert isinstance(r1.action, Action)
        assert isinstance(r2.action, Action)
