"""Unit tests for the InferencePort abstract interface."""

import pytest

from src.domain.actions import Action
from src.domain.models import InferenceRequest, InferenceResponse
from src.domain.ports import InferencePort


class FakeInferenceAdapter(InferencePort):
    """Minimal concrete implementation used only in tests."""

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(action=Action.IDLE)


class IncompleteAdapter(InferencePort):
    """Subclass that deliberately does NOT implement ``infer``."""


class TestInferencePort:
    def test_fake_adapter_can_be_instantiated(self):
        adapter = FakeInferenceAdapter()
        assert isinstance(adapter, InferencePort)

    def test_fake_adapter_infer_returns_inference_response(
        self, inference_request: InferenceRequest
    ):
        adapter = FakeInferenceAdapter()
        response = adapter.infer(inference_request)
        assert isinstance(response, InferenceResponse)

    def test_fake_adapter_infer_returns_idle_action(
        self, inference_request: InferenceRequest
    ):
        adapter = FakeInferenceAdapter()
        response = adapter.infer(inference_request)
        assert response.action == Action.IDLE

    def test_incomplete_adapter_raises_type_error_on_instantiation(self):
        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def inference_request() -> InferenceRequest:
    """Return a minimal but valid ``InferenceRequest``."""
    from src.domain.models import PetStats, SceneData

    scene = SceneData(objects=[], tick=0)
    stats = PetStats(
        hunger=0.0,
        boredom=0.0,
        social=0.0,
        toilet=0.0,
        tiredness=0.0,
    )
    return InferenceRequest(scene=scene, pet_stats=stats)
