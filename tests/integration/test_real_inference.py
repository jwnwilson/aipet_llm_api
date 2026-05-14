"""Real model inference smoke test — no mocks, requires AIPET_TEST_MODEL_PATH or models/test_aipet.gguf."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from adapters.inference import LlamaCppInferenceAdapter
from adapters.prompt import parse_response
from domain.models import InferenceRequest, PetStats, SceneData

_DEFAULT_MODEL_PATH = Path(__file__).parents[2] / "models" / "test_aipet.gguf"
MODEL_PATH = Path(os.environ.get("AIPET_TEST_MODEL_PATH", str(_DEFAULT_MODEL_PATH)))

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"Real model not found at {MODEL_PATH}",
)

_PARSE_RESPONSE = "adapters.inference.parse_response"


@pytest.fixture(scope="module")
def adapter() -> LlamaCppInferenceAdapter:
    return LlamaCppInferenceAdapter(model_path=str(MODEL_PATH), context_size=2048)


def _request(**stats) -> InferenceRequest:
    defaults = dict(hunger=0.5, boredom=0.3, social=0.2, toilet=0.1, tiredness=0.4)
    defaults.update(stats)
    return InferenceRequest(
        scene=SceneData(objects=[], tick=1),
        pet_stats=PetStats(**defaults),
    )


class TestRealInference:
    def test_model_output_is_parseable_json(self, adapter):
        with patch(_PARSE_RESPONSE, wraps=parse_response) as spy:
            response = adapter.infer(_request())

        spy.assert_called_once()
        raw_text: str = spy.call_args[0][0]
        parsed = parse_response(raw_text)
        assert parsed == response, (
            "Response does not match parse_response result — fallback was substituted"
        )
