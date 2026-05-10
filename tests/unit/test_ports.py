"""Unit tests for InferencePort and RemoteTrainingPort abstract interfaces."""

import pytest

from domain.actions import Action
from domain.models import InferenceRequest, InferenceResponse, RemoteTrainConfig
from domain.ports import InferencePort, RemoteTrainingPort


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
# RemoteTrainingPort contract
# ---------------------------------------------------------------------------


def _full_remote_adapter() -> type:
    """Minimal complete RemoteTrainingPort implementation for contract testing."""

    class _FullAdapter(RemoteTrainingPort):
        def submit(self, config: RemoteTrainConfig) -> str:
            return "run-id"

        def status(self, run_id: str):
            return "pending"

        def download(self, run_id: str, dest):
            return str(dest)

    return _FullAdapter


def _make_remote_config() -> RemoteTrainConfig:
    return RemoteTrainConfig(
        model="HuggingFaceTB/SmolLM-360M",
        train_data="data/train.jsonl",
        eval_data="data/eval.jsonl",
        epochs=1,
        patience=1,
        warmup_ratio=0.05,
        experiment_name="contract-test",
    )


class TestRemoteTrainingPort:
    def test_cannot_instantiate_abstract_port_directly(self):
        with pytest.raises(TypeError):
            RemoteTrainingPort()  # type: ignore[abstract]

    def test_missing_status_raises_type_error(self):
        class _Partial(RemoteTrainingPort):
            def submit(self, config): return ""
            def download(self, run_id, dest): return ""

        with pytest.raises(TypeError):
            _Partial()  # type: ignore[abstract]

    def test_missing_download_raises_type_error(self):
        class _Partial(RemoteTrainingPort):
            def submit(self, config): return ""
            def status(self, run_id): return "pending"

        with pytest.raises(TypeError):
            _Partial()  # type: ignore[abstract]

    def test_missing_submit_raises_type_error(self):
        class _Partial(RemoteTrainingPort):
            def status(self, run_id): return "pending"
            def download(self, run_id, dest): return ""

        with pytest.raises(TypeError):
            _Partial()  # type: ignore[abstract]

    def test_full_implementation_instantiates(self):
        adapter = _full_remote_adapter()()
        assert isinstance(adapter, RemoteTrainingPort)

    def test_submit_returns_non_empty_string(self):
        adapter = _full_remote_adapter()()
        run_id = adapter.submit(_make_remote_config())
        assert isinstance(run_id, str) and run_id

    def test_status_returns_valid_literal(self):
        valid = {"pending", "running", "done", "failed"}
        adapter = _full_remote_adapter()()
        assert adapter.status("run-id") in valid

    def test_download_returns_string_path(self, tmp_path):
        adapter = _full_remote_adapter()()
        result = adapter.download("run-id", tmp_path)
        assert isinstance(result, str)

    @pytest.mark.parametrize("status", ["pending", "running", "done", "failed"])
    def test_each_valid_status_literal_is_accepted(self, status):
        """Any implementation may return any of the four canonical status values."""

        class _FixedAdapter(RemoteTrainingPort):
            def submit(self, config): return "x"
            def status(self, run_id): return status
            def download(self, run_id, dest): return str(dest)

        assert _FixedAdapter().status("x") == status


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def inference_request() -> InferenceRequest:
    """Return a minimal but valid ``InferenceRequest``."""
    from domain.models import PetStats, SceneData

    scene = SceneData(objects=[], tick=0)
    stats = PetStats(
        hunger=0.0,
        boredom=0.0,
        social=0.0,
        toilet=0.0,
        tiredness=0.0,
    )
    return InferenceRequest(scene=scene, pet_stats=stats)
