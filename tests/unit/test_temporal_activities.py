"""Unit tests for Temporal activities — domain functions are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

from temporal.activities import (
    CheckpointPath,
    DatasetConfig,
    DatasetPaths,
    EvalConfig,
    EvalResult,
    GGUFPath,
    TrainConfig,
    _parse_valid_pct,
    evaluate_activity,
    export_activity,
    generate_dataset_activity,
    train_activity,
)


ENV = ActivityEnvironment()


# ---------------------------------------------------------------------------
# generate_dataset_activity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_dataset_activity_delegates_to_domain():
    with patch("domain.train.dataset.generate", return_value=True) as mock_gen:
        result = await ENV.run(generate_dataset_activity, DatasetConfig(data_dir="data", train_size=10, eval_size=5, seed=1))

    mock_gen.assert_called_once_with(data_dir=Path("data"), train_size=10, eval_size=5, seed=1)
    assert result == DatasetPaths(train="data/train.jsonl", eval="data/eval.jsonl")


@pytest.mark.asyncio
async def test_generate_dataset_activity_raises_on_invalid_examples():
    with patch("domain.train.dataset.generate", return_value=False):
        with pytest.raises(ApplicationError, match="invalid examples"):
            await ENV.run(generate_dataset_activity, DatasetConfig())


@pytest.mark.asyncio
async def test_generate_dataset_activity_raises_on_exception():
    with patch("domain.train.dataset.generate", side_effect=RuntimeError("disk full")):
        with pytest.raises(ApplicationError, match="generate_dataset failed"):
            await ENV.run(generate_dataset_activity, DatasetConfig())


# ---------------------------------------------------------------------------
# train_activity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_train_activity_delegates_to_domain():
    config = TrainConfig(
        model="some-model",
        train_data="data/train.jsonl",
        eval_data="data/eval.jsonl",
        output_dir="models/checkpoints",
        epochs=3,
        patience=2,
    )
    with patch("domain.train.trainer.train") as mock_train:
        result = await ENV.run(train_activity, config)

    mock_train.assert_called_once_with(
        model="some-model",
        train_data="data/train.jsonl",
        eval_data="data/eval.jsonl",
        output_dir="models/checkpoints",
        epochs=3,
        patience=2,
        warmup_ratio=0.05,
    )
    assert result == CheckpointPath(path="models/checkpoints")


@pytest.mark.asyncio
async def test_train_activity_raises_on_exception():
    with patch("domain.train.trainer.train", side_effect=ImportError("torch not found")):
        with pytest.raises(ApplicationError, match="train failed"):
            await ENV.run(train_activity, TrainConfig())


# ---------------------------------------------------------------------------
# evaluate_activity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_activity_delegates_to_domain():
    mock_pipe = MagicMock()
    with (
        patch("domain.train.evaluate.load_hf_pipeline", return_value=mock_pipe) as mock_load,
        patch("domain.train.evaluate.infer_hf", return_value='{"action": "IDLE"}'),
        patch("domain.train.evaluate.evaluate", return_value=0) as mock_eval,
    ):
        result = await ENV.run(evaluate_activity, EvalConfig(checkpoint="models/checkpoints", eval_data="data/eval.jsonl"))

    mock_load.assert_called_once_with("models/checkpoints")
    mock_eval.assert_called_once()
    assert result.passed is True


@pytest.mark.asyncio
async def test_evaluate_activity_fail_result_when_exit_code_nonzero():
    with (
        patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()),
        patch("domain.train.evaluate.infer_hf", return_value=""),
        patch("domain.train.evaluate.evaluate", return_value=1),
    ):
        result = await ENV.run(evaluate_activity, EvalConfig(checkpoint="models/checkpoints"))

    assert result.passed is False
    assert result.valid_pct == 0.0


@pytest.mark.asyncio
async def test_evaluate_activity_parses_valid_pct_from_stdout():
    def fake_evaluate(path, infer_fn):
        print("Valid: 190/200 (95.0%)  [PASS]")
        return 0

    with (
        patch("domain.train.evaluate.load_hf_pipeline", return_value=MagicMock()),
        patch("domain.train.evaluate.infer_hf", return_value=""),
        patch("domain.train.evaluate.evaluate", side_effect=fake_evaluate),
    ):
        result = await ENV.run(evaluate_activity, EvalConfig(checkpoint="models/checkpoints"))

    assert result.passed is True
    assert abs(result.valid_pct - 0.95) < 1e-6


@pytest.mark.asyncio
async def test_evaluate_activity_raises_on_exception():
    with patch("domain.train.evaluate.load_hf_pipeline", side_effect=RuntimeError("model not found")):
        with pytest.raises(ApplicationError, match="evaluate failed"):
            await ENV.run(evaluate_activity, EvalConfig(checkpoint="bad/path"))


# ---------------------------------------------------------------------------
# export_activity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_activity_delegates_to_domain():
    with patch("domain.train.export.export") as mock_export:
        result = await ENV.run(export_activity, CheckpointPath(path="models/checkpoints"))

    mock_export.assert_called_once_with(
        checkpoint=Path("models/checkpoints"),
        output=Path("models/aipet.gguf"),
    )
    assert result == GGUFPath(path="models/aipet.gguf")


@pytest.mark.asyncio
async def test_export_activity_raises_application_error_on_exception():
    with patch("domain.train.export.export", side_effect=RuntimeError("conversion failed")):
        with pytest.raises(ApplicationError, match="export failed"):
            await ENV.run(export_activity, CheckpointPath(path="models/checkpoints"))


@pytest.mark.asyncio
async def test_export_activity_raises_application_error_on_system_exit():
    with patch("domain.train.export.export", side_effect=SystemExit(1)):
        with pytest.raises(ApplicationError, match="llama.cpp setup issue"):
            await ENV.run(export_activity, CheckpointPath(path="models/checkpoints"))


# ---------------------------------------------------------------------------
# _parse_valid_pct helper
# ---------------------------------------------------------------------------


def test_parse_valid_pct_extracts_percentage():
    output = "Valid: 190/200 (95.0%)  [PASS]"
    assert abs(_parse_valid_pct(output) - 0.95) < 1e-6


def test_parse_valid_pct_returns_none_on_no_match():
    assert _parse_valid_pct("no match here") is None


# ---------------------------------------------------------------------------
# _train_remote polling loop
# ---------------------------------------------------------------------------


class TestTrainRemotePolling:
    """Verify _train_remote calls logs() and sends a structured heartbeat each poll."""

    def _make_adapter(self, statuses, log_output="step 10 loss=0.5", download_path="/tmp/ckpt"):
        adapter = MagicMock()
        adapter.submit.return_value = "run-42"
        adapter.status.side_effect = list(statuses)
        adapter.logs.return_value = log_output
        adapter.download.return_value = download_path
        return adapter

    @pytest.mark.asyncio
    async def test_calls_adapter_logs_each_poll(self, monkeypatch):
        import temporal.activities as acts

        adapter = self._make_adapter(["running", "done"])
        monkeypatch.setattr(acts.activity, "heartbeat", MagicMock())
        monkeypatch.setattr(acts.activity, "logger", MagicMock())

        config = TrainConfig(experiment_name="test-exp", output_dir="/tmp/out")
        with patch("temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        assert adapter.logs.call_count >= 1

    @pytest.mark.asyncio
    async def test_heartbeat_is_dict_with_status_elapsed_and_logs(self, monkeypatch):
        import temporal.activities as acts

        adapter = self._make_adapter(["running", "done"])
        captured: list[dict] = []
        monkeypatch.setattr(acts.activity, "heartbeat", lambda hb: captured.append(hb))
        monkeypatch.setattr(acts.activity, "logger", MagicMock())

        config = TrainConfig(experiment_name="test-exp", output_dir="/tmp/out")
        with patch("temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        assert captured, "heartbeat should have been called"
        first = captured[0]
        assert isinstance(first, dict)
        assert first["status"] == "running"
        assert "elapsed_s" in first
        assert first["logs"] == "step 10 loss=0.5"

    @pytest.mark.asyncio
    async def test_heartbeat_logs_field_is_empty_when_adapter_returns_none(self, monkeypatch):
        import temporal.activities as acts

        adapter = self._make_adapter(["done"], log_output="")
        captured: list[dict] = []
        monkeypatch.setattr(acts.activity, "heartbeat", lambda hb: captured.append(hb))
        monkeypatch.setattr(acts.activity, "logger", MagicMock())

        config = TrainConfig(experiment_name="test-exp", output_dir="/tmp/out")
        with patch("temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        assert captured[0]["logs"] == ""


def test_parse_valid_pct_handles_multiline_output():
    output = "Loading model...\nValid: 180/200 (90.0%)  [FAIL]\nAction distribution:"
    assert abs(_parse_valid_pct(output) - 0.90) < 1e-6
