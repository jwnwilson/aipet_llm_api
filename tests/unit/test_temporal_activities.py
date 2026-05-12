"""Unit tests for Temporal activities — domain functions are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

from interactors.temporal.activities import (
    CheckpointPath,
    DatasetConfig,
    DatasetPaths,
    EvalConfig,
    EvalResult,
    ExportConfig,
    GGUFPath,
    TrainConfig,
    _parse_valid_pct,
    configure_storage,
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
        dry_run=False,
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


@pytest.mark.asyncio
async def test_evaluate_remote_kaggle_fallback_passes_inner_checkpoint_to_local(monkeypatch):
    """When a backend raises NotImplementedError on eval(), download() is called and the
    path it returns (the inner HF checkpoint dir) must be forwarded to _evaluate_local."""
    import asyncio
    import interactors.temporal.activities as acts
    from unittest.mock import MagicMock

    inner_ckpt = "/tmp/dest/checkpoints"
    mock_adapter = MagicMock()
    mock_adapter.eval.side_effect = NotImplementedError
    mock_adapter.download.return_value = inner_ckpt

    local_calls: list[str] = []

    async def fake_local(config, loop):
        local_calls.append(config.checkpoint)
        return EvalResult(valid_pct=0.95, passed=True)

    monkeypatch.setattr(acts, "_evaluate_local", fake_local)
    monkeypatch.setattr(acts, "_make_remote_adapter", lambda _: mock_adapter)

    config = EvalConfig(remote_backend="kaggle", run_id="u/exp", eval_data="data/eval.jsonl")
    await acts._evaluate_remote(config, asyncio.get_event_loop())

    assert local_calls == [inner_ckpt], (
        "_evaluate_local must receive the inner checkpoint path returned by download(), not the extraction root"
    )


# ---------------------------------------------------------------------------
# export_activity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_activity_uses_model_name_for_storage_key():
    from unittest.mock import MagicMock
    from adapters.storage.local import LocalStorageAdapter

    storage = MagicMock(spec=LocalStorageAdapter)
    configure_storage(storage)

    with patch("domain.train.export.export"):
        result = await ENV.run(
            export_activity,
            ExportConfig(
                checkpoint_path="models/checkpoints",
                gguf_output="data/workflow/r1/model.gguf",
                model_name="my-pet-v2",
                pipeline_run_id="r1",
                model_id="model-uuid",
            ),
        )

    assert result.path == "gguf/my-pet-v2.gguf"


@pytest.mark.asyncio
async def test_export_activity_model_name_takes_precedence_over_pipeline_run_id():
    from unittest.mock import MagicMock
    from adapters.storage.local import LocalStorageAdapter

    storage = MagicMock(spec=LocalStorageAdapter)
    configure_storage(storage)

    with patch("domain.train.export.export"):
        result = await ENV.run(
            export_activity,
            ExportConfig(
                checkpoint_path="models/checkpoints",
                gguf_output="data/workflow/r1/model.gguf",
                model_name="my-pet-v2",
                pipeline_run_id="r1",
            ),
        )

    assert result.path == "gguf/my-pet-v2.gguf"
    assert "r1" not in result.path


@pytest.mark.asyncio
async def test_export_activity_uses_pipeline_run_id_for_storage_key():
    from unittest.mock import MagicMock
    from adapters.storage.local import LocalStorageAdapter

    storage = MagicMock(spec=LocalStorageAdapter)
    configure_storage(storage)

    with patch("domain.train.export.export"):
        result = await ENV.run(
            export_activity,
            ExportConfig(
                checkpoint_path="models/checkpoints",
                gguf_output="data/workflow/r1/model.gguf",
                pipeline_run_id="r1",
                model_id="m",
            ),
        )

    storage.upload.assert_called_once()
    assert result == GGUFPath(path="workflow/r1/model.gguf")


@pytest.mark.asyncio
async def test_export_activity_pipeline_run_id_takes_precedence_over_model_id():
    from unittest.mock import MagicMock
    from adapters.storage.local import LocalStorageAdapter

    storage = MagicMock(spec=LocalStorageAdapter)
    configure_storage(storage)

    with patch("domain.train.export.export"):
        result = await ENV.run(
            export_activity,
            ExportConfig(
                checkpoint_path="models/checkpoints",
                gguf_output="data/workflow/run-42/model.gguf",
                pipeline_run_id="run-42",
                model_id="model-99",
            ),
        )

    assert result.path == "workflow/run-42/model.gguf"


@pytest.mark.asyncio
async def test_export_activity_falls_back_to_model_id_when_no_pipeline_run_id():
    from unittest.mock import MagicMock
    from adapters.storage.local import LocalStorageAdapter

    storage = MagicMock(spec=LocalStorageAdapter)
    configure_storage(storage)

    with patch("domain.train.export.export"):
        result = await ENV.run(
            export_activity,
            ExportConfig(checkpoint_path="models/checkpoints", gguf_output="models/gguf/m.gguf", model_id="m"),
        )

    storage.upload.assert_called_once()
    assert result == GGUFPath(path="gguf/m.gguf")


@pytest.mark.asyncio
async def test_export_activity_raises_application_error_on_exception():
    with patch("domain.train.export.export", side_effect=RuntimeError("conversion failed")):
        with pytest.raises(ApplicationError, match="export failed"):
            await ENV.run(
                export_activity,
                ExportConfig(checkpoint_path="models/checkpoints", gguf_output="models/aipet.gguf"),
            )


@pytest.mark.asyncio
async def test_export_activity_raises_application_error_on_system_exit():
    with patch("domain.train.export.export", side_effect=SystemExit(1)):
        with pytest.raises(ApplicationError, match="llama.cpp setup issue"):
            await ENV.run(
                export_activity,
                ExportConfig(checkpoint_path="models/checkpoints", gguf_output="models/aipet.gguf"),
            )


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
        import interactors.temporal.activities as acts

        adapter = self._make_adapter(["running", "done"])
        monkeypatch.setattr(acts.activity, "heartbeat", MagicMock())
        monkeypatch.setattr(acts.activity, "logger", MagicMock())

        config = TrainConfig(experiment_name="test-exp", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        assert adapter.logs.call_count >= 1

    @pytest.mark.asyncio
    async def test_heartbeat_is_dict_with_status_elapsed_and_logs(self, monkeypatch):
        import interactors.temporal.activities as acts

        adapter = self._make_adapter(["running", "done"])
        captured: list[dict] = []
        monkeypatch.setattr(acts.activity, "heartbeat", lambda hb: captured.append(hb))
        monkeypatch.setattr(acts.activity, "logger", MagicMock())

        config = TrainConfig(experiment_name="test-exp", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        assert captured, "heartbeat should have been called"
        first = captured[0]
        assert isinstance(first, dict)
        assert first["status"] == "running"
        assert "elapsed_s" in first
        assert first["logs"] == "step 10 loss=0.5"

    @pytest.mark.asyncio
    async def test_heartbeat_logs_field_is_empty_when_adapter_returns_none(self, monkeypatch):
        import interactors.temporal.activities as acts

        adapter = self._make_adapter(["done"], log_output="")
        captured: list[dict] = []
        monkeypatch.setattr(acts.activity, "heartbeat", lambda hb: captured.append(hb))
        monkeypatch.setattr(acts.activity, "logger", MagicMock())

        config = TrainConfig(experiment_name="test-exp", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        assert captured[0]["logs"] == ""


def test_parse_valid_pct_handles_multiline_output():
    output = "Loading model...\nValid: 180/200 (90.0%)  [FAIL]\nAction distribution:"
    assert abs(_parse_valid_pct(output) - 0.90) < 1e-6


# ---------------------------------------------------------------------------
# _train_remote progress tracking
# ---------------------------------------------------------------------------


class TestTrainRemoteProgress:
    """Verify _train_remote calls adapter.progress() and persists it when db_run_id is set."""

    def _make_adapter(self, statuses, progress_return=(0.0, ""), download_path="/tmp/ckpt"):
        adapter = MagicMock()
        adapter.submit.return_value = "run-42"
        adapter.status.side_effect = list(statuses)
        adapter.logs.return_value = ""
        adapter.progress.return_value = progress_return
        adapter.download.return_value = download_path
        return adapter

    @pytest.mark.asyncio
    async def test_calls_adapter_progress_and_persists_when_db_run_id_set(self, monkeypatch):
        import interactors.temporal.activities as acts

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_run_store", mock_store)
        monkeypatch.setattr(acts.activity, "heartbeat", MagicMock())
        monkeypatch.setattr(acts.activity, "logger", MagicMock())

        adapter = self._make_adapter(["done"], progress_return=(0.5, "epoch=1.0  loss=0.4312"))
        config = TrainConfig(db_run_id="run-db-1", experiment_name="test", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        adapter.progress.assert_called()
        mock_store.update_progress.assert_called_with("run-db-1", 0.5, "epoch=1.0  loss=0.4312")

    @pytest.mark.asyncio
    async def test_skips_update_when_fraction_is_zero(self, monkeypatch):
        import interactors.temporal.activities as acts

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_run_store", mock_store)
        monkeypatch.setattr(acts.activity, "heartbeat", MagicMock())
        monkeypatch.setattr(acts.activity, "logger", MagicMock())

        adapter = self._make_adapter(["done"], progress_return=(0.0, ""))
        config = TrainConfig(db_run_id="run-db-1", experiment_name="test", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        adapter.progress.assert_called()
        mock_store.update_progress.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_progress_entirely_when_no_db_run_id(self, monkeypatch):
        import interactors.temporal.activities as acts

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_run_store", mock_store)
        monkeypatch.setattr(acts.activity, "heartbeat", MagicMock())
        monkeypatch.setattr(acts.activity, "logger", MagicMock())

        adapter = self._make_adapter(["done"], progress_return=(0.75, "epoch=2.0"))
        config = TrainConfig(db_run_id="", experiment_name="test", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            await acts._train_remote(config, adapter)

        adapter.progress.assert_not_called()
        mock_store.update_progress.assert_not_called()

    @pytest.mark.asyncio
    async def test_progress_errors_do_not_fail_the_activity(self, monkeypatch):
        import interactors.temporal.activities as acts

        mock_store = MagicMock()
        mock_store.update_progress.side_effect = RuntimeError("DB gone")
        monkeypatch.setattr(acts, "_run_store", mock_store)
        monkeypatch.setattr(acts.activity, "heartbeat", MagicMock())
        monkeypatch.setattr(acts.activity, "logger", MagicMock())

        adapter = self._make_adapter(["done"], progress_return=(0.5, "epoch=1.0"))
        config = TrainConfig(db_run_id="run-db-1", experiment_name="test", output_dir="/tmp/out")
        with patch("interactors.temporal.activities.asyncio.sleep"):
            result = await acts._train_remote(config, adapter)

        assert result is not None


# ---------------------------------------------------------------------------
# _poll_local_progress
# ---------------------------------------------------------------------------


class TestPollLocalProgress:
    """Verify _poll_local_progress reads progress.json and calls update_progress."""

    @pytest.mark.asyncio
    async def test_reads_progress_json_and_calls_update_progress(self, monkeypatch, tmp_path):
        import asyncio
        import interactors.temporal.activities as acts

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_run_store", mock_store)
        monkeypatch.setattr(acts.activity, "heartbeat", MagicMock())

        (tmp_path / "progress.json").write_text(
            '{"step": 50, "max_steps": 100, "epoch": 1.0, "loss": 0.4312}'
        )

        with patch("interactors.temporal.activities.asyncio.sleep", side_effect=asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await acts._poll_local_progress("run-42", str(tmp_path))

        mock_store.update_progress.assert_called_once_with("run-42", 0.5, "epoch=1.0  loss=0.4312")

    @pytest.mark.asyncio
    async def test_includes_eval_loss_in_detail(self, monkeypatch, tmp_path):
        import asyncio
        import interactors.temporal.activities as acts

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_run_store", mock_store)
        monkeypatch.setattr(acts.activity, "heartbeat", MagicMock())

        (tmp_path / "progress.json").write_text(
            '{"step": 75, "max_steps": 100, "epoch": 1.5, "eval_loss": 0.3210}'
        )

        with patch("interactors.temporal.activities.asyncio.sleep", side_effect=asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await acts._poll_local_progress("run-42", str(tmp_path))

        call_args = mock_store.update_progress.call_args
        assert "eval_loss=0.3210" in call_args[0][2]

    @pytest.mark.asyncio
    async def test_skips_update_when_db_run_id_is_empty(self, monkeypatch, tmp_path):
        import asyncio
        import interactors.temporal.activities as acts

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_run_store", mock_store)
        monkeypatch.setattr(acts.activity, "heartbeat", MagicMock())

        (tmp_path / "progress.json").write_text('{"step": 50, "max_steps": 100, "epoch": 1.0}')

        with patch("interactors.temporal.activities.asyncio.sleep", side_effect=asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await acts._poll_local_progress("", str(tmp_path))

        mock_store.update_progress.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_missing_progress_file_gracefully(self, monkeypatch, tmp_path):
        import asyncio
        import interactors.temporal.activities as acts

        mock_store = MagicMock()
        monkeypatch.setattr(acts, "_run_store", mock_store)
        monkeypatch.setattr(acts.activity, "heartbeat", MagicMock())

        with patch("interactors.temporal.activities.asyncio.sleep", side_effect=asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await acts._poll_local_progress("run-42", str(tmp_path))

        mock_store.update_progress.assert_not_called()
