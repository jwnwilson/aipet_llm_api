"""Unit tests for RunPodTrainingAdapter — runpod SDK and boto3 are mocked."""
from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domain.models import RemoteTrainConfig


def _config(**kwargs) -> RemoteTrainConfig:
    defaults = dict(
        model="HuggingFaceTB/SmolLM-360M",
        train_data="data/train.jsonl",
        eval_data="data/eval.jsonl",
        epochs=2,
        patience=2,
        warmup_ratio=0.05,
        experiment_name="test-exp",
    )
    defaults.update(kwargs)
    return RemoteTrainConfig(**defaults)


def _make_adapter(monkeypatch, tmp_path: Path):
    """Return a RunPodTrainingAdapter with mocked S3 client."""
    monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret")
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-runpod-key")

    mock_s3 = MagicMock()

    with patch("boto3.client", return_value=mock_s3):
        from adapters.compute.runpod.adapter import RunPodTrainingAdapter
        adapter = RunPodTrainingAdapter(work_dir=tmp_path / "runs")

    adapter._s3 = mock_s3
    return adapter, mock_s3


class TestRunPodAdapterSubmit:
    def _submit(self, monkeypatch, tmp_path):
        """Run submit() and return (run_id, create_pod call kwargs)."""
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        monkeypatch.setattr(adapter, "_stage_files", lambda config, staging: staging.mkdir(parents=True, exist_ok=True))
        monkeypatch.setattr(adapter, "_upload_to_s3", lambda staging, run_id, config: None)

        mock_runpod = MagicMock()
        mock_runpod.create_pod.return_value = {"id": "pod-abc123"}
        import sys
        sys.modules["runpod"] = mock_runpod

        run_id = adapter.submit(_config())
        return run_id, mock_runpod.create_pod.call_args.kwargs

    def test_docker_args_contains_no_double_quotes(self, monkeypatch, tmp_path):
        # RunPod's SDK embeds docker_args directly into a GraphQL mutation string
        # without escaping — any " character breaks the query with a syntax error.
        _, kwargs = self._submit(monkeypatch, tmp_path)
        docker_args = kwargs["docker_args"]
        assert '"' not in docker_args, (
            f'docker_args must not contain double-quote characters '
            f'(RunPod GraphQL will break). Got: {docker_args!r}'
        )

    def test_submit_creates_pod_and_uploads_to_s3(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)

        monkeypatch.setattr(adapter, "_stage_files", lambda config, staging: staging.mkdir(parents=True, exist_ok=True))
        monkeypatch.setattr(adapter, "_upload_to_s3", lambda staging, run_id, config: None)

        mock_runpod = MagicMock()
        mock_runpod.create_pod.return_value = {"id": "pod-abc123"}

        import sys
        sys.modules["runpod"] = mock_runpod

        run_id = adapter.submit(_config())

        assert run_id.startswith("runpod/test-exp-")
        mock_runpod.create_pod.assert_called_once()
        # pod_id.txt written to S3
        s3.put_object.assert_called_once()
        call_kwargs = s3.put_object.call_args.kwargs
        assert call_kwargs["Key"].endswith("/pod_id.txt")
        assert call_kwargs["Body"] == b"pod-abc123"


class TestRunPodAdapterStatus:
    def test_returns_status_from_s3(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        s3.get_object.return_value = {"Body": MagicMock(read=lambda: b"running")}

        assert adapter.status("runpod/test-exp-aabbcc") == "running"

    def test_returns_pending_when_no_status_txt_and_no_pod_id(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        s3.get_object.side_effect = Exception("NoSuchKey")

        assert adapter.status("runpod/test-exp-aabbcc") == "pending"

    def test_falls_back_to_runpod_api_on_missing_status_txt(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)

        def get_object(Bucket, Key):
            if Key.endswith("status.txt"):
                raise Exception("NoSuchKey")
            return {"Body": MagicMock(read=lambda: b"pod-xyz")}

        s3.get_object.side_effect = get_object

        import sys
        mock_runpod = MagicMock()
        mock_runpod.get_pod.return_value = {"desiredStatus": "RUNNING"}
        sys.modules["runpod"] = mock_runpod

        assert adapter.status("runpod/test-exp-aabbcc") == "running"


class TestRunPodAdapterDownload:
    def test_download_extracts_checkpoint(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)

        # Create a real tar.gz with a dummy file inside
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text('{"model": "test"}')
        archive = tmp_path / "checkpoint.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(checkpoint_dir, arcname="checkpoints")

        def fake_download(Bucket, Key, Filename):
            import shutil
            shutil.copy2(archive, Filename)

        s3.download_file.side_effect = fake_download

        dest = tmp_path / "output"
        result = adapter.download("runpod/test-exp-aabbcc", dest)

        assert Path(result) == dest
        assert (dest / "checkpoints" / "config.json").exists()
        assert not (dest / "checkpoint.tar.gz").exists()


class TestRunPodAdapterLogs:
    def test_terminate_pod_terminates_without_archiving_logs(self, monkeypatch, tmp_path):
        # logs are written by the training script; _terminate_pod only terminates the pod
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        s3.get_object.return_value = {"Body": MagicMock(read=lambda: b"pod-xyz")}

        import sys
        mock_runpod = MagicMock(spec=["terminate_pod"])
        sys.modules["runpod"] = mock_runpod

        adapter._terminate_pod("runpod/test-exp-aabbcc")

        mock_runpod.terminate_pod.assert_called_once_with("pod-xyz")
        s3.put_object.assert_not_called()

    def test_logs_reads_from_s3_via_storage_adapter(self, monkeypatch, tmp_path):
        adapter, _ = _make_adapter(monkeypatch, tmp_path)

        storage_mock = MagicMock()
        storage_mock.read_text.return_value = "epoch 1 loss=0.5\nepoch 2 loss=0.3\n"

        with patch("adapters.storage.s3.S3StorageAdapter", return_value=storage_mock):
            result = adapter.logs("runpod/test-exp-aabbcc")

        assert result == "epoch 1 loss=0.5\nepoch 2 loss=0.3\n"
        storage_mock.read_text.assert_called_once_with("runpod/test-exp-aabbcc/logs.txt")

    def test_logs_returns_empty_string_when_no_log_in_s3(self, monkeypatch, tmp_path):
        adapter, _ = _make_adapter(monkeypatch, tmp_path)

        storage_mock = MagicMock()
        storage_mock.read_text.return_value = ""

        with patch("adapters.storage.s3.S3StorageAdapter", return_value=storage_mock):
            result = adapter.logs("runpod/test-exp-aabbcc")

        assert result == ""


class TestRunPodAdapterProgress:
    def test_returns_fraction_and_detail(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        payload = json.dumps({"fraction": 0.5, "detail": "epoch=1"}).encode()
        s3.get_object.return_value = {"Body": MagicMock(read=lambda: payload)}

        fraction, detail = adapter.progress("runpod/test-exp-aabbcc")

        assert fraction == pytest.approx(0.5)
        assert detail == "epoch=1"

    def test_returns_zero_on_missing_progress_json(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        s3.get_object.side_effect = Exception("NoSuchKey")

        fraction, detail = adapter.progress("runpod/test-exp-aabbcc")

        assert fraction == 0.0
        assert detail == ""
