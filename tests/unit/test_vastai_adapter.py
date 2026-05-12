"""Unit tests for VastAiTrainingAdapter — vastai SDK and boto3 are mocked."""
from __future__ import annotations

import json
import sys
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
    """Return a VastAiTrainingAdapter with mocked S3 client."""
    monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret")
    monkeypatch.setenv("VAST_API_KEY", "fake-vast-key")

    mock_s3 = MagicMock()

    with patch("boto3.client", return_value=mock_s3):
        from adapters.compute.vastai.adapter import VastAiTrainingAdapter
        adapter = VastAiTrainingAdapter(work_dir=tmp_path / "runs")

    adapter._s3 = mock_s3
    return adapter, mock_s3


def _mock_vastai_client(adapter, offers=None, instance_info=None):
    """Inject a mock VastAI client into the adapter."""
    mock_client = MagicMock()
    mock_client.search_offers.return_value = [
        {"id": 111, "dph_total": 0.5},
        {"id": 222, "dph_total": 0.3},  # cheapest — should be chosen
    ] if offers is None else offers
    mock_client.create_instance.return_value = {"new_contract": 222}
    mock_client.show_instance.return_value = instance_info or {"actual_status": "running"}
    adapter._build_vastai_client = lambda: mock_client
    return mock_client


class TestVastAiAdapterSubmit:
    def test_submit_picks_cheapest_offer_and_creates_instance(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        mock_client = _mock_vastai_client(adapter)

        monkeypatch.setattr(adapter, "_stage_files", lambda config, staging: staging.mkdir(parents=True, exist_ok=True))
        monkeypatch.setattr(adapter, "_upload_to_s3", lambda staging, run_id, config: None)

        run_id = adapter.submit(_config())

        assert run_id.startswith("vastai/test-exp-")
        # Cheapest offer id=222 (dph_total=0.3) must be chosen
        mock_client.create_instance.assert_called_once()
        call_kwargs = mock_client.create_instance.call_args.kwargs
        assert call_kwargs["id"] == 222

    def test_submit_writes_instance_id_to_s3(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        _mock_vastai_client(adapter)

        monkeypatch.setattr(adapter, "_stage_files", lambda config, staging: staging.mkdir(parents=True, exist_ok=True))
        monkeypatch.setattr(adapter, "_upload_to_s3", lambda staging, run_id, config: None)

        adapter.submit(_config())

        s3.put_object.assert_called_once()
        call_kwargs = s3.put_object.call_args.kwargs
        assert call_kwargs["Key"].endswith("/instance_id.txt")
        assert call_kwargs["Body"] == b"222"

    def test_submit_raises_when_no_offers_found(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        _mock_vastai_client(adapter, offers=[])

        monkeypatch.setattr(adapter, "_stage_files", lambda config, staging: staging.mkdir(parents=True, exist_ok=True))
        monkeypatch.setattr(adapter, "_upload_to_s3", lambda staging, run_id, config: None)

        with pytest.raises(RuntimeError, match="No Vast.ai offers found"):
            adapter.submit(_config())


class TestVastAiAdapterStatus:
    def test_returns_status_from_s3(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        s3.get_object.return_value = {"Body": MagicMock(read=lambda: b"done")}

        assert adapter.status("vastai/test-exp-aabbcc") == "done"

    def test_returns_pending_when_no_s3_data_and_no_instance_id(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        s3.get_object.side_effect = Exception("NoSuchKey")

        assert adapter.status("vastai/test-exp-aabbcc") == "pending"

    def test_falls_back_to_vastai_api_when_no_status_txt(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)

        def get_object(Bucket, Key):
            if Key.endswith("status.txt"):
                raise Exception("NoSuchKey")
            return {"Body": MagicMock(read=lambda: b"12345")}

        s3.get_object.side_effect = get_object
        mock_client = MagicMock()
        mock_client.show_instance.return_value = {"actual_status": "loading"}
        adapter._build_vastai_client = lambda: mock_client

        assert adapter.status("vastai/test-exp-aabbcc") == "pending"

    def test_maps_exited_status_to_pending_when_no_status_txt(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)

        def get_object(Bucket, Key):
            if Key.endswith("status.txt"):
                raise Exception("NoSuchKey")
            return {"Body": MagicMock(read=lambda: b"99999")}

        s3.get_object.side_effect = get_object
        mock_client = MagicMock()
        mock_client.show_instance.return_value = {"actual_status": "exited"}
        adapter._build_vastai_client = lambda: mock_client

        # exited with no status.txt → we don't know if done or failed → pending
        assert adapter.status("vastai/test-exp-aabbcc") == "pending"


class TestVastAiAdapterDownload:
    def test_download_extracts_checkpoint(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)

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
        result = adapter.download("vastai/test-exp-aabbcc", dest)

        assert Path(result) == dest
        assert (dest / "checkpoints" / "config.json").exists()
        assert not (dest / "checkpoint.tar.gz").exists()


class TestVastAiAdapterProgress:
    def test_returns_fraction_and_detail(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        payload = json.dumps({"fraction": 0.75, "detail": "epoch=2"}).encode()
        s3.get_object.return_value = {"Body": MagicMock(read=lambda: payload)}

        fraction, detail = adapter.progress("vastai/test-exp-aabbcc")

        assert fraction == pytest.approx(0.75)
        assert detail == "epoch=2"

    def test_returns_zero_on_missing_progress_json(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        s3.get_object.side_effect = Exception("NoSuchKey")

        fraction, detail = adapter.progress("vastai/test-exp-aabbcc")

        assert fraction == 0.0
        assert detail == ""


class TestVastAiAdapterLogs:
    def test_returns_log_string(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        s3.get_object.return_value = {"Body": MagicMock(read=lambda: b"12345")}
        mock_client = MagicMock()
        mock_client.logs.return_value = "training step 1/10\n"
        adapter._build_vastai_client = lambda: mock_client

        result = adapter.logs("vastai/test-exp-aabbcc")

        assert result == "training step 1/10\n"
        mock_client.logs.assert_called_once_with(instance_id=12345, tail="200")

    def test_returns_empty_string_on_error(self, monkeypatch, tmp_path):
        adapter, s3 = _make_adapter(monkeypatch, tmp_path)
        s3.get_object.side_effect = Exception("NoSuchKey")

        assert adapter.logs("vastai/test-exp-aabbcc") == ""
