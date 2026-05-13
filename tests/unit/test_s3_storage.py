"""Unit tests for S3StorageAdapter — boto3 client is mocked."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_adapter(tmp_path: Path):
    """Return an S3StorageAdapter with a mocked boto3 client."""
    with patch("boto3.client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        # Ensure AWS_S3_BUCKET is set for the constructor
        import os
        os.environ.setdefault("AWS_S3_BUCKET", "test-bucket")
        from adapters.storage.s3 import S3StorageAdapter
        adapter = S3StorageAdapter(bucket="test-bucket")
    adapter._s3 = mock_client
    return adapter, mock_client


class TestS3StorageAdapterUpload:
    def test_upload_calls_upload_file(self, tmp_path):
        adapter, s3 = _make_adapter(tmp_path)
        local = tmp_path / "model.gguf"
        local.write_bytes(b"weights")

        adapter.upload(local, "gguf/model.gguf")

        s3.upload_file.assert_called_once_with(
            str(local), "test-bucket", "gguf/model.gguf"
        )


class TestS3StorageAdapterDownload:
    def test_download_creates_parent_dirs_and_calls_download_file(self, tmp_path):
        adapter, s3 = _make_adapter(tmp_path)
        dest = tmp_path / "deep" / "dir" / "model.gguf"

        adapter.download("gguf/model.gguf", dest)

        assert dest.parent.exists()
        s3.download_file.assert_called_once_with(
            "test-bucket", "gguf/model.gguf", str(dest)
        )


class TestS3StorageAdapterExists:
    def test_returns_true_when_head_object_succeeds(self, tmp_path):
        adapter, s3 = _make_adapter(tmp_path)
        s3.head_object.return_value = {}

        assert adapter.exists("some/key") is True
        s3.head_object.assert_called_once_with(Bucket="test-bucket", Key="some/key")

    def test_returns_false_when_head_object_raises(self, tmp_path):
        adapter, s3 = _make_adapter(tmp_path)
        s3.head_object.side_effect = Exception("NoSuchKey")

        assert adapter.exists("some/key") is False


class TestS3StorageAdapterDelete:
    def test_delete_calls_delete_object(self, tmp_path):
        adapter, s3 = _make_adapter(tmp_path)

        adapter.delete("some/key")

        s3.delete_object.assert_called_once_with(Bucket="test-bucket", Key="some/key")

    def test_delete_is_silent_on_error(self, tmp_path):
        adapter, s3 = _make_adapter(tmp_path)
        s3.delete_object.side_effect = Exception("access denied")

        adapter.delete("some/key")  # must not raise


class TestS3StorageAdapterWriteBytes:
    def test_write_bytes_calls_put_object(self, tmp_path):
        import io
        adapter, s3 = _make_adapter(tmp_path)
        adapter.write_bytes("runpod/run-1/logs.txt", b"hello logs")
        s3.put_object.assert_called_once_with(
            Bucket="test-bucket", Key="runpod/run-1/logs.txt", Body=b"hello logs"
        )


class TestS3StorageAdapterReadText:
    def test_read_text_returns_content(self, tmp_path):
        import io
        adapter, s3 = _make_adapter(tmp_path)
        s3.get_object.return_value = {"Body": io.BytesIO(b"running")}
        assert adapter.read_text("runpod/run-1/status.txt") == "running"

    def test_read_text_returns_empty_string_on_missing_key(self, tmp_path):
        from botocore.exceptions import ClientError
        adapter, s3 = _make_adapter(tmp_path)
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject"
        )
        assert adapter.read_text("runpod/run-1/status.txt") == ""
