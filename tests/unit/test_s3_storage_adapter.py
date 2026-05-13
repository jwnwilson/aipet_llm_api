"""Tests for S3StorageAdapter write_bytes / read_text helpers."""
import io
import unittest
from unittest.mock import MagicMock, patch


def _make_adapter():
    with patch("boto3.client"):
        from adapters.storage.s3 import S3StorageAdapter
        adapter = S3StorageAdapter.__new__(S3StorageAdapter)
        adapter._bucket = "test-bucket"
        adapter._s3 = MagicMock()
        return adapter


class TestWriteBytes(unittest.TestCase):
    def test_write_bytes_calls_put_object(self):
        adapter = _make_adapter()
        adapter.write_bytes("runpod/run-1/logs.txt", b"hello logs")
        adapter._s3.put_object.assert_called_once_with(
            Bucket="test-bucket", Key="runpod/run-1/logs.txt", Body=b"hello logs"
        )


class TestReadText(unittest.TestCase):
    def test_read_text_returns_content(self):
        adapter = _make_adapter()
        adapter._s3.get_object.return_value = {"Body": io.BytesIO(b"running")}
        result = adapter.read_text("runpod/run-1/status.txt")
        assert result == "running"

    def test_read_text_returns_empty_string_on_missing_key(self):
        from botocore.exceptions import ClientError
        adapter = _make_adapter()
        adapter._s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject"
        )
        result = adapter.read_text("runpod/run-1/status.txt")
        assert result == ""
