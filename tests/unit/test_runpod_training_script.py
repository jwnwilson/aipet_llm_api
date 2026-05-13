"""Tests for S3 log streaming in the RunPod training script."""
import logging
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("AWS_S3_BUCKET", "test-bucket")
os.environ.setdefault("RUN_ID", "runpod/test-run-abc123")


class TestFlushLogs(unittest.TestCase):
    def test_flush_logs_uploads_buffer_content(self):
        from adapters.compute.runpod import training_script as ts

        ts._log_buffer.truncate(0)
        ts._log_buffer.seek(0)
        ts._log_buffer.write("line one\nline two\n")

        storage = MagicMock()
        ts._flush_logs_to_s3(storage)

        storage.write_bytes.assert_called_once()
        key, content = storage.write_bytes.call_args[0]
        assert key == "runpod/test-run-abc123/logs.txt"
        assert b"line one" in content

    def test_flush_logs_skips_upload_when_buffer_empty(self):
        from adapters.compute.runpod import training_script as ts

        ts._log_buffer.truncate(0)
        ts._log_buffer.seek(0)

        storage = MagicMock()
        ts._flush_logs_to_s3(storage)
        storage.write_bytes.assert_not_called()


class TestRunSubprocessStreaming(unittest.TestCase):
    def test_streams_output_and_flushes_every_20_lines(self):
        from adapters.compute.runpod import training_script as ts

        lines = [f"line {i}\n" for i in range(25)]
        fake_proc = MagicMock()
        fake_proc.stdout = iter(lines)
        fake_proc.wait.return_value = 0

        storage = MagicMock()
        with patch("subprocess.Popen", return_value=fake_proc):
            rc = ts._run_subprocess_streaming(
                ["echo", "hi"], storage, log=logging.getLogger("test")
            )

        assert rc == 0
        # flush at line 20 + final flush = at least 2 calls
        assert storage.write_bytes.call_count >= 2

    def test_returns_nonzero_exit_code(self):
        from adapters.compute.runpod import training_script as ts

        fake_proc = MagicMock()
        fake_proc.stdout = iter([])
        fake_proc.wait.return_value = 1

        storage = MagicMock()
        with patch("subprocess.Popen", return_value=fake_proc):
            rc = ts._run_subprocess_streaming(["false"], storage, log=logging.getLogger("test"))

        assert rc == 1
