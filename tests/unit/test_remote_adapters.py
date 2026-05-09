"""Unit tests for src/adapters/kaggle_adapter.py and src/adapters/ssh_adapter.py."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from domain.models import RemoteTrainConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _ok(stdout: str = "") -> MagicMock:
    m = MagicMock()
    m.returncode = 0
    m.stdout = stdout
    return m


# ---------------------------------------------------------------------------
# KaggleTrainingAdapter
# ---------------------------------------------------------------------------


class TestKaggleAdapterSubmit:
    def test_calls_datasets_version(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            calls.append(list(cmd))
            return _ok()

        monkeypatch.setattr("adapters.kaggle_adapter.subprocess.run", fake_run)

        from adapters.kaggle_adapter import KaggleTrainingAdapter
        adapter = KaggleTrainingAdapter(work_dir=tmp_path)
        adapter.submit(_config())

        dataset_calls = [c for c in calls if "datasets" in c]
        assert len(dataset_calls) == 1
        assert "version" in dataset_calls[0]
        assert "-p" in dataset_calls[0]

    def test_calls_kernels_push(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            calls.append(list(cmd))
            return _ok()

        monkeypatch.setattr("adapters.kaggle_adapter.subprocess.run", fake_run)

        from adapters.kaggle_adapter import KaggleTrainingAdapter
        adapter = KaggleTrainingAdapter(work_dir=tmp_path)
        adapter.submit(_config())

        push_calls = [c for c in calls if "kernels" in c and "push" in c]
        assert len(push_calls) == 1

    def test_returns_slug_with_username(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "myuser")
        monkeypatch.setattr("adapters.kaggle_adapter.subprocess.run", lambda *a, **kw: _ok())

        from adapters.kaggle_adapter import KaggleTrainingAdapter
        adapter = KaggleTrainingAdapter(work_dir=tmp_path)
        slug = adapter.submit(_config(experiment_name="myexp"))

        assert slug == "myuser/myexp"

    def test_renders_notebook_with_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        monkeypatch.setattr("adapters.kaggle_adapter.subprocess.run", lambda *a, **kw: _ok())

        from adapters.kaggle_adapter import KaggleTrainingAdapter
        adapter = KaggleTrainingAdapter(work_dir=tmp_path)
        cfg = _config(experiment_name="render-test", epochs=7)
        adapter.submit(cfg)

        notebook_path = tmp_path / "render-test" / "notebook.ipynb"
        assert notebook_path.exists(), "Rendered notebook not written"
        content = notebook_path.read_text()
        assert "{{config}}" not in content, "Template placeholder was not replaced"
        assert '"epochs": 7' in content

    def test_writes_kernel_metadata(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        monkeypatch.setattr("adapters.kaggle_adapter.subprocess.run", lambda *a, **kw: _ok())

        from adapters.kaggle_adapter import KaggleTrainingAdapter
        adapter = KaggleTrainingAdapter(work_dir=tmp_path)
        adapter.submit(_config(experiment_name="meta-test"))

        meta_path = tmp_path / "meta-test" / "kernel-metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["id"] == "testuser/meta-test"
        assert meta["enable_gpu"] is True


class TestKaggleAdapterStatus:
    def _status(self, stdout: str, monkeypatch, tmp_path) -> str:
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        monkeypatch.setattr(
            "adapters.kaggle_adapter.subprocess.run",
            lambda *a, **kw: _ok(stdout),
        )
        from adapters.kaggle_adapter import KaggleTrainingAdapter
        return KaggleTrainingAdapter(work_dir=tmp_path).status("testuser/exp")

    def test_complete_maps_to_done(self, tmp_path, monkeypatch):
        assert self._status('testuser/exp has status "complete"', monkeypatch, tmp_path) == "done"

    def test_error_maps_to_failed(self, tmp_path, monkeypatch):
        assert self._status('testuser/exp has status "error"', monkeypatch, tmp_path) == "failed"

    def test_running_maps_to_running(self, tmp_path, monkeypatch):
        assert self._status('testuser/exp has status "running"', monkeypatch, tmp_path) == "running"

    def test_queued_maps_to_pending(self, tmp_path, monkeypatch):
        assert self._status('testuser/exp has status "queued"', monkeypatch, tmp_path) == "pending"

    def test_unknown_output_defaults_to_pending(self, tmp_path, monkeypatch):
        assert self._status("something unexpected", monkeypatch, tmp_path) == "pending"


class TestKaggleAdapterDownload:
    def test_calls_kernels_output(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            calls.append(list(cmd))
            return _ok()

        monkeypatch.setattr("adapters.kaggle_adapter.subprocess.run", fake_run)

        from adapters.kaggle_adapter import KaggleTrainingAdapter
        adapter = KaggleTrainingAdapter(work_dir=tmp_path)
        result = adapter.download("testuser/exp", tmp_path / "dest")

        assert any("output" in c for c in calls)
        assert result == str(tmp_path / "dest")

    def test_unpacks_archive_if_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        monkeypatch.setattr("adapters.kaggle_adapter.subprocess.run", lambda *a, **kw: _ok())

        import tarfile
        dest = tmp_path / "dest"
        dest.mkdir()
        # Create a fake checkpoint.tar.gz in dest before download is called.
        marker = tmp_path / "marker.txt"
        marker.write_text("hello")
        archive = dest / "checkpoint.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(marker, arcname="marker.txt")

        from adapters.kaggle_adapter import KaggleTrainingAdapter
        KaggleTrainingAdapter(work_dir=tmp_path).download("testuser/exp", dest)

        assert not archive.exists(), "Archive should be deleted after extraction"
        assert (dest / "marker.txt").exists(), "Archive contents should be extracted"


# ---------------------------------------------------------------------------
# SshTrainingAdapter
# ---------------------------------------------------------------------------


class TestSshAdapterSubmit:
    def _make_adapter(self, monkeypatch):
        monkeypatch.setenv("REMOTE_HOST", "gpu.example.com")
        monkeypatch.setenv("REMOTE_USER", "ubuntu")
        monkeypatch.setenv("REMOTE_KEY_PATH", "/home/user/.ssh/id_rsa")
        monkeypatch.setenv("REMOTE_WORK_DIR", "/app")
        from adapters.ssh_adapter import SshTrainingAdapter
        return SshTrainingAdapter()

    def test_calls_rsync_for_src(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)
        calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            calls.append(list(cmd))
            return _ok()

        monkeypatch.setattr("adapters.ssh_adapter.subprocess.run", fake_run)
        adapter.submit(_config())

        rsync_calls = [c for c in calls if c[0] == "rsync"]
        assert len(rsync_calls) >= 1
        src_sync = [c for c in rsync_calls if "src/" in " ".join(c)]
        assert src_sync, "Expected rsync of src/"

    def test_calls_ssh_to_start_screen_session(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)
        calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            calls.append(list(cmd))
            return _ok()

        monkeypatch.setattr("adapters.ssh_adapter.subprocess.run", fake_run)
        adapter.submit(_config(experiment_name="my-exp"))

        ssh_calls = [c for c in calls if c[0] == "ssh"]
        assert ssh_calls, "Expected at least one SSH call"
        cmd_str = " ".join(ssh_calls[-1])
        assert "screen" in cmd_str
        assert "my-exp" in cmd_str

    def test_returns_session_name(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)
        monkeypatch.setattr("adapters.ssh_adapter.subprocess.run", lambda *a, **kw: _ok())
        run_id = adapter.submit(_config(experiment_name="sess-exp"))
        assert "sess-exp" in run_id


class TestSshAdapterStatus:
    def _make_adapter(self, monkeypatch):
        monkeypatch.setenv("REMOTE_HOST", "gpu.example.com")
        monkeypatch.setenv("REMOTE_USER", "ubuntu")
        monkeypatch.setenv("REMOTE_KEY_PATH", "")
        monkeypatch.setenv("REMOTE_WORK_DIR", "/app")
        from adapters.ssh_adapter import SshTrainingAdapter
        return SshTrainingAdapter()

    def test_running_when_screen_session_alive(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)
        monkeypatch.setattr(
            "adapters.ssh_adapter.subprocess.run",
            lambda *a, **kw: _ok("aipet-my-exp\t(Detached)"),
        )
        assert adapter.status("aipet-my-exp") == "running"

    def test_done_when_session_gone_and_checkpoint_exists(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)
        responses = iter([_ok("No Sockets found"), _ok("exists")])
        monkeypatch.setattr(
            "adapters.ssh_adapter.subprocess.run",
            lambda *a, **kw: next(responses),
        )
        assert adapter.status("aipet-my-exp") == "done"

    def test_failed_when_session_gone_and_no_checkpoint(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)
        responses = iter([_ok("No Sockets found"), _ok("")])
        monkeypatch.setattr(
            "adapters.ssh_adapter.subprocess.run",
            lambda *a, **kw: next(responses),
        )
        assert adapter.status("aipet-my-exp") == "failed"


class TestSshAdapterDownload:
    def test_calls_rsync_to_fetch_checkpoint(self, monkeypatch):
        monkeypatch.setenv("REMOTE_HOST", "gpu.example.com")
        monkeypatch.setenv("REMOTE_USER", "ubuntu")
        monkeypatch.setenv("REMOTE_KEY_PATH", "")
        monkeypatch.setenv("REMOTE_WORK_DIR", "/app")
        calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            calls.append(list(cmd))
            return _ok()

        monkeypatch.setattr("adapters.ssh_adapter.subprocess.run", fake_run)

        from adapters.ssh_adapter import SshTrainingAdapter
        adapter = SshTrainingAdapter()
        tmp = Path("/tmp/ckpt")
        result = adapter.download("aipet-exp", tmp)

        rsync_calls = [c for c in calls if c[0] == "rsync"]
        assert rsync_calls, "Expected rsync to download checkpoint"
        assert "checkpoints" in " ".join(rsync_calls[-1])
        assert result == str(tmp)


# ---------------------------------------------------------------------------
# train_activity routing
# ---------------------------------------------------------------------------


class TestTrainActivityRouting:
    """Verify train_activity delegates to the correct backend without running real training."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_local_backend_calls_train_domain_function(self, monkeypatch):
        import temporal.activities as acts

        called = {}

        async def fake_local(config):
            called["config"] = config
            from temporal.activities import CheckpointPath
            return CheckpointPath(path="models/checkpoints")

        monkeypatch.setattr(acts, "_train_local", fake_local)

        from temporal.activities import TrainConfig, train_activity
        config = TrainConfig(remote_backend="")
        result = self._run(train_activity(config))
        assert called.get("config") is not None
        assert result.path == "models/checkpoints"

    def test_kaggle_backend_routes_to_kaggle_adapter(self, monkeypatch):
        import temporal.activities as acts

        submitted = {}

        async def fake_remote(config, adapter):
            submitted["adapter"] = type(adapter).__name__
            from temporal.activities import CheckpointPath
            return CheckpointPath(path="models/checkpoints")

        monkeypatch.setattr(acts, "_train_remote", fake_remote)

        from temporal.activities import TrainConfig, train_activity
        config = TrainConfig(remote_backend="kaggle")
        self._run(train_activity(config))
        assert submitted["adapter"] == "KaggleTrainingAdapter"

    def test_ssh_backend_routes_to_ssh_adapter(self, monkeypatch):
        import temporal.activities as acts

        submitted = {}

        async def fake_remote(config, adapter):
            submitted["adapter"] = type(adapter).__name__
            from temporal.activities import CheckpointPath
            return CheckpointPath(path="models/checkpoints")

        monkeypatch.setattr(acts, "_train_remote", fake_remote)

        from temporal.activities import TrainConfig, train_activity
        config = TrainConfig(remote_backend="ssh")
        self._run(train_activity(config))
        assert submitted["adapter"] == "SshTrainingAdapter"

    def test_unknown_backend_raises_application_error(self):
        from temporalio.exceptions import ApplicationError
        from temporal.activities import TrainConfig, train_activity

        config = TrainConfig(remote_backend="gcp")
        with pytest.raises(ApplicationError, match="Unknown remote_backend"):
            self._run(train_activity(config))
