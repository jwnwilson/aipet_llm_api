"""Unit tests for src/adapters/kaggle/ and src/adapters/ssh_adapter.py."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

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


def _ok(stdout: str = "", stderr: str = "") -> MagicMock:
    m = MagicMock()
    m.returncode = 0
    m.stdout = stdout
    m.stderr = stderr
    return m


def _fail(stdout: str = "", stderr: str = "") -> MagicMock:
    m = MagicMock()
    m.returncode = 1
    m.stdout = stdout
    m.stderr = stderr
    return m


# ---------------------------------------------------------------------------
# KaggleTrainingAdapter
# ---------------------------------------------------------------------------


class TestKaggleAdapterSubmit:
    def _patch_no_wait(self, monkeypatch) -> None:
        """Skip Kaggle-CLI detection and Python-API polling (not installed in dev)."""
        monkeypatch.setattr(
            "adapters.kaggle.adapter._kaggle_bin",
            lambda: "kaggle",
        )
        monkeypatch.setattr(
            "adapters.kaggle.adapter.KaggleTrainingAdapter._wait_for_dataset",
            lambda *a, **kw: None,
        )

    def test_calls_datasets_version_when_dataset_exists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        self._patch_no_wait(monkeypatch)

        calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            calls.append(list(cmd))
            # Simulate "dataset already exists" by returning non-zero for create
            if "create" in cmd:
                return _fail()
            return _ok()

        monkeypatch.setattr("adapters.kaggle.adapter.subprocess.run", fake_run)

        from adapters.kaggle import KaggleTrainingAdapter
        adapter = KaggleTrainingAdapter(work_dir=tmp_path)
        adapter.submit(_config())

        dataset_calls = [c for c in calls if "datasets" in c]
        assert len(dataset_calls) == 2  # create (non-zero) + version
        assert "version" in dataset_calls[1]
        assert "-p" in dataset_calls[1]

    def test_calls_kernels_push(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        self._patch_no_wait(monkeypatch)

        calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            calls.append(list(cmd))
            return _ok()

        monkeypatch.setattr("adapters.kaggle.adapter.subprocess.run", fake_run)

        from adapters.kaggle import KaggleTrainingAdapter
        adapter = KaggleTrainingAdapter(work_dir=tmp_path)
        adapter.submit(_config())

        push_calls = [c for c in calls if "kernels" in c and "push" in c]
        assert len(push_calls) == 1

    def test_returns_slug_with_username(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "myuser")
        self._patch_no_wait(monkeypatch)
        monkeypatch.setattr("adapters.kaggle.adapter.subprocess.run", lambda *a, **kw: _ok())

        from adapters.kaggle import KaggleTrainingAdapter
        adapter = KaggleTrainingAdapter(work_dir=tmp_path)
        slug = adapter.submit(_config(experiment_name="myexp"))

        assert slug == "myuser/myexp"

    def test_renders_notebook_with_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        self._patch_no_wait(monkeypatch)
        monkeypatch.setattr("adapters.kaggle.adapter.subprocess.run", lambda *a, **kw: _ok())

        from adapters.kaggle import KaggleTrainingAdapter
        adapter = KaggleTrainingAdapter(work_dir=tmp_path)
        cfg = _config(experiment_name="render-test", epochs=7)
        adapter.submit(cfg)

        notebook_path = tmp_path / "render-test" / "notebook.ipynb"
        assert notebook_path.exists(), "Rendered notebook not written"
        content = notebook_path.read_text()
        assert "{{config}}" not in content, "Template placeholder was not replaced"
        assert "'epochs': 7" in content  # injected as Python repr, not JSON

    def test_writes_kernel_metadata(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        self._patch_no_wait(monkeypatch)
        monkeypatch.setattr("adapters.kaggle.adapter.subprocess.run", lambda *a, **kw: _ok())

        from adapters.kaggle import KaggleTrainingAdapter
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
        monkeypatch.setattr("adapters.kaggle.adapter._kaggle_bin", lambda: "kaggle")
        monkeypatch.setattr(
            "adapters.kaggle.adapter.subprocess.run",
            lambda *a, **kw: _ok(stdout),
        )
        from adapters.kaggle import KaggleTrainingAdapter
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
        monkeypatch.setattr("adapters.kaggle.adapter._kaggle_bin", lambda: "kaggle")

        calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            calls.append(list(cmd))
            return _ok()

        monkeypatch.setattr("adapters.kaggle.adapter.subprocess.run", fake_run)

        from adapters.kaggle import KaggleTrainingAdapter
        adapter = KaggleTrainingAdapter(work_dir=tmp_path)
        result = adapter.download("testuser/exp", tmp_path / "dest")

        assert any("output" in c for c in calls)
        assert result == str(tmp_path / "dest")

    def test_unpacks_archive_if_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        monkeypatch.setattr("adapters.kaggle.adapter._kaggle_bin", lambda: "kaggle")
        monkeypatch.setattr("adapters.kaggle.adapter.subprocess.run", lambda *a, **kw: _ok())

        import tarfile
        dest = tmp_path / "dest"
        dest.mkdir()
        marker = tmp_path / "marker.txt"
        marker.write_text("hello")
        archive = dest / "checkpoint.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(marker, arcname="marker.txt")

        from adapters.kaggle import KaggleTrainingAdapter
        KaggleTrainingAdapter(work_dir=tmp_path).download("testuser/exp", dest)

        assert not archive.exists(), "Archive should be deleted after extraction"
        assert (dest / "marker.txt").exists(), "Archive contents should be extracted"


class TestKaggleAdapterLogs:
    def test_returns_kernels_status_output(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        monkeypatch.setattr("adapters.kaggle.adapter._kaggle_bin", lambda: "kaggle")

        def fake_run(cmd, **kw):
            return _ok('testuser/exp has status "running"')

        monkeypatch.setattr("adapters.kaggle.adapter.subprocess.run", fake_run)
        from adapters.kaggle import KaggleTrainingAdapter
        result = KaggleTrainingAdapter(work_dir=tmp_path).logs("testuser/exp")

        assert "running" in result

    def test_calls_kernels_status_command(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        monkeypatch.setattr("adapters.kaggle.adapter._kaggle_bin", lambda: "kaggle")
        calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            calls.append(list(cmd))
            return _ok()

        monkeypatch.setattr("adapters.kaggle.adapter.subprocess.run", fake_run)
        from adapters.kaggle import KaggleTrainingAdapter
        KaggleTrainingAdapter(work_dir=tmp_path).logs("testuser/exp")

        assert any("status" in c for cmd in calls for c in cmd)

    def test_returns_empty_string_when_no_output(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        monkeypatch.setattr("adapters.kaggle.adapter._kaggle_bin", lambda: "kaggle")
        monkeypatch.setattr("adapters.kaggle.adapter.subprocess.run", lambda *a, **kw: _ok(""))
        from adapters.kaggle import KaggleTrainingAdapter
        assert KaggleTrainingAdapter(work_dir=tmp_path).logs("testuser/exp") == ""


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


class TestSshAdapterLogs:
    def _make_adapter(self, monkeypatch):
        monkeypatch.setenv("REMOTE_HOST", "gpu.example.com")
        monkeypatch.setenv("REMOTE_USER", "ubuntu")
        monkeypatch.setenv("REMOTE_KEY_PATH", "")
        monkeypatch.setenv("REMOTE_WORK_DIR", "/app")
        from adapters.ssh_adapter import SshTrainingAdapter
        return SshTrainingAdapter()

    def test_returns_remote_log_output(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)
        monkeypatch.setattr(
            "adapters.ssh_adapter.subprocess.run",
            lambda *a, **kw: _ok("step 10/200 loss=1.23\nstep 20/200 loss=1.10"),
        )
        result = adapter.logs("aipet-my-exp")
        assert "loss=1.23" in result

    def test_ssh_command_tails_train_log(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)
        calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            calls.append(list(cmd))
            return _ok()

        monkeypatch.setattr("adapters.ssh_adapter.subprocess.run", fake_run)
        adapter.logs("aipet-my-exp")

        ssh_calls = [c for c in calls if "ssh" in c]
        assert ssh_calls, "Expected an SSH call"
        cmd_str = " ".join(ssh_calls[-1])
        assert "tail" in cmd_str
        assert "train.log" in cmd_str


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

    def test_colab_backend_routes_to_colab_adapter(self, monkeypatch):
        import temporal.activities as acts

        submitted: dict = {}

        async def fake_remote(config, adapter):
            submitted["adapter"] = type(adapter).__name__
            from temporal.activities import CheckpointPath
            return CheckpointPath(path="models/checkpoints")

        monkeypatch.setattr(acts, "_train_remote", fake_remote)
        monkeypatch.setattr(
            "adapters.colab.adapter.ColabTrainingAdapter._build_drive_client",
            lambda self: MagicMock(),
        )

        from temporal.activities import TrainConfig, train_activity
        config = TrainConfig(remote_backend="colab")
        self._run(train_activity(config))
        assert submitted["adapter"] == "ColabTrainingAdapter"


# ---------------------------------------------------------------------------
# ColabTrainingAdapter helpers
# ---------------------------------------------------------------------------


def _make_status_downloader(content: bytes):
    """Returns a MediaIoBaseDownload drop-in that writes *content* into buf."""

    class _FakeDownloader:
        def __init__(self, buf, request):
            buf.write(content)

        def next_chunk(self):
            return None, True

    return _FakeDownloader


# ---------------------------------------------------------------------------
# ColabTrainingAdapter — submit
# ---------------------------------------------------------------------------


class TestColabAdapterSubmit:
    def _make_adapter(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "adapters.colab.adapter.ColabTrainingAdapter._build_drive_client",
            lambda self: MagicMock(),
        )
        from adapters.colab.adapter import ColabTrainingAdapter

        adapter = ColabTrainingAdapter(work_dir=tmp_path)
        drive = adapter._drive
        drive.files().list().execute.return_value = {"files": []}
        drive.files().create().execute.return_value = {"id": "fake-id"}
        drive.files().update().execute.return_value = {}
        return adapter

    def _data_config(self, tmp_path, **kwargs) -> RemoteTrainConfig:
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "train.jsonl").write_text('{"x":1}\n')
        (data_dir / "eval.jsonl").write_text('{"x":2}\n')
        return _config(
            train_data=str(data_dir / "train.jsonl"),
            eval_data=str(data_dir / "eval.jsonl"),
            **kwargs,
        )

    def _patch_io(self, monkeypatch):
        monkeypatch.setattr("adapters.colab.adapter.subprocess.run", lambda *a, **kw: _ok())
        monkeypatch.setattr("googleapiclient.http.MediaFileUpload", MagicMock)

    def test_returns_non_empty_string_run_id(self, tmp_path, monkeypatch):
        adapter = self._make_adapter(tmp_path, monkeypatch)
        self._patch_io(monkeypatch)
        run_id = adapter.submit(self._data_config(tmp_path))
        assert isinstance(run_id, str) and run_id

    def test_creates_folder_hierarchy_on_drive(self, tmp_path, monkeypatch):
        adapter = self._make_adapter(tmp_path, monkeypatch)
        self._patch_io(monkeypatch)
        adapter.submit(self._data_config(tmp_path))

        create_calls = adapter._drive.files.return_value.create.call_args_list
        folder_creates = [
            c for c in create_calls
            if "application/vnd.google-apps.folder" in str(c)
        ]
        assert len(folder_creates) >= 2, "Expected ColabTraining root + experiment subfolder"

    def test_renders_notebook_replacing_both_placeholders(self, tmp_path, monkeypatch):
        adapter = self._make_adapter(tmp_path, monkeypatch)
        self._patch_io(monkeypatch)
        cfg = self._data_config(tmp_path, experiment_name="ph-test", epochs=5)
        adapter.submit(cfg)

        rendered = tmp_path / "ph-test" / "notebook.ipynb"
        assert rendered.exists()
        content = rendered.read_text()
        assert "{{config}}" not in content, "{{config}} placeholder was not replaced"
        assert "{{folder_id}}" not in content, "{{folder_id}} placeholder was not replaced"
        assert "'epochs': 5" in content

    def test_prints_colab_url_with_drive_link(self, tmp_path, monkeypatch, capsys):
        adapter = self._make_adapter(tmp_path, monkeypatch)
        self._patch_io(monkeypatch)
        adapter.submit(self._data_config(tmp_path))

        captured = capsys.readouterr()
        assert "colab.research.google.com/drive/" in captured.out

    def test_uploads_notebook_ipynb_to_drive(self, tmp_path, monkeypatch):
        adapter = self._make_adapter(tmp_path, monkeypatch)
        self._patch_io(monkeypatch)
        adapter.submit(self._data_config(tmp_path, experiment_name="nb-test"))

        create_calls = adapter._drive.files.return_value.create.call_args_list
        notebook_creates = [c for c in create_calls if "notebook.ipynb" in str(c)]
        assert notebook_creates, "notebook.ipynb should be uploaded to Drive"


# ---------------------------------------------------------------------------
# ColabTrainingAdapter — status
# ---------------------------------------------------------------------------


class TestColabAdapterStatus:
    def _make_adapter(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "adapters.colab.adapter.ColabTrainingAdapter._build_drive_client",
            lambda self: MagicMock(),
        )
        from adapters.colab.adapter import ColabTrainingAdapter

        return ColabTrainingAdapter(work_dir=tmp_path)

    def test_returns_pending_when_no_status_file_in_drive(self, tmp_path, monkeypatch):
        adapter = self._make_adapter(tmp_path, monkeypatch)
        adapter._drive.files().list().execute.return_value = {"files": []}
        assert adapter.status("folder-id") == "pending"

    def test_running_content_maps_to_running(self, tmp_path, monkeypatch):
        adapter = self._make_adapter(tmp_path, monkeypatch)
        adapter._drive.files().list().execute.return_value = {"files": [{"id": "s-id"}]}
        monkeypatch.setattr(
            "googleapiclient.http.MediaIoBaseDownload",
            _make_status_downloader(b"running"),
        )
        assert adapter.status("folder-id") == "running"

    def test_done_content_maps_to_done(self, tmp_path, monkeypatch):
        adapter = self._make_adapter(tmp_path, monkeypatch)
        adapter._drive.files().list().execute.return_value = {"files": [{"id": "s-id"}]}
        monkeypatch.setattr(
            "googleapiclient.http.MediaIoBaseDownload",
            _make_status_downloader(b"done"),
        )
        assert adapter.status("folder-id") == "done"

    def test_failed_content_maps_to_failed(self, tmp_path, monkeypatch):
        adapter = self._make_adapter(tmp_path, monkeypatch)
        adapter._drive.files().list().execute.return_value = {"files": [{"id": "s-id"}]}
        monkeypatch.setattr(
            "googleapiclient.http.MediaIoBaseDownload",
            _make_status_downloader(b"failed"),
        )
        assert adapter.status("folder-id") == "failed"

    def test_unknown_content_defaults_to_pending(self, tmp_path, monkeypatch):
        adapter = self._make_adapter(tmp_path, monkeypatch)
        adapter._drive.files().list().execute.return_value = {"files": [{"id": "s-id"}]}
        monkeypatch.setattr(
            "googleapiclient.http.MediaIoBaseDownload",
            _make_status_downloader(b"something-unexpected"),
        )
        assert adapter.status("folder-id") == "pending"


# ---------------------------------------------------------------------------
# ColabTrainingAdapter — download
# ---------------------------------------------------------------------------


class TestColabAdapterDownload:
    def _make_adapter(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "adapters.colab.adapter.ColabTrainingAdapter._build_drive_client",
            lambda self: MagicMock(),
        )
        from adapters.colab.adapter import ColabTrainingAdapter

        return ColabTrainingAdapter(work_dir=tmp_path)

    def _tar_bytes(self, tmp_path, **files: str) -> bytes:
        """Return a .tar.gz archive containing the given filename→content pairs."""
        import io as _io
        import tarfile

        buf = _io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for name, content in files.items():
                data = content.encode()
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, _io.BytesIO(data))
        return buf.getvalue()

    def test_raises_if_checkpoint_not_in_drive(self, tmp_path, monkeypatch):
        adapter = self._make_adapter(tmp_path, monkeypatch)
        adapter._drive.files().list().execute.return_value = {"files": []}
        with pytest.raises(FileNotFoundError, match="checkpoint.tar.gz"):
            adapter.download("folder-id", tmp_path / "dest")

    def test_downloads_and_extracts_archive(self, tmp_path, monkeypatch):
        archive_bytes = self._tar_bytes(tmp_path, **{"marker.txt": "hello"})

        adapter = self._make_adapter(tmp_path, monkeypatch)
        adapter._drive.files().list().execute.return_value = {"files": [{"id": "ckpt-id"}]}
        monkeypatch.setattr(
            "googleapiclient.http.MediaIoBaseDownload",
            _make_status_downloader(archive_bytes),
        )

        dest = tmp_path / "dest"
        adapter.download("folder-id", dest)

        assert (dest / "marker.txt").exists(), "Extracted file should be present"
        assert not (dest / "checkpoint.tar.gz").exists(), "Archive should be deleted after extract"

    def test_returns_dest_as_string(self, tmp_path, monkeypatch):
        archive_bytes = self._tar_bytes(tmp_path)

        adapter = self._make_adapter(tmp_path, monkeypatch)
        adapter._drive.files().list().execute.return_value = {"files": [{"id": "ckpt-id"}]}
        monkeypatch.setattr(
            "googleapiclient.http.MediaIoBaseDownload",
            _make_status_downloader(archive_bytes),
        )

        dest = tmp_path / "dest"
        result = adapter.download("folder-id", dest)
        assert result == str(dest)


# ---------------------------------------------------------------------------
# Colab notebook template integrity
# ---------------------------------------------------------------------------


class TestColabNotebookTemplate:
    @pytest.fixture
    def template(self) -> dict:
        import json
        from pathlib import Path

        path = Path(__file__).parents[2] / "src/adapters/colab/notebook_template.ipynb"
        return json.loads(path.read_text())

    def _all_source(self, template: dict) -> str:
        return " ".join(
            line
            for cell in template["cells"]
            for line in (
                cell["source"] if isinstance(cell["source"], list) else [cell["source"]]
            )
        )

    def test_template_is_valid_notebook_json(self, template):
        assert "cells" in template
        assert isinstance(template["cells"], list)
        assert len(template["cells"]) > 0

    def test_template_has_config_placeholder(self, template):
        assert "{{config}}" in self._all_source(template)

    def test_template_has_folder_id_placeholder(self, template):
        assert "{{folder_id}}" in self._all_source(template)

    def test_template_marks_done_on_success(self, template):
        assert "update_status('done')" in self._all_source(template)

    def test_template_marks_failed_on_training_error(self, template):
        assert "update_status('failed')" in self._all_source(template)
