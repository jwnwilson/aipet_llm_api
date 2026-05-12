"""Unit tests for StoragePort contract and LocalStorageAdapter implementation.

Design:
- Contract tests verify the abstract port is correctly enforced.
- Implementation tests use a real tmp_path filesystem — no mocking of I/O.
  This ensures LocalStorageAdapter actually moves bytes rather than just
  calling methods on a fake.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from domain.ports import StoragePort
from adapters.storage.local import LocalStorageAdapter


# ---------------------------------------------------------------------------
# StoragePort contract tests
# ---------------------------------------------------------------------------


class TestStoragePortContract:
    """Verify that the abstract port correctly requires all four methods."""

    def test_cannot_instantiate_storage_port_directly(self):
        with pytest.raises(TypeError):
            StoragePort()  # type: ignore[abstract]

    def test_missing_upload_raises_type_error(self):
        class _Partial(StoragePort):
            def download(self, key, dest): pass
            def exists(self, key): return False
            def delete(self, key): pass

        with pytest.raises(TypeError):
            _Partial()  # type: ignore[abstract]

    def test_missing_download_raises_type_error(self):
        class _Partial(StoragePort):
            def upload(self, local_path, key): pass
            def exists(self, key): return False
            def delete(self, key): pass

        with pytest.raises(TypeError):
            _Partial()  # type: ignore[abstract]

    def test_missing_exists_raises_type_error(self):
        class _Partial(StoragePort):
            def upload(self, local_path, key): pass
            def download(self, key, dest): pass
            def delete(self, key): pass

        with pytest.raises(TypeError):
            _Partial()  # type: ignore[abstract]

    def test_missing_delete_raises_type_error(self):
        class _Partial(StoragePort):
            def upload(self, local_path, key): pass
            def download(self, key, dest): pass
            def exists(self, key): return False

        with pytest.raises(TypeError):
            _Partial()  # type: ignore[abstract]

    def test_full_implementation_instantiates(self, tmp_path):
        adapter = LocalStorageAdapter(base_dir=tmp_path)
        assert isinstance(adapter, StoragePort)


# ---------------------------------------------------------------------------
# LocalStorageAdapter — upload
# ---------------------------------------------------------------------------


class TestLocalStorageAdapterUpload:
    def test_upload_copies_file_to_storage(self, tmp_path: Path) -> None:
        """upload() must place the file at base_dir / key."""
        base = tmp_path / "store"
        src = tmp_path / "model.gguf"
        src.write_bytes(b"weights")

        adapter = LocalStorageAdapter(base_dir=base)
        adapter.upload(src, "gguf/model.gguf")

        dest = base / "gguf" / "model.gguf"
        assert dest.exists()
        assert dest.read_bytes() == b"weights"

    def test_upload_creates_intermediate_directories(self, tmp_path: Path) -> None:
        """Parent directories of the key must be created automatically."""
        base = tmp_path / "store"
        src = tmp_path / "x.gguf"
        src.write_bytes(b"x")

        adapter = LocalStorageAdapter(base_dir=base)
        adapter.upload(src, "deep/nested/path/x.gguf")

        assert (base / "deep" / "nested" / "path" / "x.gguf").exists()

    def test_upload_same_source_and_dest_is_noop(self, tmp_path: Path) -> None:
        """When source and destination resolve to the same path, no error occurs."""
        base = tmp_path
        src = base / "model.gguf"
        src.write_bytes(b"noop")

        adapter = LocalStorageAdapter(base_dir=base)
        # Key resolves to base / "model.gguf" which is the same as src
        adapter.upload(src, "model.gguf")

        assert src.read_bytes() == b"noop"


# ---------------------------------------------------------------------------
# LocalStorageAdapter — download
# ---------------------------------------------------------------------------


class TestLocalStorageAdapterDownload:
    def test_download_copies_file_to_dest(self, tmp_path: Path) -> None:
        """download() must copy the keyed artifact to the given dest path."""
        base = tmp_path / "store"
        (base / "gguf").mkdir(parents=True)
        stored = base / "gguf" / "model.gguf"
        stored.write_bytes(b"stored-weights")

        dest = tmp_path / "local" / "model.gguf"
        adapter = LocalStorageAdapter(base_dir=base)
        adapter.download("gguf/model.gguf", dest)

        assert dest.exists()
        assert dest.read_bytes() == b"stored-weights"

    def test_download_creates_parent_dirs(self, tmp_path: Path) -> None:
        base = tmp_path / "store"
        (base / "gguf").mkdir(parents=True)
        (base / "gguf" / "x.gguf").write_bytes(b"x")

        dest = tmp_path / "deep" / "dir" / "x.gguf"
        adapter = LocalStorageAdapter(base_dir=base)
        adapter.download("gguf/x.gguf", dest)

        assert dest.parent.exists()
        assert dest.exists()

    def test_download_same_source_and_dest_is_noop(self, tmp_path: Path) -> None:
        """When source and destination resolve to the same path, no error occurs."""
        base = tmp_path
        src = base / "model.gguf"
        src.write_bytes(b"same")

        adapter = LocalStorageAdapter(base_dir=base)
        adapter.download("model.gguf", src)

        assert src.read_bytes() == b"same"


# ---------------------------------------------------------------------------
# LocalStorageAdapter — exists
# ---------------------------------------------------------------------------


class TestLocalStorageAdapterExists:
    def test_returns_true_for_existing_key(self, tmp_path: Path) -> None:
        base = tmp_path / "store"
        (base / "gguf").mkdir(parents=True)
        (base / "gguf" / "model.gguf").write_bytes(b"exists")

        adapter = LocalStorageAdapter(base_dir=base)
        assert adapter.exists("gguf/model.gguf") is True

    def test_returns_false_for_missing_key(self, tmp_path: Path) -> None:
        base = tmp_path / "store"
        base.mkdir()

        adapter = LocalStorageAdapter(base_dir=base)
        assert adapter.exists("gguf/nonexistent.gguf") is False


# ---------------------------------------------------------------------------
# LocalStorageAdapter — delete
# ---------------------------------------------------------------------------


class TestLocalStorageAdapterDelete:
    def test_delete_removes_file(self, tmp_path: Path) -> None:
        base = tmp_path / "store"
        (base / "gguf").mkdir(parents=True)
        target = base / "gguf" / "model.gguf"
        target.write_bytes(b"to-delete")

        adapter = LocalStorageAdapter(base_dir=base)
        adapter.delete("gguf/model.gguf")

        assert not target.exists()

    def test_delete_is_silent_when_key_absent(self, tmp_path: Path) -> None:
        """Deleting a non-existent key must not raise."""
        base = tmp_path / "store"
        base.mkdir()
        adapter = LocalStorageAdapter(base_dir=base)
        adapter.delete("gguf/nonexistent.gguf")  # should not raise

    def test_exists_returns_false_after_delete(self, tmp_path: Path) -> None:
        base = tmp_path / "store"
        (base / "gguf").mkdir(parents=True)
        (base / "gguf" / "model.gguf").write_bytes(b"bye")

        adapter = LocalStorageAdapter(base_dir=base)
        adapter.delete("gguf/model.gguf")

        assert adapter.exists("gguf/model.gguf") is False
