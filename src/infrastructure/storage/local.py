"""Local filesystem implementation of StoragePort."""

from __future__ import annotations

import shutil
from pathlib import Path

from domain.ports import StoragePort


class LocalStorageAdapter(StoragePort):
    """Stores model artifacts under a single base directory on the local filesystem.

    Keys are relative paths (e.g. ``gguf/{model_id}.gguf``) resolved against
    ``base_dir``.  Swapping this for an S3 adapter requires no changes to callers.
    """

    def __init__(self, base_dir: Path = Path("data")) -> None:
        self._base = base_dir

    def _resolve(self, key: str) -> Path:
        return self._base / key

    def upload(self, local_path: Path, key: str) -> None:
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if local_path.resolve() != dest.resolve():
            shutil.copy2(local_path, dest)

    def download(self, key: str, dest: Path) -> None:
        src = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def delete(self, key: str) -> None:
        self._resolve(key).unlink(missing_ok=True)
