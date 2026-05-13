"""AWS S3 implementation of StoragePort."""
from __future__ import annotations

import os
from pathlib import Path

from domain.ports import StoragePort


class S3StorageAdapter(StoragePort):
    """Stores model artifacts in an AWS S3 bucket.

    Keys are relative object names (e.g. ``workflow/{run_id}/model.gguf``).
    Auth is provided by the standard boto3 credential chain — set
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION.
    """

    def __init__(self, bucket: str | None = None) -> None:
        import boto3
        self._bucket = bucket or os.environ["AWS_S3_BUCKET"]
        self._s3 = boto3.client("s3")

    def upload(self, local_path: Path, key: str) -> None:
        self._s3.upload_file(str(local_path), self._bucket, key)

    def download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(self._bucket, key, str(dest))

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> None:
        try:
            self._s3.delete_object(Bucket=self._bucket, Key=key)
        except Exception:
            pass

    def write_bytes(self, key: str, content: bytes) -> None:
        """Write raw bytes to ``key`` in S3 (creates or overwrites)."""
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=content)

    def read_text(self, key: str, *, encoding: str = "utf-8") -> str:
        """Read ``key`` from S3 and decode; returns empty string if key is absent."""
        try:
            return self._s3.get_object(Bucket=self._bucket, Key=key)["Body"].read().decode(encoding)
        except Exception:
            return ""
