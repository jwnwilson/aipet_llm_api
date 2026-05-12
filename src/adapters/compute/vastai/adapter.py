"""Vast.ai-backed remote training adapter implementing RemoteTrainingPort."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import uuid
from pathlib import Path
from typing import Literal

from domain.models import RemoteTrainConfig
from domain.ports import RemoteTrainingPort

_DEFAULT_GPU_QUERY = "num_gpus=1 gpu_name=RTX_3090 reliability>0.99"
_DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel"

# Vast.ai actual_status values → canonical states
_INSTANCE_STATUS_MAP: dict[str, str | None] = {
    "created": "pending",
    "loading": "pending",
    "running": "running",
    "exited": None,   # resolved via S3 status.txt
    "stopped": None,  # resolved via S3 status.txt
}


class VastAiTrainingAdapter(RemoteTrainingPort):
    """RemoteTrainingPort implementation that runs training on a Vast.ai GPU instance.

    Flow:
        1. Build project wheel and upload with training data to S3 under a unique prefix.
        2. Search for the cheapest available Vast.ai offer matching VASTAI_GPU_QUERY.
        3. Create an instance that runs training_script.py, reading config from env vars.
        4. The instance writes status.txt and progress.json to S3 during training.
        5. Poll S3 status.txt; fall back to Vast.ai API (via stored instance_id.txt) to detect crashes.
        6. Download checkpoint.tar.gz from S3 when done.

    run_id is an S3 key prefix, e.g. ``vastai/my-experiment-a1b2c3``.
    """

    def __init__(self, work_dir: Path | None = None) -> None:
        self._work_dir = work_dir or Path("models/vastai_runs")
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._project_root = Path(__file__).parents[4].resolve()
        self._bucket = os.environ["AWS_S3_BUCKET"]
        self._s3 = self._build_s3_client()

    def _build_s3_client(self):
        import boto3
        return boto3.client("s3")

    def _build_vastai_client(self):
        from vastai import VastAI
        return VastAI(api_key=os.environ["VAST_API_KEY"])

    # ------------------------------------------------------------------
    # RemoteTrainingPort
    # ------------------------------------------------------------------

    def submit(self, config: RemoteTrainConfig) -> str:
        run_id = f"vastai/{config.experiment_name}-{uuid.uuid4().hex[:6]}"
        staging = self._work_dir / config.experiment_name
        self._stage_files(config, staging)
        self._upload_to_s3(staging, run_id, config)

        client = self._build_vastai_client()

        query = os.getenv("VASTAI_GPU_QUERY", _DEFAULT_GPU_QUERY)
        offers = client.search_offers(query=query, type="on-demand", limit=20)
        if not offers:
            raise RuntimeError(f"No Vast.ai offers found for query: {query!r}")
        offer = min(offers, key=lambda o: float(o.get("dph_total", float("inf"))))

        result = client.create_instance(
            id=int(offer["id"]),
            image=os.getenv("VASTAI_IMAGE", _DEFAULT_IMAGE),
            disk=float(os.getenv("VASTAI_DISK_GB", "50")),
            onstart_cmd=(
                "bash -c 'pip install -q boto3 && "
                "python -m adapters.compute.vastai.training_script'"
            ),
            env={
                "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
                "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
                "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
                "AWS_S3_BUCKET": self._bucket,
                "RUN_ID": run_id,
                "MODEL": config.model,
                "EPOCHS": str(config.epochs),
                "PATIENCE": str(config.patience),
                "WARMUP_RATIO": str(config.warmup_ratio),
            },
        )
        instance_id = str(result.get("new_contract", result.get("id", "")))
        self._s3.put_object(
            Bucket=self._bucket,
            Key=f"{run_id}/instance_id.txt",
            Body=instance_id.encode(),
        )
        return run_id

    def status(self, run_id: str) -> Literal["pending", "running", "done", "failed"]:
        # Primary: read status.txt written by the instance training script
        try:
            raw = (
                self._s3.get_object(Bucket=self._bucket, Key=f"{run_id}/status.txt")[
                    "Body"
                ]
                .read()
                .decode()
                .strip()
            )
            if raw in ("pending", "running", "done", "failed"):
                return raw  # type: ignore[return-value]
        except Exception:
            pass

        # Fallback: check Vast.ai API via stored instance_id (detects OOM / eviction)
        try:
            instance_id = int(
                self._s3.get_object(
                    Bucket=self._bucket, Key=f"{run_id}/instance_id.txt"
                )["Body"]
                .read()
                .decode()
                .strip()
            )
            client = self._build_vastai_client()
            instance = client.show_instance(id=instance_id)
            actual = instance.get("actual_status", "")
            mapped = _INSTANCE_STATUS_MAP.get(actual, "pending")
            return (mapped or "pending")  # type: ignore[return-value]
        except Exception:
            return "pending"

    def download(self, run_id: str, dest: Path) -> str:
        dest.mkdir(parents=True, exist_ok=True)
        archive = dest / "checkpoint.tar.gz"
        self._s3.download_file(
            self._bucket, f"{run_id}/checkpoint.tar.gz", str(archive)
        )
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")
        archive.unlink()
        return str(dest)

    def logs(self, run_id: str) -> str:
        try:
            instance_id = int(
                self._s3.get_object(
                    Bucket=self._bucket, Key=f"{run_id}/instance_id.txt"
                )["Body"]
                .read()
                .decode()
                .strip()
            )
            client = self._build_vastai_client()
            result = client.logs(instance_id=instance_id, tail="200")
            return str(result) if result else ""
        except Exception:
            return ""

    def progress(self, run_id: str) -> tuple[float, str]:
        try:
            raw = (
                self._s3.get_object(
                    Bucket=self._bucket, Key=f"{run_id}/progress.json"
                )["Body"]
                .read()
                .decode()
            )
            data = json.loads(raw)
            return float(data.get("fraction", 0.0)), str(data.get("detail", ""))
        except Exception:
            return 0.0, ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stage_files(self, config: RemoteTrainConfig, staging: Path) -> None:
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(staging)],
            cwd=str(self._project_root),
            check=True,
        )

        train_data = Path(config.train_data)
        if not train_data.is_absolute():
            train_data = self._project_root / train_data
        for jsonl in train_data.parent.glob("*.jsonl"):
            shutil.copy2(jsonl, staging / jsonl.name)

    def _upload_to_s3(
        self, staging: Path, run_id: str, config: RemoteTrainConfig
    ) -> None:
        for path in staging.iterdir():
            if not path.is_file():
                continue
            if path.suffix == ".whl":
                key = f"{run_id}/{path.name}"
            elif path.suffix == ".jsonl":
                key = f"{run_id}/data/{path.name}"
            else:
                continue
            self._s3.upload_file(str(path), self._bucket, key)
