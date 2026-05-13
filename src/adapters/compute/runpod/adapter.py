"""RunPod-backed remote training adapter implementing RemoteTrainingPort."""
from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import tarfile
import uuid
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

from domain.models import RemoteTrainConfig
from domain.ports import RemoteTrainingPort

_DEFAULT_GPU = "NVIDIA GeForce RTX 3090"
_DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"

# Fetcher script: downloads bootstrap.py from S3 and exec()s it.
# Base64-encoded because RunPod's SDK embeds docker_args directly into a
# GraphQL mutation string without escaping — any " character breaks the query.
_BOOTSTRAP_FETCH_B64 = base64.b64encode(
    b"import boto3,os;"
    b"boto3.client('s3').download_file("
    b"os.environ['AWS_S3_BUCKET'],"
    b"os.environ['RUN_ID']+'/bootstrap.py',"
    b"'/tmp/aipet_bootstrap.py');"
    b"exec(open('/tmp/aipet_bootstrap.py').read())"
).decode()

# RunPod desiredStatus values → canonical states (EXITED resolved via S3 status.txt)
_POD_STATUS_MAP: dict[str, str | None] = {
    "CREATED": "pending",
    "RUNNING": "running",
    "EXITED": None,
    "FAILED": "failed",
    "TERMINATED": "failed",
}


class RunPodTrainingAdapter(RemoteTrainingPort):
    """RemoteTrainingPort implementation that runs training on a RunPod GPU pod.

    Flow:
        1. Build project wheel and upload with training data to S3 under a unique prefix.
        2. Create a RunPod pod that runs training_script.py, reading config from env vars.
        3. The pod writes status.txt and progress.json to S3 during training.
        4. Poll S3 status.txt; fall back to RunPod API (via stored pod_id.txt) to detect crashes.
        5. Download checkpoint.tar.gz from S3 when done.

    run_id is an S3 key prefix, e.g. ``runpod/my-experiment-a1b2c3``.
    """

    def __init__(self, work_dir: Path | None = None) -> None:
        self._work_dir = work_dir or Path("models/runpod_runs")
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._project_root = Path(__file__).parents[4].resolve()
        self._bucket = os.environ["AWS_S3_BUCKET"]
        self._s3 = self._build_s3_client()

    def _build_s3_client(self):
        import boto3
        return boto3.client("s3")

    def _configure_runpod(self):
        import runpod
        runpod.api_key = os.environ["RUNPOD_API_KEY"]
        return runpod

    # ------------------------------------------------------------------
    # RemoteTrainingPort
    # ------------------------------------------------------------------

    def submit(self, config: RemoteTrainConfig) -> str:
        runpod = self._configure_runpod()

        run_id = f"runpod/{config.experiment_name}-{uuid.uuid4().hex[:6]}"
        staging = self._work_dir / config.experiment_name
        self._stage_files(config, staging)
        self._upload_to_s3(staging, run_id, config)

        pod = runpod.create_pod(
            name=config.experiment_name[:63],
            image_name=os.getenv("RUNPOD_IMAGE", _DEFAULT_IMAGE),
            gpu_type_id=os.getenv("RUNPOD_GPU_TYPE_ID", _DEFAULT_GPU),
            container_disk_in_gb=50,
            docker_args=(
                f"bash -c 'pip install -q boto3 && "
                f"echo {_BOOTSTRAP_FETCH_B64} | base64 -d | python'"
            ),
            env=self._build_pod_env(run_id, config),
        )
        # Persist pod_id so status() can cross-check RunPod API for crash detection
        self._s3.put_object(
            Bucket=self._bucket,
            Key=f"{run_id}/pod_id.txt",
            Body=pod["id"].encode(),
        )
        return run_id

    def status(self, run_id: str) -> Literal["pending", "running", "done", "failed"]:
        # Primary: read status.txt written by the pod training script
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
                log.info("runpod status (s3)  run_id=%s  status=%s", run_id, raw)
                if raw in ("done", "failed"):
                    self._terminate_pod(run_id)
                return raw  # type: ignore[return-value]
        except Exception:
            pass

        # Fallback: check RunPod API via stored pod_id (detects OOM / preemption)
        try:
            runpod = self._configure_runpod()

            pod_id = (
                self._s3.get_object(Bucket=self._bucket, Key=f"{run_id}/pod_id.txt")[
                    "Body"
                ]
                .read()
                .decode()
                .strip()
            )
            pod = runpod.get_pod(pod_id)
            mapped = _POD_STATUS_MAP.get(pod.get("desiredStatus", ""), "pending")
            log.info("runpod status (api)  run_id=%s  desired=%s  mapped=%s", run_id, pod.get("desiredStatus"), mapped or "pending")
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
        # Prefer the archived copy (available even after pod is gone)
        try:
            archived = (
                self._s3.get_object(Bucket=self._bucket, Key=f"{run_id}/logs.txt")[
                    "Body"
                ]
                .read()
                .decode()
            )
            if archived:
                return archived
        except Exception:
            pass

        # Fall back to live RunPod API (pod still running)
        try:
            runpod = self._configure_runpod()
            pod_id = (
                self._s3.get_object(Bucket=self._bucket, Key=f"{run_id}/pod_id.txt")[
                    "Body"
                ]
                .read()
                .decode()
                .strip()
            )
            return runpod.get_pod_log(pod_id) or ""
        except Exception:
            return ""

    def eval(self, run_id: str, eval_data: str) -> tuple[float, bool]:  # noqa: ARG002
        # Eval ran on the training pod (training_script.py) and results
        # are already on S3 by the time train_activity completes.
        raw = (
            self._s3.get_object(Bucket=self._bucket, Key=f"{run_id}/eval_result.json")[
                "Body"
            ]
            .read()
            .decode()
        )
        data = json.loads(raw)
        valid_pct, passed = float(data["valid_pct"]), bool(data["passed"])
        log.info("runpod eval (s3)  run_id=%s  valid_pct=%.1f%%  passed=%s", run_id, valid_pct * 100, passed)
        return valid_pct, passed

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

    def _build_pod_env(self, run_id: str, config: RemoteTrainConfig) -> dict:
        env = {
            "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
            "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
            "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            "AWS_S3_BUCKET": self._bucket,
            "RUN_ID": run_id,
            "MODEL": config.model,
            "EPOCHS": str(config.epochs),
            "PATIENCE": str(config.patience),
            "WARMUP_RATIO": str(config.warmup_ratio),
        }
        if tok := os.environ.get("AWS_SESSION_TOKEN"):
            env["AWS_SESSION_TOKEN"] = tok
        return env

    def _terminate_pod(self, run_id: str) -> None:
        """Terminate the training pod for run_id (best-effort, swallows all errors)."""
        try:
            runpod = self._configure_runpod()

            pod_id = (
                self._s3.get_object(Bucket=self._bucket, Key=f"{run_id}/pod_id.txt")[
                    "Body"
                ]
                .read()
                .decode()
                .strip()
            )
            try:
                raw_logs = runpod.get_pod_log(pod_id) or ""
                if raw_logs:
                    self._s3.put_object(
                        Bucket=self._bucket,
                        Key=f"{run_id}/logs.txt",
                        Body=raw_logs.encode(),
                    )
                    log.info("runpod logs archived  run_id=%s  bytes=%d", run_id, len(raw_logs))
            except Exception as exc:
                log.warning("runpod log capture failed (best-effort)  run_id=%s  error=%s", run_id, exc)

            log.info("runpod terminating pod  run_id=%s  pod_id=%s", run_id, pod_id)
            runpod.terminate_pod(pod_id)
            log.info("runpod pod terminated  run_id=%s  pod_id=%s", run_id, pod_id)
        except Exception as exc:
            log.warning("runpod terminate failed (best-effort)  run_id=%s  error=%s", run_id, exc)

    def _stage_files(self, config: RemoteTrainConfig, staging: Path) -> None:
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(staging)],
            cwd=str(self._project_root),
            check=True,
        )

        # Copy the standalone bootstrap script (no project-wheel dependency).
        shutil.copy2(Path(__file__).parent / "bootstrap.py", staging / "bootstrap.py")

        train_data = Path(config.train_data)
        if not train_data.is_absolute():
            train_data = self._project_root / train_data
        for jsonl in train_data.parent.glob("*.jsonl"):
            shutil.copy2(jsonl, staging / jsonl.name)

    def _upload_to_s3(
        self, staging: Path, run_id: str, config: RemoteTrainConfig  # noqa: ARG002
    ) -> None:
        for path in staging.iterdir():
            if not path.is_file():
                continue
            if path.suffix == ".whl":
                key = f"{run_id}/{path.name}"
            elif path.suffix == ".jsonl":
                key = f"{run_id}/data/{path.name}"
            elif path.name == "bootstrap.py":
                key = f"{run_id}/bootstrap.py"
            else:
                continue
            self._s3.upload_file(str(path), self._bucket, key)
