"""Vast.ai-backed remote training adapter implementing RemoteTrainingPort."""
from __future__ import annotations

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

_DEFAULT_GPU_QUERY = "num_gpus=1 gpu_name=RTX_3090 reliability>0.99"
_DEFAULT_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"

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
        log.info("vastai submit  run_id=%s  model=%s  epochs=%s", run_id, config.model, config.epochs)

        staging = self._work_dir / config.experiment_name
        log.info("staging files to %s", staging)
        self._stage_files(config, staging)

        log.info("uploading staged files to s3  bucket=%s  prefix=%s", self._bucket, run_id)
        self._upload_to_s3(staging, run_id, config)

        client = self._build_vastai_client()
        result = self._create_instance(
            client,
            onstart_cmd=(
                "pip install -q boto3 && "
                # Download the bootstrap script from S3 (single-quotes inside so
                # the outer double-quote wrapping by VastAI doesn't conflict).
                'python -c "'
                "import boto3,os;"
                "boto3.client('s3').download_file("
                "os.environ['AWS_S3_BUCKET'],"
                "os.environ['RUN_ID']+'/bootstrap.py',"
                "'/tmp/aipet_bootstrap.py')"
                '" && '
                "python /tmp/aipet_bootstrap.py"
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
        log.info("vastai instance created  run_id=%s  instance_id=%s", run_id, instance_id)
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
                log.info("vastai status (s3)  run_id=%s  status=%s", run_id, raw)
                if raw in ("done", "failed"):
                    self._destroy_instance(run_id)
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
            log.info("vastai status (api)  run_id=%s  actual=%s  mapped=%s", run_id, actual, mapped or "pending")
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

            actual_status = "unknown"
            try:
                instance = client.show_instance(id=instance_id)
                actual_status = instance.get("actual_status", "unknown")
            except Exception:
                pass

            header = f"[vastai] instance_id={instance_id}  actual_status={actual_status}"

            result = client.logs(instance_id=instance_id, tail="200")
            raw = str(result) if result else ""
            # Filter VastAI SSH relay noise — port collisions on their shared relay
            # (ssh*.vast.ai) don't affect training because we communicate via S3.
            lines = [
                ln for ln in raw.splitlines()
                if "remote port forwarding failed" not in ln
                and "Permanently added" not in ln
            ]
            body = "\n".join(lines)
            return f"{header}\n{body}" if body else header
        except Exception:
            return ""

    def eval(self, run_id: str, eval_data: str) -> tuple[float, bool]:
        # Eval ran on the training instance (training_script.py) and results
        # are already on S3 by the time train_activity completes.
        raw = (
            self._s3.get_object(Bucket=self._bucket, Key=f"{run_id}/eval_result.json")[
                "Body"
            ]
            .read()
            .decode()
        )
        data = json.loads(raw)
        return float(data["valid_pct"]), bool(data["passed"])

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

    def _destroy_instance(self, run_id: str) -> None:
        """Destroy the training instance for run_id (best-effort, swallows all errors)."""
        try:
            instance_id = int(
                self._s3.get_object(
                    Bucket=self._bucket, Key=f"{run_id}/instance_id.txt"
                )["Body"]
                .read()
                .decode()
                .strip()
            )
            log.info("vastai destroying instance  run_id=%s  instance_id=%s", run_id, instance_id)
            self._build_vastai_client().destroy_instance(id=instance_id)
            log.info("vastai instance destroyed  run_id=%s  instance_id=%s", run_id, instance_id)
        except Exception as exc:
            log.warning("vastai destroy failed (best-effort)  run_id=%s  error=%s", run_id, exc)

    def _create_instance(self, client, onstart_cmd: str, env: dict, max_retries: int = 3) -> dict:
        """Search for the cheapest matching offer and create an instance.

        Retries up to max_retries times if a 400 is returned, which typically
        means the selected offer was taken between search and create.
        """
        import requests

        query = os.getenv("VASTAI_GPU_QUERY", _DEFAULT_GPU_QUERY)
        image = os.getenv("VASTAI_IMAGE", _DEFAULT_IMAGE)
        disk = float(os.getenv("VASTAI_DISK_GB", "50"))
        log.info("vastai searching offers  query=%r  image=%s  disk_gb=%s", query, image, disk)

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            offers = client.search_offers(query=query, type="on-demand", limit=20)
            if not offers:
                raise RuntimeError(f"No Vast.ai offers found for query: {query!r}")
            offer = min(offers, key=lambda o: float(o.get("dph_total", float("inf"))))
            log.info(
                "vastai selected offer  attempt=%d  offer_id=%s  gpu=%s  dph=$%.4f",
                attempt, offer.get("id"), offer.get("gpu_name"), float(offer.get("dph_total", 0)),
            )
            try:
                result = client.create_instance(
                    id=int(offer["id"]),
                    image=image,
                    disk=disk,
                    onstart_cmd=onstart_cmd,
                    env=env,
                )
                log.info("vastai create_instance succeeded  offer_id=%s", offer.get("id"))
                return result
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 400:
                    body = exc.response.text or ""
                    # Billing / account errors won't be fixed by retrying a different offer.
                    if any(kw in body.lower() for kw in ("credit", "balance", "payment", "billing", "insufficient")):
                        raise RuntimeError(
                            f"Vast.ai rejected the request — likely insufficient credits. "
                            f"Add credits at https://vast.ai/console/billing/ and retry. "
                            f"API response: {body}"
                        ) from exc
                    log.warning("vastai 400 on offer %s (stale?), retrying  body=%s", offer.get("id"), body[:200])
                    last_exc = exc
                    continue
                raise
        raise RuntimeError(
            f"Failed to create Vast.ai instance after {max_retries} attempts "
            f"(offer kept disappearing): {last_exc}"
        )

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
        self, staging: Path, run_id: str, config: RemoteTrainConfig
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
