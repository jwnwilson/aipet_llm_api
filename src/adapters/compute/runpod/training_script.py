"""Training entry point executed inside a RunPod pod.

Reads all configuration from environment variables set by RunPodTrainingAdapter.submit().
Installed as part of the project wheel so it is importable as:
    python -m adapters.compute.runpod.training_script
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tarfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s  %(message)s")
log = logging.getLogger(__name__)

BUCKET = os.environ["AWS_S3_BUCKET"]
RUN_ID = os.environ["RUN_ID"]


def _s3():
    import boto3
    return boto3.client("s3")


def put_status(status: str) -> None:
    _s3().put_object(Bucket=BUCKET, Key=f"{RUN_ID}/status.txt", Body=status.encode())


def put_progress(fraction: float, detail: str) -> None:
    body = json.dumps({"fraction": fraction, "detail": detail}).encode()
    _s3().put_object(Bucket=BUCKET, Key=f"{RUN_ID}/progress.json", Body=body)


def download_prefix(s3_client, prefix: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = key[len(prefix):]
            if fname:
                s3_client.download_file(BUCKET, key, str(dest / fname))


def main() -> None:
    s3 = _s3()
    log.info("run_id=%s  bucket=%s  starting", RUN_ID, BUCKET)
    put_status("running")
    put_progress(0.0, "starting")

    # Download training data
    data_dir = Path("data")
    log.info("downloading training data  prefix=%s/data/", RUN_ID)
    download_prefix(s3, f"{RUN_ID}/data/", data_dir)
    put_progress(0.15, "data downloaded")

    # Run training
    cmd = [
        sys.executable, "-m", "interactors.cli.train",
        "--model", os.environ["MODEL"],
        "--epochs", os.environ["EPOCHS"],
        "--patience", os.environ["PATIENCE"],
        "--warmup-ratio", os.environ["WARMUP_RATIO"],
        "--train-data", "data/train.jsonl",
        "--eval-data", "data/eval.jsonl",
        "--output-dir", "models/checkpoints",
    ]
    log.info("starting training  cmd=%s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        put_status("failed")
        sys.exit(f"Training exited with code {result.returncode}")

    put_progress(0.9, "training complete, uploading checkpoint")

    # Package and upload checkpoint
    checkpoint_dir = Path("models/checkpoints")
    if not checkpoint_dir.exists():
        put_status("failed")
        sys.exit(f"Checkpoint directory not found: {checkpoint_dir}")

    archive = Path("/tmp/checkpoint.tar.gz")
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(checkpoint_dir, arcname="checkpoints")

    log.info("uploading checkpoint  key=%s/checkpoint.tar.gz", RUN_ID)
    s3.upload_file(str(archive), BUCKET, f"{RUN_ID}/checkpoint.tar.gz")
    put_progress(0.95, "evaluating checkpoint")

    # Run HF eval in-process so results land on S3 before the instance exits.
    # On failure we still mark training done — the checkpoint is usable.
    try:
        from domain.train.evaluate import PASS_THRESHOLD, evaluate, infer_hf, load_hf_pipeline

        pipe = load_hf_pipeline(str(checkpoint_dir))
        exit_code, valid_pct = evaluate(Path("data/eval.jsonl"), lambda p: infer_hf(pipe, p))
        passed = valid_pct >= PASS_THRESHOLD
        log.info("eval complete  valid_pct=%.1f%%  passed=%s", valid_pct * 100, passed)
    except Exception as exc:
        log.error("eval failed — storing 0%%: %s", exc, exc_info=True)
        valid_pct, passed = 0.0, False

    s3.put_object(
        Bucket=BUCKET,
        Key=f"{RUN_ID}/eval_result.json",
        Body=json.dumps({"valid_pct": valid_pct, "passed": passed}).encode(),
    )
    put_progress(1.0, "done")
    put_status("done")
    log.info("run complete  run_id=%s", RUN_ID)


if __name__ == "__main__":
    main()
