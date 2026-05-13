"""Training entry point executed inside a RunPod pod.

Reads all configuration from environment variables set by RunPodTrainingAdapter.submit().
Installed as part of the project wheel so it is importable as:
    python -m adapters.compute.runpod.training_script
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from contextlib import redirect_stdout
from pathlib import Path


def _s3():
    import boto3
    return boto3.client("s3")


BUCKET = os.environ["AWS_S3_BUCKET"]
RUN_ID = os.environ["RUN_ID"]


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
    put_status("running")
    put_progress(0.0, "starting")

    # Download and install project wheel
    whl_prefix = f"{RUN_ID}/"
    paginator = s3.get_paginator("list_objects_v2")
    whl_key = None
    for page in paginator.paginate(Bucket=BUCKET, Prefix=whl_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".whl"):
                whl_key = obj["Key"]
                break
        if whl_key:
            break

    if not whl_key:
        put_status("failed")
        sys.exit("No .whl found in S3 prefix — re-run submit to rebuild.")

    whl_path = Path("/tmp") / whl_key.split("/")[-1]
    s3.download_file(BUCKET, whl_key, str(whl_path))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-q", str(whl_path)],
        check=True,
    )
    put_progress(0.1, "wheel installed")

    # Download training data
    data_dir = Path("data")
    download_prefix(s3, f"{RUN_ID}/data/", data_dir)
    put_progress(0.15, "data downloaded")

    # Run training
    cmd = [
        sys.executable, "-m", "cli.train",
        "--model", os.environ["MODEL"],
        "--epochs", os.environ["EPOCHS"],
        "--patience", os.environ["PATIENCE"],
        "--warmup-ratio", os.environ["WARMUP_RATIO"],
        "--train-data", "data/train.jsonl",
        "--eval-data", "data/eval.jsonl",
        "--output-dir", "models/checkpoints",
    ]
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

    s3.upload_file(str(archive), BUCKET, f"{RUN_ID}/checkpoint.tar.gz")
    put_progress(0.95, "evaluating checkpoint")

    # Run HF eval in-process so results land on S3 before the instance exits.
    # On failure we still mark training done — the checkpoint is usable.
    try:
        from domain.train.evaluate import PASS_THRESHOLD, evaluate, infer_hf, load_hf_pipeline

        pipe = load_hf_pipeline(str(checkpoint_dir))
        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = evaluate(Path("data/eval.jsonl"), lambda p: infer_hf(pipe, p))

        output = buf.getvalue()
        valid_pct = _parse_valid_pct(output)
        if valid_pct is None:
            valid_pct = 1.0 if exit_code == 0 else 0.0
        passed = valid_pct >= PASS_THRESHOLD
    except Exception as exc:
        print(f"[eval] failed — storing 0%: {exc}", file=sys.stderr)
        valid_pct, passed = 0.0, False

    s3.put_object(
        Bucket=BUCKET,
        Key=f"{RUN_ID}/eval_result.json",
        Body=json.dumps({"valid_pct": valid_pct, "passed": passed}).encode(),
    )
    put_progress(1.0, "done")
    put_status("done")


def _parse_valid_pct(output: str) -> float | None:
    for line in output.splitlines():
        if line.startswith("Valid:") and "(" in line and "%)" in line:
            try:
                return float(line.split("(")[1].split("%")[0].strip()) / 100.0
            except (IndexError, ValueError):
                pass
    return None


if __name__ == "__main__":
    main()
