"""Evaluation entry point executed inside a Vast.ai instance.

Reads config from environment variables, evaluates the trained checkpoint
against the eval dataset already staged on S3 by VastAiTrainingAdapter.submit(),
and writes results back to S3.

Environment variables (set by VastAiTrainingAdapter.eval()):
  AWS_S3_BUCKET   S3 bucket
  RUN_ID          S3 key prefix for this run (e.g. "vastai/my-exp-a1b2c3")

Run as: python -m adapters.compute.vastai.evaluation_script
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

BUCKET = os.environ["AWS_S3_BUCKET"]
RUN_ID = os.environ["RUN_ID"]


def _s3():
    import boto3
    return boto3.client("s3")


def _put_eval_status(s3_client, status: str) -> None:
    s3_client.put_object(Bucket=BUCKET, Key=f"{RUN_ID}/eval_status.txt", Body=status.encode())


def _download_prefix(s3_client, prefix: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = key[len(prefix):]
            if fname:
                s3_client.download_file(BUCKET, key, str(dest / fname))


def _parse_valid_pct(output: str) -> float | None:
    for line in output.splitlines():
        if line.startswith("Valid:") and "(" in line and "%)" in line:
            try:
                return float(line.split("(")[1].split("%")[0].strip()) / 100.0
            except (IndexError, ValueError):
                pass
    return None


def main() -> None:
    s3 = _s3()
    _put_eval_status(s3, "running")

    # Install project wheel
    paginator = s3.get_paginator("list_objects_v2")
    whl_key = None
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{RUN_ID}/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".whl"):
                whl_key = obj["Key"]
                break
        if whl_key:
            break

    if not whl_key:
        _put_eval_status(s3, "failed")
        sys.exit("No .whl found in S3 prefix")

    whl_path = Path("/tmp") / whl_key.split("/")[-1]
    s3.download_file(BUCKET, whl_key, str(whl_path))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-q", str(whl_path)],
        check=True,
    )

    # Download checkpoint (uploaded as checkpoint.tar.gz by the training script)
    checkpoint_archive = Path("/tmp/checkpoint.tar.gz")
    try:
        s3.download_file(BUCKET, f"{RUN_ID}/checkpoint.tar.gz", str(checkpoint_archive))
    except Exception as exc:
        _put_eval_status(s3, "failed")
        sys.exit(f"Failed to download checkpoint: {exc}")

    checkpoint_dir = Path("models/checkpoints")
    with tarfile.open(checkpoint_archive) as tf:
        tf.extractall(Path("models"), filter="data")
    checkpoint_archive.unlink()

    # Download eval data (uploaded under {RUN_ID}/data/ by VastAiTrainingAdapter.submit())
    data_dir = Path("data")
    _download_prefix(s3, f"{RUN_ID}/data/", data_dir)

    eval_data = data_dir / "eval.jsonl"
    if not eval_data.exists():
        _put_eval_status(s3, "failed")
        sys.exit(f"eval.jsonl not found at {eval_data}")

    # Run HF evaluation in-process
    try:
        from domain.train.evaluate import PASS_THRESHOLD, evaluate, infer_hf, load_hf_pipeline

        pipe = load_hf_pipeline(str(checkpoint_dir))
        infer_fn = lambda prompt: infer_hf(pipe, prompt)  # noqa: E731

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = evaluate(eval_data, infer_fn)

        output = buf.getvalue()
        print(output, end="")

        valid_pct = _parse_valid_pct(output)
        if valid_pct is None:
            valid_pct = 1.0 if exit_code == 0 else 0.0
        passed = valid_pct >= PASS_THRESHOLD

    except Exception as exc:
        _put_eval_status(s3, "failed")
        sys.exit(f"Evaluation failed: {exc}")

    s3.put_object(
        Bucket=BUCKET,
        Key=f"{RUN_ID}/eval_result.json",
        Body=json.dumps({"valid_pct": valid_pct, "passed": passed}).encode(),
    )
    _put_eval_status(s3, "done")


if __name__ == "__main__":
    main()
