"""Standalone bootstrap executed by VastAI onstart_cmd.

Only depends on boto3 + stdlib — the project wheel is not yet installed
when this runs.  Downloads and installs the wheel from S3, then delegates
to the training script (which lives inside the wheel).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BUCKET = os.environ["AWS_S3_BUCKET"]
RUN_ID = os.environ["RUN_ID"]


def _s3():
    import boto3
    return boto3.client("s3")


def main() -> None:
    s3 = _s3()
    print(f"[bootstrap] run_id={RUN_ID}  bucket={BUCKET}", flush=True)
    s3.put_object(Bucket=BUCKET, Key=f"{RUN_ID}/status.txt", Body=b"pending")

    # Find the project wheel in S3
    pag = s3.get_paginator("list_objects_v2")
    whl_key = next(
        (
            obj["Key"]
            for page in pag.paginate(Bucket=BUCKET, Prefix=f"{RUN_ID}/")
            for obj in page.get("Contents", [])
            if obj["Key"].endswith(".whl")
        ),
        None,
    )
    if not whl_key:
        s3.put_object(Bucket=BUCKET, Key=f"{RUN_ID}/status.txt", Body=b"failed")
        sys.exit("ERROR: no .whl found in S3 — re-submit to rebuild.")

    whl = Path("/tmp") / whl_key.split("/")[-1]
    print(f"[bootstrap] downloading wheel  key={whl_key}", flush=True)
    s3.download_file(BUCKET, whl_key, str(whl))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", str(whl)],
        check=True,
    )

    # Belt-and-suspenders: conda images sometimes confuse pip's resolver and
    # skip installing heavy deps that are "already present" at a lower version.
    # Explicitly install training packages so they are always present.
    print("[bootstrap] ensuring training dependencies are installed", flush=True)
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "transformers", "datasets", "accelerate", "peft",
            "bitsandbytes", "sentencepiece",
        ],
        check=True,
    )

    # Verify key import is resolvable before handing off to training script.
    import importlib.util
    if importlib.util.find_spec("transformers") is None:
        s3.put_object(Bucket=BUCKET, Key=f"{RUN_ID}/status.txt", Body=b"failed")
        sys.exit("ERROR: transformers still not importable after explicit install — aborting.")
    print("[bootstrap] transformers OK — starting training script", flush=True)

    # Wheel is now installed; delegate to the training script in the same process.
    # runpy sets __name__ = "__main__" so the if-block at the bottom fires.
    import runpy
    runpy.run_module("adapters.compute.vastai.training_script", run_name="__main__")


if __name__ == "__main__":
    main()
