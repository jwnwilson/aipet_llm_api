"""Training entry point executed inside a RunPod pod.

Reads all configuration from environment variables set by RunPodTrainingAdapter.submit().
Installed as part of the project wheel so it is importable as:
    python -m adapters.compute.runpod.training_script
"""
from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import tarfile
from pathlib import Path

# Module-level log buffer captures all Python log records emitted by this process.
_log_buffer: io.StringIO = io.StringIO()


class _BufferHandler(logging.Handler):
    """Appends formatted records to _log_buffer so they land in the S3 log file."""

    def emit(self, record: logging.LogRecord) -> None:
        _log_buffer.write(self.format(record) + "\n")


_fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s  %(message)s")
_buf_handler = _BufferHandler()
_buf_handler.setFormatter(_fmt)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s  %(message)s")
logging.getLogger().addHandler(_buf_handler)

log = logging.getLogger(__name__)

BUCKET = os.environ["AWS_S3_BUCKET"]
RUN_ID = os.environ["RUN_ID"]


def _storage():
    from adapters.storage.s3 import S3StorageAdapter
    return S3StorageAdapter()


def _flush_logs_to_s3(storage) -> None:
    """Upload current log buffer to S3 (no-op when buffer is empty)."""
    content = _log_buffer.getvalue().encode("utf-8", errors="replace")
    if content:
        storage.write_bytes(f"{RUN_ID}/logs.txt", content)


def _run_subprocess_streaming(cmd: list[str], storage, *, log: logging.Logger) -> int:
    """Run cmd, routing stdout+stderr through logger and flushing to S3 every 20 lines."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    line_count = 0
    for line in proc.stdout:  # type: ignore[union-attr]
        log.info("[subprocess] %s", line.rstrip())
        line_count += 1
        if line_count % 20 == 0:
            _flush_logs_to_s3(storage)
    returncode = proc.wait()
    _flush_logs_to_s3(storage)
    return returncode


def download_prefix(storage, prefix: str, dest: Path) -> None:
    from adapters.storage.s3 import S3StorageAdapter
    assert isinstance(storage, S3StorageAdapter)
    dest.mkdir(parents=True, exist_ok=True)
    paginator = storage._s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = key[len(prefix):]
            if fname:
                dest_file = dest / fname
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                storage.download(key, dest_file)


def main() -> None:
    storage = _storage()
    log.info("run_id=%s  bucket=%s  starting", RUN_ID, BUCKET)
    storage.write_bytes(f"{RUN_ID}/status.txt", b"running")
    storage.write_bytes(
        f"{RUN_ID}/progress.json",
        json.dumps({"fraction": 0.0, "detail": "starting"}).encode(),
    )

    # Download training data
    data_dir = Path("data")
    log.info("downloading training data  prefix=%s/data/", RUN_ID)
    download_prefix(storage, f"{RUN_ID}/data/", data_dir)
    storage.write_bytes(
        f"{RUN_ID}/progress.json",
        json.dumps({"fraction": 0.15, "detail": "data downloaded"}).encode(),
    )
    _flush_logs_to_s3(storage)

    # Run training — streams subprocess output to logger and flushes to S3 every 20 lines
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
    returncode = _run_subprocess_streaming(cmd, storage, log=log)
    if returncode != 0:
        storage.write_bytes(f"{RUN_ID}/status.txt", b"failed")
        _flush_logs_to_s3(storage)
        sys.exit(f"Training exited with code {returncode}")

    storage.write_bytes(
        f"{RUN_ID}/progress.json",
        json.dumps({"fraction": 0.9, "detail": "training complete, uploading checkpoint"}).encode(),
    )
    _flush_logs_to_s3(storage)

    # Package and upload checkpoint
    checkpoint_dir = Path("models/checkpoints")
    if not checkpoint_dir.exists():
        storage.write_bytes(f"{RUN_ID}/status.txt", b"failed")
        _flush_logs_to_s3(storage)
        sys.exit(f"Checkpoint directory not found: {checkpoint_dir}")

    archive = Path("/tmp/checkpoint.tar.gz")
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(checkpoint_dir, arcname="checkpoints")

    log.info("uploading checkpoint  key=%s/checkpoint.tar.gz", RUN_ID)
    storage.upload(archive, f"{RUN_ID}/checkpoint.tar.gz")
    storage.write_bytes(
        f"{RUN_ID}/progress.json",
        json.dumps({"fraction": 0.95, "detail": "evaluating checkpoint"}).encode(),
    )
    _flush_logs_to_s3(storage)

    # Run HF eval in-process so results land on S3 before the instance exits.
    try:
        from domain.train.evaluate import PASS_THRESHOLD, evaluate, infer_hf, load_hf_pipeline

        pipe = load_hf_pipeline(str(checkpoint_dir))
        exit_code, valid_pct = evaluate(Path("data/eval.jsonl"), lambda p: infer_hf(pipe, p))
        passed = valid_pct >= PASS_THRESHOLD
        log.info("eval complete  valid_pct=%.1f%%  passed=%s", valid_pct * 100, passed)
    except Exception as exc:
        log.error("eval failed — storing 0%%: %s", exc, exc_info=True)
        valid_pct, passed = 0.0, False

    storage.write_bytes(
        f"{RUN_ID}/eval_result.json",
        json.dumps({"valid_pct": valid_pct, "passed": passed}).encode(),
    )
    storage.write_bytes(
        f"{RUN_ID}/progress.json",
        json.dumps({"fraction": 1.0, "detail": "done"}).encode(),
    )
    storage.write_bytes(f"{RUN_ID}/status.txt", b"done")
    log.info("run complete  run_id=%s", RUN_ID)
    _flush_logs_to_s3(storage)


if __name__ == "__main__":
    main()
