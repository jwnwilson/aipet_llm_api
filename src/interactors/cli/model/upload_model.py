"""CLI: gzip-compress and upload models/aipet.gguf to S3 for CI model caching.

One-time operation — run locally whenever the model changes.

Usage:
    uv run python -m interactors.cli.model.upload_model

Required environment variables:
    AWS_S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compress and upload the aipet GGUF model to S3 for CI caching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default="models/aipet.gguf",
        dest="model_path",
        help="Local path to the GGUF model file.",
    )
    parser.add_argument(
        "--s3-key",
        default="models/aipet.gguf.gz",
        dest="s3_key",
        help="S3 object key (destination path inside the bucket).",
    )
    args = parser.parse_args(argv)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Model:  {model_path}  ({size_mb:.1f} MB uncompressed)")
    print(f"S3 key: {args.s3_key}")

    try:
        from adapters.storage.s3 import S3StorageAdapter
    except ImportError as exc:
        print(f"ERROR: boto3 not installed — {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        storage = S3StorageAdapter()
    except KeyError as exc:
        print(f"ERROR: missing environment variable {exc}", file=sys.stderr)
        sys.exit(1)

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".gguf.gz")
    tmp_path = Path(tmp_name)
    try:
        os.close(tmp_fd)
        print("Compressing … (gzip level 6, ~1–2 minutes for 1 GB)")
        total = model_path.stat().st_size
        with (
            open(model_path, "rb") as f_in,
            gzip.open(tmp_path, "wb", compresslevel=6) as f_out,
            tqdm(total=total, unit="B", unit_scale=True, desc="compress") as bar,
        ):
            for chunk in iter(lambda: f_in.read(1 << 20), b""):
                f_out.write(chunk)
                bar.update(len(chunk))

        compressed_mb = tmp_path.stat().st_size / (1024 * 1024)
        ratio = tmp_path.stat().st_size / model_path.stat().st_size
        print(f"Compressed: {compressed_mb:.1f} MB  ({ratio:.0%} of original)")

        print("Uploading …")
        try:
            with tqdm(total=tmp_path.stat().st_size, unit="B", unit_scale=True, desc="upload") as bar:
                storage.upload(tmp_path, args.s3_key, callback=bar.update)
        except Exception as exc:
            print(f"ERROR: upload failed — {exc}", file=sys.stderr)
            sys.exit(1)
    finally:
        tmp_path.unlink(missing_ok=True)

    print("Upload complete.")
    if storage.exists(args.s3_key):
        print(f"Verified: s3://<bucket>/{args.s3_key} exists.")
    else:
        print("WARNING: post-upload existence check failed — verify manually.", file=sys.stderr)


if __name__ == "__main__":
    main()
