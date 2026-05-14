"""CLI: download and decompress a gzip-compressed GGUF model from S3.

Used by CI to fetch the test model when the GitHub Actions cache misses.
Also runnable locally to simulate a cache miss.

Usage:
    uv run python -m interactors.cli.download_model

Required environment variables:
    AWS_S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download and decompress a gzip-compressed GGUF model from S3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--s3-key",
        default="models/test_aipet.gguf.gz",
        dest="s3_key",
        help="S3 object key to download.",
    )
    parser.add_argument(
        "--dest",
        default="models/test_aipet.gguf",
        dest="dest",
        help="Local path to write the decompressed model to.",
    )
    args = parser.parse_args(argv)

    dest = Path(args.dest)

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

    gz_path = dest.parent / (dest.name + ".gz")
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading s3://<bucket>/{args.s3_key} → {gz_path}")
    try:
        storage.download(args.s3_key, gz_path)
    except Exception as exc:
        print(f"ERROR: download failed — {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Decompressing → {dest}")
    try:
        with gzip.open(gz_path, "rb") as f_in, open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    except Exception as exc:
        print(f"ERROR: decompression failed — {exc}", file=sys.stderr)
        gz_path.unlink(missing_ok=True)
        sys.exit(1)
    finally:
        gz_path.unlink(missing_ok=True)

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"Done. {dest} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
