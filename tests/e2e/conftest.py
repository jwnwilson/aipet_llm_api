"""Fixtures shared across all E2E tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_LLAMA_CPP_DIR = Path(__file__).parents[2] / "llama.cpp"


@pytest.fixture(scope="session")
def llama_cpp_ready() -> Path:
    """Return the llama.cpp directory, or skip if it isn't built."""
    convert = _LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    quantize = _LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
    if not convert.exists() or not quantize.exists():
        pytest.skip(
            "llama.cpp not built — run 'make setup-llama' first. "
            f"Expected: {convert} and {quantize}"
        )
    return _LLAMA_CPP_DIR


@pytest.fixture(scope="session")
def kaggle_credentials() -> None:
    """Skip if Kaggle credentials are missing. Accepts KAGGLE_KEY or KAGGLE_API_TOKEN."""
    has_username = bool(os.environ.get("KAGGLE_USERNAME"))
    has_key = bool(os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN"))
    if not has_username or not has_key:
        pytest.skip(
            "Kaggle credentials not set — export KAGGLE_USERNAME and "
            "KAGGLE_KEY (or KAGGLE_API_TOKEN)"
        )


@pytest.fixture(scope="session")
def runpod_credentials() -> None:
    """Skip if RunPod or AWS credentials are missing."""
    required = ["RUNPOD_API_KEY", "AWS_S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        pytest.skip(f"RunPod credentials not set — missing env vars: {', '.join(missing)}")


@pytest.fixture(scope="session")
def vastai_credentials() -> None:
    """Skip if Vast.ai or AWS credentials are missing."""
    required = ["VAST_API_KEY", "AWS_S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        pytest.skip(f"Vast.ai credentials not set — missing env vars: {', '.join(missing)}")
