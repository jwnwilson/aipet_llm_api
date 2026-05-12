"""Integration test fixtures shared across all integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_LLAMA_CPP_DIR = Path(__file__).parents[2] / "llama.cpp"


@pytest.fixture(scope="session")
def llama_cpp_ready() -> Path:
    """Return the llama.cpp directory, or skip the test if it isn't built."""
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
    """Skip the test if Kaggle credentials are not configured in the environment.

    Accepts either the legacy KAGGLE_KEY or the newer KAGGLE_API_TOKEN format.
    """
    has_username = bool(os.environ.get("KAGGLE_USERNAME"))
    has_key = bool(os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN"))
    if not has_username or not has_key:
        pytest.skip(
            "Kaggle credentials not set — export KAGGLE_USERNAME and "
            "KAGGLE_KEY (or KAGGLE_API_TOKEN) to run this test"
        )
