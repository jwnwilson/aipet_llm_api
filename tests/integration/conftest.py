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
    """Skip the test if Kaggle credentials are not configured in the environment."""
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        pytest.skip(
            "Kaggle credentials not set — export KAGGLE_USERNAME and KAGGLE_KEY to run this test"
        )
