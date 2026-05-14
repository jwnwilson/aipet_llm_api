"""Integration tests for the train CLI — runs real HuggingFace training."""
from __future__ import annotations

from pathlib import Path

import pytest

from domain.train.dataset import generate


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    generate(data_dir=tmp_path, train_size=20, eval_size=10, seed=0)
    return tmp_path


class TestTrainCli:
    @pytest.mark.slow
    def test_dry_run_exits_0(self, data_dir: Path, tmp_path: Path) -> None:
        from interactors.cli.train import main
        output_dir = tmp_path / "checkpoints"
        main([
            "--dry-run",
            "--train-data", str(data_dir / "train.jsonl"),
            "--eval-data", str(data_dir / "eval.jsonl"),
            "--output-dir", str(output_dir),
        ])
        assert output_dir.exists()
