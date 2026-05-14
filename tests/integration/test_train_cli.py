"""Integration tests for the train CLI — runs real HuggingFace training."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from domain.train.dataset import generate

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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

    @pytest.mark.slow
    def test_train_dry_run_cli(self, tmp_path: Path) -> None:
        """Subprocess smoke-test: same invocation as make train DRY_RUN=1."""
        import subprocess
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
        subprocess.run(
            [sys.executable, "src/interactors/cli/generate_dataset.py",
             "--data-dir", str(tmp_path), "--train-size", "20", "--eval-size", "5"],
            check=True, capture_output=True, cwd=PROJECT_ROOT, env=env,
        )
        result = subprocess.run(
            [sys.executable, "src/interactors/cli/train.py",
             "--dry-run",
             "--train-data", str(tmp_path / "train.jsonl"),
             "--eval-data", str(tmp_path / "eval.jsonl"),
             "--output-dir", str(tmp_path / "checkpoints")],
            capture_output=True, cwd=PROJECT_ROOT, env=env,
        )
        assert result.returncode == 0, result.stderr.decode()
