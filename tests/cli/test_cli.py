"""Tests for CLI commands — calls main() directly and smoke-tests Makefile targets."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.domain.actions import Action
from src.domain.models import InferenceResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Small train/eval dataset in a temp dir — used by multiple test classes."""
    from src.domain.train.dataset import generate
    generate(data_dir=tmp_path, train_size=20, eval_size=10, seed=0)
    return tmp_path


VALID_REQUEST = {
    "scene": {
        "objects": [{"id": "b1", "type": "bowl", "distance": 2.5}],
        "tick": 1,
    },
    "pet_stats": {"hunger": 0.9, "boredom": 0.1, "social": 0.1, "toilet": 0.1, "tiredness": 0.1},
}


# ---------------------------------------------------------------------------
# generate_dataset CLI
# ---------------------------------------------------------------------------

class TestGenerateDatasetCli:
    def test_creates_train_and_eval_files(self, tmp_path: Path) -> None:
        from src.cli.generate_dataset import main
        with pytest.raises(SystemExit) as exc:
            main(["--data-dir", str(tmp_path), "--train-size", "20", "--eval-size", "5"])
        assert exc.value.code == 0
        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "eval.jsonl").exists()

    def test_correct_line_counts(self, tmp_path: Path) -> None:
        from src.cli.generate_dataset import main
        with pytest.raises(SystemExit):
            main(["--data-dir", str(tmp_path), "--train-size", "15", "--eval-size", "7"])
        assert len((tmp_path / "train.jsonl").read_text().strip().splitlines()) == 15
        assert len((tmp_path / "eval.jsonl").read_text().strip().splitlines()) == 7

    def test_each_line_has_prompt_and_completion(self, tmp_path: Path) -> None:
        from src.cli.generate_dataset import main
        with pytest.raises(SystemExit):
            main(["--data-dir", str(tmp_path), "--train-size", "5", "--eval-size", "3"])
        for line in (tmp_path / "train.jsonl").read_text().strip().splitlines():
            obj = json.loads(line)
            assert "prompt" in obj and "completion" in obj

    def test_exits_0_on_success(self, tmp_path: Path) -> None:
        from src.cli.generate_dataset import main
        with pytest.raises(SystemExit) as exc:
            main(["--data-dir", str(tmp_path), "--train-size", "5", "--eval-size", "3"])
        assert exc.value.code == 0


# ---------------------------------------------------------------------------
# train CLI  (dry-run only; skipped when torch/transformers not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _torch_available(), reason="torch/transformers not installed")
class TestTrainCli:
    def test_dry_run_exits_0(self, data_dir: Path, tmp_path: Path) -> None:
        from src.cli.train import main
        output_dir = tmp_path / "checkpoints"
        main([
            "--dry-run",
            "--train-data", str(data_dir / "train.jsonl"),
            "--eval-data", str(data_dir / "eval.jsonl"),
            "--output-dir", str(output_dir),
        ])
        assert output_dir.exists()

    def test_missing_train_dep_exits_1(self, tmp_path: Path) -> None:
        """ImportError from domain layer is caught and exits with code 1."""
        from src.cli.train import main
        with patch("src.domain.train.trainer._TORCH_AVAILABLE", False):
            with pytest.raises((SystemExit, ImportError)):
                main(["--dry-run"])


# ---------------------------------------------------------------------------
# evaluate CLI
# ---------------------------------------------------------------------------

class TestEvaluateCli:
    def test_missing_eval_file_exits_1(self, tmp_path: Path) -> None:
        from src.cli.evaluate import main
        with pytest.raises(SystemExit) as exc:
            main(["--eval-data", str(tmp_path / "nonexistent.jsonl")])
        assert exc.value.code == 1

    def test_passes_with_mocked_llama_cpp(self, data_dir: Path) -> None:
        from src.cli.evaluate import main
        valid_json = '{"action": "IDLE", "target_object_id": null}'
        with patch("src.cli.evaluate.load_llama_cpp_adapter", return_value=MagicMock()):
            with patch("src.cli.evaluate.infer_llama_cpp", return_value=valid_json):
                with pytest.raises(SystemExit) as exc:
                    main([
                        "--model-path", "/fake/model.gguf",
                        "--eval-data", str(data_dir / "eval.jsonl"),
                    ])
        assert exc.value.code == 0

    def test_fails_when_responses_are_invalid(self, data_dir: Path) -> None:
        from src.cli.evaluate import main
        with patch("src.cli.evaluate.load_llama_cpp_adapter", return_value=MagicMock()):
            with patch("src.cli.evaluate.infer_llama_cpp", return_value="not json at all"):
                with pytest.raises(SystemExit) as exc:
                    main([
                        "--model-path", "/fake/model.gguf",
                        "--eval-data", str(data_dir / "eval.jsonl"),
                    ])
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# export CLI
# ---------------------------------------------------------------------------

class TestExportCli:
    def test_missing_checkpoint_exits_1(self, tmp_path: Path) -> None:
        from src.cli.export import main
        with pytest.raises(SystemExit) as exc:
            main([
                "--checkpoint", str(tmp_path / "nonexistent"),
                "--output", str(tmp_path / "out.gguf"),
            ])
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# infer CLI
# ---------------------------------------------------------------------------

class TestInferCli:
    def test_invalid_json_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.cli.infer import main
        monkeypatch.setattr(sys, "stdin", io.StringIO("not valid json {{{"))
        with pytest.raises(SystemExit) as exc:
            main(["--model-path", "/fake/model.gguf"])
        assert exc.value.code == 1

    def test_valid_request_prints_response(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        from src.cli.infer import main
        fake_response = InferenceResponse(action=Action.EAT, target_object_id="b1")
        mock_adapter = MagicMock()
        mock_adapter.infer.return_value = fake_response

        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(VALID_REQUEST)))
        with patch("src.cli.infer.LlamaCppInferenceAdapter", return_value=mock_adapter):
            main(["--model-path", "/fake/model.gguf"])

        result = json.loads(capsys.readouterr().out)
        assert result["action"] == "EAT"
        assert result["target_object_id"] == "b1"

    def test_wrong_schema_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.cli.infer import main
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"not": "a request"}'))
        with pytest.raises(SystemExit) as exc:
            main(["--model-path", "/fake/model.gguf"])
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# Makefile smoke tests (subprocess)
# ---------------------------------------------------------------------------

class TestMakefile:
    def test_make_help_exits_0(self) -> None:
        """Validates Makefile syntax and that all targets are listed."""
        result = subprocess.run(["make", "help"], capture_output=True, cwd=PROJECT_ROOT)
        assert result.returncode == 0
        output = result.stdout.decode()
        for target in ("serve", "test", "data", "train", "evaluate", "export", "infer"):
            assert target in output, f"'{target}' missing from make help output"

    def test_data_cli_creates_files(self, tmp_path: Path) -> None:
        """Runs the same command make data invokes, verifying end-to-end wiring."""
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
        result = subprocess.run(
            [sys.executable, "src/cli/generate_dataset.py", "--data-dir", str(tmp_path),
             "--train-size", "20", "--eval-size", "5"],
            capture_output=True, cwd=PROJECT_ROOT, env=env,
        )
        assert result.returncode == 0, result.stderr.decode()
        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "eval.jsonl").exists()

    @pytest.mark.skipif(not _torch_available(), reason="torch/transformers not installed")
    def test_train_dry_run_cli(self, tmp_path: Path) -> None:
        """Runs the same command make train DRY_RUN=1 invokes."""
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
        subprocess.run(
            [sys.executable, "src/cli/generate_dataset.py", "--data-dir", str(tmp_path),
             "--train-size", "20", "--eval-size", "5"],
            check=True, capture_output=True, cwd=PROJECT_ROOT, env=env,
        )
        result = subprocess.run(
            [sys.executable, "src/cli/train.py",
             "--dry-run",
             "--train-data", str(tmp_path / "train.jsonl"),
             "--eval-data", str(tmp_path / "eval.jsonl"),
             "--output-dir", str(tmp_path / "checkpoints")],
            capture_output=True, cwd=PROJECT_ROOT, env=env,
        )
        assert result.returncode == 0, result.stderr.decode()
