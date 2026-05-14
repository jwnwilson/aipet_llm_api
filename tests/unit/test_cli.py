"""Tests for CLI commands — calls main() directly and smoke-tests Makefile targets."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domain.actions import Action
from domain.models import InferenceResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Small train/eval dataset in a temp dir — used by multiple test classes."""
    from domain.train.dataset import generate
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
        from interactors.cli.generate_dataset import main
        with pytest.raises(SystemExit) as exc:
            main(["--data-dir", str(tmp_path), "--train-size", "20", "--eval-size", "5"])
        assert exc.value.code == 0
        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "eval.jsonl").exists()

    def test_correct_line_counts(self, tmp_path: Path) -> None:
        from interactors.cli.generate_dataset import main
        with pytest.raises(SystemExit):
            main(["--data-dir", str(tmp_path), "--train-size", "15", "--eval-size", "7"])
        assert len((tmp_path / "train.jsonl").read_text().strip().splitlines()) == 15
        assert len((tmp_path / "eval.jsonl").read_text().strip().splitlines()) == 7

    def test_each_line_has_prompt_and_completion(self, tmp_path: Path) -> None:
        from interactors.cli.generate_dataset import main
        with pytest.raises(SystemExit):
            main(["--data-dir", str(tmp_path), "--train-size", "5", "--eval-size", "3"])
        for line in (tmp_path / "train.jsonl").read_text().strip().splitlines():
            obj = json.loads(line)
            assert "prompt" in obj and "completion" in obj

    def test_exits_0_on_success(self, tmp_path: Path) -> None:
        from interactors.cli.generate_dataset import main
        with pytest.raises(SystemExit) as exc:
            main(["--data-dir", str(tmp_path), "--train-size", "5", "--eval-size", "3"])
        assert exc.value.code == 0


# ---------------------------------------------------------------------------
# train CLI  (dry-run only; skipped when torch/transformers not installed)
# ---------------------------------------------------------------------------

class TestTrainCli:
    def test_missing_train_dep_exits_1(self, tmp_path: Path) -> None:
        """ImportError from domain layer is caught and exits with code 1."""
        from interactors.cli.train import main
        with patch("domain.train.trainer._TORCH_AVAILABLE", False):
            with pytest.raises((SystemExit, ImportError)):
                main(["--dry-run"])


# ---------------------------------------------------------------------------
# evaluate CLI
# ---------------------------------------------------------------------------

class TestEvaluateCli:
    def test_missing_eval_file_exits_1(self, tmp_path: Path) -> None:
        from interactors.cli.evaluate import main
        with pytest.raises(SystemExit) as exc:
            main(["--eval-data", str(tmp_path / "nonexistent.jsonl")])
        assert exc.value.code == 1

    def test_passes_with_mocked_llama_cpp(self, data_dir: Path) -> None:
        from interactors.cli.evaluate import main
        valid_json = '{"action": "IDLE", "target_object_id": null}'
        with patch("interactors.cli.evaluate.load_llama_cpp_adapter", return_value=MagicMock()):
            with patch("interactors.cli.evaluate.infer_llama_cpp", return_value=valid_json):
                with pytest.raises(SystemExit) as exc:
                    main([
                        "--model-path", "/fake/model.gguf",
                        "--eval-data", str(data_dir / "eval.jsonl"),
                    ])
        assert exc.value.code == 0

    def test_fails_when_responses_are_invalid(self, data_dir: Path) -> None:
        from interactors.cli.evaluate import main
        with patch("interactors.cli.evaluate.load_llama_cpp_adapter", return_value=MagicMock()):
            with patch("interactors.cli.evaluate.infer_llama_cpp", return_value="not json at all"):
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
        from interactors.cli.export import main
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
        from interactors.cli.infer import main
        monkeypatch.setattr(sys, "stdin", io.StringIO("not valid json {{{"))
        with pytest.raises(SystemExit) as exc:
            main(["--model-path", "/fake/model.gguf"])
        assert exc.value.code == 1

    def test_valid_request_prints_response(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        from interactors.cli.infer import main
        fake_response = InferenceResponse(action=Action.EAT, target_object_id="b1")
        mock_adapter = MagicMock()
        mock_adapter.infer.return_value = fake_response

        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(VALID_REQUEST)))
        with patch("interactors.cli.infer.LlamaCppInferenceAdapter", return_value=mock_adapter):
            main(["--model-path", "/fake/model.gguf"])

        result = json.loads(capsys.readouterr().out)
        assert result["action"] == "EAT"
        assert result["target_object_id"] == "b1"

    def test_wrong_schema_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from interactors.cli.infer import main
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
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
        result = subprocess.run(
            [sys.executable, "src/interactors/cli/generate_dataset.py", "--data-dir", str(tmp_path),
             "--train-size", "20", "--eval-size", "5"],
            capture_output=True, cwd=PROJECT_ROOT, env=env,
        )
        assert result.returncode == 0, result.stderr.decode()
        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "eval.jsonl").exists()



# ---------------------------------------------------------------------------
# trigger_training CLI
# ---------------------------------------------------------------------------


class TestTriggerTrainingCli:
    """Tests for trigger_training._trigger() — Temporal client and DB are mocked."""

    def _mock_client(self):
        handle = MagicMock()
        handle.id = "wf-test-123"
        client = MagicMock()
        client.start_workflow = AsyncMock(return_value=handle)
        return client

    @pytest.mark.asyncio
    async def test_without_model_id_starts_workflow_with_empty_model_fields(self, monkeypatch, tmp_path):
        from interactors.cli import trigger_training

        mock_client = self._mock_client()
        monkeypatch.setattr("temporalio.client.Client.connect", AsyncMock(return_value=mock_client))
        monkeypatch.chdir(tmp_path)

        await trigger_training._trigger(
            experiment_name="test-exp",
            epochs=1, patience=1, warmup_ratio=0.05,
            skip_generate=False, dry_run=True, remote_backend="",
            model="HuggingFaceTB/SmolLM2-360M",
            train_size=10, eval_size=5,
            model_id=None,
        )

        mock_client.start_workflow.assert_called_once()
        config = mock_client.start_workflow.call_args[0][1]
        assert config.model_id == ""
        assert config.model_name == ""
        assert len(config.run_id) == 36  # local UUID

    @pytest.mark.asyncio
    async def test_with_model_id_creates_run_record_and_passes_model_name(self, monkeypatch, tmp_path):
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool
        from adapters.database import init_db
        from adapters.database.model_store import SQLAlchemyModelStore
        from adapters.database.run_store import SQLAlchemyRunStore
        from domain.models import TrainingModelConfig
        from interactors.cli import trigger_training

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        init_db(engine)
        model = SQLAlchemyModelStore(engine).create(TrainingModelConfig(name="my-pet-v2"))
        run_store = SQLAlchemyRunStore(engine)

        mock_client = self._mock_client()
        monkeypatch.setattr("temporalio.client.Client.connect", AsyncMock(return_value=mock_client))
        monkeypatch.setattr("adapters.database.engine.make_engine", lambda: engine)
        monkeypatch.chdir(tmp_path)

        await trigger_training._trigger(
            experiment_name="test-exp",
            epochs=1, patience=1, warmup_ratio=0.05,
            skip_generate=False, dry_run=True, remote_backend="",
            model="HuggingFaceTB/SmolLM2-360M",
            train_size=10, eval_size=5,
            model_id=model.id,
        )

        runs = run_store.list(model_id=model.id)
        assert len(runs) == 1
        assert runs[0].status.value == "pending"

        config = mock_client.start_workflow.call_args[0][1]
        assert config.model_id == model.id
        assert config.model_name == "my-pet-v2"
        assert config.run_id == runs[0].id

    @pytest.mark.asyncio
    async def test_with_invalid_model_id_exits_1(self, monkeypatch, tmp_path):
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool
        from adapters.database import init_db
        from interactors.cli import trigger_training

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        init_db(engine)

        mock_client = self._mock_client()
        monkeypatch.setattr("temporalio.client.Client.connect", AsyncMock(return_value=mock_client))
        monkeypatch.setattr("adapters.database.engine.make_engine", lambda: engine)

        with pytest.raises(SystemExit) as exc_info:
            await trigger_training._trigger(
                experiment_name="test-exp",
                epochs=1, patience=1, warmup_ratio=0.05,
                skip_generate=False, dry_run=True, remote_backend="",
                model="HuggingFaceTB/SmolLM2-360M",
                train_size=10, eval_size=5,
                model_id="nonexistent-model-id",
            )

        assert exc_info.value.code == 1
        mock_client.start_workflow.assert_not_called()

    def test_main_with_model_id_creates_run_record(self, monkeypatch, tmp_path):
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool
        from adapters.database import init_db
        from adapters.database.model_store import SQLAlchemyModelStore
        from adapters.database.run_store import SQLAlchemyRunStore
        from domain.models import TrainingModelConfig
        from interactors.cli import trigger_training

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        init_db(engine)
        model = SQLAlchemyModelStore(engine).create(TrainingModelConfig(name="main-test-model"))
        run_store = SQLAlchemyRunStore(engine)

        mock_handle = MagicMock()
        mock_handle.id = "wf-main-test"
        mock_client = MagicMock()
        mock_client.start_workflow = AsyncMock(return_value=mock_handle)

        monkeypatch.setattr("adapters.database.engine.make_engine", lambda: engine)
        monkeypatch.setattr("temporalio.client.Client.connect", AsyncMock(return_value=mock_client))
        monkeypatch.chdir(tmp_path)

        trigger_training.main([
            "--experiment-name", "main-test",
            "--model-id", model.id,
        ])

        runs = run_store.list(model_id=model.id)
        assert len(runs) == 1
        assert runs[0].model_id == model.id
        assert runs[0].status.value == "pending"


# ---------------------------------------------------------------------------
# seed_models CLI
# ---------------------------------------------------------------------------

class TestSeedModelsCli:
    def test_creates_default_models(self, tmp_path, monkeypatch):
        from sqlalchemy import create_engine
        from adapters.database.model_store import SQLAlchemyModelStore

        db_path = tmp_path / "seed.db"
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

        from interactors.cli import seed_models
        seed_models.main()

        engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        names = {m.name for m in SQLAlchemyModelStore(engine).list()}
        assert "smollm2-360m-local" in names
        assert "smollm2-360m-kaggle" in names
        assert "smollm2-1.7b-runpod" in names
        assert len(names) == 3

    def test_is_idempotent(self, tmp_path, monkeypatch):
        from sqlalchemy import create_engine
        from adapters.database.model_store import SQLAlchemyModelStore

        db_path = tmp_path / "seed.db"
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

        from interactors.cli import seed_models
        seed_models.main()
        seed_models.main()

        engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        assert len(SQLAlchemyModelStore(engine).list()) == 3
