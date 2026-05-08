"""Training pipeline workflow — orchestrates dataset generation through GGUF export."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from domain.train.dataset import EVAL_SIZE, SEED, TRAIN_SIZE
    from domain.train.trainer import DEFAULT_EPOCHS, DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_PATIENCE, DEFAULT_WARMUP_RATIO
    from temporal.activities import (
        CheckpointPath,
        DatasetConfig,
        DatasetPaths,
        EvalConfig,
        EvalResult,
        GGUFPath,
        TrainConfig,
        evaluate_activity,
        export_activity,
        generate_dataset_activity,
        train_activity,
    )


@dataclass
class ExperimentConfig:
    experiment_name: str = ""
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    skip_generate: bool = False
    data_dir: str = "data"
    output_dir: str = DEFAULT_OUTPUT_DIR
    model: str = DEFAULT_MODEL
    train_size: int = TRAIN_SIZE
    eval_size: int = EVAL_SIZE
    seed: int = SEED
    # "local", "kaggle", or "ssh" — controls where fine-tuning runs.
    remote_backend: str = ""


@dataclass
class PipelineResult:
    experiment_name: str = ""
    dataset_paths: DatasetPaths = field(default_factory=DatasetPaths)
    checkpoint: CheckpointPath = field(default_factory=CheckpointPath)
    eval_result: EvalResult = field(default_factory=EvalResult)
    gguf_path: GGUFPath = field(default_factory=GGUFPath)
    passed: bool = False


_RETRY = RetryPolicy(maximum_attempts=3, backoff_coefficient=2.0)
_NO_RETRY = RetryPolicy(maximum_attempts=1)


@workflow.defn
class TrainingPipelineWorkflow:
    def __init__(self) -> None:
        self._failed = False

    @workflow.signal
    def WorkflowFailed(self) -> None:
        """Signal received when a caller marks this workflow as externally failed."""
        self._failed = True

    @workflow.run
    async def run(self, config: ExperimentConfig) -> PipelineResult:
        result = PipelineResult(experiment_name=config.experiment_name)

        if config.skip_generate:
            result.dataset_paths = DatasetPaths(
                train=f"{config.data_dir}/train.jsonl",
                eval=f"{config.data_dir}/eval.jsonl",
            )
            workflow.logger.info("skip_generate=True: reusing existing dataset at %s", config.data_dir)
        else:
            result.dataset_paths = await workflow.execute_activity(
                generate_dataset_activity,
                DatasetConfig(
                    data_dir=config.data_dir,
                    train_size=config.train_size,
                    eval_size=config.eval_size,
                    seed=config.seed,
                ),
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=_RETRY,
            )

        result.checkpoint = await workflow.execute_activity(
            train_activity,
            TrainConfig(
                model=config.model,
                train_data=result.dataset_paths.train,
                eval_data=result.dataset_paths.eval,
                output_dir=config.output_dir,
                epochs=config.epochs,
                patience=config.patience,
                warmup_ratio=config.warmup_ratio,
                remote_backend=config.remote_backend,
                experiment_name=config.experiment_name,
            ),
            start_to_close_timeout=timedelta(hours=6),
            retry_policy=_NO_RETRY,
        )

        result.eval_result = await workflow.execute_activity(
            evaluate_activity,
            EvalConfig(
                checkpoint=result.checkpoint.path,
                eval_data=result.dataset_paths.eval,
            ),
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=_RETRY,
        )

        result.passed = result.eval_result.passed

        if result.eval_result.passed:
            result.gguf_path = await workflow.execute_activity(
                export_activity,
                result.checkpoint,
                start_to_close_timeout=timedelta(hours=1),
                retry_policy=_NO_RETRY,
            )
            workflow.logger.info(
                "experiment=%s PASS valid_pct=%.1f%% gguf=%s",
                config.experiment_name,
                result.eval_result.valid_pct * 100,
                result.gguf_path.path,
            )
        else:
            workflow.logger.warning(
                "experiment=%s FAIL valid_pct=%.1f%% (threshold=95%%) — export skipped",
                config.experiment_name,
                result.eval_result.valid_pct * 100,
            )

        return result
