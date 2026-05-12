"""Training pipeline workflow — orchestrates dataset generation through GGUF export."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from domain.models import RunStatus
    from domain.train.dataset import EVAL_SIZE, SEED, TRAIN_SIZE
    from domain.train.trainer import DEFAULT_EPOCHS, DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_PATIENCE, DEFAULT_WARMUP_RATIO
    from interactors.temporal.activities import (
        CheckpointPath,
        DatasetConfig,
        DatasetPaths,
        EvalConfig,
        EvalResult,
        ExportConfig,
        GGUFPath,
        TrainConfig,
        evaluate_activity,
        export_activity,
        finalise_run_activity,
        generate_dataset_activity,
        save_gguf_path_activity,
        train_activity,
        update_run_status_activity,
    )


@dataclass
class ExperimentConfig:
    experiment_name: str = ""
    model_id: str = ""
    model_name: str = ""
    run_id: str = ""
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    skip_generate: bool = False
    dry_run: bool = False
    data_dir: str = "data"
    output_dir: str = DEFAULT_OUTPUT_DIR
    gguf_output: str = "models/aipet.gguf"
    model: str = DEFAULT_MODEL
    train_size: int = TRAIN_SIZE
    eval_size: int = EVAL_SIZE
    seed: int = SEED
    # "local", "kaggle", or "ssh" — controls where fine-tuning runs.
    remote_backend: str = ""
    # None = auto-detect based on model size; True = always QLoRA; False = never QLoRA.
    force_qlora: bool | None = None


@dataclass
class PipelineResult:
    run_id: str = ""
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
        result = PipelineResult(run_id=config.run_id, experiment_name=config.experiment_name)

        if config.skip_generate:
            result.dataset_paths = DatasetPaths(
                train=f"{config.data_dir}/train.jsonl",
                eval=f"{config.data_dir}/eval.jsonl",
            )
            workflow.logger.info("skip_generate=True: reusing existing dataset at %s", config.data_dir)
        else:
            if config.run_id:
                await workflow.execute_activity(
                    update_run_status_activity,
                    args=[config.run_id, RunStatus.GENERATING.value],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_RETRY,
                )
            result.dataset_paths = await workflow.execute_activity(
                generate_dataset_activity,
                DatasetConfig(
                    data_dir=config.data_dir,
                    train_size=config.train_size,
                    eval_size=config.eval_size,
                    seed=config.seed,
                ),
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(minutes=2),
                retry_policy=_RETRY,
            )

        if config.run_id:
            await workflow.execute_activity(
                update_run_status_activity,
                args=[config.run_id, RunStatus.TRAINING.value],
                start_to_close_timeout=timedelta(minutes=5),
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
                dry_run=config.dry_run,
                remote_backend=config.remote_backend,
                experiment_name=config.experiment_name,
                db_run_id=config.run_id,
                force_qlora=config.force_qlora,
            ),
            start_to_close_timeout=timedelta(hours=6),
            heartbeat_timeout=timedelta(minutes=2),
            retry_policy=_NO_RETRY,
        )

        if config.run_id:
            await workflow.execute_activity(
                update_run_status_activity,
                args=[config.run_id, RunStatus.EVALUATING.value],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=_RETRY,
            )

        result.eval_result = await workflow.execute_activity(
            evaluate_activity,
            EvalConfig(
                checkpoint=result.checkpoint.path,
                eval_data=result.dataset_paths.eval,
                run_id=result.checkpoint.run_id,
                remote_backend=result.checkpoint.remote_backend,
                output_dir=config.output_dir,
                db_run_id=config.run_id,
            ),
            start_to_close_timeout=timedelta(minutes=30),
            heartbeat_timeout=timedelta(minutes=2),
            retry_policy=_RETRY,
        )

        result.passed = result.eval_result.passed

        if result.eval_result.passed:
            if config.run_id:
                await workflow.execute_activity(
                    update_run_status_activity,
                    args=[config.run_id, RunStatus.EXPORTING.value],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_RETRY,
                )

            result.gguf_path = await workflow.execute_activity(
                export_activity,
                ExportConfig(
                    checkpoint_path=result.checkpoint.path,
                    gguf_output=config.gguf_output,
                    run_id=result.checkpoint.run_id,
                    remote_backend=result.checkpoint.remote_backend,
                    model_id=config.model_id,
                    pipeline_run_id=config.run_id,
                    model_name=config.model_name,
                ),
                start_to_close_timeout=timedelta(hours=1),
                heartbeat_timeout=timedelta(minutes=2),
                retry_policy=_NO_RETRY,
            )

            if config.model_id:
                await workflow.execute_activity(
                    save_gguf_path_activity,
                    args=[config.model_id, result.gguf_path.path],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_RETRY,
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

        if config.run_id:
            await workflow.execute_activity(
                finalise_run_activity,
                args=[config.run_id, result.passed, result.eval_result.valid_pct],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=_RETRY,
            )

        return result


# ---------------------------------------------------------------------------
# Standalone evaluate workflow (re-eval an existing run without retraining)
# ---------------------------------------------------------------------------


@dataclass
class EvaluateWorkflowConfig:
    run_id: str = ""
    remote_backend: str = ""
    remote_run_id: str = ""
    eval_data: str = "data/eval.jsonl"
    checkpoint_path: str = ""
    output_dir: str = ""


@workflow.defn
class EvaluateWorkflow:
    @workflow.run
    async def run(self, config: EvaluateWorkflowConfig) -> EvalResult:
        result = await workflow.execute_activity(
            evaluate_activity,
            EvalConfig(
                checkpoint=config.checkpoint_path,
                eval_data=config.eval_data,
                run_id=config.remote_run_id,
                remote_backend=config.remote_backend,
                output_dir=config.output_dir,
            ),
            start_to_close_timeout=timedelta(minutes=30),
            heartbeat_timeout=timedelta(minutes=2),
            retry_policy=_RETRY,
        )
        if config.run_id:
            await workflow.execute_activity(
                finalise_run_activity,
                args=[config.run_id, result.passed, result.valid_pct],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=_RETRY,
            )
        return result


# ---------------------------------------------------------------------------
# Standalone export workflow (download checkpoint + export GGUF)
# ---------------------------------------------------------------------------


@dataclass
class ExportWorkflowConfig:
    run_id: str = ""
    model_id: str = ""
    remote_backend: str = ""
    remote_run_id: str = ""
    checkpoint_path: str = ""
    gguf_output: str = "models/aipet.gguf"


@workflow.defn
class ExportWorkflow:
    @workflow.run
    async def run(self, config: ExportWorkflowConfig) -> GGUFPath:
        gguf = await workflow.execute_activity(
            export_activity,
            ExportConfig(
                checkpoint_path=config.checkpoint_path,
                gguf_output=config.gguf_output,
                run_id=config.remote_run_id,
                remote_backend=config.remote_backend,
                model_id=config.model_id,
                pipeline_run_id=config.run_id,
            ),
            start_to_close_timeout=timedelta(hours=1),
            heartbeat_timeout=timedelta(minutes=2),
            retry_policy=_NO_RETRY,
        )
        if config.model_id:
            await workflow.execute_activity(
                save_gguf_path_activity,
                args=[config.model_id, gguf.path],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=_RETRY,
            )
        if config.run_id:
            await workflow.execute_activity(
                update_run_status_activity,
                args=[config.run_id, "completed"],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=_RETRY,
            )
        return gguf
