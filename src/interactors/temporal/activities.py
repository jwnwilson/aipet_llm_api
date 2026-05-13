"""Temporal activities — one per pipeline stage, each wrapping a domain function."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from temporalio import activity
from temporalio.exceptions import ApplicationError

from domain.ports import ModelStorePort, RemoteTrainingPort, RunStorePort, StoragePort
from domain.train.dataset import EVAL_SIZE, SEED, TRAIN_SIZE
from domain.train.config import DEFAULT_EPOCHS, DEFAULT_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_PATIENCE, DEFAULT_WARMUP_RATIO


# ---------------------------------------------------------------------------
# Module-level singletons — injected by the worker (or tests)
# ---------------------------------------------------------------------------

_model_store: ModelStorePort | None = None
_run_store: RunStorePort | None = None
_storage: StoragePort | None = None


def configure_model_store(store: ModelStorePort) -> None:
    global _model_store
    _model_store = store


def configure_run_store(store: RunStorePort) -> None:
    global _run_store
    _run_store = store


def configure_storage(storage: StoragePort) -> None:
    global _storage
    _storage = storage


def _get_model_store() -> ModelStorePort:
    if _model_store is None:
        raise RuntimeError("ModelStorePort has not been configured in activities.")
    return _model_store


def _get_run_store() -> RunStorePort:
    if _run_store is None:
        raise RuntimeError("RunStorePort has not been configured in activities.")
    return _run_store


def _get_storage() -> StoragePort:
    if _storage is None:
        raise RuntimeError("StoragePort has not been configured in activities.")
    return _storage


# ---------------------------------------------------------------------------
# Config / result dataclasses (Temporal serialises these as JSON)
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    data_dir: str = "data"
    train_size: int = TRAIN_SIZE
    eval_size: int = EVAL_SIZE
    seed: int = SEED


@dataclass
class DatasetPaths:
    train: str = ""
    eval: str = ""


@dataclass
class TrainConfig:
    model: str = DEFAULT_MODEL
    train_data: str = "data/train.jsonl"
    eval_data: str = "data/eval.jsonl"
    output_dir: str = DEFAULT_OUTPUT_DIR
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    dry_run: bool = False
    # Remote backend: "" or "local" → run locally; "kaggle" or "ssh" → remote.
    remote_backend: str = ""
    experiment_name: str = ""
    db_run_id: str = ""  # DB RunRecord.id for progress updates; "" = no tracking
    # None = auto-detect based on model size; True = always QLoRA; False = never QLoRA.
    force_qlora: bool | None = None


@dataclass
class CheckpointPath:
    path: str = ""            # local path; empty when checkpoint is still on the remote
    run_id: str = ""          # opaque id from adapter.submit(); non-empty for remote runs
    remote_backend: str = ""  # "kaggle", "ssh", etc.; "" means local


@dataclass
class EvalConfig:
    checkpoint: str = ""
    eval_data: str = "data/eval.jsonl"
    run_id: str = ""          # non-empty → run eval on the remote machine
    remote_backend: str = ""
    output_dir: str = ""      # local download dest when falling back from remote to local eval
    db_run_id: str = ""       # DB RunRecord.id for progress updates; "" = no tracking


@dataclass
class EvalResult:
    valid_pct: float = 0.0
    passed: bool = False


@dataclass
class ExportConfig:
    checkpoint_path: str = ""
    gguf_output: str = "models/aipet.gguf"
    run_id: str = ""           # non-empty → download checkpoint from remote before export
    remote_backend: str = ""
    model_id: str = ""         # fallback storage key when pipeline_run_id is unset
    pipeline_run_id: str = ""  # pipeline UUID; drives storage key workflow/{id}/model.gguf
    model_name: str = ""       # human-readable name; drives gguf/{model_name}.gguf key


@dataclass
class GGUFPath:
    path: str = ""            # storage key (e.g. "gguf/{model_id}.gguf")


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------


async def _heartbeat_loop(stage: str, interval: int = 30) -> None:
    """Send a liveness heartbeat every `interval` seconds while a blocking call runs."""
    while True:
        activity.heartbeat({"stage": stage})
        await asyncio.sleep(interval)


async def _poll_local_progress(db_run_id: str, output_dir: str, interval: int = 30) -> None:
    """Heartbeat loop for local training: also polls progress.json and persists it."""
    import json as _json
    progress_path = Path(output_dir) / "progress.json"
    while True:
        activity.heartbeat({"stage": "train_local"})
        if db_run_id:
            try:
                data = _json.loads(progress_path.read_text())
                step = data.get("step", 0)
                max_steps = data.get("max_steps", 1)
                frac = step / max_steps if max_steps else 0.0
                epoch = data.get("epoch", "?")
                parts = [f"epoch={epoch}"]
                for key in ("loss", "eval_loss"):
                    if key in data:
                        parts.append(f"{key}={data[key]:.4f}")
                _get_run_store().update_progress(db_run_id, frac, "  ".join(parts))
            except Exception:
                pass
        await asyncio.sleep(interval)


@activity.defn
async def generate_dataset_activity(config: DatasetConfig) -> DatasetPaths:
    from domain.train.dataset import generate

    loop = asyncio.get_event_loop()
    heartbeat_task = asyncio.ensure_future(_heartbeat_loop("generate_dataset"))
    try:
        ok = await loop.run_in_executor(
            None,
            lambda: generate(
                data_dir=Path(config.data_dir),
                train_size=config.train_size,
                eval_size=config.eval_size,
                seed=config.seed,
            ),
        )
    except Exception as exc:
        raise ApplicationError(f"generate_dataset failed: {exc}") from exc
    finally:
        heartbeat_task.cancel()

    if not ok:
        raise ApplicationError("Dataset generation failed: invalid examples or distribution out of bounds")

    return DatasetPaths(
        train=str(Path(config.data_dir) / "train.jsonl"),
        eval=str(Path(config.data_dir) / "eval.jsonl"),
    )


def _make_remote_adapter(backend: str) -> RemoteTrainingPort:
    if backend == "kaggle":
        from adapters.compute.kaggle import KaggleTrainingAdapter
        return KaggleTrainingAdapter()
    if backend == "ssh":
        from adapters.compute.ssh import SshTrainingAdapter
        return SshTrainingAdapter()
    if backend == "colab":
        from adapters.compute.colab.adapter import ColabTrainingAdapter
        return ColabTrainingAdapter()
    if backend == "runpod":
        from adapters.compute.runpod import RunPodTrainingAdapter
        return RunPodTrainingAdapter()
    if backend == "vastai":
        from adapters.compute.vastai import VastAiTrainingAdapter
        return VastAiTrainingAdapter()
    raise ApplicationError(f"Unknown remote_backend: {backend!r}")


@activity.defn
async def train_activity(config: TrainConfig) -> CheckpointPath:
    backend = config.remote_backend or "local"

    if backend == "local":
        return await _train_local(config)

    adapter = _make_remote_adapter(backend)
    return await _train_remote(config, adapter)


async def _train_local(config: TrainConfig) -> CheckpointPath:
    from domain.train.trainer import train

    loop = asyncio.get_event_loop()
    heartbeat_task = asyncio.ensure_future(
        _poll_local_progress(config.db_run_id, config.output_dir)
    )
    try:
        await loop.run_in_executor(
            None,
            lambda: train(
                model=config.model,
                train_data=config.train_data,
                eval_data=config.eval_data,
                output_dir=config.output_dir,
                epochs=config.epochs,
                patience=config.patience,
                warmup_ratio=config.warmup_ratio,
                dry_run=config.dry_run,
                force_qlora=config.force_qlora,
            ),
        )
    except Exception as exc:
        raise ApplicationError(f"train failed: {exc}") from exc
    finally:
        heartbeat_task.cancel()

    return CheckpointPath(path=config.output_dir)


async def _train_remote(config: TrainConfig, adapter: RemoteTrainingPort) -> CheckpointPath:
    from domain.models import RemoteTrainConfig

    remote_config = RemoteTrainConfig(
        model=config.model,
        train_data=config.train_data,
        eval_data=config.eval_data,
        epochs=config.epochs,
        patience=config.patience,
        warmup_ratio=config.warmup_ratio,
        experiment_name=config.experiment_name or "aipet",
    )

    loop = asyncio.get_event_loop()

    # Run submit in an executor — it calls subprocess + time.sleep (blocks event loop).
    heartbeat_task = asyncio.ensure_future(_heartbeat_loop("train_submit"))
    try:
        run_id = await loop.run_in_executor(None, lambda: adapter.submit(remote_config))
    except Exception as exc:
        raise ApplicationError(f"Remote submit failed: {exc}") from exc
    finally:
        heartbeat_task.cancel()

    activity.logger.info("Remote job submitted: adapter=%s run_id=%s", type(adapter).__name__, run_id)

    started_at = time.time()
    while True:
        # status/logs/progress all run subprocess — keep them off the event loop.
        try:
            status = await loop.run_in_executor(None, lambda: adapter.status(run_id))
        except Exception as exc:
            raise ApplicationError(f"Remote status check failed: {exc}") from exc

        elapsed_s = int(time.time() - started_at)
        logs = await loop.run_in_executor(None, lambda: adapter.logs(run_id))

        if logs:
            activity.logger.info(
                "Training progress [%s] elapsed=%ds:\n%s",
                type(adapter).__name__, elapsed_s, logs,
            )
        else:
            activity.logger.info(
                "Remote status: adapter=%s run_id=%s status=%s elapsed=%ds",
                type(adapter).__name__, run_id, status, elapsed_s,
            )

        activity.heartbeat({"status": status, "elapsed_s": elapsed_s, "logs": logs})

        if config.db_run_id:
            try:
                frac, detail = await loop.run_in_executor(None, lambda: adapter.progress(run_id))
                if frac > 0:
                    _get_run_store().update_progress(config.db_run_id, frac, detail)
            except Exception:
                pass

        if status == "done":
            # Defer download — eval may run on the remote, so we avoid pulling
            # gigabytes of checkpoint data until we know the model actually passes.
            return CheckpointPath(
                run_id=run_id,
                remote_backend=config.remote_backend,
            )

        if status == "failed":
            raise ApplicationError(
                f"Remote training failed (adapter={type(adapter).__name__}, run_id={run_id})"
            )

        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            activity.logger.warning(
                "train_activity cancelled while polling (adapter=%s, run_id=%s, elapsed=%ds)",
                type(adapter).__name__,
                run_id,
                int(time.time() - started_at),
            )
            raise


@activity.defn
async def evaluate_activity(config: EvalConfig) -> EvalResult:
    loop = asyncio.get_event_loop()
    heartbeat_task = asyncio.ensure_future(_heartbeat_loop("evaluate"))
    try:
        if config.remote_backend:
            result = await _evaluate_remote(config, loop)
        else:
            result = await _evaluate_local(config, loop)
    except ApplicationError:
        raise
    except Exception as exc:
        raise ApplicationError(f"evaluate failed: {exc}") from exc
    finally:
        heartbeat_task.cancel()

    activity.logger.info(
        "Eval complete: valid_pct=%.1f%% passed=%s",
        result.valid_pct * 100,
        result.passed,
    )
    return result


async def _evaluate_remote(config: EvalConfig, loop: asyncio.AbstractEventLoop) -> EvalResult:
    adapter = _make_remote_adapter(config.remote_backend)
    try:
        valid_pct, passed = await loop.run_in_executor(
            None, lambda: adapter.eval(config.run_id, config.eval_data)
        )
        return EvalResult(valid_pct=valid_pct, passed=passed)
    except NotImplementedError:
        # Backend doesn't support remote eval (e.g. Kaggle). Download the checkpoint
        # now and fall back to local HF eval.
        activity.logger.info(
            "Remote backend %r does not support remote eval — downloading checkpoint for local eval",
            config.remote_backend,
        )
        dest = Path(config.output_dir) if config.output_dir else Path("models/checkpoints") / config.run_id
        try:
            checkpoint_path = await loop.run_in_executor(
                None, lambda: adapter.download(config.run_id, dest)
            )
        except Exception as exc:
            raise ApplicationError(
                f"Remote eval not supported and checkpoint download failed: {exc}"
            ) from exc

        local_config = EvalConfig(checkpoint=checkpoint_path, eval_data=config.eval_data)
        return await _evaluate_local(local_config, loop)


async def _evaluate_local(config: EvalConfig, loop: asyncio.AbstractEventLoop) -> EvalResult:
    from domain.train.evaluate import evaluate, infer_hf, load_hf_pipeline

    pipe = load_hf_pipeline(config.checkpoint)
    infer_fn = lambda prompt: infer_hf(pipe, prompt)  # noqa: E731

    exit_code, valid_pct = await loop.run_in_executor(
        None, lambda: evaluate(Path(config.eval_data), infer_fn)
    )
    return EvalResult(valid_pct=valid_pct, passed=exit_code == 0)


@activity.defn
async def export_activity(config: ExportConfig) -> GGUFPath:
    from domain.train.export import export as export_gguf

    loop = asyncio.get_event_loop()
    heartbeat_task = asyncio.ensure_future(_heartbeat_loop("export"))
    try:
        # For remote runs, download the checkpoint first (deferred from train_activity).
        if config.remote_backend:
            adapter = _make_remote_adapter(config.remote_backend)
            dest = Path(config.gguf_output).parent / "checkpoint"
            try:
                checkpoint_path = await loop.run_in_executor(
                    None, lambda: adapter.download(config.run_id, dest)
                )
            except Exception as exc:
                raise ApplicationError(f"Remote download failed: {exc}") from exc
        else:
            checkpoint_path = config.checkpoint_path

        local_gguf = Path(config.gguf_output)
        await loop.run_in_executor(
            None,
            lambda: export_gguf(checkpoint=Path(checkpoint_path), output=local_gguf),
        )

        # Upload to storage so the API can retrieve it by key.
        storage = _get_storage()
        if config.model_name:
            key = f"gguf/{config.model_name}.gguf"
        elif config.pipeline_run_id:
            key = f"workflow/{config.pipeline_run_id}/model.gguf"
        elif config.model_id:
            key = f"gguf/{config.model_id}.gguf"
        else:
            key = config.gguf_output
        storage.upload(local_gguf, key)

    except SystemExit as exc:
        raise ApplicationError(f"export failed: llama.cpp setup issue (exit {exc.code})") from exc
    except ApplicationError:
        raise
    except Exception as exc:
        raise ApplicationError(f"export failed: {exc}") from exc
    finally:
        heartbeat_task.cancel()

    return GGUFPath(path=key)


@activity.defn
async def finalise_run_activity(run_id: str, passed: bool, valid_pct: float) -> None:
    """Mark the run as completed or failed and persist the eval result."""
    from domain.models import RunStatus

    store = _get_run_store()
    store.update_eval(run_id, valid_pct)
    store.update_status(run_id, RunStatus.COMPLETED if passed else RunStatus.FAILED)
    activity.logger.info(
        "Run %s finalised: status=%s valid_pct=%.1f%%",
        run_id,
        RunStatus.COMPLETED.value if passed else RunStatus.FAILED.value,
        valid_pct * 100,
    )


@activity.defn
async def update_run_status_activity(run_id: str, status_value: str) -> None:
    """Set run status without touching eval_valid_pct (used by export-only workflows)."""
    from domain.models import RunStatus
    store = _get_run_store()
    store.update_status(run_id, RunStatus(status_value))


@activity.defn
async def save_gguf_path_activity(model_id: str, gguf_path: str) -> None:
    """Persist the storage key of the exported GGUF back to the model record."""
    store = _get_model_store()
    model = store.get(model_id)
    if model is None:
        activity.logger.warning("save_gguf_path: model %s not found — skipping", model_id)
        return

    from domain.models import TrainingModelConfig
    config = TrainingModelConfig(
        name=model.name,
        description=model.description,
        base_model=model.base_model,
        train_data=model.train_data,
        eval_data=model.eval_data,
        epochs=model.epochs,
        patience=model.patience,
        warmup_ratio=model.warmup_ratio,
        remote_backend=model.remote_backend,
        skip_generate=model.skip_generate,
        gguf_path=gguf_path,
        is_active=model.is_active,
    )
    store.update(model_id, config)
    activity.logger.info("Saved gguf_path=%s for model %s", gguf_path, model_id)
