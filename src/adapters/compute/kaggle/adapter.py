"""Kaggle-backed remote training adapter implementing RemoteTrainingPort."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Literal

from domain.models import RemoteTrainConfig
from domain.ports import RemoteTrainingPort

def _kaggle_bin() -> str:
    found = shutil.which("kaggle")
    if found:
        return found
    candidate = Path(sys.executable).parent / "kaggle"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        "kaggle CLI not found in PATH or alongside Python interpreter. "
        "Install with: uv sync"
    )


_STATUS_MAP: dict[str, str] = {
    "complete": "done",
    "error": "failed",
    "failed": "failed",
    "running": "running",
    "queued": "pending",
    "cancelacknowledged": "failed",
}


class KaggleTrainingAdapter(RemoteTrainingPort):
    """RemoteTrainingPort implementation that submits training as a Kaggle kernel.

    Credentials are read from env vars ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``.
    No internet access is required on the Kaggle kernel — the project is built
    into a wheel locally and uploaded as part of the dataset.

    Typical flow:
        1. Build a wheel of the project and stage it alongside the .jsonl data
           files as a Kaggle Dataset (all flat files, no subdirectories).
        2. Render ``notebook_template.ipynb`` with the run config.
        3. Write ``kernel-metadata.json`` pointing at the dataset.
        4. Push the kernel — Kaggle queues it for GPU execution.
        5. Poll ``kaggle kernels status`` until done/failed.
        6. Pull the checkpoint archive via ``kaggle kernels output``.
    """

    def __init__(self, work_dir: Path | None = None) -> None:
        self._username = os.environ.get("KAGGLE_USERNAME", "")
        self._work_dir = work_dir or Path("models/kaggle_kernels")
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._project_root = Path(__file__).parents[4].resolve()

    # ------------------------------------------------------------------
    # RemoteTrainingPort
    # ------------------------------------------------------------------

    def submit(self, config: RemoteTrainConfig) -> str:
        dataset_slug = f"{config.experiment_name}-data"
        dataset_ref = f"{self._username}/{dataset_slug}"

        staging = self._work_dir / dataset_slug
        self._stage_dataset(config, staging, dataset_slug)
        self._push_dataset(staging)
        self._wait_for_dataset(dataset_ref)

        kernel_dir = self._work_dir / config.experiment_name
        kernel_dir.mkdir(parents=True, exist_ok=True)
        self._render_notebook(config, kernel_dir)

        slug = f"{self._username}/{config.experiment_name}"
        metadata = {
            "id": slug,
            "title": config.experiment_name,
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [dataset_ref],
        }
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        subprocess.run(
            [_kaggle_bin(), "kernels", "push", "-p", str(kernel_dir), "--accelerator", config.gpu_type],
            check=True,
        )

        return slug

    def status(self, run_id: str) -> Literal["pending", "running", "done", "failed"]:
        result = subprocess.run(
            [_kaggle_bin(), "kernels", "status", run_id],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout.lower()
        for keyword, canonical in _STATUS_MAP.items():
            if keyword in output:
                return canonical  # type: ignore[return-value]
        return "pending"

    def logs(self, run_id: str) -> str:
        """Return a progress string from the training sidecar file if available.

        The training notebook writes /kaggle/working/progress.json after each
        HF Trainer log step. We try to fetch it via ``kaggle kernels output``
        with a short timeout; whether Kaggle exposes working-dir files during
        execution depends on their API version. On failure we fall back to the
        plain kernel status line.
        """
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                subprocess.run(
                    [_kaggle_bin(), "kernels", "output", run_id, "-p", tmpdir, "--quiet"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                matches = list(Path(tmpdir).glob("**/progress.json"))
                if matches:
                    data = json.loads(matches[0].read_text())
                    step = data.get("step", "?")
                    max_steps = data.get("max_steps", "?")
                    epoch = data.get("epoch", "?")
                    elapsed = data.get("elapsed_s", "?")
                    parts = [f"step={step}/{max_steps}", f"epoch={epoch}", f"elapsed={elapsed}s"]
                    for key in ("loss", "eval_loss", "grad_norm"):
                        if key in data:
                            parts.append(f"{key}={data[key]:.4f}")
                    return "  ".join(parts)
        except Exception:
            pass

        result = subprocess.run(
            [_kaggle_bin(), "kernels", "status", run_id],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def download(self, run_id: str, dest: Path) -> str:
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [_kaggle_bin(), "kernels", "output", run_id, "-p", str(dest)],
            check=True,
        )
        archive = dest / "checkpoint.tar.gz"
        if archive.exists():
            with tarfile.open(archive) as tf:
                tf.extractall(dest)
            archive.unlink()
        # Find the directory that contains the HF model config
        for config_path in sorted(dest.rglob("config.json")):
            if config_path.read_text().find('"model_type"') != -1:
                return str(config_path.parent)
        return str(dest)

    def eval(self, run_id: str, eval_data: str) -> tuple[float, bool]:
        experiment_name = run_id.split("/")[-1]
        dataset_ref = f"{self._username}/{experiment_name}-data"

        eval_kernel_id = f"{experiment_name}-eval"
        eval_slug = f"{self._username}/{eval_kernel_id}"
        kernel_dir = self._work_dir / eval_kernel_id
        kernel_dir.mkdir(parents=True, exist_ok=True)

        self._render_eval_notebook(run_id, eval_data, experiment_name, kernel_dir)

        metadata = {
            "id": eval_slug,
            "title": eval_kernel_id,
            "code_file": "eval_notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [dataset_ref],
        }
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        subprocess.run(
            [_kaggle_bin(), "kernels", "push", "-p", str(kernel_dir), "--accelerator", "NvidiaTeslaT4"],
            check=True,
        )

        started = time.time()
        while True:
            status = self.status(eval_slug)
            elapsed = int(time.time() - started)
            print(f"Eval status: {status} elapsed={elapsed}s", flush=True)
            if status == "done":
                break
            if status == "failed":
                raise RuntimeError(f"Eval kernel failed: {eval_slug}")
            time.sleep(30)

        result_dir = self._work_dir / f"{eval_kernel_id}-output"
        result_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [_kaggle_bin(), "kernels", "output", eval_slug, "-p", str(result_dir)],
            check=True,
        )

        result_file = result_dir / "eval_result.json"
        if not result_file.exists():
            raise RuntimeError(f"eval_result.json missing from eval kernel output at {result_dir}")
        result = json.loads(result_file.read_text())
        return result["valid_pct"], result["passed"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stage_dataset(
        self, config: RemoteTrainConfig, staging: Path, dataset_slug: str
    ) -> None:
        """Build a project wheel and stage it with the training data for Kaggle upload.

        All files are placed flat in the staging directory (no subdirectories) to
        avoid Kaggle CLI upload reliability issues with nested directory structures.
        """
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        # Build a wheel of the project and copy it into staging
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(staging)],
            cwd=str(self._project_root),
            check=True,
        )

        # Copy flat .jsonl training data files
        train_data = Path(config.train_data)
        if not train_data.is_absolute():
            train_data = self._project_root / train_data
        for jsonl in train_data.parent.glob("*.jsonl"):
            shutil.copy2(jsonl, staging / jsonl.name)

        meta = {
            "title": f"{config.experiment_name} Training Data",
            "id": f"{self._username}/{dataset_slug}",
            "licenses": [{"name": "CC0-1.0"}],
        }
        (staging / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    def _push_dataset(self, staging: Path) -> None:
        """Create the dataset on first run; add a new version on subsequent runs."""
        staged_files = [f.name for f in staging.iterdir() if f.is_file()]
        print(f"Staged files for upload: {staged_files}", flush=True)

        create_result = subprocess.run(
            [_kaggle_bin(), "datasets", "create", "-p", str(staging)],
            capture_output=True, text=True,
        )
        create_output = (create_result.stdout + create_result.stderr).strip()
        dataset_exists = create_result.returncode != 0 or "error" in create_output.lower()

        if not dataset_exists:
            print(f"Dataset created: {create_output}", flush=True)
        else:
            print(f"Dataset exists, uploading new version … ({create_output})", flush=True)
            version_result = subprocess.run(
                [_kaggle_bin(), "datasets", "version", "-p", str(staging), "-m", "update"],
                capture_output=True, text=True,
            )
            version_output = (version_result.stdout + version_result.stderr).strip()
            if version_result.returncode != 0 or "error" in version_output.lower():
                raise RuntimeError(f"Dataset version upload failed:\n{version_output}")
            print(f"Dataset version uploaded: {version_output}", flush=True)

    def _wait_for_dataset(self, dataset_ref: str, timeout: int = 300, interval: int = 15) -> None:
        """Poll via the Kaggle Python API until a .whl is visible in the dataset."""
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        print(f"Polling for .whl in dataset {dataset_ref} …", flush=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                response = api.dataset_list_files(dataset_ref)
                if response and getattr(response, "files", None):
                    names = [f.name for f in response.files]
                    whl = [n for n in names if n.endswith(".whl")]
                    if whl:
                        print(f"Dataset ready — {whl[0]} visible.", flush=True)
                        return
                    print(f"  visible files: {names[:6]} — no .whl yet …", flush=True)
                else:
                    print("  no files visible yet …", flush=True)
            except Exception as exc:
                print(f"  poll error: {exc}", flush=True)
            time.sleep(interval)

        print(f"WARNING: .whl not confirmed in {dataset_ref} after {timeout}s — proceeding anyway.", flush=True)

    def _render_notebook(self, config: RemoteTrainConfig, kernel_dir: Path) -> None:
        template_path = Path(__file__).parent / "notebook_template.ipynb"
        notebook = json.loads(template_path.read_text())

        config_repr = repr({
            "model": config.model,
            "epochs": config.epochs,
            "patience": config.patience,
            "warmup_ratio": config.warmup_ratio,
            "experiment_name": config.experiment_name,
        })

        replacements = {"{{config}}": config_repr}
        for cell in notebook["cells"]:
            src = cell["source"]
            if isinstance(src, str):
                cell["source"] = _replace_all(src, replacements)
            else:
                cell["source"] = [_replace_all(line, replacements) for line in src]

        (kernel_dir / "notebook.ipynb").write_text(json.dumps(notebook, indent=1))

    def _render_eval_notebook(
        self, training_run_id: str, eval_data: str, experiment_name: str, kernel_dir: Path
    ) -> None:
        template_path = Path(__file__).parent / "eval_notebook_template.ipynb"
        notebook = json.loads(template_path.read_text())

        config_repr = repr({
            "training_run_id": training_run_id,
            "experiment_name": experiment_name,
            "eval_data_file": Path(eval_data).name,
        })

        replacements = {"{{config}}": config_repr}
        for cell in notebook["cells"]:
            src = cell["source"]
            if isinstance(src, str):
                cell["source"] = _replace_all(src, replacements)
            else:
                cell["source"] = [_replace_all(line, replacements) for line in src]

        (kernel_dir / "eval_notebook.ipynb").write_text(json.dumps(notebook, indent=1))


def _replace_all(s: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s
