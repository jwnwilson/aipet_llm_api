"""Kaggle-backed remote training adapter implementing RemoteTrainingPort."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Literal

from domain.models import RemoteTrainConfig
from domain.ports import RemoteTrainingPort

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
        self._project_root = Path(__file__).parents[3].resolve()

    # ------------------------------------------------------------------
    # RemoteTrainingPort
    # ------------------------------------------------------------------

    def submit(self, config: RemoteTrainConfig) -> str:
        dataset_slug = f"{config.experiment_name}-data"
        dataset_ref = f"{self._username}/{dataset_slug}"

        staging = self._work_dir / dataset_slug
        self._stage_dataset(config, staging, dataset_slug)
        self._push_dataset(staging)

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
        subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], check=True)

        return slug

    def status(self, run_id: str) -> Literal["pending", "running", "done", "failed"]:
        result = subprocess.run(
            ["kaggle", "kernels", "status", run_id],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout.lower()
        for keyword, canonical in _STATUS_MAP.items():
            if keyword in output:
                return canonical  # type: ignore[return-value]
        return "pending"

    def download(self, run_id: str, dest: Path) -> str:
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["kaggle", "kernels", "output", run_id, "-p", str(dest)],
            check=True,
        )
        archive = dest / "checkpoint.tar.gz"
        if archive.exists():
            with tarfile.open(archive) as tf:
                tf.extractall(dest)
            archive.unlink()
        return str(dest)

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
        try:
            subprocess.run(
                ["kaggle", "datasets", "create", "-p", str(staging), "--quiet"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            subprocess.run(
                ["kaggle", "datasets", "version", "-p", str(staging), "-m", "update", "--quiet"],
                check=True,
            )

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


def _replace_all(s: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s
