"""Kaggle-backed remote training adapter implementing RemoteTrainingPort."""

from __future__ import annotations

import json
import os
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
    The Kaggle CLI (``kaggle``) must be installed and on PATH.

    Typical flow:
        1. Push ``data/`` as a versioned Kaggle Dataset.
        2. Render ``notebook_template.ipynb`` with the run config.
        3. Write ``kernel-metadata.json`` pointing at the dataset.
        4. Push the kernel — Kaggle queues it for GPU execution.
        5. Poll ``kaggle kernels status`` until done/failed.
        6. Pull the checkpoint archive via ``kaggle kernels output``.
    """

    def __init__(self, work_dir: Path | None = None) -> None:
        self._username = os.environ.get("KAGGLE_USERNAME", "")
        self._work_dir = work_dir or Path(".kaggle_kernels")

    # ------------------------------------------------------------------
    # RemoteTrainingPort
    # ------------------------------------------------------------------

    def submit(self, config: RemoteTrainConfig) -> str:
        data_dir = Path(config.train_data).parent

        # Step 1: push training data as a versioned Kaggle Dataset.
        subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(data_dir), "-m", config.experiment_name],
            check=True,
        )

        # Step 2: render the notebook template.
        kernel_dir = self._work_dir / config.experiment_name
        kernel_dir.mkdir(parents=True, exist_ok=True)

        template_path = Path(__file__).parent / "notebook_template.ipynb"
        template = template_path.read_text()
        config_json = json.dumps({
            "model": config.model,
            "epochs": config.epochs,
            "patience": config.patience,
            "warmup_ratio": config.warmup_ratio,
            "experiment_name": config.experiment_name,
        })
        rendered = template.replace("{{config}}", config_json)
        (kernel_dir / "notebook.ipynb").write_text(rendered)

        # Step 3: write kernel metadata.
        slug = f"{self._username}/{config.experiment_name}"
        dataset_ref = f"{self._username}/{config.experiment_name}-data"
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

        # Step 4: push the kernel.
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
