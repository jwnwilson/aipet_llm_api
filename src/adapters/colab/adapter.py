"""Google Colab–backed remote training adapter implementing RemoteTrainingPort."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Literal

from domain.models import RemoteTrainConfig
from domain.ports import RemoteTrainingPort

_DRIVE_ROOT_FOLDER = "ColabTraining"


class ColabTrainingAdapter(RemoteTrainingPort):
    """RemoteTrainingPort implementation that submits training as a Colab notebook.

    Credentials are read from GOOGLE_APPLICATION_CREDENTIALS (service account JSON)
    or from Application Default Credentials (gcloud auth application-default login).

    Typical flow:
        1. Build a project wheel and stage it alongside the .jsonl data files.
        2. Upload staged files to a Google Drive folder.
        3. Render notebook_template.ipynb with the run config and Drive folder ID.
        4. Upload the notebook to Drive and print the Colab URL.
        5. User opens the URL and runs the notebook (or it runs via Colab API).
        6. Notebook writes status.txt sentinel back to Drive on start/end.
        7. Poll Drive for status.txt; download checkpoint.tar.gz when done.
    """

    def __init__(self, work_dir: Path | None = None) -> None:
        self._work_dir = work_dir or Path("models/colab_runs")
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._project_root = Path(__file__).parents[3].resolve()
        self._drive = self._build_drive_client()

    # ------------------------------------------------------------------
    # RemoteTrainingPort
    # ------------------------------------------------------------------

    def submit(self, config: RemoteTrainConfig) -> str:
        staging = self._work_dir / config.experiment_name
        self._stage_files(config, staging)

        folder_id = self._get_or_create_folder(config.experiment_name)
        self._upload_directory(staging, folder_id)

        notebook_path = self._render_notebook(config, folder_id, staging)
        notebook_id = self._upload_file(notebook_path, folder_id, "notebook.ipynb")

        colab_url = f"https://colab.research.google.com/drive/{notebook_id}"
        print(f"\n  Open in Colab and click 'Run All':\n  {colab_url}\n")

        return folder_id

    def status(self, run_id: str) -> Literal["pending", "running", "done", "failed"]:
        from googleapiclient.http import MediaIoBaseDownload

        file_id = self._find_file(run_id, "status.txt")
        if file_id is None:
            return "pending"

        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(
            buf, self._drive.files().get_media(fileId=file_id)
        )
        done = False
        while not done:
            _, done = downloader.next_chunk()
        raw = buf.getvalue().decode().strip().lower()
        return {"pending": "pending", "running": "running", "done": "done", "failed": "failed"}.get(raw, "pending")  # type: ignore[return-value]

    def download(self, run_id: str, dest: Path) -> str:
        from googleapiclient.http import MediaIoBaseDownload

        dest.mkdir(parents=True, exist_ok=True)
        file_id = self._find_file(run_id, "checkpoint.tar.gz")
        if file_id is None:
            raise FileNotFoundError("checkpoint.tar.gz not found in Drive folder")

        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(
            buf, self._drive.files().get_media(fileId=file_id)
        )
        done = False
        while not done:
            _, done = downloader.next_chunk()

        archive = dest / "checkpoint.tar.gz"
        archive.write_bytes(buf.getvalue())
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
        archive.unlink()
        return str(dest)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_drive_client(self):
        import google.auth
        from googleapiclient.discovery import build

        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=["https://www.googleapis.com/auth/drive"],
            )
        else:
            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/drive"]
            )
        return build("drive", "v3", credentials=creds)

    def _stage_files(self, config: RemoteTrainConfig, staging: Path) -> None:
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(staging)],
            cwd=str(self._project_root),
            check=True,
        )

        train_data = Path(config.train_data)
        if not train_data.is_absolute():
            train_data = self._project_root / train_data
        for jsonl in train_data.parent.glob("*.jsonl"):
            shutil.copy2(jsonl, staging / jsonl.name)

    def _get_or_create_folder(self, experiment_name: str) -> str:
        root_id = self._get_or_create_root_folder()
        query = (
            f"name='{experiment_name}' and '{root_id}' in parents and "
            "mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
        results = self._drive.files().list(q=query, fields="files(id)").execute()
        files = results.get("files", [])
        if files:
            return files[0]["id"]
        meta = {
            "name": experiment_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [root_id],
        }
        return self._drive.files().create(body=meta, fields="id").execute()["id"]

    def _get_or_create_root_folder(self) -> str:
        query = (
            f"name='{_DRIVE_ROOT_FOLDER}' and "
            "mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
        results = self._drive.files().list(q=query, fields="files(id)").execute()
        files = results.get("files", [])
        if files:
            return files[0]["id"]
        meta = {
            "name": _DRIVE_ROOT_FOLDER,
            "mimeType": "application/vnd.google-apps.folder",
        }
        return self._drive.files().create(body=meta, fields="id").execute()["id"]

    def _upload_directory(self, staging: Path, folder_id: str) -> None:
        for path in staging.iterdir():
            if path.is_file():
                self._upload_file(path, folder_id, path.name)

    def _upload_file(self, path: Path, folder_id: str, name: str) -> str:
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(str(path), resumable=True)
        existing_id = self._find_file(folder_id, name)
        if existing_id:
            self._drive.files().update(fileId=existing_id, media_body=media).execute()
            return existing_id
        meta = {"name": name, "parents": [folder_id]}
        result = self._drive.files().create(body=meta, media_body=media, fields="id").execute()
        return result["id"]

    def _find_file(self, folder_id: str, name: str) -> str | None:
        query = f"name='{name}' and '{folder_id}' in parents and trashed=false"
        results = self._drive.files().list(q=query, fields="files(id)").execute()
        files = results.get("files", [])
        return files[0]["id"] if files else None

    def _render_notebook(
        self, config: RemoteTrainConfig, folder_id: str, staging: Path
    ) -> Path:
        template_path = Path(__file__).parent / "notebook_template.ipynb"
        notebook = json.loads(template_path.read_text())

        config_repr = repr({
            "model": config.model,
            "epochs": config.epochs,
            "patience": config.patience,
            "warmup_ratio": config.warmup_ratio,
            "experiment_name": config.experiment_name,
        })
        replacements = {"{{config}}": config_repr, "{{folder_id}}": folder_id}

        for cell in notebook["cells"]:
            src = cell["source"]
            if isinstance(src, str):
                cell["source"] = _replace_all(src, replacements)
            else:
                cell["source"] = [_replace_all(line, replacements) for line in src]

        out = staging / "notebook.ipynb"
        out.write_text(json.dumps(notebook, indent=1))
        return out


def _replace_all(s: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s
