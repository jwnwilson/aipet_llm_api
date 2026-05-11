"""SSH-backed remote training adapter implementing RemoteTrainingPort."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Literal

from domain.models import RemoteTrainConfig
from domain.ports import RemoteTrainingPort


class SshTrainingAdapter(RemoteTrainingPort):
    """RemoteTrainingPort implementation that runs training on a remote machine via SSH.

    Configuration is read from env vars:
        REMOTE_HOST      — hostname or IP of the remote machine
        REMOTE_USER      — SSH username
        REMOTE_KEY_PATH  — path to the private key file (optional; omit to use agent)
        REMOTE_WORK_DIR  — working directory on the remote host (default: ~/aipet)

    The training process is launched inside a ``screen`` session so it survives
    SSH disconnects.  ``rsync`` transfers data and the source tree; the checkpoint
    is synced back to the local ``dest`` directory after the job completes.
    """

    def __init__(self) -> None:
        self._host = os.environ.get("REMOTE_HOST", "")
        self._user = os.environ.get("REMOTE_USER", "")
        self._key = os.environ.get("REMOTE_KEY_PATH", "")
        self._work_dir = os.environ.get("REMOTE_WORK_DIR", "~/aipet")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ssh_args(self) -> list[str]:
        args = ["ssh"]
        if self._key:
            args += ["-i", self._key]
        args += ["-o", "StrictHostKeyChecking=no", f"{self._user}@{self._host}"]
        return args

    def _rsync_args(self) -> list[str]:
        args = ["rsync", "-az", "--delete"]
        if self._key:
            args += ["-e", f"ssh -i {self._key} -o StrictHostKeyChecking=no"]
        return args

    # ------------------------------------------------------------------
    # RemoteTrainingPort
    # ------------------------------------------------------------------

    def submit(self, config: RemoteTrainConfig) -> str:
        remote = f"{self._user}@{self._host}"

        # Step 1: sync source tree and data.
        subprocess.run(
            self._rsync_args() + ["src/", f"{remote}:{self._work_dir}/src/"],
            check=True,
        )
        data_dir = str(Path(config.train_data).parent) + "/"
        subprocess.run(
            self._rsync_args() + [data_dir, f"{remote}:{self._work_dir}/data/"],
            check=True,
        )

        # Step 2: start training in a detached screen session.
        session = f"aipet-{config.experiment_name}"
        train_cmd = (
            f"cd {self._work_dir} && "
            f"uv run python -m src.cli.train"
            f" --model {config.model}"
            f" --epochs {config.epochs}"
            f" --patience {config.patience}"
            f" --warmup-ratio {config.warmup_ratio}"
            f" > train.log 2>&1"
        )
        subprocess.run(
            self._ssh_args() + [f"screen -dmS {session} bash -c '{train_cmd}'"],
            check=True,
        )
        return session

    def status(self, run_id: str) -> Literal["pending", "running", "done", "failed"]:
        # Check whether the screen session is still alive.
        result = subprocess.run(
            self._ssh_args() + [f"screen -list {run_id}"],
            capture_output=True,
            text=True,
        )
        if run_id in result.stdout:
            return "running"

        # Session ended — check for checkpoint directory as success indicator.
        check = subprocess.run(
            self._ssh_args() + [f"test -d {self._work_dir}/models/checkpoints && echo exists"],
            capture_output=True,
            text=True,
        )
        if "exists" in check.stdout:
            return "done"
        return "failed"

    def logs(self, run_id: str) -> str:  # noqa: ARG002
        result = subprocess.run(
            self._ssh_args() + [f"tail -n 50 {self._work_dir}/train.log"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def download(self, run_id: str, dest: Path) -> str:
        dest.mkdir(parents=True, exist_ok=True)
        remote = f"{self._user}@{self._host}"
        subprocess.run(
            self._rsync_args() + [
                f"{remote}:{self._work_dir}/models/checkpoints/",
                str(dest) + "/",
            ],
            check=True,
        )
        return str(dest)

    def eval(self, run_id: str, eval_data: str) -> tuple[float, bool]:
        """Run evaluation on the remote machine and return ``(valid_pct, passed)``.

        Syncs the eval dataset to the remote, runs ``src.cli.evaluate`` there,
        and parses the "Valid: N/M (P%)" summary line from stdout.
        """
        remote = f"{self._user}@{self._host}"

        # Sync eval data to remote.
        eval_path = Path(eval_data)
        subprocess.run(
            self._rsync_args() + [str(eval_path), f"{remote}:{self._work_dir}/{eval_data}"],
            check=True,
        )

        # Run evaluate CLI on the remote checkpoint.
        result = subprocess.run(
            self._ssh_args() + [
                f"cd {self._work_dir} && "
                f"uv run python -m src.cli.evaluate"
                f" --checkpoint models/checkpoints"
                f" --eval-data {eval_data}"
            ],
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr
        valid_pct = self._parse_valid_pct(output)
        if valid_pct is None:
            raise RuntimeError(f"Could not parse eval output from remote:\n{output}")

        passed = valid_pct >= 0.95
        return valid_pct, passed

    @staticmethod
    def _parse_valid_pct(output: str) -> float | None:
        for line in output.splitlines():
            if line.startswith("Valid:") and "(" in line and "%)" in line:
                try:
                    return float(line.split("(")[1].split("%")[0].strip()) / 100.0
                except (IndexError, ValueError):
                    pass
        return None
