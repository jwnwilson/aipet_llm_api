"""Local simulation of the Kaggle notebook cells.

Runs all cells in order against a local staging directory so errors can be
caught before pushing to Kaggle. Driven by env vars set by the Makefile:

    KAGGLE_INPUT_BASE  path to the local fake /kaggle/input/<dataset-slug>/
    EXPERIMENT         experiment name (default: experiment-01)
    MODEL              HF model ID (default: HuggingFaceTB/SmolLM2-360M)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

EXPERIMENT = os.environ.get("EXPERIMENT", "experiment-01")
MODEL = os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-360M")
INPUT_BASE = Path(os.environ["KAGGLE_INPUT_BASE"])

CONFIG = {
    "experiment_name": EXPERIMENT,
    "model": MODEL,
    "epochs": 1,
    "patience": 2,
    "warmup_ratio": 0.05,
}

# ---------------------------------------------------------------------------
# Cell 1 — install wheel
# ---------------------------------------------------------------------------
print(f"\n=== [cell-1] install wheel from {INPUT_BASE} ===")
wheels = list(INPUT_BASE.glob("*.whl"))
if not wheels:
    raise FileNotFoundError(
        f"No .whl found in {INPUT_BASE} — wheel build or staging failed."
    )
subprocess.run(["uv", "pip", "install", "-q", str(wheels[0])], check=True)
print(f"Installed {wheels[0].name}")

# ---------------------------------------------------------------------------
# Cell 2 — copy training data
# ---------------------------------------------------------------------------
print("\n=== [cell-2] copy training data ===")
data_dst = Path("data")
data_dst.mkdir(exist_ok=True)
for jsonl in INPUT_BASE.glob("*.jsonl"):
    shutil.copy(jsonl, data_dst / jsonl.name)
    print(f"Copied {jsonl.name} ({jsonl.stat().st_size // 1000} KB)")

# ---------------------------------------------------------------------------
# Cell 3 — training (dry-run locally)
# ---------------------------------------------------------------------------
print("\n=== [cell-3] training (dry-run) ===")
cmd = [
    sys.executable, "-m", "cli.train",
    "--dry-run",
    "--model", CONFIG["model"],
    "--epochs", str(CONFIG["epochs"]),
    "--patience", str(CONFIG["patience"]),
    "--warmup-ratio", str(CONFIG["warmup_ratio"]),
    "--train-data", "data/train.jsonl",
    "--eval-data", "data/eval.jsonl",
    "--output-dir", "/tmp/aipet-test-checkpoint",
]
subprocess.run(cmd, check=True)

print("\n=== All notebook cells passed locally ===")
