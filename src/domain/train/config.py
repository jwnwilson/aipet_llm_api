"""Lightweight constants for the training pipeline.

Kept in a separate module so that activities.py and workflows.py can import
default values without triggering the torch / transformers / peft imports that
live in trainer.py.  trainer.py re-exports these for backward compatibility.
"""

MAX_LENGTH = 512
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"
DEFAULT_TRAIN_DATA = "data/train.jsonl"
DEFAULT_EVAL_DATA = "data/eval.jsonl"
DEFAULT_OUTPUT_DIR = "models/checkpoints"
DEFAULT_EPOCHS = 5
DEFAULT_PATIENCE = 3
DEFAULT_WARMUP_RATIO = 0.05
