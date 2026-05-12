"""Seed the database with default training model configurations."""

from __future__ import annotations

import os

from adapters.database.engine import make_engine, run_migrations
from adapters.database.model_store import SQLAlchemyModelStore
from domain.models import TrainingModelConfig

_DEFAULT_MODELS: list[TrainingModelConfig] = [
    TrainingModelConfig(
        name="smollm2-360m-local",
        description="SmolLM2 360M fine-tuned locally — fast iteration, no GPU required",
        base_model="HuggingFaceTB/SmolLM2-360M",
        train_data="data/train.jsonl",
        eval_data="data/eval.jsonl",
        epochs=5,
        patience=3,
        warmup_ratio=0.05,
        remote_backend="local",
        skip_generate=False,
    ),
    TrainingModelConfig(
        name="smollm2-360m-kaggle",
        description="SmolLM2 360M fine-tuned on Kaggle T4 GPU — no local GPU needed",
        base_model="HuggingFaceTB/SmolLM2-360M",
        train_data="data/train.jsonl",
        eval_data="data/eval.jsonl",
        epochs=5,
        patience=3,
        warmup_ratio=0.05,
        remote_backend="kaggle",
        skip_generate=False,
    ),
    TrainingModelConfig(
        name="smollm2-1.7b-runpod",
        description="SmolLM2 1.7B on RunPod RTX 3090 — higher quality, requires RUNPOD_API_KEY + AWS",
        base_model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        train_data="data/train.jsonl",
        eval_data="data/eval.jsonl",
        epochs=3,
        patience=2,
        warmup_ratio=0.05,
        remote_backend="runpod",
        skip_generate=False,
    ),
]


def main() -> None:
    url = os.environ.get("DATABASE_URL", "sqlite:///data/aipet.db")
    engine = make_engine(url)
    run_migrations(engine)
    store = SQLAlchemyModelStore(engine)

    existing = {m.name for m in store.list()}
    created = 0
    for config in _DEFAULT_MODELS:
        if config.name in existing:
            print(f"  skip  {config.name} (already exists)")
            continue
        store.create(config)
        print(f"  added {config.name}")
        created += 1

    print(f"\nDone: {created} created, {len(existing)} already existed.")


if __name__ == "__main__":
    main()
