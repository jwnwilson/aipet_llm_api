from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from domain.actions import Action


class PetStats(BaseModel):
    hunger: float = Field(..., ge=0.0, le=1.0)
    boredom: float = Field(..., ge=0.0, le=1.0)
    social: float = Field(..., ge=0.0, le=1.0)
    toilet: float = Field(..., ge=0.0, le=1.0)
    tiredness: float = Field(..., ge=0.0, le=1.0)


class SceneObject(BaseModel):
    id: str
    type: Literal["bowl", "bed", "toy", "player", "pet"]
    distance: float


class SceneData(BaseModel):
    objects: list[SceneObject]
    tick: int


class InferenceRequest(BaseModel):
    scene: SceneData
    pet_stats: PetStats


class InferenceResponse(BaseModel):
    stat: str | None = None
    action: Action
    target_object_id: str | None = None
    confidence: float | None = None


class RemoteTrainConfig(BaseModel):
    model: str
    train_data: str
    eval_data: str
    epochs: int
    patience: int
    warmup_ratio: float
    experiment_name: str
    gpu_type: str = "NvidiaTeslaT4"


class TrainingModelConfig(BaseModel):
    name: str
    description: str = ""
    base_model: str = "HuggingFaceTB/SmolLM2-360M"
    train_data: str = "data/train.jsonl"
    eval_data: str = "data/eval.jsonl"
    epochs: int = 5
    patience: int = 3
    warmup_ratio: float = 0.05
    remote_backend: str = "local"
    skip_generate: bool = False
    gguf_path: str = ""
    is_active: bool = False


class TrainingModel(TrainingModelConfig):
    id: str
    created_at: datetime
    updated_at: datetime


class RunStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    TRAINING = "training"
    EVALUATING = "evaluating"
    EXPORTING = "exporting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunConfig(BaseModel):
    model_id: str
    workflow_id: str


class RunRecord(RunConfig):
    id: str
    status: RunStatus
    eval_valid_pct: float | None = None
    progress: float | None = None
    progress_detail: str | None = None
    created_at: datetime
    updated_at: datetime


