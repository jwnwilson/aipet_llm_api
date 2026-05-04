from __future__ import annotations

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
    action: Action
    target_object_id: str | None = None
    confidence: float | None = None


