"""Prompt builder and response parser for the LLM inference adapter."""

from __future__ import annotations

import json
import re

from domain.actions import Action
from domain.models import InferenceRequest, InferenceResponse

# Actions that are always available regardless of scene contents.
_ALWAYS_AVAILABLE = {Action.TOILET, Action.IDLE, Action.EXPLORE}

# Mapping from action → object types that must be present in the scene.
_ACTION_REQUIRES: dict[Action, set[str]] = {
    Action.EAT: {"bowl"},
    Action.DRINK: {"bowl"},
    Action.PLAY: {"toy"},
    Action.FETCH: {"toy"},
    Action.SLEEP: {"bed"},
    Action.SOCIAL: {"player", "pet"},
    Action.FOLLOW: {"player", "pet"},
}


def _available_actions(request: InferenceRequest) -> list[Action]:
    """Return actions available given the objects present in the scene."""
    present_types = {obj.type for obj in request.scene.objects}
    available: list[Action] = []
    for action in Action:
        if action in _ALWAYS_AVAILABLE:
            available.append(action)
        elif action in _ACTION_REQUIRES:
            if _ACTION_REQUIRES[action] & present_types:
                available.append(action)
    return available


def build_prompt(request: InferenceRequest) -> str:
    """Build a compact prompt string for the LLM from an InferenceRequest."""
    stats = request.pet_stats
    objects = request.scene.objects

    stats_str = (
        f"hunger={stats.hunger:.2f} boredom={stats.boredom:.2f} "
        f"social={stats.social:.2f} toilet={stats.toilet:.2f} "
        f"tiredness={stats.tiredness:.2f}"
    )

    if objects:
        obj_parts = [f"{o.type}(id={o.id},dist={o.distance:.1f})" for o in objects]
        scene_str = ", ".join(obj_parts)
    else:
        scene_str = "empty"

    actions = _available_actions(request)
    actions_str = ", ".join(a.value for a in actions)

    prompt = (
        f"You are an AI pet brain. Choose the best action for the pet.\n"
        f"Stats: {stats_str}\n"
        f"Scene: {scene_str}\n"
        f"Available actions: {actions_str}\n"
        f"Respond with JSON only."
    )
    return prompt


def parse_response(raw: str) -> InferenceResponse:
    """Extract and validate an InferenceResponse JSON object from raw LLM output.

    Raises ValueError if no valid JSON block is found or validation fails.
    """
    # Try to find a JSON object (possibly surrounded by extra text).
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {raw!r}")

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in response: {exc}") from exc

    try:
        return InferenceResponse.model_validate(data)
    except Exception as exc:
        raise ValueError(f"Response does not match InferenceResponse schema: {exc}") from exc
