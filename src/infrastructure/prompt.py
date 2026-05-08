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
    """Build a compact prompt string for the LLM from an InferenceRequest.

    Stats are sorted highest-first so the dominant need is immediately visible.
    The rule is stated explicitly so small models don't need to infer it.
    Scene objects are sorted nearest-first so the closest valid target is easy to read.
    """
    stats = request.pet_stats

    # Sort stats high → low so the dominant stat is always first.
    stat_dict = {
        "hunger": stats.hunger,
        "boredom": stats.boredom,
        "social": stats.social,
        "toilet": stats.toilet,
        "tiredness": stats.tiredness,
    }
    sorted_stats = sorted(stat_dict.items(), key=lambda x: -x[1])
    stats_parts = [
        f"{name}={value:.2f}" + (" (highest)" if i == 0 else "")
        for i, (name, value) in enumerate(sorted_stats)
    ]
    stats_str = ", ".join(stats_parts)

    # Sort objects nearest-first so the closest target is always at the front.
    sorted_objects = sorted(request.scene.objects, key=lambda o: o.distance)
    if sorted_objects:
        obj_parts = [f"{o.type}(id={o.id},dist={o.distance:.1f})" for o in sorted_objects]
        scene_str = ", ".join(obj_parts)
    else:
        scene_str = "empty"

    actions = _available_actions(request)
    actions_str = ", ".join(a.value for a in actions)

    prompt = (
        f"You are an AI pet brain. Choose the best action for the pet.\n"
        f"Stats (highest first): {stats_str}\n"
        f"Rule: choose the action that satisfies the highest stat. "
        f"If a target object is required, select the closest one.\n"
        f"Scene (nearest first): {scene_str}\n"
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
