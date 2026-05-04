"""Dataset generation logic — pure functions, no I/O side-effects except writing files."""

from __future__ import annotations

import json
import random
from pathlib import Path

from domain.actions import Action
from domain.models import (
    InferenceRequest,
    InferenceResponse,
    PetStats,
    SceneData,
    SceneObject,
)
from infrastructure.prompt import build_prompt

SEED = 42
TRAIN_SIZE = 2000
EVAL_SIZE = 200

OBJECT_TYPES = ["bowl", "bed", "toy", "player", "pet"]
STAT_NAMES = ["hunger", "tiredness", "boredom", "social", "toilet"]

# Each stat maps to one or more valid actions; tick parity selects between equivalents.
_STAT_TO_ACTIONS: dict[str, list[Action]] = {
    "hunger": [Action.EAT, Action.DRINK],
    "tiredness": [Action.SLEEP],
    "boredom": [Action.PLAY, Action.FETCH],
    "social": [Action.SOCIAL, Action.FOLLOW],
    "toilet": [Action.TOILET],
}

# Scene object types required to satisfy each stat. None = no object needed.
_STAT_REQUIRES_TYPES: dict[str, list[str] | None] = {
    "hunger": ["bowl"],
    "tiredness": ["bed"],
    "boredom": ["toy"],
    "social": ["player", "pet"],
    "toilet": None,
}


# ---------------------------------------------------------------------------
# Random scene / stat generation
# ---------------------------------------------------------------------------


def _stats_with_dominant(rng: random.Random, dominant: str) -> PetStats:
    """One stat is high (0.70–1.0); all others are low (0.0–0.35)."""
    values = {name: rng.uniform(0.0, 0.35) for name in STAT_NAMES}
    values[dominant] = rng.uniform(0.70, 1.0)
    return PetStats(**values)


def _scene_with_required(rng: random.Random, required_types: list[str] | None) -> SceneData:
    """Scene that guarantees at least one object of a required type is present."""
    objects: list[SceneObject] = []

    if required_types:
        req_type = rng.choice(required_types)
        objects.append(SceneObject(
            id="obj_0",
            type=req_type,
            distance=round(rng.uniform(1.0, 50.0), 1),
        ))

    for i in range(1, rng.randint(1, 8)):
        objects.append(SceneObject(
            id=f"obj_{i}",
            type=rng.choice(OBJECT_TYPES),
            distance=round(rng.uniform(1.0, 50.0), 1),
        ))

    rng.shuffle(objects)
    return SceneData(objects=objects, tick=rng.randint(0, 10_000))


def _scene_without_required(rng: random.Random, excluded_types: list[str]) -> SceneData:
    """Scene that intentionally omits excluded types (teaches fallback behaviour)."""
    allowed = [t for t in OBJECT_TYPES if t not in excluded_types]
    n = rng.randint(0, 5)
    objects = [
        SceneObject(
            id=f"obj_{i}",
            type=rng.choice(allowed) if allowed else "bowl",
            distance=round(rng.uniform(1.0, 50.0), 1),
        )
        for i in range(n)
    ]
    return SceneData(objects=objects, tick=rng.randint(0, 10_000))


# ---------------------------------------------------------------------------
# Rule-based labeller
# ---------------------------------------------------------------------------


def label(request: InferenceRequest) -> InferenceResponse:
    """Assign a ground-truth action using deterministic rules.

    Priority: dominant stat (≥ 0.5) → preferred action → closest matching object.
    Tick parity varies between equivalent action pairs (EAT/DRINK, PLAY/FETCH, SOCIAL/FOLLOW).
    Falls back to IDLE/EXPLORE when no suitable object exists or all stats are low.
    """
    stats = request.pet_stats
    stat_values: dict[str, float] = {
        "hunger": stats.hunger,
        "tiredness": stats.tiredness,
        "boredom": stats.boredom,
        "social": stats.social,
        "toilet": stats.toilet,
    }

    dominant_stat = max(stat_values, key=lambda k: stat_values[k])
    dominant_value = stat_values[dominant_stat]

    if dominant_value < 0.5:
        action = Action.IDLE if request.scene.tick % 2 == 0 else Action.EXPLORE
        return InferenceResponse(action=action, target_object_id=None, confidence=round(1.0 - dominant_value, 4))

    actions = _STAT_TO_ACTIONS[dominant_stat]
    preferred_action = actions[request.scene.tick % len(actions)]
    required_types = _STAT_REQUIRES_TYPES[dominant_stat]

    if required_types is None:
        return InferenceResponse(action=preferred_action, target_object_id=None, confidence=round(dominant_value, 4))

    candidates = [o for o in request.scene.objects if o.type in required_types]
    if not candidates:
        action = Action.IDLE if request.scene.tick % 2 == 0 else Action.EXPLORE
        return InferenceResponse(action=action, target_object_id=None, confidence=round(dominant_value, 4))

    closest = min(candidates, key=lambda o: o.distance)
    return InferenceResponse(
        action=preferred_action,
        target_object_id=closest.id,
        confidence=round(dominant_value, 4),
    )


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def make_example(rng: random.Random) -> dict[str, str]:
    """Generate one labelled example.

    75 % of examples: dominant stat is high and the required scene object is present
    so the correct targeted action is always reachable.
    25 % of examples: required object is absent to teach IDLE/EXPLORE fallback behaviour.
    Toilet stat never needs a scene object, so it is always a 'present' example.
    """
    dominant = rng.choice(STAT_NAMES)
    stats = _stats_with_dominant(rng, dominant)
    required_types = _STAT_REQUIRES_TYPES[dominant]

    if required_types is None or rng.random() < 0.75:
        scene = _scene_with_required(rng, required_types)
    else:
        scene = _scene_without_required(rng, required_types)

    request = InferenceRequest(scene=scene, pet_stats=stats)
    response = label(request)
    return {"prompt": build_prompt(request), "completion": response.model_dump_json(exclude_none=True)}


def generate_examples(n: int, rng: random.Random) -> list[dict[str, str]]:
    return [make_example(rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, examples: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def validate_jsonl(path: Path) -> dict[str, int]:
    """Parse every line and validate prompt/completion fields. Returns total/valid/invalid counts."""
    total = valid = invalid = 0
    errors: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            try:
                obj = json.loads(raw)
                InferenceResponse.model_validate_json(obj["completion"])
                assert "Stats:" in obj["prompt"], "prompt missing Stats section"
                valid += 1
            except Exception as exc:
                invalid += 1
                errors.append(f"  line {line_no}: {exc}")

    if errors:
        for msg in errors[:10]:
            print(msg)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return {"total": total, "valid": valid, "invalid": invalid}


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def generate(
    data_dir: Path = Path("data"),
    train_size: int = TRAIN_SIZE,
    eval_size: int = EVAL_SIZE,
    seed: int = SEED,
) -> bool:
    """Generate and validate training + eval datasets. Returns True if all examples are valid."""
    rng = random.Random(seed)

    print(f"Generating {train_size} training examples …")
    train_path = data_dir / "train.jsonl"
    write_jsonl(train_path, generate_examples(train_size, rng))
    print(f"  Written → {train_path}")

    print(f"Generating {eval_size} eval examples …")
    eval_path = data_dir / "eval.jsonl"
    write_jsonl(eval_path, generate_examples(eval_size, rng))
    print(f"  Written → {eval_path}")

    print("\nValidating train.jsonl …")
    train_summary = validate_jsonl(train_path)
    print(f"  total={train_summary['total']}  valid={train_summary['valid']}  invalid={train_summary['invalid']}")

    print("Validating eval.jsonl …")
    eval_summary = validate_jsonl(eval_path)
    print(f"  total={eval_summary['total']}  valid={eval_summary['valid']}  invalid={eval_summary['invalid']}")

    all_valid = train_summary["invalid"] == 0 and eval_summary["invalid"] == 0
    if all_valid:
        print("\nAll examples valid.")
    else:
        total_invalid = train_summary["invalid"] + eval_summary["invalid"]
        print(f"\n{total_invalid} invalid example(s) found — review errors above.")
    return all_valid
