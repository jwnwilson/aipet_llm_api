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

_STAT_TO_ACTION: dict[str, Action] = {
    "hunger": Action.EAT,
    "tiredness": Action.SLEEP,
    "boredom": Action.PLAY,
    "social": Action.SOCIAL,
    "toilet": Action.TOILET,
}

_ACTION_REQUIRES_TYPE: dict[Action, list[str] | None] = {
    Action.EAT: ["bowl"],
    Action.SLEEP: ["bed"],
    Action.PLAY: ["toy"],
    Action.SOCIAL: ["player", "pet"],
    Action.TOILET: None,
    Action.IDLE: None,
    Action.EXPLORE: None,
}


# ---------------------------------------------------------------------------
# Random scene / stat generation
# ---------------------------------------------------------------------------


def _random_stats(rng: random.Random) -> PetStats:
    stats = {name: rng.uniform(0.0, 1.0) for name in ["hunger", "tiredness", "boredom", "social", "toilet"]}
    if rng.random() < 0.70:
        dominant = rng.choice(list(stats.keys()))
        stats[dominant] = rng.uniform(0.55, 1.0)
        if rng.random() < 0.5:
            for k in stats:
                if k != dominant:
                    stats[k] = rng.uniform(0.0, 0.45)
    return PetStats(**stats)


def _random_scene(rng: random.Random) -> SceneData:
    n_objects = rng.randint(0, 10)
    objects: list[SceneObject] = []
    for i in range(n_objects):
        obj_type = rng.choice(OBJECT_TYPES)
        distance = round(rng.uniform(1.0, 50.0), 2)
        objects.append(SceneObject(id=f"obj_{i}", type=obj_type, distance=distance))
    return SceneData(objects=objects, tick=rng.randint(0, 10_000))


def _random_request(rng: random.Random) -> InferenceRequest:
    return InferenceRequest(scene=_random_scene(rng), pet_stats=_random_stats(rng))


# ---------------------------------------------------------------------------
# Rule-based labeller
# ---------------------------------------------------------------------------


def label(request: InferenceRequest) -> InferenceResponse:
    """Assign a ground-truth action using deterministic rules.

    Priority: dominant stat → preferred action → closest matching object.
    Falls back to IDLE when no suitable object exists or all stats are low.
    """
    stats = request.pet_stats
    stat_values = {
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

    preferred_action = _STAT_TO_ACTION[dominant_stat]
    required_types = _ACTION_REQUIRES_TYPE[preferred_action]

    if required_types is None:
        return InferenceResponse(action=preferred_action, target_object_id=None, confidence=round(dominant_value, 4))

    candidates = [o for o in request.scene.objects if o.type in required_types]
    if not candidates:
        return InferenceResponse(action=Action.IDLE, target_object_id=None, confidence=round(dominant_value, 4))

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
    request = _random_request(rng)
    response = label(request)
    return {"prompt": build_prompt(request), "completion": response.model_dump_json()}


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
