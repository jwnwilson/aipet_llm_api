"""Generate a labelled (scene + pet stats) → action dataset for fine-tuning.

Usage:
    uv run python scripts/generate_dataset.py

Outputs:
    data/train.jsonl  — 2000 training examples
    data/eval.jsonl   — 200 eval examples

Each line is JSON: {"prompt": "...", "completion": "..."}
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.actions import Action
from src.domain.models import (
    InferenceRequest,
    InferenceResponse,
    PetStats,
    SceneData,
    SceneObject,
)
from src.infrastructure.prompt import build_prompt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
TRAIN_SIZE = 2000
EVAL_SIZE = 200

OBJECT_TYPES = ["bowl", "bed", "toy", "player", "pet"]

# Mapping from stat name → preferred action (if the required object is present).
_STAT_TO_ACTION: dict[str, Action] = {
    "hunger": Action.EAT,
    "tiredness": Action.SLEEP,
    "boredom": Action.PLAY,
    "social": Action.SOCIAL,
    "toilet": Action.TOILET,
}

# Required object type for each action (None means no target required).
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
    """Return a PetStats instance with one dominant stat most of the time."""
    stats = {name: rng.uniform(0.0, 1.0) for name in ["hunger", "tiredness", "boredom", "social", "toilet"]}

    # 70% of the time, amplify one stat to make the label clear.
    if rng.random() < 0.70:
        dominant = rng.choice(list(stats.keys()))
        # Push the dominant stat into [0.55, 1.0].
        stats[dominant] = rng.uniform(0.55, 1.0)
        # Optionally suppress others so dominant is clearly highest.
        if rng.random() < 0.5:
            for k in stats:
                if k != dominant:
                    stats[k] = rng.uniform(0.0, 0.45)

    return PetStats(**stats)


def _random_scene(rng: random.Random) -> SceneData:
    """Return a SceneData with 0–10 randomly typed objects at varied distances."""
    n_objects = rng.randint(0, 10)
    objects: list[SceneObject] = []
    for i in range(n_objects):
        obj_type = rng.choice(OBJECT_TYPES)
        distance = round(rng.uniform(1.0, 50.0), 2)
        objects.append(SceneObject(id=f"obj_{i}", type=obj_type, distance=distance))
    tick = rng.randint(0, 10_000)
    return SceneData(objects=objects, tick=tick)


def _random_request(rng: random.Random) -> InferenceRequest:
    return InferenceRequest(scene=_random_scene(rng), pet_stats=_random_stats(rng))


# ---------------------------------------------------------------------------
# Rule-based labeller
# ---------------------------------------------------------------------------


def _label(request: InferenceRequest) -> InferenceResponse:
    """Assign a ground-truth action using deterministic rules.

    Rules (checked in order):
    1. Find the highest pet stat.
    2. If highest stat < 0.5 → IDLE or EXPLORE randomly (seeded via scene tick).
    3. Map stat → preferred action:
       - hunger  → EAT  (requires bowl; fallback IDLE)
       - tiredness → SLEEP (requires bed; fallback IDLE)
       - boredom  → PLAY (requires toy; fallback IDLE)
       - social   → SOCIAL (requires player or pet; fallback IDLE)
       - toilet   → TOILET (no target required)
    4. When an action requires a target, set target_object_id to the closest
       matching object (lowest distance). If no match exists → IDLE.
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

    # Below-threshold: choose IDLE or EXPLORE at random (deterministic via tick).
    if dominant_value < 0.5:
        action = Action.IDLE if request.scene.tick % 2 == 0 else Action.EXPLORE
        return InferenceResponse(action=action, target_object_id=None, confidence=round(1.0 - dominant_value, 4))

    preferred_action = _STAT_TO_ACTION[dominant_stat]
    required_types = _ACTION_REQUIRES_TYPE[preferred_action]

    # No target required (toilet).
    if required_types is None:
        return InferenceResponse(action=preferred_action, target_object_id=None, confidence=round(dominant_value, 4))

    # Find closest object of a required type.
    candidates = [o for o in request.scene.objects if o.type in required_types]
    if not candidates:
        # No suitable object in scene → fallback to IDLE.
        return InferenceResponse(action=Action.IDLE, target_object_id=None, confidence=round(dominant_value, 4))

    closest = min(candidates, key=lambda o: o.distance)
    return InferenceResponse(
        action=preferred_action,
        target_object_id=closest.id,
        confidence=round(dominant_value, 4),
    )


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def _make_example(rng: random.Random) -> dict[str, str]:
    """Generate a single {"prompt": ..., "completion": ...} dict."""
    request = _random_request(rng)
    response = _label(request)
    prompt = build_prompt(request)
    completion = response.model_dump_json()
    return {"prompt": prompt, "completion": completion}


def _generate(n: int, rng: random.Random) -> list[dict[str, str]]:
    return [_make_example(rng) for _ in range(n)]


def _write_jsonl(path: Path, examples: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------


def _validate(path: Path) -> dict[str, int]:
    """Parse every line in a JSONL file and validate prompt/completion fields.

    Returns a summary dict with counts: total, valid, invalid.
    Raises on the first line that cannot be decoded as JSON.
    """
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
                prompt: str = obj["prompt"]
                completion_str: str = obj["completion"]

                # Validate completion parses as InferenceResponse.
                InferenceResponse.model_validate_json(completion_str)

                # Basic sanity: prompt must be non-empty and contain "Stats:".
                assert "Stats:" in prompt, "prompt missing Stats section"

                valid += 1
            except Exception as exc:
                invalid += 1
                errors.append(f"  line {line_no}: {exc}")

    if errors:
        print(f"  [!] {len(errors)} validation error(s):")
        for msg in errors[:10]:
            print(msg)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return {"total": total, "valid": valid, "invalid": invalid}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    rng = random.Random(SEED)

    data_dir = ROOT / "data"

    print(f"Generating {TRAIN_SIZE} training examples …")
    train_examples = _generate(TRAIN_SIZE, rng)
    train_path = data_dir / "train.jsonl"
    _write_jsonl(train_path, train_examples)
    print(f"  Written → {train_path}")

    print(f"Generating {EVAL_SIZE} eval examples …")
    eval_examples = _generate(EVAL_SIZE, rng)
    eval_path = data_dir / "eval.jsonl"
    _write_jsonl(eval_path, eval_examples)
    print(f"  Written → {eval_path}")

    print("\nValidating train.jsonl …")
    train_summary = _validate(train_path)
    print(
        f"  total={train_summary['total']}  valid={train_summary['valid']}  invalid={train_summary['invalid']}"
    )

    print("Validating eval.jsonl …")
    eval_summary = _validate(eval_path)
    print(
        f"  total={eval_summary['total']}  valid={eval_summary['valid']}  invalid={eval_summary['invalid']}"
    )

    all_valid = train_summary["invalid"] == 0 and eval_summary["invalid"] == 0
    if all_valid:
        print("\nAll examples valid.")
    else:
        total_invalid = train_summary["invalid"] + eval_summary["invalid"]
        print(f"\n{total_invalid} invalid example(s) found — review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
