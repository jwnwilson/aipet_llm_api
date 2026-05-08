"""Dataset generation logic — pure functions, no I/O side-effects except writing files."""

from __future__ import annotations

import json
import random
from collections import Counter
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
TRAIN_SIZE = 5000
EVAL_SIZE = 500

OBJECT_TYPES = ["bowl", "bed", "toy", "player", "pet"]
STAT_NAMES = ["hunger", "tiredness", "boredom", "social", "toilet"]

# Each stat maps to one or more valid actions; make_example picks randomly between equivalents.
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

# Actions that never need a target object.
_NO_TARGET_ACTIONS = {Action.TOILET, Action.IDLE, Action.EXPLORE}


# ---------------------------------------------------------------------------
# Random scene / stat generation
# ---------------------------------------------------------------------------


def _stats_with_dominant(rng: random.Random, dominant: str) -> PetStats:
    """One stat is high (0.70–1.0); all others are low (0.0–0.35)."""
    values = {name: rng.uniform(0.0, 0.35) for name in STAT_NAMES}
    values[dominant] = rng.uniform(0.70, 1.0)
    return PetStats(**values)


def _stats_with_two_dominant(rng: random.Random, dominant: str, second: str) -> PetStats:
    """Dominant stat is 0.80–1.0; second is 0.60–0.79; rest are low."""
    values = {name: rng.uniform(0.0, 0.35) for name in STAT_NAMES}
    values[dominant] = rng.uniform(0.80, 1.0)
    values[second] = rng.uniform(0.60, 0.79)
    return PetStats(**values)


def _scene_with_required_multi_target(
    rng: random.Random, required_types: list[str] | None
) -> SceneData:
    """Scene with 2–4 competing objects of the required type plus 0–3 distractors.

    Multiple targets at varied distances teach the model to select the closest one
    rather than defaulting to the first id in the list.
    """
    objects: list[SceneObject] = []

    if required_types:
        req_type = rng.choice(required_types)
        n_required = rng.randint(2, 4)
        for i in range(n_required):
            objects.append(SceneObject(
                id=f"obj_{i}",
                type=req_type,
                distance=round(rng.uniform(1.0, 50.0), 1),
            ))

    n_distractors = rng.randint(0, 3)
    for _ in range(n_distractors):
        idx = len(objects)
        objects.append(SceneObject(
            id=f"obj_{idx}",
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


def label(request: InferenceRequest, rng: random.Random | None = None) -> InferenceResponse:
    """Assign a ground-truth action using deterministic rules.

    Priority: dominant stat (≥ 0.5) → preferred action → closest matching object.
    When rng is provided, picks randomly between equivalent action pairs (EAT/DRINK,
    PLAY/FETCH, SOCIAL/FOLLOW) so each example has an independently chosen label.
    Without rng, always returns the first action in each group (deterministic default
    used by unit tests and one-off calls).
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
    preferred_action = rng.choice(actions) if rng is not None else actions[0]
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


def make_example(rng: random.Random, dominant: str | None = None) -> dict[str, str]:
    """Generate one labelled example.

    dominant — if given, fixes which stat is high; otherwise chosen randomly.

    75% of examples: required object is present (with multiple competing targets).
    25% of examples: required object is absent to teach IDLE/EXPLORE fallback.
    Toilet stat never needs a scene object so it is always a 'present' example.

    Action within equivalent pairs (EAT/DRINK etc.) is chosen randomly per example
    so the model never sees the same prompt paired with contradictory labels.
    """
    if dominant is None:
        dominant = rng.choice(STAT_NAMES)

    # 15% chance of a priority-conflict example (two high stats)
    if rng.random() < 0.15:
        second = rng.choice([s for s in STAT_NAMES if s != dominant])
        stats = _stats_with_two_dominant(rng, dominant, second)
    else:
        stats = _stats_with_dominant(rng, dominant)

    required_types = _STAT_REQUIRES_TYPES[dominant]

    if required_types is None or rng.random() < 0.75:
        scene = _scene_with_required_multi_target(rng, required_types)
    else:
        scene = _scene_without_required(rng, required_types)

    request = InferenceRequest(scene=scene, pet_stats=stats)
    response = label(request, rng=rng)
    # Build completion with stat first so the model reasons about the dominant
    # stat before committing to an action (chain-of-thought lite).
    comp = {"stat": dominant, **json.loads(response.model_dump_json(exclude_none=True))}
    return {"prompt": build_prompt(request, rng=rng), "completion": json.dumps(comp)}


def generate_examples(n: int, rng: random.Random) -> list[dict[str, str]]:
    """Generate n examples using stratified sampling over dominant stats.

    Each stat drives exactly n//5 examples (remainder distributed round-robin)
    so every action category is equally represented in the labelled output.
    """
    per_stat = n // len(STAT_NAMES)
    counts = {stat: per_stat for stat in STAT_NAMES}
    remainder = n - per_stat * len(STAT_NAMES)
    for stat in STAT_NAMES[:remainder]:
        counts[stat] += 1

    examples: list[dict[str, str]] = []
    for stat, count in counts.items():
        for _ in range(count):
            examples.append(make_example(rng, dominant=stat))

    rng.shuffle(examples)
    return examples


# ---------------------------------------------------------------------------
# Distribution validation
# ---------------------------------------------------------------------------


def check_dataset_distribution(path: Path) -> None:
    """Print per-action counts and raise AssertionError if any action is under/over represented.

    Thresholds: no action < 5% or > 25% of total labelled examples.
    """
    counts: Counter[str] = Counter()
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                completion = json.loads(obj["completion"])
                counts[completion["action"]] += 1
                total += 1
            except Exception:
                pass

    print(f"\nAction distribution in {path.name} (n={total}):")
    violations: list[str] = []
    for action in sorted(counts):
        pct = counts[action] / total if total else 0
        flag = ""
        if pct < 0.05:
            violations.append(f"{action} underrepresented ({pct:.1%} < 5%)")
            flag = " ← UNDER"
        elif pct > 0.25:
            violations.append(f"{action} overrepresented ({pct:.1%} > 25%)")
            flag = " ← OVER"
        print(f"  {action:<12} {counts[action]:5d}  ({pct:.1%}){flag}")

    if violations:
        raise AssertionError("Dataset distribution out of bounds:\n  " + "\n  ".join(violations))


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
                assert "Stats" in obj["prompt"], "prompt missing Stats section"
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
    if not all_valid:
        total_invalid = train_summary["invalid"] + eval_summary["invalid"]
        print(f"\n{total_invalid} invalid example(s) found — review errors above.")
        return False

    try:
        check_dataset_distribution(train_path)
        check_dataset_distribution(eval_path)
    except AssertionError as exc:
        print(f"\nDistribution check failed: {exc}")
        return False

    print("\nAll examples valid and distribution within bounds.")
    return True
