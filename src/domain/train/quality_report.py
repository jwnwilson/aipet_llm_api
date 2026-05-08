"""Quality report — per-stat accuracy, target accuracy, and action frequency distribution."""

from __future__ import annotations

import random
from collections import Counter
from typing import Callable

from domain.actions import Action
from domain.models import InferenceRequest, InferenceResponse, PetStats, SceneData, SceneObject

STAT_NAMES = ["hunger", "tiredness", "boredom", "social", "toilet"]

_STAT_TO_EXPECTED_ACTIONS: dict[str, set[Action]] = {
    "hunger": {Action.EAT, Action.DRINK},
    "tiredness": {Action.SLEEP},
    "boredom": {Action.PLAY, Action.FETCH},
    "social": {Action.SOCIAL, Action.FOLLOW},
    "toilet": {Action.TOILET},
}

_STAT_REQUIRES_TYPES: dict[str, list[str] | None] = {
    "hunger": ["bowl"],
    "tiredness": ["bed"],
    "boredom": ["toy"],
    "social": ["player", "pet"],
    "toilet": None,
}

_ALL_OBJECT_TYPES = ["bowl", "bed", "toy", "player", "pet"]

PER_STAT_ACCURACY_THRESHOLD = 0.90
TARGET_ACCURACY_THRESHOLD = 0.90
MAX_ACTION_SHARE = 0.30


def _make_stat_request(
    stat: str, rng: random.Random
) -> tuple[InferenceRequest, str | None]:
    """Build a request with one dominant stat (0.9) and return (request, expected_target_id).

    Places 3 objects of the required type at clearly distinct distances so that
    both action selection and nearest-target selection are exercised.
    """
    stat_values = {s: 0.1 for s in STAT_NAMES}
    stat_values[stat] = 0.9

    required_types = _STAT_REQUIRES_TYPES[stat]
    objects: list[SceneObject] = []
    expected_target_id: str | None = None

    if required_types:
        req_type = required_types[0]
        # Three objects at clearly separated distances; first is always closest.
        distances = [
            round(rng.uniform(1.0, 4.0), 1),
            round(rng.uniform(8.0, 15.0), 1),
            round(rng.uniform(20.0, 40.0), 1),
        ]
        for i, dist in enumerate(distances):
            objects.append(SceneObject(id=f"target_{i}", type=req_type, distance=dist))
        expected_target_id = "target_0"  # the closest one

        # One distractor of an unrelated type
        other_types = [t for t in _ALL_OBJECT_TYPES if t not in required_types]
        if other_types:
            objects.append(SceneObject(
                id="distractor_0",
                type=other_types[0],
                distance=round(rng.uniform(1.0, 3.0), 1),
            ))

    rng.shuffle(objects)
    return InferenceRequest(
        scene=SceneData(objects=objects, tick=rng.randint(0, 10_000)),
        pet_stats=PetStats(**stat_values),
    ), expected_target_id


def _make_conflict_request(rng: random.Random) -> tuple[InferenceRequest, set[Action]]:
    """Build a request where two stats are high (one clearly dominant).

    Returns (request, expected_actions_for_dominant_stat).
    """
    dominant, second = rng.sample(STAT_NAMES, 2)
    stat_values = {s: 0.1 for s in STAT_NAMES}
    stat_values[dominant] = round(rng.uniform(0.85, 1.0), 2)
    stat_values[second] = round(rng.uniform(0.60, 0.79), 2)

    objects: list[SceneObject] = []
    for stat in (dominant, second):
        req_types = _STAT_REQUIRES_TYPES[stat]
        if req_types:
            objects.append(SceneObject(
                id=f"{stat}_obj",
                type=req_types[0],
                distance=round(rng.uniform(2.0, 10.0), 1),
            ))

    return InferenceRequest(
        scene=SceneData(objects=objects, tick=rng.randint(0, 10_000)),
        pet_stats=PetStats(**stat_values),
    ), _STAT_TO_EXPECTED_ACTIONS[dominant]


def _make_absent_request(rng: random.Random) -> tuple[InferenceRequest, str]:
    """Build a request where the dominant stat's required object is absent.

    Returns (request, dominant_stat).
    """
    stat = rng.choice([s for s in STAT_NAMES if _STAT_REQUIRES_TYPES[s] is not None])
    stat_values = {s: 0.1 for s in STAT_NAMES}
    stat_values[stat] = 0.9

    excluded = _STAT_REQUIRES_TYPES[stat] or []
    allowed_types = [t for t in _ALL_OBJECT_TYPES if t not in excluded]
    n = rng.randint(1, 4)
    objects = [
        SceneObject(
            id=f"obj_{i}",
            type=rng.choice(allowed_types) if allowed_types else "toy",
            distance=round(rng.uniform(1.0, 20.0), 1),
        )
        for i in range(n)
    ]
    return InferenceRequest(
        scene=SceneData(objects=objects, tick=rng.randint(0, 10_000)),
        pet_stats=PetStats(**stat_values),
    ), stat


def _make_random_request(rng: random.Random) -> InferenceRequest:
    """Build a uniformly random request for action frequency distribution analysis."""
    stat_values = {s: round(rng.uniform(0.0, 1.0), 2) for s in STAT_NAMES}
    n_objects = rng.randint(0, 6)
    objects = [
        SceneObject(
            id=f"obj_{i}",
            type=rng.choice(_ALL_OBJECT_TYPES),
            distance=round(rng.uniform(1.0, 50.0), 1),
        )
        for i in range(n_objects)
    ]
    return InferenceRequest(
        scene=SceneData(objects=objects, tick=rng.randint(0, 10_000)),
        pet_stats=PetStats(**stat_values),
    )


def run_quality_report(
    infer_fn: Callable[[InferenceRequest], InferenceResponse],
    n_per_stat: int = 40,
    n_conflict: int = 20,
    n_absent: int = 20,
    n_distribution: int = 200,
    seed: int = 999,
) -> dict:
    """Run the full quality report and return a JSON-serialisable result dict.

    Sections:
      per_stat_accuracy  — does the model pick the right action for each dominant stat?
      target_accuracy    — does it pick the closest valid target object?
      priority_conflict  — does it prefer the higher stat when two are high?
      fallback_accuracy  — does it fall back to IDLE/EXPLORE when required object absent?
      action_distribution — raw counts over uniform-random inputs
    """
    rng = random.Random(seed)

    # --- per-stat accuracy + target accuracy ---
    per_stat: dict[str, dict] = {}
    target_correct = 0
    target_total = 0

    for stat in STAT_NAMES:
        correct = 0
        for _ in range(n_per_stat):
            request, expected_target = _make_stat_request(stat, rng)
            response = infer_fn(request)
            expected_actions = _STAT_TO_EXPECTED_ACTIONS[stat]
            action_ok = response.action in expected_actions
            if action_ok:
                correct += 1
            if expected_target is not None and action_ok and response.target_object_id is not None:
                target_total += 1
                if response.target_object_id == expected_target:
                    target_correct += 1

        accuracy = correct / n_per_stat
        per_stat[stat] = {
            "correct": correct,
            "total": n_per_stat,
            "accuracy": round(accuracy, 4),
            "pass": accuracy >= PER_STAT_ACCURACY_THRESHOLD,
        }

    target_accuracy = target_correct / target_total if target_total > 0 else 0.0
    target_pass = target_accuracy >= TARGET_ACCURACY_THRESHOLD or target_total == 0

    # --- priority conflict ---
    conflict_correct = 0
    for _ in range(n_conflict):
        request, expected_actions = _make_conflict_request(rng)
        response = infer_fn(request)
        if response.action in expected_actions:
            conflict_correct += 1
    conflict_accuracy = conflict_correct / n_conflict

    # --- fallback when object absent ---
    fallback_correct = 0
    _fallback_actions = {Action.IDLE, Action.EXPLORE}
    for _ in range(n_absent):
        request, _ = _make_absent_request(rng)
        response = infer_fn(request)
        if response.action in _fallback_actions:
            fallback_correct += 1
    fallback_accuracy = fallback_correct / n_absent

    # --- action distribution ---
    action_counts: Counter[str] = Counter()
    for _ in range(n_distribution):
        resp = infer_fn(_make_random_request(rng))
        action_counts[resp.action.value] += 1

    max_share = max(action_counts.values()) / n_distribution if action_counts else 0.0

    return {
        "per_stat_accuracy": per_stat,
        "target_accuracy": {
            "correct": target_correct,
            "total": target_total,
            "accuracy": round(target_accuracy, 4),
            "pass": target_pass,
        },
        "priority_conflict": {
            "correct": conflict_correct,
            "total": n_conflict,
            "accuracy": round(conflict_accuracy, 4),
            "pass": conflict_accuracy >= 0.80,
        },
        "fallback_accuracy": {
            "correct": fallback_correct,
            "total": n_absent,
            "accuracy": round(fallback_accuracy, 4),
            "pass": fallback_accuracy >= 0.90,
        },
        "action_distribution": dict(action_counts),
        "max_action_share": round(max_share, 4),
        "pass": (
            all(r["pass"] for r in per_stat.values())
            and target_pass
            and conflict_accuracy >= 0.80
            and fallback_accuracy >= 0.90
            and max_share <= MAX_ACTION_SHARE
        ),
    }


def print_report(report: dict) -> None:
    print("\n=== Quality Report ===")

    print(f"\nPer-stat accuracy (threshold ≥{PER_STAT_ACCURACY_THRESHOLD:.0%}):")
    for stat, res in report["per_stat_accuracy"].items():
        status = "PASS" if res["pass"] else "FAIL"
        bar = "#" * res["correct"]
        print(f"  {stat:<12} {res['correct']:3d}/{res['total']}  ({res['accuracy']:.1%})  [{status}]")

    ta = report["target_accuracy"]
    t_status = "PASS" if ta["pass"] else "FAIL"
    print(
        f"\nTarget accuracy   (threshold ≥{TARGET_ACCURACY_THRESHOLD:.0%}): "
        f"{ta['correct']}/{ta['total']} ({ta['accuracy']:.1%})  [{t_status}]"
    )

    pc = report["priority_conflict"]
    pc_status = "PASS" if pc["pass"] else "FAIL"
    print(
        f"Priority conflict (threshold ≥80%): "
        f"{pc['correct']}/{pc['total']} ({pc['accuracy']:.1%})  [{pc_status}]"
    )

    fb = report["fallback_accuracy"]
    fb_status = "PASS" if fb["pass"] else "FAIL"
    print(
        f"Fallback accuracy (threshold ≥90%): "
        f"{fb['correct']}/{fb['total']} ({fb['accuracy']:.1%})  [{fb_status}]"
    )

    total_dist = sum(report["action_distribution"].values())
    print(f"\nAction distribution (n={total_dist} random inputs, threshold ≤{MAX_ACTION_SHARE:.0%} each):")
    for action, count in sorted(report["action_distribution"].items(), key=lambda x: -x[1]):
        pct = count / total_dist if total_dist else 0
        bar = "#" * min(40, count)
        flag = " ← DOMINANT" if pct > MAX_ACTION_SHARE else ""
        print(f"  {action:<12} {count:4d}  ({pct:.1%})  {bar}{flag}")

    overall = "PASS" if report["pass"] else "FAIL"
    print(f"\nOverall: [{overall}]")
