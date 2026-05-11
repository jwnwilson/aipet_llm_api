"""Integration tests for real model quality — skipped if models/aipet.gguf is absent.

Measures:
  - Per-stat action accuracy (model picks right action for each dominant stat)
  - Target object accuracy (model picks closest valid object)
  - No single action dominates the distribution (≤30% share)
  - Priority conflict resolution (higher stat wins when two are elevated)
  - Fallback behaviour when required object is absent
"""

from __future__ import annotations

from pathlib import Path

import pytest

from domain.train.quality_report import (
    MAX_ACTION_SHARE,
    PER_STAT_ACCURACY_THRESHOLD,
    TARGET_ACCURACY_THRESHOLD,
    run_quality_report,
)
from adapters.inference import LlamaCppInferenceAdapter

MODEL_PATH = Path(__file__).parents[2] / "models" / "aipet.gguf"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"Real model not found at {MODEL_PATH}",
)


@pytest.fixture(scope="module")
def report():
    """Run the full quality report once per test session and share the result."""
    adapter = LlamaCppInferenceAdapter(model_path=str(MODEL_PATH), context_size=2048)
    return run_quality_report(adapter.infer, n_per_stat=40, n_conflict=20, n_absent=20, n_distribution=200)


class TestPerStatAccuracy:
    def test_hunger_accuracy(self, report):
        res = report["per_stat_accuracy"]["hunger"]
        assert res["pass"], (
            f"hunger: accuracy {res['accuracy']:.1%} < {PER_STAT_ACCURACY_THRESHOLD:.0%} "
            f"({res['correct']}/{res['total']} correct)"
        )

    def test_tiredness_accuracy(self, report):
        res = report["per_stat_accuracy"]["tiredness"]
        assert res["pass"], (
            f"tiredness: accuracy {res['accuracy']:.1%} < {PER_STAT_ACCURACY_THRESHOLD:.0%} "
            f"({res['correct']}/{res['total']} correct)"
        )

    def test_boredom_accuracy(self, report):
        res = report["per_stat_accuracy"]["boredom"]
        assert res["pass"], (
            f"boredom: accuracy {res['accuracy']:.1%} < {PER_STAT_ACCURACY_THRESHOLD:.0%} "
            f"({res['correct']}/{res['total']} correct)"
        )

    def test_social_accuracy(self, report):
        res = report["per_stat_accuracy"]["social"]
        assert res["pass"], (
            f"social: accuracy {res['accuracy']:.1%} < {PER_STAT_ACCURACY_THRESHOLD:.0%} "
            f"({res['correct']}/{res['total']} correct)"
        )

    def test_toilet_accuracy(self, report):
        res = report["per_stat_accuracy"]["toilet"]
        assert res["pass"], (
            f"toilet: accuracy {res['accuracy']:.1%} < {PER_STAT_ACCURACY_THRESHOLD:.0%} "
            f"({res['correct']}/{res['total']} correct)"
        )


class TestTargetAccuracy:
    def test_target_object_accuracy_meets_threshold(self, report):
        ta = report["target_accuracy"]
        if ta["total"] == 0:
            pytest.skip("No targeted responses in report sample — increase n_per_stat")
        assert ta["pass"], (
            f"Target accuracy {ta['accuracy']:.1%} < {TARGET_ACCURACY_THRESHOLD:.0%} "
            f"({ta['correct']}/{ta['total']} correct)"
        )


class TestActionDistribution:
    def test_no_action_dominates(self, report):
        dist = report["action_distribution"]
        total = sum(dist.values())
        dominant = {a: c for a, c in dist.items() if c / total > MAX_ACTION_SHARE}
        assert not dominant, (
            f"Action(s) exceed {MAX_ACTION_SHARE:.0%} share: "
            + ", ".join(f"{a}={c/total:.1%}" for a, c in dominant.items())
        )

    def test_multiple_actions_observed(self, report):
        dist = report["action_distribution"]
        assert len(dist) >= 5, (
            f"Only {len(dist)} distinct actions observed in 200 random inputs — "
            "model may be collapsing to a few actions"
        )


class TestPriorityConflict:
    def test_dominant_stat_wins_when_two_are_high(self, report):
        pc = report["priority_conflict"]
        assert pc["pass"], (
            f"Priority conflict accuracy {pc['accuracy']:.1%} < 80% "
            f"({pc['correct']}/{pc['total']} correct) — "
            "model is not consistently choosing the higher stat's action"
        )


class TestFallbackBehaviour:
    def test_idle_or_explore_when_object_absent(self, report):
        fb = report["fallback_accuracy"]
        assert fb["pass"], (
            f"Fallback accuracy {fb['accuracy']:.1%} < 90% "
            f"({fb['correct']}/{fb['total']} correct) — "
            "model is not falling back to IDLE/EXPLORE when required object is absent"
        )
