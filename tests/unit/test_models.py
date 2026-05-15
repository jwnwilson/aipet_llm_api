"""Unit tests for domain model schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from domain.models import (
    CategoryAccuracyResult,
    EvaluationData,
    QualityReport,
    RunStatus,
    StatAccuracyResult,
)


def _valid_stat() -> dict:
    return {"correct": 38, "total": 40, "accuracy": 0.95, "passed": True}


def _valid_category() -> dict:
    return {"correct": 18, "total": 20, "accuracy": 0.9, "passed": True}


def _valid_report() -> dict:
    return {
        "per_stat_accuracy": {s: _valid_stat() for s in ["hunger", "boredom", "social", "tiredness", "toilet"]},
        "target_accuracy": _valid_category(),
        "priority_conflict": _valid_category(),
        "fallback_accuracy": _valid_category(),
        "action_distribution": {"EAT": 42, "SLEEP": 35, "IDLE": 20},
        "max_action_share": 0.21,
        "passed": True,
    }


class TestStatAccuracyResult:
    def test_valid(self):
        r = StatAccuracyResult(**_valid_stat())
        assert r.accuracy == 0.95
        assert r.passed is True

    def test_accuracy_range(self):
        with pytest.raises(ValidationError):
            StatAccuracyResult(correct=5, total=10, accuracy=1.5, passed=False)


class TestQualityReport:
    def test_valid(self):
        r = QualityReport(**_valid_report())
        assert r.passed is True
        assert r.per_stat_accuracy["hunger"].correct == 38

    def test_missing_stat_raises(self):
        data = _valid_report()
        data["per_stat_accuracy"].pop("hunger")
        with pytest.raises(ValidationError):
            QualityReport(**data)


class TestEvaluationData:
    def test_without_report(self):
        d = EvaluationData(run_id="abc", status=RunStatus.EVALUATING)
        assert d.quality_report is None
        assert d.eval_valid_pct is None

    def test_with_report(self):
        d = EvaluationData(
            run_id="abc",
            status=RunStatus.COMPLETED,
            eval_valid_pct=0.95,
            quality_report=QualityReport(**_valid_report()),
        )
        assert d.quality_report.passed is True
