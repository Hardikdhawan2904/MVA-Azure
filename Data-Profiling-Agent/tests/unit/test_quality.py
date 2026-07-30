"""Tests for data quality assessment dimensions."""

import pytest
import pandas as pd

from app.core.config import Settings
from app.core.enums import QualityStatus, RuleType, RuleSource
from app.services.profiling.column_profiler import ColumnProfiler
from app.services.quality.completeness import assess_completeness
from app.services.quality.uniqueness import assess_uniqueness
from app.services.quality.validity import assess_validity
from app.services.quality.conformity import assess_conformity
from app.services.quality.consistency import assess_consistency
from app.services.quality.timeliness import assess_timeliness
from app.services.quality.integrity import assess_integrity
from app.services.quality.accuracy import assess_accuracy
from app.services.quality.business_rule_compliance import assess_business_rule_compliance
from app.services.quality.semantic_quality import assess_semantic_quality
from app.services.quality.overall_score import calculate_overall_score
from app.services.rules.rule_engine import RuleEvaluationResult


@pytest.fixture
def profiler() -> ColumnProfiler:
    settings = Settings(DATABASE_URL="postgresql://x:x@localhost/test", MAX_SAMPLE_VALUES=5)
    return ColumnProfiler(settings)


class TestCompleteness:
    def test_assessed_with_mandatory_columns(self, profiler):
        df = pd.DataFrame({"amount": ["100", None, "300", None, "500"]})
        profiles = profiler.profile_all(df, ["amount"])
        result = assess_completeness(profiles, mandatory_columns=["amount"])
        assert result.status == QualityStatus.ASSESSED
        assert result.score == 0.6  # 3/5 non-null

    def test_not_assessable_without_mandatory(self, profiler):
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        profiles = profiler.profile_all(df, ["x"])
        result = assess_completeness(profiles, mandatory_columns=[])
        assert result.status == QualityStatus.NOT_ASSESSABLE
        assert result.score is None


class TestUniqueness:
    def test_assessed_with_expected_unique(self, profiler):
        df = pd.DataFrame({"id": ["A", "B", "C", "A", "D"]})
        profiles = profiler.profile_all(df, ["id"])
        result = assess_uniqueness(profiles, expected_unique_columns=["id"])
        assert result["status"] == QualityStatus.ASSESSED.value
        assert result["score"] == 0.8  # 1 dupe out of 5

    def test_not_assessable_without_expected_unique(self, profiler):
        df = pd.DataFrame({"status": ["a", "a", "b"]})
        profiles = profiler.profile_all(df, ["status"])
        result = assess_uniqueness(profiles, expected_unique_columns=[])
        assert result["status"] == QualityStatus.NOT_ASSESSABLE.value


class TestValidity:
    def test_assessed_with_range_rules(self):
        results = [RuleEvaluationResult(
            rule_key="r1", rule_type=RuleType.NUMERIC_RANGE,
            records_checked=100, pass_count=90, fail_count=10, score=0.9, target_columns=["x"],
            source=RuleSource.REQUEST,
        )]
        r = assess_validity(results)
        assert r["status"] == QualityStatus.ASSESSED.value
        assert r["score"] == 0.9

    def test_not_assessable_without_rules(self):
        r = assess_validity([])
        assert r["status"] == QualityStatus.NOT_ASSESSABLE.value


class TestConformity:
    def test_assessed_with_regex_rules(self):
        results = [RuleEvaluationResult(
            rule_key="regex1", rule_type=RuleType.REGEX_MATCH,
            records_checked=50, pass_count=45, fail_count=5, score=0.9, target_columns=["code"],
            source=RuleSource.REQUEST,
        )]
        r = assess_conformity(results)
        assert r["status"] == QualityStatus.ASSESSED.value
        assert r["score"] == 0.9


class TestConsistency:
    def test_assessed_with_comparison_rules(self):
        results = [RuleEvaluationResult(
            rule_key="cmp1", rule_type=RuleType.COLUMN_COMPARISON,
            records_checked=100, pass_count=95, fail_count=5, score=0.95, target_columns=["a", "b"],
            source=RuleSource.REQUEST,
        )]
        r = assess_consistency(results)
        assert r["status"] == QualityStatus.ASSESSED.value
        assert r["score"] == 0.95


class TestNotAssessable:
    def test_timeliness_not_assessable(self):
        r = assess_timeliness()
        assert r["status"] == QualityStatus.NOT_ASSESSABLE.value
        assert r["score"] is None

    def test_integrity_not_assessable(self):
        r = assess_integrity()
        assert r["status"] == QualityStatus.NOT_ASSESSABLE.value

    def test_accuracy_not_assessable(self):
        r = assess_accuracy()
        assert r["status"] == QualityStatus.NOT_ASSESSABLE.value


class TestBusinessRuleCompliance:
    def test_assessed(self):
        results = [
            RuleEvaluationResult("r1", RuleType.NON_NULL, 100, 95, 5, 0.95, ["col"], source=RuleSource.REQUEST),
            RuleEvaluationResult("r2", RuleType.NUMERIC_RANGE, 100, 80, 20, 0.80, ["col"], source=RuleSource.REQUEST),
        ]
        r = assess_business_rule_compliance(results)
        assert r["status"] == QualityStatus.ASSESSED.value
        assert r["score"] == 175 / 200  # 0.875


class TestSemanticQuality:
    def test_calculated(self):
        r = assess_semantic_quality(
            schema_confidence_avg=0.90,
            classification_confidence_avg=0.85,
            secondary_domain_confidence=0.80,
            hierarchy_confidence=0.75,
        )
        assert r["status"] == QualityStatus.ASSESSED.value
        assert r["score"] is not None
        assert 0.0 <= r["score"] <= 1.0

    def test_partial_components(self):
        r = assess_semantic_quality(
            schema_confidence_avg=0.90,
            classification_confidence_avg=None,
            secondary_domain_confidence=0.80,
            hierarchy_confidence=None,
        )
        assert r["status"] == QualityStatus.ASSESSED.value


class TestOverallScore:
    def test_excludes_not_assessable(self):
        dimensions = [
            {"dimension": "completeness", "score": 0.90, "status": "assessed"},
            {"dimension": "uniqueness", "score": 0.85, "status": "assessed"},
            {"dimension": "timeliness", "score": None, "status": "not_assessable"},
            {"dimension": "accuracy", "score": None, "status": "not_assessable"},
        ]
        weights = {"completeness": 0.3, "uniqueness": 0.2, "timeliness": 0.1, "accuracy": 0.1}
        result = calculate_overall_score(dimensions, weights)

        assert "timeliness" in result["excluded_dimensions"]
        assert "accuracy" in result["excluded_dimensions"]
        assert "completeness" in result["assessed_dimensions"]
        # Score = (0.3*0.9 + 0.2*0.85) / (0.3 + 0.2) = 0.44/0.5 = 0.88
        assert abs(result["overall_score"] - 0.88) < 0.01

    def test_never_treats_not_assessable_as_zero(self):
        dimensions = [
            {"dimension": "completeness", "score": 0.50, "status": "assessed"},
            {"dimension": "accuracy", "score": None, "status": "not_assessable"},
        ]
        weights = {"completeness": 0.5, "accuracy": 0.5}
        result = calculate_overall_score(dimensions, weights)
        # If accuracy were treated as 0: score = (0.5*0.5 + 0.5*0)/(0.5+0.5) = 0.25
        # Correct: score = (0.5*0.5)/(0.5) = 0.50
        assert abs(result["overall_score"] - 0.50) < 0.01

    def test_all_not_assessable(self):
        dimensions = [
            {"dimension": "timeliness", "score": None, "status": "not_assessable"},
        ]
        weights = {"timeliness": 0.1}
        result = calculate_overall_score(dimensions, weights)
        assert result["overall_score"] is None
