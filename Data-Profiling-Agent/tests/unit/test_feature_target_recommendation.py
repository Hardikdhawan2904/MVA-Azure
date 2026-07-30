"""Tests for the target/feature/drop recommendation: the deterministic
fallback splitter (app/agents/data_profiling_agent/nodes/pipeline.py) and its
integration into ReadinessEngine's ml/llm scoring."""

import pandas as pd
import pytest

from app.core.enums import ColumnRole, RefinedDataType
from app.services.profiling.column_profiler import ColumnProfiler
from app.services.profiling.identifier_detector import GrainDetectionResult
from app.services.profiling.semantic_candidate_generator import SemanticCandidate
from app.services.readiness.readiness_engine import ReadinessEngine
from app.agents.data_profiling_agent.nodes.pipeline import (
    _deterministic_feature_split, _infer_target_column_from_question,
)


def _candidate(name: str, role: ColumnRole, semantic_type: str | None = None) -> SemanticCandidate:
    return SemanticCandidate(
        column_name=name,
        normalized_key=name,
        refined_type=RefinedDataType.UNKNOWN,
        candidate_semantic_type=semantic_type,
        candidate_column_role=role,
        candidate_confidence=0.9,
        evidence=[],
    )


class TestInferTargetColumnFromQuestion:
    """Regression tests for a bug caught via live testing: on a real
    141-column dataset, the target-picking LLM was never even shown
    columns past index 60 (a truncation bug fixed separately in
    feature_target_agent/graph.py), so "Forecast underwriting result..."
    resolved to underwriting_result_actual (index 88), a wrong-but-visible
    column, or nothing at all -- non-deterministically, across identical
    repeated calls. This deterministic pre-check removes the guessing for
    exactly the case that doesn't need it: the question already names the
    column in plain English."""

    def test_resolves_the_obviously_named_column(self):
        candidates = [
            _candidate("underwriting_result_actual", ColumnRole.METRIC),
            _candidate("gross_written_premium_actual", ColumnRole.METRIC),
        ]
        result = _infer_target_column_from_question(
            "Forecast underwriting result for next 6 months", candidates,
        )
        assert result == "underwriting_result_actual"

    def test_longer_match_wins_over_shorter_substring_match(self):
        # Both "profit" and "net profit" are literal substrings of the
        # question -- the longer, more specific match must win.
        candidates = [
            _candidate("profit_actual", ColumnRole.METRIC),
            _candidate("net_profit_actual", ColumnRole.METRIC),
        ]
        result = _infer_target_column_from_question(
            "Why did net profit decline vs budget?", candidates,
        )
        assert result == "net_profit_actual"

    def test_no_match_returns_none_not_a_guess(self):
        candidates = [_candidate("marketing_spend_actual", ColumnRole.METRIC)]
        result = _infer_target_column_from_question(
            "Forecast underwriting result for next 6 months", candidates,
        )
        assert result is None

    def test_ambiguous_tie_returns_none_not_a_guess(self):
        candidates = [
            _candidate("revenue_actual", ColumnRole.METRIC),
            _candidate("revenue_budget", ColumnRole.METRIC),
        ]
        # Both base-phrase to "revenue" -- an exact tie, must not guess.
        result = _infer_target_column_from_question("Explain revenue", candidates)
        assert result is None

    def test_non_metric_columns_are_never_candidates(self):
        candidates = [_candidate("underwriting_department", ColumnRole.DIMENSION)]
        result = _infer_target_column_from_question(
            "Forecast underwriting department for next 6 months", candidates,
        )
        assert result is None


class TestDeterministicFeatureSplit:
    def test_identifiers_are_dropped_not_featured(self):
        candidates = [
            _candidate("txn_id", ColumnRole.IDENTIFIER),
            _candidate("amount", ColumnRole.METRIC, "monetary_amount"),
        ]
        grain_result = GrainDetectionResult(grain_columns=["txn_id"], identifier_candidates=[])

        result = _deterministic_feature_split(candidates, grain_result)

        feature_names = [f["column"] for f in result["feature_columns"]]
        drop_names = [d["column"] for d in result["drop_columns"]]
        assert "txn_id" not in feature_names
        assert "amount" in feature_names
        assert "txn_id" in drop_names

    def test_text_and_unknown_roles_excluded_from_features(self):
        candidates = [
            _candidate("notes", ColumnRole.TEXT_FIELD),
            _candidate("misc", ColumnRole.UNKNOWN),
            _candidate("region", ColumnRole.DIMENSION),
        ]
        grain_result = GrainDetectionResult(grain_columns=[], identifier_candidates=[])

        result = _deterministic_feature_split(candidates, grain_result)

        feature_names = [f["column"] for f in result["feature_columns"]]
        assert "notes" not in feature_names
        assert "misc" not in feature_names
        assert "region" in feature_names

    def test_no_target_or_approach_without_a_question(self):
        candidates = [_candidate("amount", ColumnRole.METRIC)]
        grain_result = GrainDetectionResult(grain_columns=[], identifier_candidates=[])

        result = _deterministic_feature_split(candidates, grain_result)

        assert result["target_column"] is None
        assert result["recommended_approach"] == "unknown"
        assert result["confidence"] == 0.0
        assert result["approach_reasoning"] is None

    def test_deterministic_features_are_weighted_high(self):
        """Regression guard: deterministic-fallback features must be "high"
        usefulness, not "medium" — a weaker weight would silently deflate
        ml_readiness's feature-coverage score for every question-less run."""
        candidates = [_candidate("amount", ColumnRole.METRIC)]
        grain_result = GrainDetectionResult(grain_columns=[], identifier_candidates=[])

        result = _deterministic_feature_split(candidates, grain_result)

        assert result["feature_columns"][0]["usefulness"] == "high"


@pytest.fixture
def profiler() -> ColumnProfiler:
    from app.core.config import Settings
    settings = Settings(DATABASE_URL="postgresql://x:x@localhost/test", MAX_SAMPLE_VALUES=5)
    return ColumnProfiler(settings)


@pytest.fixture
def engine() -> ReadinessEngine:
    return ReadinessEngine()


class TestReadinessEngineFeatureRecommendation:
    def _profiles_and_quality(self, profiler):
        df = pd.DataFrame({
            "id": [f"R{i}" for i in range(100)],
            "amount": [str(i * 10) for i in range(100)],
            "status": ["active", "inactive"] * 50,
        })
        profiles = profiler.profile_all(df, ["id", "amount", "status"])
        quality = [{"dimension": "completeness", "score": 0.95, "status": "assessed"}]
        return profiles, quality

    def test_none_feature_recommendation_preserves_legacy_behavior(self, profiler, engine):
        """Callers that don't pass feature_recommendation (e.g. not-yet-updated
        tests) must see identical behavior to before this feature existed."""
        profiles, quality = self._profiles_and_quality(profiler)

        results = engine.assess_all(
            profiles, quality, ["id"], True, 1, 1, 0.5, 100, feature_recommendation=None,
        )
        ml = next(r for r in results if r.assessment_type.value == "ml_readiness")
        assert not any(s.get("code") == "TARGET_IDENTIFIED" for s in ml.strengths)
        assert not any(b.get("code") == "NO_TARGET_IDENTIFIED" for b in ml.blocking_issues)

    def test_target_identified_strength_when_recommendation_has_target(self, profiler, engine):
        profiles, quality = self._profiles_and_quality(profiler)
        recommendation = {
            "target_column": "status",
            "feature_columns": [{"column": "amount", "usefulness": "high"}],
            "drop_columns": [{"column": "id"}],
            "recommended_approach": "ml",
            "approach_reasoning": "Structured target with numeric features.",
            "confidence": 0.8,
        }

        results = engine.assess_all(
            profiles, quality, ["id"], True, 1, 1, 0.5, 100, feature_recommendation=recommendation,
        )
        ml = next(r for r in results if r.assessment_type.value == "ml_readiness")
        assert any(s.get("code") == "TARGET_IDENTIFIED" and s.get("value") == "status" for s in ml.strengths)

    def test_no_target_blocker_when_question_asked_but_unresolved(self, profiler, engine):
        profiles, quality = self._profiles_and_quality(profiler)
        recommendation = {
            "target_column": None,
            "feature_columns": [],
            "drop_columns": [],
            "recommended_approach": "unknown",
            "approach_reasoning": "The question doesn't map onto a single column.",
            "confidence": 0.3,
        }

        results = engine.assess_all(
            profiles, quality, [], True, 1, 1, 0.5, 100, feature_recommendation=recommendation,
        )
        ml = next(r for r in results if r.assessment_type.value == "ml_readiness")
        assert any(b.get("code") == "NO_TARGET_IDENTIFIED" for b in ml.blocking_issues)

    def test_llm_readiness_boosted_when_approach_recommends_llm(self, profiler, engine):
        profiles, quality = self._profiles_and_quality(profiler)
        recommendation = {
            "target_column": None,
            "feature_columns": [],
            "drop_columns": [],
            "recommended_approach": "llm",
            "approach_reasoning": "Open-ended question best answered narratively.",
            "confidence": 0.9,
        }

        without = engine.assess_all(profiles, quality, [], True, 1, 1, 0.5, 100, feature_recommendation=None)
        with_llm = engine.assess_all(
            profiles, quality, [], True, 1, 1, 0.5, 100, feature_recommendation=recommendation,
        )

        llm_without = next(r for r in without if r.assessment_type.value == "llm_readiness")
        llm_with = next(r for r in with_llm if r.assessment_type.value == "llm_readiness")
        assert llm_with.score >= llm_without.score
        assert any(s.get("code") == "QUESTION_SUITABLE_FOR_LLM" for s in llm_with.strengths)
