"""Tests for AI readiness assessments."""

import pytest
import pandas as pd

from app.core.config import Settings
from app.core.enums import ReadinessType, ReadinessStatus
from app.services.profiling.column_profiler import ColumnProfiler
from app.services.readiness.readiness_engine import ReadinessEngine, _determine_status


@pytest.fixture
def profiler() -> ColumnProfiler:
    settings = Settings(DATABASE_URL="postgresql://x:x@localhost/test", MAX_SAMPLE_VALUES=5)
    return ColumnProfiler(settings)


@pytest.fixture
def engine() -> ReadinessEngine:
    return ReadinessEngine()


class TestReadinessThresholds:
    def test_ready(self):
        assert _determine_status(80.0) == ReadinessStatus.READY
        assert _determine_status(100.0) == ReadinessStatus.READY

    def test_partially_ready(self):
        assert _determine_status(60.0) == ReadinessStatus.PARTIALLY_READY
        assert _determine_status(79.99) == ReadinessStatus.PARTIALLY_READY

    def test_not_ready(self):
        assert _determine_status(0.0) == ReadinessStatus.NOT_READY
        assert _determine_status(59.99) == ReadinessStatus.NOT_READY


class TestReadinessEngine:
    def test_all_four_assessments_returned(self, profiler, engine):
        df = pd.DataFrame({
            "id": [f"R{i}" for i in range(100)],
            "amount": [str(i * 10) for i in range(100)],
            "status": ["active", "inactive"] * 50,
            "date": ["2024-01-01"] * 100,
        })
        profiles = profiler.profile_all(df, ["id", "amount", "status", "date"])
        quality = [{"dimension": "completeness", "score": 0.95, "status": "assessed"}]

        results = engine.assess_all(
            profiles=profiles,
            quality_results=quality,
            grain_columns=["id"],
            has_temporal=True,
            metric_count=1,
            dimension_count=1,
            description_coverage=0.5,
            row_count=100,
        )

        types = [r.assessment_type for r in results]
        assert ReadinessType.ANALYTICS in types
        assert ReadinessType.ML in types
        assert ReadinessType.LLM in types
        assert ReadinessType.OVERALL in types

    def test_structured_evidence(self, profiler, engine):
        df = pd.DataFrame({"x": [str(i) for i in range(50)]})
        profiles = profiler.profile_all(df, ["x"])
        results = engine.assess_all(profiles, [], [], False, 0, 0, 0.0, 50)

        for r in results:
            assert isinstance(r.evidence, list)
            assert isinstance(r.strengths, list)
            assert isinstance(r.blocking_issues, list)

    def test_scores_between_0_and_100(self, profiler, engine):
        df = pd.DataFrame({"x": ["a"] * 20})
        profiles = profiler.profile_all(df, ["x"])
        results = engine.assess_all(profiles, [], [], False, 0, 0, 0.0, 20)
        for r in results:
            assert 0.0 <= r.score <= 100.0

    def test_reuses_common_evidence(self, profiler, engine):
        """All readiness types should reference quality scores from same source."""
        quality = [
            {"dimension": "completeness", "score": 0.95, "status": "assessed"},
            {"dimension": "consistency", "score": 0.88, "status": "assessed"},
        ]
        df = pd.DataFrame({"x": [str(i) for i in range(100)]})
        profiles = profiler.profile_all(df, ["x"])
        results = engine.assess_all(profiles, quality, [], True, 2, 3, 0.8, 100)

        analytics = next(r for r in results if r.assessment_type == ReadinessType.ANALYTICS)
        ml = next(r for r in results if r.assessment_type == ReadinessType.ML)
        # Both should reference completeness in evidence
        analytics_dims = [e.get("dimension") for e in analytics.evidence]
        ml_dims = [e.get("dimension") for e in ml.evidence]
        assert "completeness" in analytics_dims
        assert "completeness" in ml_dims

    def test_analytics_readiness_dataset_score_mirrors_score(self, profiler, engine):
        """Analytics readiness has no task-dependent input at all (no
        feature_recommendation is ever passed to _assess_analytics) --
        dataset_score must equal score exactly, and there's no task to
        report compatibility with."""
        df = pd.DataFrame({"x": [str(i) for i in range(50)]})
        profiles = profiler.profile_all(df, ["x"])
        results = engine.assess_all(profiles, [], [], False, 0, 0, 0.0, 50)
        analytics = next(r for r in results if r.assessment_type == ReadinessType.ANALYTICS)
        assert analytics.dataset_score == analytics.score
        assert analytics.task_compatibility_score is None

    def test_ml_readiness_no_feature_recommendation_has_no_task_score(self, profiler, engine):
        """No business_question -> feature_recommendation=None -> nothing
        task-dependent was ever evaluated, so task_compatibility_score must
        be None (there's no task), not 0 (which would read as 'bad task
        fit')."""
        df = pd.DataFrame({"x": [str(i) for i in range(100)]})
        profiles = profiler.profile_all(df, ["x"])
        results = engine.assess_all(profiles, [], [], False, 0, 0, 0.5, 100, feature_recommendation=None)
        ml = next(r for r in results if r.assessment_type == ReadinessType.ML)
        assert ml.task_compatibility_score is None
        assert ml.dataset_score is not None
        assert 0.0 <= ml.dataset_score <= 100.0

    def test_ml_readiness_with_feature_recommendation_splits_dataset_and_task(self, profiler, engine):
        """With a real feature_recommendation (a business_question was
        asked), both components must be present and independently in
        [0, 100] -- and `score` itself (already covered by the existing
        exact-value tests elsewhere) must be unaffected by this split."""
        df = pd.DataFrame({
            "revenue": [str(i) for i in range(100)],
            "id": [f"R{i}" for i in range(100)],
        })
        profiles = profiler.profile_all(df, ["revenue", "id"])
        feature_recommendation = {
            "target_column": "revenue",
            "feature_columns": [{"column": "revenue", "usefulness": "high"}],
            "drop_columns": [{"column": "id"}],
            "approach_reasoning": "numeric target",
        }
        results = engine.assess_all(
            profiles, [], ["id"], False, 1, 0, 0.5, 100, feature_recommendation=feature_recommendation,
        )
        ml = next(r for r in results if r.assessment_type == ReadinessType.ML)
        assert ml.dataset_score is not None
        assert ml.task_compatibility_score is not None
        assert 0.0 <= ml.dataset_score <= 100.0
        assert 0.0 <= ml.task_compatibility_score <= 100.0

    def test_feature_coverage_not_penalized_for_dataset_width(self, profiler, engine):
        """Regression test for a bug caught via live testing: the same,
        equally-good recommendation (10 well-chosen 'high'-usefulness
        feature columns) scored feature_coverage ~1.0 on a 20-column
        dataset but ~0.06 on a real 141-column dataset under the old
        len(profiles)-relative ratio -- purely because the dataset was
        wider, not because the recommendation was worse. Naming 10 good
        features must score identically regardless of how many other,
        deliberately-unmentioned low-signal columns the dataset has."""
        ten_features = [
            {"column": f"feat{i}", "usefulness": "high"} for i in range(10)
        ]

        def _profiles_and_recommendation(total_columns: int):
            cols = {f"feat{i}": [str(j) for j in range(20)] for i in range(min(10, total_columns))}
            for i in range(10, total_columns):
                cols[f"other{i}"] = [str(j) for j in range(20)]
            df = pd.DataFrame(cols)
            profiles = profiler.profile_all(df, list(df.columns))
            feature_recommendation = {
                "target_column": "feat0",
                "feature_columns": ten_features,
                "drop_columns": [],
                "approach_reasoning": "test",
            }
            return profiles, feature_recommendation

        narrow_profiles, narrow_rec = _profiles_and_recommendation(20)
        wide_profiles, wide_rec = _profiles_and_recommendation(141)

        narrow_results = engine.assess_all(narrow_profiles, [], [], False, 1, 0, 0.5, 100, feature_recommendation=narrow_rec)
        wide_results = engine.assess_all(wide_profiles, [], [], False, 1, 0, 0.5, 100, feature_recommendation=wide_rec)

        narrow_ml = next(r for r in narrow_results if r.assessment_type == ReadinessType.ML)
        wide_ml = next(r for r in wide_results if r.assessment_type == ReadinessType.ML)

        narrow_coverage = next(e["value"] for e in narrow_ml.evidence if e["dimension"] == "feature_coverage")
        wide_coverage = next(e["value"] for e in wide_ml.evidence if e["dimension"] == "feature_coverage")

        assert narrow_coverage == wide_coverage == 1.0
        assert narrow_ml.task_compatibility_score == wide_ml.task_compatibility_score

    def test_identifier_contamination_not_triggered_by_a_thorough_correct_drop_list(self, profiler, engine):
        """Regression test for a bug caught via live testing: the check was
        computing len(drop_columns)/len(profiles) and calling that
        "identifier contamination" -- so a genuinely good, selective
        recommendation on a 20-column dataset (10 good features, 9 other
        columns correctly dropped including the real identifier) got
        blocked with IDENTIFIER_CONTAMINATION at 0.45 and a -10 point
        penalty, purely for being thorough about what NOT to use. The
        real signal is whether an actual identifier/grain column ended up
        IN feature_columns, not how many columns got dropped overall."""
        cols = {"id": [str(i) for i in range(100)]}
        for i in range(19):
            cols[f"col{i}"] = [str(j) for j in range(100)]
        df = pd.DataFrame(cols)
        profiles = profiler.profile_all(df, list(df.columns))
        recommendation = {
            "target_column": "col0",
            "feature_columns": [{"column": f"col{i}", "usefulness": "high"} for i in range(1, 11)],
            "drop_columns": (
                [{"column": "id", "reason": "identifier"}]
                + [{"column": f"col{i}", "reason": "low signal"} for i in range(11, 19)]
            ),
            "approach_reasoning": "test", "confidence": 0.9,
        }
        results = engine.assess_all(profiles, [], ["id"], False, 10, 0, 0.5, 100, feature_recommendation=recommendation)
        ml = next(r for r in results if r.assessment_type == ReadinessType.ML)
        assert not any(b["code"] == "IDENTIFIER_CONTAMINATION" for b in ml.blocking_issues)

    def test_identifier_contamination_detects_a_real_leak_into_feature_columns(self, profiler, engine):
        """Companion to the test above: the actual failure mode -- an
        identifier/grain column left directly in feature_columns, nothing
        dropped at all -- must now be caught. Previously this triggered
        zero blocking issues under the old len(drop_columns)-relative
        formula, since drop_columns was empty."""
        cols = {"id": [str(i) for i in range(100)]}
        for i in range(19):
            cols[f"col{i}"] = [str(j) for j in range(100)]
        df = pd.DataFrame(cols)
        profiles = profiler.profile_all(df, list(df.columns))
        recommendation = {
            "target_column": "col0",
            "feature_columns": [{"column": "id", "usefulness": "high"}, {"column": "col1", "usefulness": "high"}],
            "drop_columns": [],
            "approach_reasoning": "test", "confidence": 0.9,
        }
        results = engine.assess_all(profiles, [], ["id"], False, 10, 0, 0.5, 100, feature_recommendation=recommendation)
        ml = next(r for r in results if r.assessment_type == ReadinessType.ML)
        assert any(b["code"] == "IDENTIFIER_CONTAMINATION" for b in ml.blocking_issues)

    def test_llm_readiness_task_compatibility_reflects_confidence(self, profiler, engine):
        """task_compatibility_score for LLM readiness is the underlying
        recommendation confidence rescaled to 0-100 -- distinct from the
        boost itself, which is capped at 10 points."""
        df = pd.DataFrame({"x": [str(i) for i in range(50)]})
        profiles = profiler.profile_all(df, ["x"])
        feature_recommendation = {"recommended_approach": "llm", "confidence": 0.85}
        results = engine.assess_all(
            profiles, [], [], False, 0, 0, 0.8, 50, feature_recommendation=feature_recommendation,
        )
        llm = next(r for r in results if r.assessment_type == ReadinessType.LLM)
        assert llm.task_compatibility_score == pytest.approx(85.0)

    def test_llm_readiness_no_task_when_approach_is_not_llm(self, profiler, engine):
        """The boost (and therefore task_compatibility_score) only applies
        when recommended_approach == 'llm' -- an 'ml'-approach question
        must leave task_compatibility_score as None, not 0."""
        df = pd.DataFrame({"x": [str(i) for i in range(50)]})
        profiles = profiler.profile_all(df, ["x"])
        feature_recommendation = {"recommended_approach": "ml", "confidence": 0.9}
        results = engine.assess_all(
            profiles, [], [], False, 0, 0, 0.8, 50, feature_recommendation=feature_recommendation,
        )
        llm = next(r for r in results if r.assessment_type == ReadinessType.LLM)
        assert llm.task_compatibility_score is None

    def test_overall_readiness_reports_a_dataset_score_and_task_compatibility_score(self, profiler, engine):
        """overall_ai_readiness previously left dataset_score/
        task_compatibility_score as None always, even though all 3 inputs
        (analytics/ml/llm) carry real values -- a caller wanting "how good
        is this dataset overall, independent of any question" had to
        average the components' own dataset_score fields themselves. This
        is that aggregate: mean of whichever components have a value."""
        df = pd.DataFrame({
            "revenue": [str(i) for i in range(100)],
            "id": [f"R{i}" for i in range(100)],
        })
        profiles = profiler.profile_all(df, ["revenue", "id"])
        feature_recommendation = {
            "target_column": "revenue",
            "feature_columns": [{"column": "revenue", "usefulness": "high"}],
            "drop_columns": [{"column": "id"}],
            "approach_reasoning": "numeric target",
        }
        results = engine.assess_all(
            profiles, [], ["id"], False, 1, 0, 0.5, 100, feature_recommendation=feature_recommendation,
        )
        overall = next(r for r in results if r.assessment_type == ReadinessType.OVERALL)
        analytics = next(r for r in results if r.assessment_type == ReadinessType.ANALYTICS)
        ml = next(r for r in results if r.assessment_type == ReadinessType.ML)
        llm = next(r for r in results if r.assessment_type == ReadinessType.LLM)

        assert overall.dataset_score is not None
        expected = (analytics.dataset_score + ml.dataset_score + llm.dataset_score) / 3.0
        assert overall.dataset_score == pytest.approx(expected)
        # ml.task_compatibility_score is populated (a real target_column/
        # feature_columns task was in play) but llm.task_compatibility_score
        # is None (recommended_approach wasn't "llm") -- overall must
        # average only the non-None component, not treat the missing one
        # as 0.
        assert llm.task_compatibility_score is None
        assert overall.task_compatibility_score == pytest.approx(ml.task_compatibility_score)

    def test_overall_readiness_task_compatibility_score_averages_ml_and_llm_when_both_present(self, profiler, engine):
        df = pd.DataFrame({"revenue": [str(i) for i in range(100)], "id": [f"R{i}" for i in range(100)]})
        profiles = profiler.profile_all(df, ["revenue", "id"])
        feature_recommendation = {
            "target_column": "revenue",
            "feature_columns": [{"column": "revenue", "usefulness": "high"}],
            "drop_columns": [{"column": "id"}],
            "recommended_approach": "llm",
            "confidence": 0.8,
            "approach_reasoning": "test",
        }
        results = engine.assess_all(
            profiles, [], ["id"], False, 1, 0, 0.5, 100, feature_recommendation=feature_recommendation,
        )
        overall = next(r for r in results if r.assessment_type == ReadinessType.OVERALL)
        ml = next(r for r in results if r.assessment_type == ReadinessType.ML)
        llm = next(r for r in results if r.assessment_type == ReadinessType.LLM)

        assert overall.task_compatibility_score is not None
        assert overall.task_compatibility_score == pytest.approx((ml.task_compatibility_score + llm.task_compatibility_score) / 2.0)

    def test_cardinality_dimension_not_assessed_when_no_non_grain_columns_exist(self, profiler, engine):
        """Regression test for a bug caught via live testing: the
        cardinality-health check trivially awarded a perfect 10/10 "no
        problems" score when there were zero non-grain columns to check
        (e.g. a dataset that's entirely its own grain key) -- "nothing to
        assess" was being reported as "assessed clean". Now the dimension
        is simply not scored (no dataset_max contribution) when there's
        nothing to check, matching every other quality dimension's
        not-assessable-vs-clean distinction."""
        df = pd.DataFrame({"id": [str(i) for i in range(50)]})
        profiles = profiler.profile_all(df, ["id"])
        results = engine.assess_all(profiles, [], ["id"], False, 0, 0, 0.5, 50, feature_recommendation=None)
        ml = next(r for r in results if r.assessment_type == ReadinessType.ML)
        assert not any(r["code"] == "HIGH_CARDINALITY_FEATURES" for r in ml.recommendations)
