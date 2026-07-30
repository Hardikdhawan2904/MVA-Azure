"""tests/test_planning.py — Agent 3 redesign, Phase 1, Stage 4 (plan
"zany-giggling-crayon"): AnalyticsPlanner.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from app.services.capability_resolution.analytics_capability_resolver import AnalyticsCapabilityResolver
from app.services.capability_resolution.models import ALL_CAPABILITIES, CapabilityProfile, CapabilityResult, ExecutionResult, StructuralResult
from app.services.dataset_context.local_schema_inferer import LocalSchemaInferer
from app.services.dataset_context.models import ColumnContext, DatasetContext, HierarchyInfo
from app.services.kpi_discovery.models import DiscoveredKPI, FINANCIAL_MEASURE
from app.services.kpi_discovery.semantic_kpi_discovery import SemanticKPIDiscovery
from app.services.planning.analytics_planner import AnalyticsPlanner
from app.services.question_interpreter.business_question_interpreter import BusinessQuestionInterpreter
from app.services.question_interpreter.models import QuestionIntent

_INSURANCE_CSV = Path(__file__).parent / "fixtures" / "insurance_variance_data_native.csv"
_HR_CSV = Path(r"C:\Users\dhawa\mva\Data-Profiling-Agent\tests\fixtures\hr_employee_payroll.csv")
pytestmark_insurance = pytest.mark.skipif(not _INSURANCE_CSV.exists(), reason="Insurance dataset not found")
pytestmark_hr = pytest.mark.skipif(not _HR_CSV.exists(), reason="HR fixture not found")


def _all_possible_profile() -> CapabilityProfile:
    caps = {name: CapabilityResult(StructuralResult(True, "ok"), ExecutionResult(True, confidence=0.9)) for name in ALL_CAPABILITIES}
    return CapabilityProfile(capabilities=caps)


def _none_possible_profile() -> CapabilityProfile:
    caps = {name: CapabilityResult(StructuralResult(False, "nope"), None) for name in ALL_CAPABILITIES}
    return CapabilityProfile(capabilities=caps)


def test_target_column_present_plans_classification_or_regression_plus_feature_importance():
    ctx = DatasetContext(
        row_count=100, column_count=1,
        columns=[ColumnContext(name="churned", semantic_role="metric")],
        context_source="local_fallback",
        feature_recommendation={"target_column": "churned", "problem_type": "classification"},
    )
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [])
    types = {p.analysis_type for p in plan}
    assert "classification" in types
    assert "feature_importance" in types
    assert "regression" not in types


def test_target_column_driven_plan_includes_llm_recommended_features_in_usefulness_order():
    """Regression test for a bug caught via live testing: rule_target_column_driven
    used to propose target_columns=[target] only, dropping
    feature_recommendation's feature_columns (the LLM's usefulness-rated
    feature list) entirely. ClassificationAnalyzer/RegressionAnalyzer/
    FeatureImportanceAnalyzer all read target_columns[1:] as their feature
    set, so they silently received zero features and produced no evidence
    -- reproduced live: 'Insufficient labelled rows: 0' / 'requires
    target_columns=[target_col, feature_col, ...]'. Best-usefulness-first
    ordering and stale-column filtering are both exercised here."""
    ctx = DatasetContext(
        row_count=100, column_count=4,
        columns=[
            ColumnContext(name="churned", semantic_role="metric"),
            ColumnContext(name="revenue", semantic_role="metric"),
            ColumnContext(name="cost", semantic_role="metric"),
            ColumnContext(name="region", semantic_role="dimension"),
        ],
        context_source="local_fallback",
        feature_recommendation={
            "target_column": "churned",
            "problem_type": "classification",
            "feature_columns": [
                {"column": "cost", "usefulness": "medium"},
                {"column": "revenue", "usefulness": "high"},
                {"column": "region", "usefulness": "low"},
                {"column": "no_longer_in_dataset", "usefulness": "high"},  # stale name -- must not crash or appear
            ],
        },
    )
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [])
    classification = next(p for p in plan if p.analysis_type == "classification")
    feature_importance = next(p for p in plan if p.analysis_type == "feature_importance")

    assert classification.target_columns == ["churned", "revenue", "cost", "region"]
    assert feature_importance.target_columns == ["churned", "revenue", "cost", "region"]
    assert "no_longer_in_dataset" not in classification.target_columns


def test_target_column_driven_plan_with_no_feature_columns_falls_back_to_bare_target():
    ctx = DatasetContext(
        row_count=100, column_count=1,
        columns=[ColumnContext(name="churned", semantic_role="metric")],
        context_source="local_fallback",
        feature_recommendation={"target_column": "churned", "problem_type": "classification"},
    )
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [])
    classification = next(p for p in plan if p.analysis_type == "classification")
    assert classification.target_columns == ["churned"]


def test_kpi_grounded_root_cause_always_proposed_when_kpi_exists():
    ctx = DatasetContext(row_count=100, column_count=1, columns=[], context_source="local_fallback")
    kpi = DiscoveredKPI(name="Profit Margin", formula="x", source_columns=["revenue", "cost"], semantic_basis="", category=FINANCIAL_MEASURE, kpi_type="ratio")
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [kpi])
    root_cause = next((p for p in plan if p.analysis_type == "root_cause"), None)
    assert root_cause is not None
    assert root_cause.is_kpi_grounded is True
    # Only the primary metric — never the rest of source_columns, which are
    # KPI *inputs* (e.g. "cost" here), not a real driver decomposition.
    # RootCauseAnalyzer falls through to correlation-based mode for this
    # single-column case rather than treating "cost" as if it were a
    # pre-labeled driver explaining 100% of the variance.
    assert root_cause.target_columns == ["revenue"]


def test_no_kpi_no_root_cause_proposed():
    ctx = DatasetContext(row_count=100, column_count=1, columns=[], context_source="local_fallback")
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [])
    assert not any(p.analysis_type == "root_cause" for p in plan)


def test_dimension_metric_pair_no_longer_proposes_clustering_and_segmentation_drops_the_dimension():
    """Regression test for a bug caught via live testing: rule_dimension_metric
    used to propose "clustering" AND "segmentation" with
    target_columns=[metric_name, dim_name] -- both purposes register
    cluster-on-features algorithms (clustering: K-Means; segmentation:
    K-Means/DBSCAN/Hierarchical Clustering) that treat the entire
    target_columns list as a purely-numeric feature set, so one of those
    got selected with the categorical dimension in its feature set and
    KMeans.fit_predict() raised "Input X contains NaN" at execution time --
    reproduced live for both purposes.

    clustering is dropped entirely from this rule (rule_numeric_pair_
    no_target already proposes it correctly with two real metrics).
    segmentation is kept but with target_columns=[metric_name] only --
    single-metric binning is what its deterministic strategies actually
    expect, and no other rule proposes segmentation generically (dropping
    it entirely broke the real "Segment portfolio by risk profile"
    Insurance end-to-end test -- confirmed by running it)."""
    ctx = DatasetContext(
        row_count=100, column_count=2,
        columns=[
            ColumnContext(name="revenue", semantic_role="metric", cardinality_ratio=0.9),
            ColumnContext(name="region", semantic_role="dimension", cardinality_ratio=0.05),
        ],
        context_source="local_fallback",
    )
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [])
    types = {p.analysis_type for p in plan}
    assert "clustering" not in types
    assert "comparative_analysis" in types
    assert "ranking" in types

    segmentation = next(p for p in plan if p.analysis_type == "segmentation")
    assert segmentation.target_columns == ["revenue"]


def test_two_metrics_plus_dimension_still_gets_correct_clustering_via_numeric_pair_rule():
    """Companion to the test above: when >= 2 real metric columns exist,
    rule_numeric_pair_no_target still correctly proposes "clustering" with
    a purely-numeric pair -- clustering isn't silently lost, it's produced
    by the rule that was always structurally correct for it."""
    ctx = DatasetContext(
        row_count=100, column_count=3,
        columns=[
            ColumnContext(name="revenue", semantic_role="metric", cardinality_ratio=0.9),
            ColumnContext(name="cost", semantic_role="metric", cardinality_ratio=0.9),
            ColumnContext(name="region", semantic_role="dimension", cardinality_ratio=0.05),
        ],
        context_source="local_fallback",
    )
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [])
    clustering = next(p for p in plan if p.analysis_type == "clustering")
    assert clustering.target_columns == ["revenue", "cost"]
    assert "region" not in clustering.target_columns


def test_distribution_analysis_gets_a_real_metric_not_two_categoricals():
    """Regression test for a bug caught via live testing:
    rule_two_categoricals proposed "distribution_analysis" with
    target_columns=[dim1, dim2] -- two categorical columns, no metric at
    all -- but DistributionAnalyzer's own documented convention is
    [metric_col] or [metric_col, dimension_col]; target_columns[0] MUST
    be numeric (DescriptiveDistributionStrategy does
    pd.to_numeric(..., errors="coerce").dropna() on it). Feeding it a
    categorical column turned the "metric" entirely to NaN and produced
    "Insufficient data for distribution analysis: 0 rows" every time.
    rule_dimension_metric (which already has a real metric+dimension pair
    in scope) proposes it now instead."""
    ctx = DatasetContext(
        row_count=100, column_count=3,
        columns=[
            ColumnContext(name="revenue", semantic_role="metric", cardinality_ratio=0.9),
            ColumnContext(name="region", semantic_role="dimension", cardinality_ratio=0.02),
            ColumnContext(name="segment", semantic_role="dimension", cardinality_ratio=0.015),
        ],
        context_source="local_fallback",
    )
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [])
    distribution = next(p for p in plan if p.analysis_type == "distribution_analysis")
    assert distribution.target_columns[0] == "revenue"

    association = next(p for p in plan if p.analysis_type == "association_rules")
    assert set(association.target_columns) == {"region", "segment"}


# ── Phase 4.6: question_intent.preferred_metrics/preferred_dimensions ──────

def test_root_cause_prefers_question_named_metric_over_kpi_discovery_order():
    """The bug found via live testing: a spuriously-discovered KPI winning
    kpis[0] purely by rule order shouldn't decide root-cause's target when
    the question itself names a real column."""
    ctx = DatasetContext(row_count=100, column_count=1, columns=[], context_source="local_fallback")
    kpi = DiscoveredKPI(name="Success Rate", formula="x", source_columns=["irrelevant_column"], semantic_basis="", category=FINANCIAL_MEASURE, kpi_type="rate")
    intent = QuestionIntent(candidate_analysis_types={"root_cause"}, preferred_metrics=["net_profit_actual"])
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [kpi], question_intent=intent)
    root_cause = next((p for p in plan if p.analysis_type == "root_cause"), None)
    assert root_cause is not None
    assert root_cause.target_columns == ["net_profit_actual"]
    assert root_cause.is_kpi_grounded is False


def test_root_cause_proposed_from_preferred_metric_with_zero_discovered_kpis():
    """New capability: a question-named metric is enough on its own —
    root_cause no longer strictly requires a discovered/curated KPI."""
    ctx = DatasetContext(row_count=100, column_count=1, columns=[], context_source="local_fallback")
    intent = QuestionIntent(candidate_analysis_types={"root_cause"}, preferred_metrics=["net_profit_actual"])
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [], question_intent=intent)
    root_cause = next((p for p in plan if p.analysis_type == "root_cause"), None)
    assert root_cause is not None
    assert root_cause.target_columns == ["net_profit_actual"]


def test_root_cause_falls_back_to_kpi_discovery_when_no_preferred_metric():
    ctx = DatasetContext(row_count=100, column_count=1, columns=[], context_source="local_fallback")
    kpi = DiscoveredKPI(name="Profit Margin", formula="x", source_columns=["revenue", "cost"], semantic_basis="", category=FINANCIAL_MEASURE, kpi_type="ratio")
    intent = QuestionIntent(candidate_analysis_types={"root_cause"})  # no preferred_metrics
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [kpi], question_intent=intent)
    root_cause = next((p for p in plan if p.analysis_type == "root_cause"), None)
    assert root_cause.target_columns == ["revenue"]
    assert root_cause.is_kpi_grounded is True


def test_dimension_metric_prefers_question_named_metric_and_dimension():
    ctx = DatasetContext(
        row_count=100, column_count=3,
        columns=[
            ColumnContext(name="actual_exchange_rate", semantic_role="metric"),
            ColumnContext(name="net_profit_actual", semantic_role="metric"),
            ColumnContext(name="business_segment", semantic_role="dimension", cardinality_ratio=0.1),
        ],
        context_source="local_fallback",
    )
    intent = QuestionIntent(
        candidate_analysis_types={"comparative_analysis"},
        preferred_metrics=["net_profit_actual"], preferred_dimensions=["business_segment"],
    )
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [], question_intent=intent)
    comparative = next(p for p in plan if p.analysis_type == "comparative_analysis")
    assert comparative.target_columns == ["net_profit_actual", "business_segment"]


def test_dimension_metric_falls_back_when_preferred_column_not_structurally_valid():
    """A preferred_metrics entry that isn't actually a metric-role column
    in this dataset (or a preferred_dimensions entry that got cardinality-
    filtered out) must fall back cleanly to the existing first-in-list
    behavior, not error or silently drop the analysis."""
    ctx = DatasetContext(
        row_count=100, column_count=2,
        columns=[
            ColumnContext(name="actual_exchange_rate", semantic_role="metric"),
            ColumnContext(name="business_segment", semantic_role="dimension", cardinality_ratio=0.1),
        ],
        context_source="local_fallback",
    )
    intent = QuestionIntent(
        candidate_analysis_types={"comparative_analysis"}, preferred_metrics=["nonexistent_column"],
    )
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [], question_intent=intent)
    comparative = next(p for p in plan if p.analysis_type == "comparative_analysis")
    assert comparative.target_columns == ["actual_exchange_rate", "business_segment"]


def test_temporal_metric_plans_trend_forecast_correlation_anomaly_timeseries():
    ctx = DatasetContext(
        row_count=100, column_count=2,
        columns=[
            ColumnContext(name="date", semantic_role="temporal_dimension", is_temporal=True),
            ColumnContext(name="revenue", semantic_role="metric"),
        ],
        context_source="local_fallback",
    )
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [])
    types = {p.analysis_type for p in plan}
    assert {"trend", "forecast", "correlation", "anomaly_detection", "time_series_analysis"} <= types


def test_temporal_metric_prefers_question_named_metric_over_file_order():
    """The bug found via live testing: "forecast revenue" defaulted to
    forecasting units_sold because it appeared first in the file, silently
    answering a different question than the one asked."""
    ctx = DatasetContext(
        row_count=100, column_count=3,
        columns=[
            ColumnContext(name="date", semantic_role="temporal_dimension", is_temporal=True),
            ColumnContext(name="units_sold", semantic_role="metric"),
            ColumnContext(name="revenue_actual", semantic_role="metric"),
        ],
        context_source="local_fallback",
    )
    intent = QuestionIntent(candidate_analysis_types={"forecast"}, preferred_metrics=["revenue_actual"])
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [], question_intent=intent)
    forecast = next(p for p in plan if p.analysis_type == "forecast")
    assert forecast.target_columns == ["date", "revenue_actual"]


def test_temporal_metric_falls_back_to_file_order_when_no_preferred_metric():
    ctx = DatasetContext(
        row_count=100, column_count=3,
        columns=[
            ColumnContext(name="date", semantic_role="temporal_dimension", is_temporal=True),
            ColumnContext(name="units_sold", semantic_role="metric"),
            ColumnContext(name="revenue_actual", semantic_role="metric"),
        ],
        context_source="local_fallback",
    )
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [])
    forecast = next(p for p in plan if p.analysis_type == "forecast")
    assert forecast.target_columns == ["date", "units_sold"]


def test_forecast_includes_llm_recommended_features_when_target_matches_the_metric():
    """Regression test for an inconsistency caught via live testing:
    ForecastAnalyzer's XGBoost Regressor path already reads
    target_columns[2:] as extra feature columns (its own documented
    convention), but rule_temporal_metric never populated them from
    feature_recommendation.feature_columns the way rule_target_column_driven
    now does for classification/regression -- so forecast's XGBoost variant
    fell back to structurally-derived features even when the LLM had
    already rated better ones. Only applies when the recommendation's own
    target_column IS this forecast's metric -- using features picked for a
    different target would be a category error."""
    ctx = DatasetContext(
        row_count=100, column_count=4,
        columns=[
            ColumnContext(name="date", semantic_role="temporal_dimension", is_temporal=True),
            ColumnContext(name="revenue_actual", semantic_role="metric"),
            ColumnContext(name="marketing_spend", semantic_role="metric"),
            ColumnContext(name="region", semantic_role="dimension"),
        ],
        context_source="local_fallback",
        feature_recommendation={
            "target_column": "revenue_actual",
            "feature_columns": [
                {"column": "region", "usefulness": "medium"},
                {"column": "marketing_spend", "usefulness": "high"},
            ],
        },
    )
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [])
    forecast = next(p for p in plan if p.analysis_type == "forecast")
    assert forecast.target_columns == ["date", "revenue_actual", "marketing_spend", "region"]

    # trend/correlation/anomaly_detection/time_series_analysis must NOT
    # get the extra features -- CorrelationAnalyzer/AnomalyAnalyzer consume
    # the whole target_columns list as their feature set, so injecting
    # extras there would silently change what they analyze.
    for analysis_type in ("trend", "correlation", "anomaly_detection", "time_series_analysis"):
        other = next(p for p in plan if p.analysis_type == analysis_type)
        assert other.target_columns == ["date", "revenue_actual"]


def test_forecast_ignores_feature_recommendation_for_a_different_target():
    """The recommendation's feature_columns were picked for a different
    target_column than what's being forecasted -- must not be reused."""
    ctx = DatasetContext(
        row_count=100, column_count=3,
        columns=[
            ColumnContext(name="date", semantic_role="temporal_dimension", is_temporal=True),
            ColumnContext(name="units_sold", semantic_role="metric"),
            ColumnContext(name="marketing_spend", semantic_role="metric"),
        ],
        context_source="local_fallback",
        feature_recommendation={
            "target_column": "revenue_actual",  # a different metric, not even in this dataset
            "feature_columns": [{"column": "marketing_spend", "usefulness": "high"}],
        },
    )
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [])
    forecast = next(p for p in plan if p.analysis_type == "forecast")
    assert forecast.target_columns == ["date", "units_sold"]


def test_structurally_impossible_analysis_never_planned():
    """The Planner only checks structural.supported — when a capability's
    resolver already said structurally impossible, it must never appear
    in the plan, full stop (not even as a low-priority entry)."""
    ctx = DatasetContext(
        row_count=100, column_count=2,
        columns=[
            ColumnContext(name="date", semantic_role="temporal_dimension", is_temporal=True),
            ColumnContext(name="revenue", semantic_role="metric"),
        ],
        context_source="local_fallback",
    )
    plan = AnalyticsPlanner().plan(ctx, _none_possible_profile(), [])
    # forecast/time_series_analysis/anomaly_detection are capability-gated and blocked;
    # trend/correlation are NOT gated (no entry in ANALYSIS_TYPE_TO_CAPABILITY... wait,
    # correlation IS gated) -- only ungated types should survive
    types = {p.analysis_type for p in plan}
    assert "forecast" not in types
    assert "time_series_analysis" not in types
    assert "anomaly_detection" not in types
    assert "correlation" not in types
    assert "trend" in types  # trend has no capability gate


def test_question_intent_narrows_the_final_plan():
    ctx = DatasetContext(
        row_count=100, column_count=2,
        columns=[
            ColumnContext(name="date", semantic_role="temporal_dimension", is_temporal=True),
            ColumnContext(name="revenue", semantic_role="metric"),
        ],
        context_source="local_fallback",
    )
    profile = _all_possible_profile()
    unrestricted = AnalyticsPlanner().plan(ctx, profile, [])
    intent = QuestionIntent(candidate_analysis_types={"forecast", "trend"})
    narrowed = AnalyticsPlanner().plan(ctx, profile, [], question_intent=intent)
    assert {p.analysis_type for p in narrowed} == {"forecast", "trend"}
    assert len(narrowed) < len(unrestricted)


def test_plan_is_deduplicated_and_sorted_by_priority():
    ctx = DatasetContext(
        row_count=100, column_count=2,
        columns=[
            ColumnContext(name="date", semantic_role="temporal_dimension", is_temporal=True),
            ColumnContext(name="revenue", semantic_role="metric"),
        ],
        context_source="local_fallback",
    )
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [])
    types = [p.analysis_type for p in plan]
    assert len(types) == len(set(types))  # no duplicates
    priorities = [p.priority for p in plan]
    assert priorities == sorted(priorities)  # sorted ascending


def test_hierarchy_detected_plans_comparative_and_ranking():
    ctx = DatasetContext(
        row_count=100, column_count=0, columns=[], context_source="agent2",
        hierarchy=HierarchyInfo(status="accepted", template_key="geo", level_columns=["region", "country"]),
    )
    plan = AnalyticsPlanner().plan(ctx, _all_possible_profile(), [])
    types = {p.analysis_type for p in plan}
    assert "comparative_analysis" in types
    assert "ranking" in types


# ── Real, cross-domain end-to-end (Stages 0-4 chained) ───────────────────────

@pytestmark_insurance
def test_full_pipeline_report_mode_against_real_insurance_dataset():
    df = pd.read_csv(_INSURANCE_CSV)
    ctx = LocalSchemaInferer().infer(df)
    profile = AnalyticsCapabilityResolver().resolve(ctx, 92.0, 95.0)
    kpis = SemanticKPIDiscovery().discover(ctx)
    plan = AnalyticsPlanner().plan(ctx, profile, kpis, question_intent=None)
    assert len(plan) > 0
    assert any(p.analysis_type == "root_cause" and p.is_kpi_grounded for p in plan)


@pytestmark_insurance
def test_full_pipeline_question_driven_narrows_correctly():
    df = pd.read_csv(_INSURANCE_CSV)
    ctx = LocalSchemaInferer().infer(df)
    profile = AnalyticsCapabilityResolver().resolve(ctx, 92.0, 95.0)
    kpis = SemanticKPIDiscovery().discover(ctx)
    question_intent = BusinessQuestionInterpreter().interpret(
        "Forecast underwriting result for next 6 months", profile, kpis,
    )
    plan = AnalyticsPlanner().plan(ctx, profile, kpis, question_intent=question_intent)
    types = {p.analysis_type for p in plan}
    assert types <= {"forecast", "trend", "time_series_analysis", "anomaly_detection"}
    assert "clustering" not in types
    assert "association_rules" not in types


def test_rule_dimension_metric_deprioritizes_period_fragment_dimensions():
    """Regression test for a bug caught via live testing against a 2-year
    banking dataset: rule_dimension_metric fell back to dims[0] in file
    order, which happened to be a bare "month" column (1-12, no year) --
    comparing by it on a multi-year dataset silently sums different years
    of the same month together (confirmed live: the reported "highest in
    March" was actually March 2023 + March 2024 summed, not a real single
    period). "region" (a genuine business dimension) must be preferred
    over "month" even though "month" appears first in column order."""
    from app.services.planning.planning_rules import rule_dimension_metric

    ctx = DatasetContext(
        row_count=100, column_count=3,
        columns=[
            ColumnContext(name="month", semantic_role="dimension", cardinality_ratio=12 / 100),
            ColumnContext(name="region", semantic_role="dimension", cardinality_ratio=20 / 100),
            ColumnContext(name="revenue", semantic_role="metric"),
        ],
        context_source="local_fallback",
    )
    plans = rule_dimension_metric(ctx, _all_possible_profile(), [], None)
    comparative = next(p for p in plans if p.analysis_type == "comparative_analysis")
    assert comparative.target_columns[1] == "region"


def test_rule_dimension_metric_still_uses_period_fragment_when_only_option():
    """Deprioritized, not excluded -- still available when it's the only
    workable dimension in the dataset."""
    from app.services.planning.planning_rules import rule_dimension_metric

    ctx = DatasetContext(
        row_count=1200, column_count=2,
        columns=[
            ColumnContext(name="month", semantic_role="dimension", cardinality_ratio=12 / 1200),
            ColumnContext(name="revenue", semantic_role="metric"),
        ],
        context_source="local_fallback",
    )
    plans = rule_dimension_metric(ctx, _all_possible_profile(), [], None)
    comparative = next(p for p in plans if p.analysis_type == "comparative_analysis")
    assert comparative.target_columns[1] == "month"


@pytestmark_hr
def test_full_pipeline_generalizes_to_hr_dataset():
    """The actual point of the whole chain: works on a domain it's never
    seen, using only semantic vocabulary, zero HR-specific code."""
    df = pd.read_csv(_HR_CSV)
    ctx = LocalSchemaInferer().infer(df)
    profile = AnalyticsCapabilityResolver().resolve(ctx, 90.0, 90.0)
    kpis = SemanticKPIDiscovery().discover(ctx)
    plan = AnalyticsPlanner().plan(ctx, profile, kpis, question_intent=None)
    assert len(plan) > 0
    types = {p.analysis_type for p in plan}
    assert "trend" in types or "forecast" in types  # hire_date + salary
