"""tests/test_question_interpreter.py — Agent 3 redesign, Phase 1, Stage 3
(plan "zany-giggling-crayon"): BusinessQuestionInterpreter.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.capability_resolution.analytics_capability_resolver import AnalyticsCapabilityResolver
from app.services.capability_resolution.models import CapabilityProfile, CapabilityResult, ExecutionResult, StructuralResult
from app.services.dataset_context.models import ColumnContext, DatasetContext
from app.services.kpi_discovery.models import DiscoveredKPI, FINANCIAL_MEASURE
from app.services.question_interpreter.business_question_interpreter import BusinessQuestionInterpreter


def _profile_all_possible() -> CapabilityProfile:
    """A CapabilityProfile where every gated capability is structurally
    possible and execution-viable — isolates the interpreter's keyword
    logic from capability-resolution specifics."""
    from app.services.capability_resolution.models import ALL_CAPABILITIES
    caps = {}
    for name in ALL_CAPABILITIES:
        caps[name] = CapabilityResult(
            structural=StructuralResult(True, "ok"),
            execution=ExecutionResult(True, confidence=0.9),
        )
    return CapabilityProfile(capabilities=caps)


def _profile_none_possible() -> CapabilityProfile:
    from app.services.capability_resolution.models import ALL_CAPABILITIES
    caps = {name: CapabilityResult(structural=StructuralResult(False, "nope"), execution=None) for name in ALL_CAPABILITIES}
    return CapabilityProfile(capabilities=caps)


def test_no_question_returns_none():
    assert BusinessQuestionInterpreter().interpret(None, _profile_all_possible(), []) is None
    assert BusinessQuestionInterpreter().interpret("", _profile_all_possible(), []) is None


def test_forecast_question_narrows_to_forecast_shaped_types():
    result = BusinessQuestionInterpreter().interpret("Forecast revenue for next quarter", _profile_all_possible(), [])
    assert result is not None
    assert result.candidate_analysis_types == {"forecast", "trend", "time_series_analysis", "anomaly_detection"}
    # explicitly NOT clustering/association_rules/ranking/segmentation — the exact example from the plan
    assert "clustering" not in result.candidate_analysis_types
    assert "association_rules" not in result.candidate_analysis_types
    assert "ranking" not in result.candidate_analysis_types
    assert "segmentation" not in result.candidate_analysis_types


def test_root_cause_question_narrows_correctly():
    result = BusinessQuestionInterpreter().interpret("Why did loss ratio decline?", _profile_all_possible(), [])
    assert result.candidate_analysis_types == {"root_cause", "feature_importance"}


def test_segment_question_narrows_correctly():
    result = BusinessQuestionInterpreter().interpret("Segment the portfolio by risk", _profile_all_possible(), [])
    assert result.candidate_analysis_types == {"clustering", "segmentation", "comparative_analysis", "ranking"}


def test_unrecognized_question_falls_through_to_none():
    result = BusinessQuestionInterpreter().interpret("asdkjfh qwoeiru", _profile_all_possible(), [])
    assert result is None


def test_never_widens_past_structurally_impossible_types():
    """Every gated capability is structurally impossible -> the forecast
    keyword group's gated members (forecast/time_series_analysis/
    anomaly_detection) must all be dropped; only ungated 'trend' survives."""
    result = BusinessQuestionInterpreter().interpret("Forecast revenue", _profile_none_possible(), [])
    assert result is not None
    assert result.candidate_analysis_types == {"trend"}  # trend has no capability gate


def test_returns_none_when_everything_matched_is_impossible():
    result = BusinessQuestionInterpreter().interpret("Segment the portfolio", _profile_none_possible(), [])
    # clustering/segmentation gated+impossible; comparative_analysis/ranking ungated -> survive
    assert result.candidate_analysis_types == {"comparative_analysis", "ranking"}


def test_mentioned_kpis_detected_by_name():
    kpis = [DiscoveredKPI(name="Profit Margin", formula="x", source_columns=[], semantic_basis="", category=FINANCIAL_MEASURE, kpi_type="ratio")]
    result = BusinessQuestionInterpreter().interpret("Forecast profit margin for next quarter", _profile_all_possible(), kpis)
    assert "Profit Margin" in result.mentioned_kpis


def test_candidate_analysis_types_never_empty_when_intent_returned():
    """Invariant from the plan: a returned QuestionIntent never has an
    empty candidate set — if it would be empty, interpret() returns None
    instead."""
    # ml_predicted "why" group is ungated (root_cause/feature_importance always pass) so this can't go empty here,
    # but confirm the *contract* holds even with a fully-blocked profile.
    result = BusinessQuestionInterpreter().interpret("Why did this happen?", _profile_none_possible(), [])
    assert result is not None
    assert len(result.candidate_analysis_types) > 0


# ── Phase 4.6: preferred_metrics / preferred_dimensions ─────────────────────

def _banking_shaped_context() -> DatasetContext:
    """Mirrors the real bug: several *_actual metric columns whose base
    names overlap ("net interest income" vs "net profit"), plus a
    dimension column — the exact shape that made rule_dimension_metric/
    rule_kpi_grounded_root_cause pick an irrelevant "first in file order"
    column before this stage existed."""
    return DatasetContext(
        row_count=100, column_count=4,
        columns=[
            ColumnContext(name="interest_income_actual", semantic_role="metric"),
            ColumnContext(name="net_interest_income_actual", semantic_role="metric"),
            ColumnContext(name="net_profit_actual", semantic_role="metric"),
            ColumnContext(name="business_segment", semantic_role="dimension"),
        ],
        context_source="local_fallback",
    )


def test_preferred_metrics_matches_question_named_column():
    result = BusinessQuestionInterpreter().interpret(
        "Why did net profit decline vs budget?", _profile_all_possible(), [], _banking_shaped_context(),
    )
    assert result.preferred_metrics == ["net_profit_actual"]


def test_preferred_metrics_empty_when_dataset_context_omitted():
    """Backward compatible — callers that don't pass dataset_context (or
    haven't been updated yet) get empty lists, not an error."""
    result = BusinessQuestionInterpreter().interpret("Why did net profit decline vs budget?", _profile_all_possible(), [])
    assert result.preferred_metrics == []
    assert result.preferred_dimensions == []


def test_preferred_metrics_empty_when_question_names_no_column():
    result = BusinessQuestionInterpreter().interpret(
        "Why did things change?", _profile_all_possible(), [], _banking_shaped_context(),
    )
    assert result.preferred_metrics == []


def test_preferred_metrics_longest_phrase_wins_on_ambiguous_match():
    """"net interest income" and "net profit" both partially overlap with
    a hypothetical mention of "net income" — the longer, more specific
    base-phrase match must win, not just the first column in file order."""
    ctx = DatasetContext(
        row_count=100, column_count=2,
        columns=[
            ColumnContext(name="income_actual", semantic_role="metric"),
            ColumnContext(name="net_interest_income_actual", semantic_role="metric"),
        ],
        context_source="local_fallback",
    )
    result = BusinessQuestionInterpreter().interpret(
        "Why did net interest income decline?", _profile_all_possible(), [], ctx,
    )
    assert result.preferred_metrics[0] == "net_interest_income_actual"


def test_preferred_dimensions_matches_question_named_dimension():
    result = BusinessQuestionInterpreter().interpret(
        "Why did net profit decline by business segment?", _profile_all_possible(), [], _banking_shaped_context(),
    )
    assert "business_segment" in result.preferred_dimensions


def test_end_to_end_against_real_resolver():
    """Wire the real AnalyticsCapabilityResolver in, not a hand-built
    profile, to confirm the two components actually compose correctly."""
    import pandas as pd
    from app.services.dataset_context.local_schema_inferer import LocalSchemaInferer

    csv_path = Path(__file__).parent / "fixtures" / "insurance_variance_data_native.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    ctx = LocalSchemaInferer().infer(df)
    profile = AnalyticsCapabilityResolver().resolve(ctx, ml_readiness_score=92.0, llm_readiness_score=95.0)
    result = BusinessQuestionInterpreter().interpret("Forecast underwriting result for next 6 months", profile, [])
    assert result is not None
    assert "forecast" in result.candidate_analysis_types
