"""tests/test_models_registry.py — Agent 3 redesign, Phase 1, Stage 6
(plan "zany-giggling-crayon"): ModelRegistry and ModelSelector.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from app.services.capability_resolution.analytics_capability_resolver import AnalyticsCapabilityResolver
from app.services.capability_resolution.models import (
    ALL_CAPABILITIES, FORECASTING, SEGMENTATION, CapabilityProfile, CapabilityResult, ExecutionResult, StructuralResult,
)
from app.services.dataset_context.local_schema_inferer import LocalSchemaInferer
from app.services.dataset_context.models import ColumnContext, DatasetContext
from app.services.models_registry.algorithm_spec import COST_EXPENSIVE
from app.services.models_registry.model_registry import ModelRegistry
from app.services.models_registry.model_selector import ModelSelector
from app.services.planning.models import PlannedAnalysis
from app.services.scheduling.budget_config import BudgetConfig
from app.services.scheduling.models import ScheduledAnalysis

_INSURANCE_CSV = Path(__file__).parent / "fixtures" / "insurance_variance_data_native.csv"
pytestmark_insurance = pytest.mark.skipif(not _INSURANCE_CSV.exists(), reason="Insurance dataset not found")


# ── ModelRegistry ────────────────────────────────────────────────────────────

def test_registry_loads_algorithms_from_yaml():
    reg = ModelRegistry()
    assert len(reg.all()) > 0


def test_every_analysis_type_has_a_deterministic_fallback():
    """Null Object / graceful-degradation guarantee from the plan: no
    analysis type can ever produce zero evidence. Enforced structurally
    here rather than trusting the YAML by inspection alone."""
    reg = ModelRegistry()
    purposes = sorted(set(a.purpose for a in reg.all()))
    missing = [p for p in purposes if not reg.deterministic_algorithms_for(p)]
    assert missing == [], f"Analysis types with no deterministic fallback registered: {missing}"


def test_disabled_algorithms_excluded_by_default():
    reg = ModelRegistry()
    forecast_algos = {a.algorithm for a in reg.for_purpose("forecast")}
    assert "ARIMA" not in forecast_algos  # disabled — needs statsmodels/pmdarima, Phase 6


def test_ml_and_deterministic_split_for_forecast():
    reg = ModelRegistry()
    ml = {a.algorithm for a in reg.ml_algorithms_for("forecast")}
    det = {a.algorithm for a in reg.deterministic_algorithms_for("forecast")}
    assert "Prophet" in ml
    assert "Moving Average" in det
    assert ml.isdisjoint(det)


# ── ModelSelector — pure unit tests with hand-built fixtures ────────────────

def _forecast_ctx(row_count=100):
    return DatasetContext(
        row_count=row_count, column_count=2,
        columns=[
            ColumnContext(name="date", semantic_role="temporal_dimension", is_temporal=True),
            ColumnContext(name="revenue", semantic_role="metric"),
        ],
        context_source="local_fallback",
    )


def _profile(capability: str, structural=True, execution=True, confidence=0.9):
    caps = {name: CapabilityResult(StructuralResult(False, "n/a"), None) for name in ALL_CAPABILITIES}
    caps[capability] = CapabilityResult(
        StructuralResult(structural, "ok"),
        ExecutionResult(execution, confidence=confidence) if structural else None,
    )
    return CapabilityProfile(capabilities=caps)


def _scheduled(analysis_type, ml_execution_allowed=True):
    pa = PlannedAnalysis(analysis_type=analysis_type, target_columns=["revenue", "date"], priority=1)
    return ScheduledAnalysis(sequence=1, planned_analysis=pa, ml_execution_allowed=ml_execution_allowed)


def test_selector_picks_real_ml_model_when_allowed_and_viable():
    ctx = _forecast_ctx()
    profile = _profile(FORECASTING, execution=True)
    budget = BudgetConfig(max_parallel_analyses=8, max_ml_analyses=3, max_expensive_operations=2)
    selector = ModelSelector(ModelRegistry(), budget)
    selected = selector.select(_scheduled("forecast", ml_execution_allowed=True), ctx, profile)
    assert selected.algorithm == "Prophet"
    assert selected.requires_ml is True


def test_selector_falls_back_to_deterministic_when_execution_not_supported():
    ctx = _forecast_ctx()
    profile = _profile(FORECASTING, execution=False)
    budget = BudgetConfig(max_parallel_analyses=8, max_ml_analyses=3, max_expensive_operations=2)
    selector = ModelSelector(ModelRegistry(), budget)
    selected = selector.select(_scheduled("forecast", ml_execution_allowed=True), ctx, profile)
    assert selected.requires_ml is False
    assert selected.algorithm in {"Moving Average", "Linear Trend", "Exponential Smoothing"}


def test_selector_falls_back_to_deterministic_when_scheduler_denied_ml_slot():
    ctx = _forecast_ctx()
    profile = _profile(FORECASTING, execution=True)  # would be ML-viable...
    budget = BudgetConfig(max_parallel_analyses=8, max_ml_analyses=3, max_expensive_operations=2)
    selector = ModelSelector(ModelRegistry(), budget)
    selected = selector.select(_scheduled("forecast", ml_execution_allowed=False), ctx, profile)  # ...but budget said no
    assert selected.requires_ml is False


def test_ungated_analysis_type_always_deterministic_regardless_of_ml_execution_allowed():
    ctx = _forecast_ctx()
    profile = _profile(FORECASTING)  # irrelevant capability; trend/root_cause have no gate at all
    budget = BudgetConfig(max_parallel_analyses=8, max_ml_analyses=3, max_expensive_operations=2)
    selector = ModelSelector(ModelRegistry(), budget)
    selected = selector.select(_scheduled("trend", ml_execution_allowed=True), ctx, profile)
    assert selected.requires_ml is False
    assert selected.algorithm == "Trend (linear regression over time)"


def test_ungated_analysis_type_reason_never_claims_an_execution_gate_failed():
    """Regression test for a bug caught via live testing: 'trend' has no
    capability gate at all (ANALYSIS_TYPE_TO_CAPABILITY has no entry for
    it), so is_analysis_type_ml_viable('trend') is always False by design,
    not because any readiness threshold check failed. Reusing the same
    "...execution gate did not..." reason text for both cases made
    base.py's substring-matched ml_readiness_blocked come out True for
    'trend' even when the real ml_readiness_score was above threshold —
    graph.py's trace builder then rendered a self-contradictory
    {score: 76.1, threshold: 75, passed: false}. The reason text for a
    genuinely ungated analysis type must never contain "execution gate"."""
    ctx = _forecast_ctx()
    # Even with a capability profile that says ML is fully viable for
    # FORECASTING, 'trend' itself has no capability entry — its reason
    # must reflect "no gate", not "gate failed".
    profile = _profile(FORECASTING, execution=True, confidence=0.99)
    budget = BudgetConfig(max_parallel_analyses=8, max_ml_analyses=3, max_expensive_operations=2)
    selector = ModelSelector(ModelRegistry(), budget)
    selected = selector.select(_scheduled("trend", ml_execution_allowed=True), ctx, profile)
    assert not any("execution gate" in r for r in selected.reasons)
    assert any("no ML-capable algorithm registered" in r for r in selected.reasons)


def test_genuinely_gated_analysis_type_still_reports_execution_gate_failure():
    """Companion to the test above: a real capability-gated analysis type
    (forecast) whose execution gate genuinely fails must still get the
    "execution gate did not" reason -- this fix must not blur the two
    cases together in the other direction."""
    ctx = _forecast_ctx()
    profile = _profile(FORECASTING, execution=False)
    budget = BudgetConfig(max_parallel_analyses=8, max_ml_analyses=3, max_expensive_operations=2)
    selector = ModelSelector(ModelRegistry(), budget)
    selected = selector.select(_scheduled("forecast", ml_execution_allowed=True), ctx, profile)
    assert any("execution gate did not" in r for r in selected.reasons)


def test_expensive_budget_exhaustion_downgrades_subsequent_selections():
    ctx = _forecast_ctx()
    profile = _profile(FORECASTING, execution=True)
    budget = BudgetConfig(max_parallel_analyses=8, max_ml_analyses=8, max_expensive_operations=1)
    selector = ModelSelector(ModelRegistry(), budget)

    first = selector.select(_scheduled("forecast"), ctx, profile)
    assert first.cost_tier == COST_EXPENSIVE
    assert first.algorithm == "Prophet"

    second = selector.select(_scheduled("forecast"), ctx, profile)
    assert second.cost_tier != COST_EXPENSIVE
    assert second.algorithm != "Prophet"


def test_requirements_filtering_excludes_algorithms_below_min_rows():
    ctx = _forecast_ctx(row_count=5)  # below Prophet's min_rows=24
    profile = _profile(FORECASTING, execution=True)
    budget = BudgetConfig(max_parallel_analyses=8, max_ml_analyses=3, max_expensive_operations=2)
    selector = ModelSelector(ModelRegistry(), budget)
    selected = selector.select(_scheduled("forecast"), ctx, profile)
    assert selected.algorithm != "Prophet"  # excluded by min_rows requirement


def test_requirements_filtering_excludes_kmeans_when_target_columns_has_fewer_than_two_columns():
    """Regression test for a bug caught via live testing: segmentation's
    target_columns is correctly just [metric_col] for its single-metric-
    binning deterministic strategies (Quantile Binning, etc.), but K-Means
    still got selected as segmentation's first ML candidate -- nothing
    filtered it out despite K-Means needing >= 2 columns to cluster on --
    and then failed its own internal check at execution time, producing
    empty evidence and wasting the ML slot. min_feature_columns must
    exclude K-Means/DBSCAN/Hierarchical Clustering whenever fewer columns
    than they need are actually available, falling through to a
    deterministic strategy that can genuinely run with just one column."""
    ctx = _forecast_ctx()  # 2 real columns exist in the dataset...
    profile = _profile(SEGMENTATION, execution=True)
    budget = BudgetConfig(max_parallel_analyses=8, max_ml_analyses=3, max_expensive_operations=2)
    selector = ModelSelector(ModelRegistry(), budget)
    pa = PlannedAnalysis(analysis_type="segmentation", target_columns=["revenue"], priority=1)  # ...but only 1 is proposed
    sa = ScheduledAnalysis(sequence=1, planned_analysis=pa, ml_execution_allowed=True)
    selected = selector.select(sa, ctx, profile)
    assert selected.algorithm != "K-Means"
    assert selected.requires_ml is False


def test_selector_never_returns_none_algorithm_for_a_covered_analysis_type():
    """Direct enforcement of the Null Object guarantee end-to-end through
    the selector, not just the registry's own coverage check. Uses a
    context with a target column set — representative of how the Selector
    is actually invoked in practice, since the Planner itself never
    proposes classification/regression/feature_importance without one
    (see rule_target_column_driven)."""
    reg = ModelRegistry()
    budget = BudgetConfig(max_parallel_analyses=8, max_ml_analyses=0, max_expensive_operations=0)  # force deterministic-only
    ctx = DatasetContext(
        row_count=100, column_count=2,
        columns=[
            ColumnContext(name="date", semantic_role="temporal_dimension", is_temporal=True),
            ColumnContext(name="revenue", semantic_role="metric"),
        ],
        context_source="local_fallback",
        feature_recommendation={"target_column": "revenue", "problem_type": "regression"},
    )
    for purpose in sorted(set(a.purpose for a in reg.all())):
        selector = ModelSelector(reg, budget)
        profile = _profile(FORECASTING)  # no capability match -> always deterministic path for gated types too
        selected = selector.select(_scheduled(purpose, ml_execution_allowed=False), ctx, profile)
        assert selected.algorithm is not None, f"'{purpose}' produced no selectable algorithm at all"


# ── Real, end-to-end (Stages 0-6 chained) ────────────────────────────────────

@pytestmark_insurance
def test_full_pipeline_against_real_insurance_dataset():
    from app.services.kpi_discovery.semantic_kpi_discovery import SemanticKPIDiscovery
    from app.services.planning.analytics_planner import AnalyticsPlanner
    from app.services.scheduling.analytics_scheduler import AnalyticsScheduler
    from app.services.scheduling.budget_config import load_budget_config

    df = pd.read_csv(_INSURANCE_CSV)
    ctx = LocalSchemaInferer().infer(df)
    profile = AnalyticsCapabilityResolver().resolve(ctx, ml_readiness_score=92.0, llm_readiness_score=95.0)
    kpis = SemanticKPIDiscovery().discover(ctx)
    plan = AnalyticsPlanner().plan(ctx, profile, kpis, question_intent=None)
    budget = load_budget_config()
    scheduled = AnalyticsScheduler().schedule(plan, budget, question_intent=None)

    selector = ModelSelector(ModelRegistry(), budget)
    for sa in scheduled:
        selected = selector.select(sa, ctx, profile)
        assert selected.algorithm is not None, f"No algorithm selected for '{sa.planned_analysis.analysis_type}'"

    forecast_selection = next(
        (selector.select(sa, ctx, profile) for sa in scheduled if sa.planned_analysis.analysis_type == "forecast"), None,
    )
    if forecast_selection:
        assert forecast_selection.algorithm in {"Prophet", "XGBoost Regressor", "Moving Average", "Linear Trend", "Exponential Smoothing"}
