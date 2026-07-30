"""tests/test_kpi_analyzers.py — Agent 3 redesign, Phase 4 (plan
"zany-giggling-crayon"): KPISummaryStrategy/KPIVarianceStrategy and
RootCauseAnalyzer's extra_context["total_variance"] threading.

These close the gap discovered before Phase 4 implementation: the old
handle_show_kpi/handle_variance/handle_root_cause aren't "run an algorithm
over some columns" — they resolve a curated business KPI object and
compare against it. See the plan's "Phase 4 Detailed Design" section.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from app.services.analyzers.kpi_strategies import KPISummaryStrategy, KPIVarianceStrategy
from app.services.analyzers.kpi_summary_analyzer import KPISummaryAnalyzer
from app.services.analyzers.kpi_variance_analyzer import KPIVarianceAnalyzer
from app.services.analyzers.root_cause_analyzer import RootCauseAnalyzer
from app.services.dataset_context.models import ColumnContext, DatasetContext
from app.services.models_registry.model_registry import ModelRegistry
from app.services.models_registry.model_selector import SelectedModel
from app.services.planning.models import PlannedAnalysis
from app.services.scheduling.models import ScheduledAnalysis

_GWP_KPI = {
    "key": "gross_written_premium",
    "label": "Gross Written Premium",
    "unit": "USD",
    "actual_column": "gwp_actual",
    "budget_column": "gwp_budget",
    "prior_year_column": "gwp_prior_year",
    "higher_is_better": True,
}


def _ctx() -> DatasetContext:
    return DatasetContext(row_count=2, column_count=3, columns=[ColumnContext(name="x")], context_source="local_fallback")


def _scheduled(analysis_type: str) -> ScheduledAnalysis:
    return ScheduledAnalysis(
        sequence=1,
        planned_analysis=PlannedAnalysis(analysis_type=analysis_type, target_columns=["gwp_actual", "gwp_budget"], rationale="test"),
        ml_execution_allowed=False,
    )


def _model_for(purpose: str, algorithm: str) -> SelectedModel:
    registry = ModelRegistry()
    spec = next(s for s in registry.for_purpose(purpose) if s.algorithm == algorithm)
    return SelectedModel(algorithm=spec.algorithm, implementation_class=spec.implementation_class, requires_ml=spec.requirements.requires_ml, cost_tier=spec.cost_tier, reasons=["test"])


# ── KPISummaryStrategy ───────────────────────────────────────────────────────

def test_kpi_summary_strategy_actual_and_budget_with_variance():
    df = pd.DataFrame({"gwp_actual": [100.0, 200.0], "gwp_budget": [90.0, 180.0]})
    result = KPISummaryStrategy().compute(df, _GWP_KPI)
    ev = result["evidence"]
    assert ev["actual"] == pytest.approx(300.0)
    assert ev["budget"] == pytest.approx(270.0)
    assert ev["variance_amount"] == pytest.approx(30.0)
    assert ev["direction"] == "favorable"


def test_kpi_summary_strategy_actual_only_no_variance_keys():
    df = pd.DataFrame({"gwp_actual": [100.0]})
    kpi = {**_GWP_KPI, "budget_column": None}
    result = KPISummaryStrategy().compute(df, kpi)
    ev = result["evidence"]
    assert ev["actual"] == 100.0
    assert "variance_amount" not in ev


def test_kpi_summary_strategy_empty_df_is_error():
    result = KPISummaryStrategy().compute(pd.DataFrame(), _GWP_KPI)
    assert "error" in result


def test_kpi_summary_strategy_missing_actual_column_is_an_explicit_error():
    """Real bug, found during a handover code review: when a curated KPI's
    actual_column doesn't exist in the uploaded dataset (the likely case
    for the new Finance/HR/Payments/Customer starter plugins, whose
    definitions assume specific column names), this used to silently
    return {"evidence": {"kpi": ..., "unit": ...}} with no actual/budget
    values and no explanation. Must now name the missing column instead."""
    df = pd.DataFrame({"some_other_column": [1.0, 2.0]})
    result = KPISummaryStrategy().compute(df, _GWP_KPI)
    assert "error" in result
    assert "gwp_actual" in result["error"]
    assert "evidence" not in result


def test_kpi_variance_strategy_missing_actual_column_is_an_explicit_error():
    df = pd.DataFrame({"some_other_column": [1.0, 2.0]})
    result = KPIVarianceStrategy().compute(df, _GWP_KPI)
    assert "error" in result
    assert "gwp_actual" in result["error"]
    assert "evidence" not in result


def test_kpi_summary_strategy_uses_mean_for_ratio_unit():
    ratio_kpi = {**_GWP_KPI, "unit": "%", "actual_column": "loss_ratio_actual", "budget_column": None}
    df = pd.DataFrame({"loss_ratio_actual": [80.0, 100.0]})
    result = KPISummaryStrategy().compute(df, ratio_kpi)
    assert result["evidence"]["actual"] == pytest.approx(90.0)  # mean, not sum


# ── KPIVarianceStrategy ──────────────────────────────────────────────────────

def test_kpi_variance_strategy_uses_distinct_key_names_from_summary():
    df = pd.DataFrame({"gwp_actual": [100.0], "gwp_budget": [90.0], "gwp_prior_year": [80.0]})
    result = KPIVarianceStrategy().compute(df, _GWP_KPI)
    ev = result["evidence"]
    assert ev["variance_vs_budget_amount"] == pytest.approx(10.0)
    assert ev["variance_vs_prior_year_amount"] == pytest.approx(20.0)
    # Distinct from KPISummaryStrategy's key names for the same underlying computation.
    assert "variance_amount" not in ev
    assert "variance_pct" not in ev


def test_kpi_variance_strategy_prior_year_only():
    df = pd.DataFrame({"gwp_actual": [100.0], "gwp_prior_year": [50.0]})
    kpi = {**_GWP_KPI, "budget_column": None}
    result = KPIVarianceStrategy().compute(df, kpi)
    ev = result["evidence"]
    assert ev["prior_year"] == 50.0
    assert ev["variance_vs_prior_year_amount"] == pytest.approx(50.0)
    assert "variance_vs_budget_amount" not in ev


# ── Analyzer wiring (extra_context["kpi"]) ──────────────────────────────────

def test_kpi_summary_analyzer_requires_kpi_in_extra_context():
    df = pd.DataFrame({"gwp_actual": [100.0]})
    selected = _model_for("kpi_summary", "KPI Aggregation")
    ev = KPISummaryAnalyzer().execute(df, _ctx(), _scheduled("kpi_summary"), selected)
    assert ev.evidence == {}
    assert any("extra_context" in r for r in ev.reasons)


def test_kpi_summary_analyzer_attaches_filters_from_extra_context():
    df = pd.DataFrame({"gwp_actual": [100.0], "gwp_budget": [90.0]})
    selected = _model_for("kpi_summary", "KPI Aggregation")
    ev = KPISummaryAnalyzer().execute(
        df, _ctx(), _scheduled("kpi_summary"), selected,
        extra_context={"kpi": _GWP_KPI, "filters": {"fiscal_year": "FY2025"}},
    )
    assert ev.evidence["filters"] == {"fiscal_year": "FY2025"}
    assert ev.evidence["actual"] == 100.0
    assert ev.fallback_metadata["fallback_applied"] == "KPI Aggregation"


def test_kpi_variance_analyzer_end_to_end():
    df = pd.DataFrame({"gwp_actual": [100.0], "gwp_budget": [90.0], "gwp_prior_year": [70.0]})
    selected = _model_for("kpi_variance", "KPI Variance Analysis")
    ev = KPIVarianceAnalyzer().execute(
        df, _ctx(), _scheduled("kpi_variance"), selected, extra_context={"kpi": _GWP_KPI, "filters": {}},
    )
    assert ev.evidence["variance_vs_budget_amount"] == pytest.approx(10.0)
    assert ev.evidence["variance_vs_prior_year_amount"] == pytest.approx(30.0)


# ── RootCauseAnalyzer + extra_context["total_variance"] ─────────────────────

def test_root_cause_labeled_mode_computes_unexplained_variance_from_extra_context():
    df = pd.DataFrame({
        "underwriting_result_actual": [100.0, 200.0],
        "exposure_growth_variance": [40.0, 10.0],
        "premium_rate_variance": [-10.0, 20.0],
    })
    selected = SelectedModel(
        algorithm="Deterministic Driver Decomposition",
        implementation_class="app.services.tools.root_cause_tool.RootCauseTool",
        requires_ml=False, cost_tier="cheap", reasons=["test"],
    )
    scheduled = ScheduledAnalysis(
        sequence=1,
        planned_analysis=PlannedAnalysis(
            analysis_type="root_cause",
            target_columns=["underwriting_result_actual", "exposure_growth_variance", "premium_rate_variance"],
            rationale="test",
        ),
        ml_execution_allowed=False,
    )
    ev = RootCauseAnalyzer().execute(df, _ctx(), scheduled, selected, extra_context={"total_variance": 100.0})
    # explained_variance = sum of all driver amounts = 40+10-10+20 = 60
    assert ev.evidence["explained_variance"] == pytest.approx(60.0)
    assert ev.evidence["unexplained_variance"] == pytest.approx(40.0)  # 100 - 60


def test_root_cause_labeled_mode_without_total_variance_leaves_unexplained_none():
    df = pd.DataFrame({
        "underwriting_result_actual": [100.0],
        "exposure_growth_variance": [40.0],
    })
    selected = SelectedModel(
        algorithm="Deterministic Driver Decomposition",
        implementation_class="app.services.tools.root_cause_tool.RootCauseTool",
        requires_ml=False, cost_tier="cheap", reasons=["test"],
    )
    scheduled = ScheduledAnalysis(
        sequence=1,
        planned_analysis=PlannedAnalysis(
            analysis_type="root_cause", target_columns=["underwriting_result_actual", "exposure_growth_variance"], rationale="test",
        ),
        ml_execution_allowed=False,
    )
    ev = RootCauseAnalyzer().execute(df, _ctx(), scheduled, selected)  # no extra_context at all
    assert ev.evidence["unexplained_variance"] is None
