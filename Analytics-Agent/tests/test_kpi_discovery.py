"""tests/test_kpi_discovery.py — Agent 3 redesign, Phase 1, Stage 2 (plan
"zany-giggling-crayon"): SemanticKPIDiscovery.

Every rule is tested with hand-built DatasetContext fixtures (precise
control over which semantic_type combination fires which rule), plus two
real, structurally different datasets confirming genuine dataset-agnostic
generalization — the entire point of this stage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from app.services.dataset_context.local_schema_inferer import LocalSchemaInferer
from app.services.dataset_context.models import ColumnContext, DatasetContext
from app.services.kpi_discovery.kpi_discovery_rules import classify_measure_category
from app.services.kpi_discovery.models import (
    COMPENSATION_MEASURE, FINANCIAL_MEASURE, INSURANCE_MEASURE, PAYMENT_MEASURE,
)
from app.services.kpi_discovery.semantic_kpi_discovery import SemanticKPIDiscovery

_INSURANCE_CSV = Path(__file__).parent / "fixtures" / "insurance_variance_data_native.csv"
_HR_CSV = Path(r"C:\Users\dhawa\mva\Data-Profiling-Agent\tests\fixtures\hr_employee_payroll.csv")

pytestmark_insurance = pytest.mark.skipif(not _INSURANCE_CSV.exists(), reason="Insurance dataset not found")
pytestmark_hr = pytest.mark.skipif(not _HR_CSV.exists(), reason="HR fixture not found")


def _ctx(columns: list[ColumnContext]) -> DatasetContext:
    return DatasetContext(row_count=100, column_count=len(columns), columns=columns, context_source="local_fallback")


# ── Layer 1: category classification ────────────────────────────────────────

def test_classify_measure_category_by_semantic_type():
    assert classify_measure_category(ColumnContext(name="x", semantic_type="revenue_amount")) == FINANCIAL_MEASURE
    assert classify_measure_category(ColumnContext(name="x", semantic_type="salary_amount")) == COMPENSATION_MEASURE
    assert classify_measure_category(ColumnContext(name="x", semantic_type="premium_amount")) == INSURANCE_MEASURE
    assert classify_measure_category(ColumnContext(name="x", semantic_type="claims_amount")) == INSURANCE_MEASURE
    assert classify_measure_category(ColumnContext(name="x", semantic_type="monetary_amount")) == PAYMENT_MEASURE
    assert classify_measure_category(ColumnContext(name="x", semantic_type=None)) is None
    assert classify_measure_category(ColumnContext(name="x", semantic_type="unrelated_thing")) is None


# ── Layer 2: individual rules, precise control via hand-built fixtures ──────

def test_profit_margin_needs_revenue_and_expense_pair():
    ctx = _ctx([
        ColumnContext(name="revenue_actual", semantic_type="revenue_amount", semantic_role="metric"),
        ColumnContext(name="cost_actual", semantic_type="expense_amount", semantic_role="metric"),
    ])
    kpis = SemanticKPIDiscovery().discover(ctx)
    profit_margin = next((k for k in kpis if k.name == "Profit Margin"), None)
    assert profit_margin is not None
    assert profit_margin.source_columns == ["revenue_actual", "cost_actual"]
    assert profit_margin.kpi_type == "ratio"


def test_profit_margin_absent_without_both_columns():
    ctx = _ctx([ColumnContext(name="revenue_actual", semantic_type="revenue_amount", semantic_role="metric")])
    kpis = SemanticKPIDiscovery().discover(ctx)
    assert not any(k.name == "Profit Margin" for k in kpis)


def test_success_rate_needs_payment_and_status():
    ctx = _ctx([
        ColumnContext(name="transaction_amount", semantic_type="monetary_amount", semantic_role="metric"),
        ColumnContext(name="auth_status", semantic_type="status", semantic_role="dimension"),
    ])
    kpis = SemanticKPIDiscovery().discover(ctx)
    success_rate = next((k for k in kpis if k.name == "Success Rate"), None)
    assert success_rate is not None
    assert success_rate.kpi_type == "rate"


def test_success_rate_not_discovered_from_generic_status_typed_flag():
    """Regression guard (found via live testing against a Finance dataset):
    a column whose Agent-2 semantic_type happens to be "status" isn't
    necessarily a transaction success/fail outcome — e.g. a balance-sheet
    asset/liability classification flag. Only a column whose NAME actually
    looks like a status/outcome column should qualify; type label alone
    is too generic (Agent 2 can assign "status" to any flag-shaped
    dimension) and previously caused a false-positive "Success Rate" KPI."""
    ctx = _ctx([
        ColumnContext(name="net_interest_income_prior_year", semantic_type="monetary_amount", semantic_role="metric"),
        ColumnContext(name="asset_liability_flag", semantic_type="status", semantic_role="dimension"),
    ])
    kpis = SemanticKPIDiscovery().discover(ctx)
    assert not any(k.name == "Success Rate" for k in kpis)


def test_success_rate_still_fires_on_genuinely_named_status_column():
    """Regression guard for the fix above: a real, name-matched status
    column (e.g. "auth_status", mirroring payments_authorization.csv)
    must still be detected — the fix only removes the loose type-label
    branch, not the name-based check."""
    ctx = _ctx([
        ColumnContext(name="amount", semantic_type="monetary_amount", semantic_role="metric"),
        ColumnContext(name="auth_status", semantic_type="text", semantic_role="text_field"),
    ])
    kpis = SemanticKPIDiscovery().discover(ctx)
    assert any(k.name == "Success Rate" for k in kpis)


def test_salary_distribution_fires_on_compensation_measure():
    ctx = _ctx([
        ColumnContext(name="salary", semantic_type="salary_amount", semantic_role="metric"),
        ColumnContext(name="department", semantic_role="dimension"),
    ])
    kpis = SemanticKPIDiscovery().discover(ctx)
    salary_dist = next((k for k in kpis if k.name == "Salary Distribution"), None)
    assert salary_dist is not None
    assert "department" in salary_dist.formula


def test_claim_frequency_fires_on_insurance_measure():
    ctx = _ctx([ColumnContext(name="claims_reported", semantic_type="claims_amount", semantic_role="metric")])
    kpis = SemanticKPIDiscovery().discover(ctx)
    assert any(k.name == "Claim Frequency" and k.category == INSURANCE_MEASURE for k in kpis)


def test_variance_vs_budget_needs_actual_and_budget_pair():
    ctx = _ctx([
        ColumnContext(name="revenue_actual", semantic_type="actual_amount", semantic_role="metric"),
        ColumnContext(name="revenue_budget", semantic_type="budget_amount", semantic_role="metric"),
    ])
    kpis = SemanticKPIDiscovery().discover(ctx)
    variance = next((k for k in kpis if k.name == "Variance vs Budget"), None)
    assert variance is not None
    assert variance.source_columns == ["revenue_actual", "revenue_budget"]


def test_volume_trend_needs_metric_and_temporal():
    ctx = _ctx([
        ColumnContext(name="amount", semantic_type="monetary_amount", semantic_role="metric"),
        ColumnContext(name="txn_date", semantic_role="temporal_dimension", is_temporal=True),
    ])
    kpis = SemanticKPIDiscovery().discover(ctx)
    assert any(k.name == "Volume Trend" for k in kpis)


def test_ratio_analysis_only_fires_on_unclaimed_metrics():
    # both metrics are already claimed by a more specific category
    # (revenue/expense) -> ratio_analysis should NOT redundantly fire on them
    ctx = _ctx([
        ColumnContext(name="revenue_actual", semantic_type="revenue_amount", semantic_role="metric"),
        ColumnContext(name="cost_actual", semantic_type="expense_amount", semantic_role="metric"),
    ])
    kpis = SemanticKPIDiscovery().discover(ctx)
    assert not any(k.name == "Ratio Analysis" for k in kpis)

    # two generic, unclassified numeric columns -> ratio_analysis is the
    # only rule that can find a KPI here
    ctx2 = _ctx([
        ColumnContext(name="metric_a", semantic_role="metric"),
        ColumnContext(name="metric_b", semantic_role="metric"),
    ])
    kpis2 = SemanticKPIDiscovery().discover(ctx2)
    assert any(k.name == "Ratio Analysis" for k in kpis2)


def test_no_semantic_vocabulary_yields_empty_list_not_an_error():
    ctx = _ctx([ColumnContext(name="mystery_column", semantic_role="unknown")])
    kpis = SemanticKPIDiscovery().discover(ctx)
    assert kpis == []


# ── Real, cross-domain datasets ──────────────────────────────────────────────

@pytestmark_insurance
def test_discovery_against_real_insurance_dataset():
    df = pd.read_csv(_INSURANCE_CSV)
    ctx = LocalSchemaInferer().infer(df)
    kpis = SemanticKPIDiscovery().discover(ctx)
    assert len(kpis) > 0
    assert any(k.name == "Claim Frequency" for k in kpis)  # claims_reported_count_actual present


@pytestmark_hr
def test_discovery_generalizes_to_hr_dataset():
    """The actual point of this stage: no Insurance-specific hardcoding,
    dynamic discovery works on a domain it's never seen."""
    df = pd.read_csv(_HR_CSV)
    ctx = LocalSchemaInferer().infer(df)
    kpis = SemanticKPIDiscovery().discover(ctx)
    assert any(k.name == "Salary Distribution" for k in kpis)
    assert not any(k.name == "Claim Frequency" for k in kpis)  # no insurance vocabulary in this dataset
