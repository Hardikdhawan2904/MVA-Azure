"""app/services/kpi_discovery/kpi_discovery_rules.py — Layer 1 (role →
measure category) and Layer 2 (specific named-KPI synthesis) for Stage 2
of the Agent 3 redesign.

Every rule matches on `semantic_type`/`semantic_role` vocabulary only —
never a literal column name — so the same rule fires identically whether
the column is called `gross_written_premium_actual` (Insurance) or
`net_interest_income_actual` (Finance): both classify as a Financial
Measure via the same keyword pattern.

Rules are evaluated independently and their contributions concatenated
(a Chain of Responsibility where every link may contribute, not a
first-match-wins dispatch) — deliberately the same shape as
`planning.planning_rules` (Stage 4), since the two are obviously related.
"""

from __future__ import annotations

import re
from typing import Callable

from app.services.dataset_context.models import ColumnContext, DatasetContext, SEMANTIC_ROLE_METRIC
from app.services.kpi_discovery.models import (
    COMPENSATION_MEASURE, DiscoveredKPI, FINANCIAL_MEASURE, INSURANCE_MEASURE, PAYMENT_MEASURE,
)

# ── Layer 1: semantic_type -> measure category ──────────────────────────────
# Order matters — more specific categories are checked before the generic
# "Payment Measure" catch-all, so e.g. a salary column doesn't get
# miscategorized as a generic payment just because it also matches "amount".

_INSURANCE_RE = re.compile(r"claim|premium|insurance|policy|underwrit", re.IGNORECASE)
_COMPENSATION_RE = re.compile(r"salary|compensation|wage|bonus|payroll", re.IGNORECASE)
_REVENUE_RE = re.compile(r"revenue|sales_amount|income|profit", re.IGNORECASE)
_EXPENSE_RE = re.compile(r"expense|cost|budget_amount|actual_amount|forecast_amount|variance_amount", re.IGNORECASE)
_PAYMENT_RE = re.compile(r"monetary_amount|payment|transaction_amount|price|value", re.IGNORECASE)

_STATUS_RE = re.compile(r"status|state|outcome|result_flag", re.IGNORECASE)
_SUCCESS_VALUES = {"success", "approved", "completed", "paid", "settled", "won"}


def classify_measure_category(column: ColumnContext) -> str | None:
    semantic_type = (column.semantic_type or "").lower()
    if not semantic_type:
        return None
    if _INSURANCE_RE.search(semantic_type):
        return INSURANCE_MEASURE
    if _COMPENSATION_RE.search(semantic_type):
        return COMPENSATION_MEASURE
    if _REVENUE_RE.search(semantic_type) or _EXPENSE_RE.search(semantic_type):
        return FINANCIAL_MEASURE
    if _PAYMENT_RE.search(semantic_type):
        return PAYMENT_MEASURE
    return None


def _metrics_by_category(ctx: DatasetContext, category: str) -> list[ColumnContext]:
    return [c for c in ctx.columns_with_role(SEMANTIC_ROLE_METRIC) if classify_measure_category(c) == category]


def _is_revenue_like(c: ColumnContext) -> bool:
    return bool(_REVENUE_RE.search((c.semantic_type or "")))


def _is_expense_like(c: ColumnContext) -> bool:
    return bool(_EXPENSE_RE.search((c.semantic_type or "")))


def _is_budget_like(c: ColumnContext) -> bool:
    return "budget" in (c.semantic_type or "").lower()


def _is_actual_like(c: ColumnContext) -> bool:
    return "actual" in (c.semantic_type or "").lower()


def _is_forecast_like(c: ColumnContext) -> bool:
    return "forecast" in (c.semantic_type or "").lower()


# ── Layer 2 rules ────────────────────────────────────────────────────────────

def rule_profit_margin(ctx: DatasetContext) -> list[DiscoveredKPI]:
    financial = _metrics_by_category(ctx, FINANCIAL_MEASURE)
    revenue_cols = [c for c in financial if _is_revenue_like(c)]
    expense_cols = [c for c in financial if _is_expense_like(c)]
    if not revenue_cols or not expense_cols:
        return []
    revenue, expense = revenue_cols[0], expense_cols[0]
    return [DiscoveredKPI(
        name="Profit Margin",
        formula=f"({revenue.name} - {expense.name}) / {revenue.name}",
        source_columns=[revenue.name, expense.name],
        semantic_basis="Financial Measure pair (revenue-like + expense-like)",
        category=FINANCIAL_MEASURE,
        kpi_type="ratio",
        target_columns=[revenue.name, expense.name],
    )]


def rule_success_rate(ctx: DatasetContext) -> list[DiscoveredKPI]:
    payment_cols = _metrics_by_category(ctx, PAYMENT_MEASURE)
    # Name-based only, deliberately — Agent 2's semantic_type=="status" label
    # is generic (any flag-shaped dimension can get it, e.g. a balance-sheet
    # asset/liability classification flag) and isn't a reliable signal that
    # a column is actually a transaction success/fail outcome. A real
    # status/outcome column is reliably named as such (e.g. "auth_status"),
    # which the regex already catches — matching on the loose type label too
    # produced a false-positive "Success Rate" KPI on a Finance dataset with
    # no genuine payment-outcome column at all (caught via live testing).
    status_cols = [c for c in ctx.columns if _STATUS_RE.search(c.name)]
    if not payment_cols or not status_cols:
        return []
    payment, status = payment_cols[0], status_cols[0]
    return [DiscoveredKPI(
        name="Success Rate",
        formula=f"count({status.name} == success) / count(*)",
        source_columns=[payment.name, status.name],
        semantic_basis="Payment Measure + a success/fail-like status dimension",
        category=PAYMENT_MEASURE,
        kpi_type="rate",
        target_columns=[status.name],
    )]


def rule_salary_distribution(ctx: DatasetContext) -> list[DiscoveredKPI]:
    comp_cols = _metrics_by_category(ctx, COMPENSATION_MEASURE)
    if not comp_cols:
        return []
    comp = comp_cols[0]
    dims = ctx.columns_with_role("dimension")
    grouping = dims[0].name if dims else None
    formula = "descriptive stats + percentile bands"
    formula += f", grouped by {grouping}" if grouping else ""
    return [DiscoveredKPI(
        name="Salary Distribution",
        formula=formula,
        source_columns=[comp.name] + ([grouping] if grouping else []),
        semantic_basis="Compensation Measure" + (" + dimension" if grouping else ""),
        category=COMPENSATION_MEASURE,
        kpi_type="distribution",
        target_columns=[comp.name],
    )]


def rule_claim_frequency(ctx: DatasetContext) -> list[DiscoveredKPI]:
    insurance_cols = _metrics_by_category(ctx, INSURANCE_MEASURE)
    if not insurance_cols:
        return []
    claims = insurance_cols[0]
    return [DiscoveredKPI(
        name="Claim Frequency",
        formula=f"count({claims.name}) / count(exposure)",
        source_columns=[claims.name],
        semantic_basis="Insurance Measure",
        category=INSURANCE_MEASURE,
        kpi_type="rate",
        target_columns=[claims.name],
    )]


def rule_variance_vs_budget(ctx: DatasetContext) -> list[DiscoveredKPI]:
    financial = _metrics_by_category(ctx, FINANCIAL_MEASURE)
    actual_cols = [c for c in financial if _is_actual_like(c)]
    reference_cols = [c for c in financial if _is_budget_like(c) or _is_forecast_like(c)]
    if not actual_cols or not reference_cols:
        return []
    actual, reference = actual_cols[0], reference_cols[0]
    return [DiscoveredKPI(
        name="Variance vs Budget",
        formula=f"({actual.name} - {reference.name}) / {reference.name}",
        source_columns=[actual.name, reference.name],
        semantic_basis="actual_amount + budget_amount/forecast_amount",
        category=FINANCIAL_MEASURE,
        kpi_type="ratio",
        target_columns=[actual.name, reference.name],
    )]


def rule_ratio_analysis(ctx: DatasetContext) -> list[DiscoveredKPI]:
    """Two ratio-shaped numeric columns sharing a scale — the generic
    fallback for any metric pair not already claimed by a more specific
    rule above (e.g. a domain with no revenue/expense vocabulary at all,
    just two related numeric measures)."""
    metrics = ctx.columns_with_role(SEMANTIC_ROLE_METRIC)
    already_claimed = {c.name for c in metrics if classify_measure_category(c) is not None}
    unclaimed = [c for c in metrics if c.name not in already_claimed]
    if len(unclaimed) < 2:
        return []
    a, b = unclaimed[0], unclaimed[1]
    return [DiscoveredKPI(
        name="Ratio Analysis",
        formula=f"{a.name} / {b.name}",
        source_columns=[a.name, b.name],
        semantic_basis="Two unclassified numeric metric columns",
        category=FINANCIAL_MEASURE,
        kpi_type="ratio",
        target_columns=[a.name, b.name],
    )]


def rule_volume_trend(ctx: DatasetContext) -> list[DiscoveredKPI]:
    metrics = ctx.columns_with_role(SEMANTIC_ROLE_METRIC)
    temporal = [c for c in ctx.columns if c.is_temporal]
    if not metrics or not temporal:
        return []
    metric, date_col = metrics[0], temporal[0]
    return [DiscoveredKPI(
        name="Volume Trend",
        formula=f"groupby({date_col.name}).sum({metric.name})",
        source_columns=[metric.name, date_col.name],
        semantic_basis="Any metric + Date Dimension",
        category=classify_measure_category(metric) or FINANCIAL_MEASURE,
        kpi_type="sum",
        target_columns=[metric.name],
    )]


ALL_RULES: list[Callable[[DatasetContext], list[DiscoveredKPI]]] = [
    rule_profit_margin,
    rule_success_rate,
    rule_salary_distribution,
    rule_claim_frequency,
    rule_variance_vs_budget,
    rule_ratio_analysis,
    rule_volume_trend,
]
