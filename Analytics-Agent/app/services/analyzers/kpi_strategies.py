"""app/services/analyzers/kpi_strategies.py — Stage 7, Phase 4 (plan
"zany-giggling-crayon"): KPISummaryStrategy / KPIVarianceStrategy.

Direct ports of nodes/pipeline.py's old handle_show_kpi/handle_variance
bodies (byte-identical evidence key names — see the plan's Phase 4
Detailed Design, Decision 1) — kept as two separate strategies rather than
merged, because the two handlers produce genuinely different evidence
shapes for the same underlying variance computation
(variance_amount/variance_pct/direction vs.
variance_vs_budget_amount/variance_vs_budget_pct/direction, the latter
also adding prior_year/variance_vs_prior_year_*). Both always-deterministic
(no ML variant registered for either purpose) — a curated KPI lookup and
comparison was never a model-selection decision.

Neither strategy fetches its own data or resolves the KPI itself — both
receive an already-fetched, already-period-filtered DataFrame (built once
per request by the orchestration node's shared SQLTool fetch — see the
plan's Decision 3) and the resolved KPI dict via
KPISummaryAnalyzer/KPIVarianceAnalyzer's extra_context["kpi"].
"""

from __future__ import annotations

import pandas as pd

from app.services.tools.analytics_tool import AnalyticsTool


def _aggregation_method(kpi: dict) -> str:
    unit = kpi.get("unit", "")
    return "mean" if unit in ("%", "ratio", "rate") or kpi.get("category") == "ratio" else "sum"


class KPISummaryStrategy:
    """Port of handle_show_kpi's body: actual (+ budget + variance vs.
    budget, when both columns are present)."""

    def compute(self, df: pd.DataFrame, kpi: dict) -> dict:
        if df.empty:
            return {"error": f"No data found for '{kpi.get('label', kpi.get('key'))}' with the given filters."}

        actual_col = kpi.get("actual_column")
        budget_col = kpi.get("budget_column")
        if not actual_col or actual_col not in df.columns:
            # Previously fell through to a near-empty {"kpi": ..., "unit": ...}
            # evidence dict with no explanation — the single most likely
            # first complaint on the new Finance/HR/Payments/Customer
            # starter plugins, whose curated KPIs assume specific column
            # names a real-world dataset may not use.
            return {"error": f"'{kpi.get('label', kpi.get('key'))}' expects a column named "
                              f"'{actual_col}', which this dataset doesn't have."}
        method = _aggregation_method(kpi)
        analytics = AnalyticsTool()

        evidence: dict = {"kpi": kpi.get("label"), "unit": kpi.get("unit", "")}
        if actual_col and actual_col in df.columns:
            evidence["actual"] = analytics.aggregate(df, actual_col, method=method)
        if budget_col and budget_col in df.columns:
            evidence["budget"] = analytics.aggregate(df, budget_col, method=method)
        if "actual" in evidence and "budget" in evidence:
            var = analytics.variance_vs_budget(evidence["actual"], evidence["budget"], kpi.get("higher_is_better", True))
            evidence.update(var)

        return {"evidence": evidence}


class KPIVarianceStrategy:
    """Port of handle_variance's body: actual, budget (+ variance vs.
    budget), prior_year (+ variance vs. prior year) — whichever of those
    columns the KPI definition actually has."""

    def compute(self, df: pd.DataFrame, kpi: dict) -> dict:
        if df.empty:
            return {"error": f"No data for '{kpi.get('label', kpi.get('key'))}' with the given filters."}

        actual_col = kpi.get("actual_column")
        budget_col = kpi.get("budget_column")
        prior_yr_col = kpi.get("prior_year_column")
        if not actual_col or actual_col not in df.columns:
            return {"error": f"'{kpi.get('label', kpi.get('key'))}' expects a column named "
                              f"'{actual_col}', which this dataset doesn't have."}
        method = _aggregation_method(kpi)
        analytics = AnalyticsTool()

        evidence: dict = {"kpi": kpi.get("label"), "unit": kpi.get("unit")}

        if actual_col and actual_col in df.columns:
            evidence["actual"] = analytics.aggregate(df, actual_col, method=method)
        if budget_col and budget_col in df.columns:
            evidence["budget"] = analytics.aggregate(df, budget_col, method=method)
            if "actual" in evidence:
                v = analytics.variance_vs_budget(evidence["actual"], evidence["budget"], kpi.get("higher_is_better", True))
                evidence["variance_vs_budget_amount"] = v["variance_amount"]
                evidence["variance_vs_budget_pct"] = v["variance_pct"]
                evidence["direction"] = v["direction"]
        if prior_yr_col and prior_yr_col in df.columns:
            evidence["prior_year"] = analytics.aggregate(df, prior_yr_col, method=method)
            if "actual" in evidence:
                v = analytics.variance_vs_prior_year(evidence["actual"], evidence["prior_year"], kpi.get("higher_is_better", True))
                evidence["variance_vs_prior_year_amount"] = v["variance_amount"]
                evidence["variance_vs_prior_year_pct"] = v["variance_pct"]

        return {"evidence": evidence}
