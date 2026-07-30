"""
tools/analytics_tool.py — Tool 3: Deterministic KPI Calculation Engine

Performs ALL mathematical operations:
- KPI aggregation
- Growth % (YoY, QoQ, MoM)
- Variance (vs budget, vs prior year)
- Rankings
- Ratios

Never retrieves data — that is the SQL Tool's job.
Never explains results — that is the Explanation Tool's job.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


class AnalyticsTool:
    """
    Performs all deterministic calculations on DataFrames
    returned by the SQL Tool.
    """

    # ── Aggregation ───────────────────────────────────────────────────────────

    def aggregate(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = "sum",
    ) -> float:
        """
        Aggregate a column with sum, mean, median, min, max.
        Returns scalar. Raises if column missing.
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame. "
                             f"Available: {list(df.columns)}")
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            logger.warning(f"No numeric data in column '{column}'")
            return 0.0

        fn_map = {
            "sum":    series.sum,
            "mean":   series.mean,
            "median": series.median,
            "min":    series.min,
            "max":    series.max,
            "count":  series.count,
        }
        if method not in fn_map:
            raise ValueError(f"Unknown aggregation method '{method}'. "
                             f"Choose from: {list(fn_map.keys())}")
        result = fn_map[method]()
        logger.info(f"aggregate({column}, {method}) = {result:,.2f}")
        return float(result)

    # ── Variance Calculation ──────────────────────────────────────────────────

    def variance(
        self,
        actual: float,
        reference: float,
        label: str = "Reference",
        higher_is_better: bool = True,
    ) -> dict:
        """
        Calculate absolute and percentage variance.

        higher_is_better flips which sign counts as "favorable" — for a
        ratio KPI like loss_ratio or expense_ratio (higher_is_better=False,
        per kpi_definitions.json), coming in *below* budget is the good
        outcome, not above. Defaults to True (revenue/premium-shaped KPIs,
        where more than budget is favorable) so existing callers that don't
        pass a KPI's flag keep their current behavior.

        Returns:
            {
              "actual": ...,
              "reference": ...,
              "variance_amount": actual - reference,
              "variance_pct": (actual - reference) / |reference| * 100,
              "direction": "favorable" | "unfavorable" | "neutral",
            }
        """
        variance_amount = actual - reference
        variance_pct = (
            (variance_amount / abs(reference) * 100) if reference != 0 else None
        )
        direction = "neutral"
        if variance_amount > 0:
            direction = "favorable" if higher_is_better else "unfavorable"
        elif variance_amount < 0:
            direction = "unfavorable" if higher_is_better else "favorable"

        result = {
            "actual": round(actual, 2),
            label.lower().replace(" ", "_"): round(reference, 2),
            "variance_amount": round(variance_amount, 2),
            "variance_pct": round(variance_pct, 2) if variance_pct is not None else None,
            "direction": direction,
        }
        pct_str = f"{variance_pct:.1f}%" if variance_pct is not None else "undefined"
        logger.info(f"variance(actual={actual:,.2f}, {label}={reference:,.2f}) "
                    f"= {variance_amount:,.2f} ({pct_str})")
        return result

    def variance_vs_budget(self, actual: float, budget: float, higher_is_better: bool = True) -> dict:
        return self.variance(actual, budget, "budget", higher_is_better)

    def variance_vs_prior_year(self, actual: float, prior_year: float, higher_is_better: bool = True) -> dict:
        return self.variance(actual, prior_year, "prior_year", higher_is_better)

    # ── Growth % ──────────────────────────────────────────────────────────────

    def growth_pct(
        self,
        current: float,
        previous: float,
        period_label: str = "Period",
    ) -> dict:
        """
        Calculates percentage growth between two periods.
        Returns growth_pct = (current - previous) / |previous| * 100.
        """
        if previous == 0:
            pct = None
            note = "Prior period is zero — growth % undefined"
        else:
            pct = ((current - previous) / abs(previous)) * 100
            note = None

        result = {
            f"current_{period_label}": round(current, 2),
            f"previous_{period_label}": round(previous, 2),
            "growth_pct": round(pct, 2) if pct is not None else None,
            "note": note,
        }
        growth_str = f"{pct:.1f}%" if pct is not None else "undefined"
        logger.info(f"growth_pct({period_label}): current={current:,.2f}, "
                    f"previous={previous:,.2f}, growth={growth_str}")
        return result

    def yoy_growth(self, current_year: float, prior_year: float) -> dict:
        return self.growth_pct(current_year, prior_year, "year")

    # ── Multi-Period Trend ────────────────────────────────────────────────────

    def trend(self, df: pd.DataFrame, date_col: str, value_col: str) -> dict:
        """
        Compute period-over-period trend from a time-series DataFrame.

        Returns:
            {
              "periods": [...],
              "values": [...],
              "overall_change_pct": ...,
              "direction": "increasing" | "decreasing" | "flat",
            }
        """
        ts = df[[date_col, value_col]].copy()
        ts[value_col] = pd.to_numeric(ts[value_col], errors="coerce")
        ts = ts.dropna().sort_values(date_col)

        if len(ts) < 2:
            return {"error": "Insufficient data points for trend analysis (need >= 2)"}

        first_val = ts[value_col].iloc[0]
        last_val  = ts[value_col].iloc[-1]
        overall_change_pct = (
            ((last_val - first_val) / abs(first_val) * 100) if first_val != 0 else None
        )

        direction = "flat"
        if overall_change_pct and overall_change_pct > 1:
            direction = "increasing"
        elif overall_change_pct and overall_change_pct < -1:
            direction = "decreasing"

        result = {
            "periods":             ts[date_col].astype(str).tolist(),
            "values":              [round(v, 2) for v in ts[value_col].tolist()],
            "first_value":         round(first_val, 2),
            "last_value":          round(last_val, 2),
            "overall_change_pct":  round(overall_change_pct, 2) if overall_change_pct else None,
            "direction":           direction,
            "data_points":         len(ts),
        }
        logger.info(f"trend({value_col}): {direction}, {overall_change_pct:.1f}% over {len(ts)} periods"
                    if overall_change_pct else f"trend({value_col}): {direction}")
        return result

    # ── Ranking ───────────────────────────────────────────────────────────────

    def rank(
        self,
        df: pd.DataFrame,
        value_col: str,
        label_col: str,
        top_n: int = 5,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Rank records by a numeric column.

        Args:
            df        : Input DataFrame.
            value_col : Column to rank by.
            label_col : Column to use as the label (e.g. 'region').
            top_n     : Return top N records.
            ascending : False = highest first (default).

        Returns: Ranked DataFrame with 'rank' column added.
        """
        ranked = df[[label_col, value_col]].copy()
        ranked[value_col] = pd.to_numeric(ranked[value_col], errors="coerce")
        ranked = (
            ranked.dropna()
                  .sort_values(value_col, ascending=ascending)
                  .reset_index(drop=True)
                  .head(top_n)
        )
        ranked.index = ranked.index + 1
        ranked.index.name = "rank"
        ranked = ranked.reset_index()
        logger.info(f"rank({value_col}): top {top_n} records returned")
        return ranked

    # ── Contribution Analysis ─────────────────────────────────────────────────

    def contribution(
        self,
        df: pd.DataFrame,
        value_col: str,
        label_col: str,
    ) -> pd.DataFrame:
        """
        Calculate each category's contribution (%) to the total.

        Args:
            df        : DataFrame with numeric value_col and label_col.
            value_col : Numeric column to analyse.
            label_col : Categorical column for grouping.

        Returns: DataFrame with 'contribution_pct' column.
        """
        df = df.copy()
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        total = df[value_col].abs().sum()
        if total == 0:
            df["contribution_pct"] = 0.0
        else:
            df["contribution_pct"] = (df[value_col].abs() / total * 100).round(2)
        df = df.sort_values("contribution_pct", ascending=False).reset_index(drop=True)
        logger.info(f"contribution({value_col}): computed for {len(df)} categories")
        return df[[label_col, value_col, "contribution_pct"]]

    # ── Group Comparison ──────────────────────────────────────────────────────

    def compare_groups(
        self,
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        method: str = "sum",
    ) -> dict:
        """
        Compare a metric across a categorical dimension's groups —
        Stage 7's "comparative_analysis" purpose (Agent 3 redesign, plan
        "zany-giggling-crayon"): the domain-agnostic generalization of
        today's ad-hoc groupby calls, registered in model_registry.yml
        alongside trend()/rank() under the same AnalyticsTool entry.

        Returns:
            {
              "groups": [{group_col: ..., value_col: ..., "rank": ...}],
              "highest": {...}, "lowest": {...},
              "group_count": int,
            }
        """
        if value_col not in df.columns or group_col not in df.columns:
            return {"error": f"Columns not found for comparison: {value_col!r}, {group_col!r}"}

        work = df[[group_col, value_col]].copy()
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.dropna()
        if work.empty:
            return {"error": f"No numeric data in column '{value_col}' to compare."}

        fn_map = {"sum": "sum", "mean": "mean", "median": "median", "min": "min", "max": "max", "count": "count"}
        if method not in fn_map:
            raise ValueError(f"Unknown aggregation method '{method}'. Choose from: {list(fn_map.keys())}")

        grouped = (
            work.groupby(group_col)[value_col]
                .agg(fn_map[method])
                .sort_values(ascending=False)
                .reset_index()
        )
        grouped.index = grouped.index + 1
        grouped.index.name = "rank"
        grouped = grouped.reset_index()
        records = grouped.round(2).to_dict(orient="records")

        result = {
            "groups": records,
            "highest": records[0] if records else None,
            "lowest": records[-1] if records else None,
            "group_count": len(records),
            "method": method,
        }
        logger.info(f"compare_groups({value_col} by {group_col}): {len(records)} groups")
        return result
