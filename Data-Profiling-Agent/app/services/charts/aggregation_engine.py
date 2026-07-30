"""Aggregation engine — computes chart data from the transient DataFrame."""

from typing import Any

import pandas as pd
import numpy as np

from app.core.logging import get_logger
from app.services.charts.candidate_generator import ChartSpec

logger = get_logger(__name__)


class AggregationEngine:
    """Computes aggregated data for chart specifications."""

    def __init__(self, max_data_points: int = 100, default_limit: int = 15):
        self._max_points = max_data_points
        self._default_limit = default_limit

    def aggregate(self, df: pd.DataFrame, chart: ChartSpec) -> ChartSpec:
        """Compute aggregated data for a chart spec. Modifies chart.data in place."""
        try:
            if chart.chart_type.value in ("bar", "pie", "donut", "stacked_bar"):
                chart.data = self._aggregate_categorical(df, chart)
            elif chart.chart_type.value == "line":
                chart.data = self._aggregate_temporal(df, chart)
            elif chart.chart_type.value == "kpi":
                chart.data = self._aggregate_kpi(df, chart)
            elif chart.chart_type.value == "histogram":
                chart.data = self._aggregate_histogram(df, chart)
            # profiling charts may already have data
        except Exception as e:
            logger.warning("aggregation_failed", chart_key=chart.chart_key, error=str(e))
            chart.warnings.append(f"Aggregation failed: {str(e)}")

        return chart

    def aggregate_all(self, df: pd.DataFrame, charts: list[ChartSpec]) -> list[ChartSpec]:
        """Aggregate data for all charts."""
        return [self.aggregate(df, c) for c in charts]

    def _aggregate_categorical(self, df: pd.DataFrame, chart: ChartSpec) -> list[dict[str, Any]]:
        """Aggregate by category dimension."""
        if not chart.dimension or chart.dimension not in df.columns:
            return chart.data or []

        if chart.metric and chart.metric in df.columns:
            numeric_col = pd.to_numeric(df[chart.metric], errors="coerce")
            grouped = df.assign(_metric=numeric_col).groupby(chart.dimension)["_metric"]
            if chart.aggregation == "sum":
                agg = grouped.sum()
            elif chart.aggregation == "mean":
                agg = grouped.mean()
            elif chart.aggregation == "max":
                agg = grouped.max()
            elif chart.aggregation == "min":
                agg = grouped.min()
            else:
                agg = grouped.count()
        else:
            agg = df[chart.dimension].value_counts()

        agg = agg.sort_values(ascending=False).head(self._default_limit)
        return [
            {"label": str(label), "value": round(float(val), 2) if not np.isnan(val) else 0}
            for label, val in agg.items()
        ]

    def _aggregate_temporal(self, df: pd.DataFrame, chart: ChartSpec) -> list[dict[str, Any]]:
        """Aggregate over time dimension."""
        if not chart.dimension or chart.dimension not in df.columns:
            return chart.data or []

        temporal = pd.to_datetime(df[chart.dimension], errors="coerce")
        if temporal.isna().all():
            return []

        temp_df = df.assign(_time=temporal).dropna(subset=["_time"])

        if chart.metric and chart.metric in temp_df.columns:
            temp_df = temp_df.assign(_metric=pd.to_numeric(temp_df[chart.metric], errors="coerce"))
            grouped = temp_df.groupby(temp_df["_time"].dt.to_period("M"))["_metric"]
            if chart.aggregation == "sum":
                agg = grouped.sum()
            elif chart.aggregation == "mean":
                agg = grouped.mean()
            elif chart.aggregation == "max":
                agg = grouped.max()
            elif chart.aggregation == "min":
                agg = grouped.min()
            else:
                agg = grouped.count()
        else:
            agg = temp_df.groupby(temp_df["_time"].dt.to_period("M")).size()

        # NaN guard for the same reason _aggregate_categorical already has
        # one: a group whose metric values are all null/non-numeric (mean/
        # max/min, not sum) produces NaN, which json.dumps happily emits
        # as the bare, non-standard token `NaN` -- not valid JSON, and a
        # real risk for any strict client-side parser. Confirmed live via
        # AggregationEngine.aggregate() directly.
        data = [
            {"label": str(period), "value": round(float(val), 2) if not np.isnan(val) else 0}
            for period, val in agg.items()
        ]
        return data[:self._max_points]

    def _aggregate_kpi(self, df: pd.DataFrame, chart: ChartSpec) -> list[dict[str, Any]]:
        """Compute single KPI value."""
        if not chart.metric or chart.metric not in df.columns:
            return []

        numeric = pd.to_numeric(df[chart.metric], errors="coerce")
        if chart.aggregation == "sum":
            value = float(numeric.sum())
        elif chart.aggregation == "mean":
            value = float(numeric.mean())
        elif chart.aggregation == "count":
            value = float(numeric.count())
        elif chart.aggregation == "max":
            value = float(numeric.max())
        elif chart.aggregation == "min":
            value = float(numeric.min())
        else:
            value = float(numeric.sum())

        # NaN guard: mean/max/min on a metric column that's entirely
        # null/non-numeric (e.g. a borderline "metric" classification, or
        # a genuinely all-null column) returns NaN, and json.dumps
        # happily emits the bare, non-standard token `NaN` for it --
        # invalid JSON, a real risk for any strict client-side parser.
        # Matches the guard _aggregate_categorical already has.
        if np.isnan(value):
            value = 0.0
        return [{"label": chart.title, "value": round(value, 2)}]

    def _aggregate_histogram(self, df: pd.DataFrame, chart: ChartSpec) -> list[dict[str, Any]]:
        """Compute histogram bins."""
        col = chart.metric or chart.dimension
        if not col or col not in df.columns:
            return []

        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(numeric) == 0:
            return []

        counts, edges = np.histogram(numeric, bins=min(20, max(5, len(numeric) // 10)))
        return [
            {"label": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "value": int(counts[i])}
            for i in range(len(counts))
        ]
