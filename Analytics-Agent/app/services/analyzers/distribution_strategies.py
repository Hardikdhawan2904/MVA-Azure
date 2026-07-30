"""app/services/analyzers/distribution_strategies.py — Stage 7 (plan
"zany-giggling-crayon"): the sole "distribution_analysis" purpose
strategy registered in config/model_registry.yml.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DescriptiveDistributionStrategy:
    """Percentiles + a fixed-bin histogram, optionally broken down by a
    categorical dimension — always deterministic by design, no ML variant
    is registered for this purpose."""

    def compute(self, df: pd.DataFrame, metric_col: str, dimension_col: str | None = None, n_bins: int = 10) -> dict:
        if metric_col not in df.columns:
            return {"error": f"Column '{metric_col}' not found for distribution analysis."}

        series = pd.to_numeric(df[metric_col], errors="coerce").dropna()
        if len(series) < 2:
            return {"error": f"Insufficient data for distribution analysis: {len(series)} rows (need >= 2)"}

        percentiles = {
            f"p{p}": round(float(series.quantile(p / 100)), 2)
            for p in (5, 25, 50, 75, 95)
        }
        counts, edges = np.histogram(series, bins=min(n_bins, series.nunique()) or 1)
        histogram_bins = [
            {"range_start": round(float(edges[i]), 2), "range_end": round(float(edges[i + 1]), 2), "count": int(counts[i])}
            for i in range(len(counts))
        ]

        evidence = {
            "metric_column": metric_col,
            "count": int(series.count()),
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std(ddof=0)), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "percentiles": percentiles,
            "histogram_bins": histogram_bins,
        }

        if dimension_col and dimension_col in df.columns:
            by_group = df[[dimension_col, metric_col]].copy()
            by_group[metric_col] = pd.to_numeric(by_group[metric_col], errors="coerce")
            summary = by_group.dropna().groupby(dimension_col)[metric_col].agg(["mean", "std", "count"]).round(2)
            evidence["by_dimension"] = summary.reset_index().to_dict(orient="records")

        return {"evidence": evidence}
