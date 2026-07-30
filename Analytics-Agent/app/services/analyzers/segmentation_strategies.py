"""app/services/analyzers/segmentation_strategies.py — Stage 7 "Improve
Current Fallbacks" (plan "zany-giggling-crayon"): deterministic
segmentation/binning strategies registered in config/model_registry.yml
alongside K-Means/DBSCAN/Agglomerative Clustering.

All return the `evidence["segments"]` shape (label -> count) that
ExplanationTool's template formatter already special-cases (see
handle_segment()'s deterministic fallback in nodes/pipeline.py) — the
Insurance-specific fixed Combined-Ratio buckets in
domain_plugins/insurance/plugin.py's InsuranceCombinedRatioBucketsStrategy
follow the exact same shape, just with hardcoded thresholds instead of
data-driven bin edges.
"""

from __future__ import annotations

import pandas as pd


def _numeric_series(df: pd.DataFrame, metric_col: str) -> pd.Series:
    return pd.to_numeric(df[metric_col], errors="coerce").dropna()


class QuantileBinningStrategy:
    """Fixed quartile buckets (Bottom/Lower-Middle/Upper-Middle/Top
    Quartile) — the business-familiar 4-way split."""

    def segment(self, df: pd.DataFrame, metric_col: str) -> dict:
        if metric_col not in df.columns:
            return {"error": f"Column '{metric_col}' not found for Quantile Binning."}
        series = _numeric_series(df, metric_col)
        if len(series) < 4:
            return {"error": f"Insufficient data for Quantile Binning: {len(series)} rows (need >= 4)"}

        labels = ["Bottom Quartile", "Lower-Middle Quartile", "Upper-Middle Quartile", "Top Quartile"]
        binned = pd.qcut(series, q=4, labels=labels, duplicates="drop")
        counts = binned.value_counts().reindex(labels, fill_value=0)
        edges = pd.qcut(series, q=4, duplicates="drop").cat.categories

        return {
            "evidence": {
                "model": "Quantile Binning",
                "segments": {str(k): int(v) for k, v in counts.items()},
                "bin_edges": [round(float(e.left), 2) for e in edges] + [round(float(edges[-1].right), 2)],
                "total_records": len(series),
                "metric_column": metric_col,
            }
        }


class EqualWidthBinningStrategy:
    """Equally-sized numeric ranges (pd.cut) — bins reflect the metric's
    value range, not its distribution, so bucket sizes can be uneven."""

    def segment(self, df: pd.DataFrame, metric_col: str, n_bins: int = 3) -> dict:
        if metric_col not in df.columns:
            return {"error": f"Column '{metric_col}' not found for Equal-Width Binning."}
        series = _numeric_series(df, metric_col)
        if len(series) < n_bins:
            return {"error": f"Insufficient data for Equal-Width Binning: {len(series)} rows (need >= {n_bins})"}

        binned, edges = pd.cut(series, bins=n_bins, retbins=True, duplicates="drop")
        labels = [f"Bin {i + 1} ({round(edges[i], 2)}–{round(edges[i + 1], 2)})" for i in range(len(edges) - 1)]
        binned = binned.cat.rename_categories(labels)
        counts = binned.value_counts().reindex(labels, fill_value=0)

        return {
            "evidence": {
                "model": "Equal-Width Binning",
                "segments": {str(k): int(v) for k, v in counts.items()},
                "bin_edges": [round(float(e), 2) for e in edges],
                "total_records": len(series),
                "metric_column": metric_col,
            }
        }


class EqualFrequencyBinningStrategy:
    """Same number of records per bucket (pd.qcut), configurable bin
    count — the generic version of QuantileBinningStrategy's fixed
    quartile split."""

    def segment(self, df: pd.DataFrame, metric_col: str, n_bins: int = 3) -> dict:
        if metric_col not in df.columns:
            return {"error": f"Column '{metric_col}' not found for Equal-Frequency Binning."}
        series = _numeric_series(df, metric_col)
        if len(series) < n_bins:
            return {"error": f"Insufficient data for Equal-Frequency Binning: {len(series)} rows (need >= {n_bins})"}

        binned, edges = pd.qcut(series, q=n_bins, retbins=True, duplicates="drop")
        labels = [f"Bin {i + 1} ({round(edges[i], 2)}–{round(edges[i + 1], 2)})" for i in range(len(edges) - 1)]
        binned = binned.cat.rename_categories(labels)
        counts = binned.value_counts().reindex(labels, fill_value=0)

        return {
            "evidence": {
                "model": "Equal-Frequency Binning",
                "segments": {str(k): int(v) for k, v in counts.items()},
                "bin_edges": [round(float(e), 2) for e in edges],
                "total_records": len(series),
                "metric_column": metric_col,
            }
        }


class CategoricalGroupingStrategy:
    """No numeric metric available at all — segments are simply the
    existing categories of a dimension column, counted."""

    def segment(self, df: pd.DataFrame, dimension_col: str) -> dict:
        if dimension_col not in df.columns:
            return {"error": f"Column '{dimension_col}' not found for Categorical Grouping."}
        counts = df[dimension_col].dropna().value_counts()
        if counts.empty:
            return {"error": f"No non-null values in '{dimension_col}' to group."}

        return {
            "evidence": {
                "model": "Categorical Grouping",
                "segments": {str(k): int(v) for k, v in counts.items()},
                "total_records": int(counts.sum()),
                "dimension_column": dimension_col,
            }
        }
