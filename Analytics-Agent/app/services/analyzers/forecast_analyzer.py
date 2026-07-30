"""app/services/analyzers/forecast_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): ForecastAnalyzer.

target_columns convention: [temporal_col, metric_col, *extra_feature_cols].
extra_feature_cols only matters for the multi-feature regression algorithm
(today's LightGBM path, registered under the "XGBoost Regressor" name —
see model_registry.yml's forecast section); Prophet and the deterministic
strategies only ever look at [temporal_col, metric_col].
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers._dataset_helpers import build_time_series, categorical_columns, infer_pandas_freq, numeric_columns
from app.services.analyzers.base import Analyzer, import_class
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class ForecastAnalyzer(Analyzer):
    analysis_type = "forecast"

    def _run(
        self,
        *,
        df: pd.DataFrame,
        dataset_context: DatasetContext,
        target_columns: list[str],
        selected_model: SelectedModel,
        extra_context: dict[str, Any],
    ) -> dict[str, Any]:
        if len(target_columns) < 2:
            return {"error": "ForecastAnalyzer requires target_columns=[temporal_col, metric_col, ...]"}
        temporal_col, metric_col = target_columns[0], target_columns[1]
        if temporal_col not in df.columns or metric_col not in df.columns:
            return {"error": f"Forecast target columns not found in data: {temporal_col!r}, {metric_col!r}"}

        strategy_cls = import_class(selected_model.implementation_class)
        algorithm = selected_model.algorithm

        if algorithm == "Prophet":
            ts_df = build_time_series(df, temporal_col, metric_col)
            freq = infer_pandas_freq(ts_df)
            result = strategy_cls().fit_and_forecast(ts_df, periods=6, freq=freq)
            return {"error": result["error"]} if "error" in result else {"evidence": result}

        if algorithm == "XGBoost Regressor":
            feature_cols = target_columns[2:] or [
                c for c in numeric_columns(dataset_context, exclude={metric_col}) if c in df.columns
            ]
            # Must intersect with feature_cols, not just "every categorical
            # column in the dataset" -- LightGBM's categorical_feature
            # param has to name columns actually present in X, or it raises
            # "Wrong type(str) or unknown name(...) in categorical_feature".
            # Confirmed live once feature_cols became a real, partial list
            # (from feature_recommendation) instead of always being every
            # numeric column in the dataset (which never overlapped with a
            # categorical column, so this bug had nothing to expose it).
            cat_cols = [c for c in categorical_columns(dataset_context) if c in feature_cols]
            result = strategy_cls().fit_and_predict(
                df, target_col=metric_col, feature_cols=feature_cols, categorical_cols=cat_cols,
            )
            return {"error": result["error"]} if "error" in result else {"evidence": result}

        # Deterministic strategies (Moving Average, Linear Trend, Exponential
        # Smoothing) all share the same .forecast(ts_df, periods) interface.
        ts_df = build_time_series(df, temporal_col, metric_col)
        return strategy_cls().forecast(ts_df, periods=6)
