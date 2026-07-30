"""app/services/analyzers/time_series_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): TimeSeriesAnalyzer.

target_columns convention: [temporal_col, metric_col].
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers._dataset_helpers import build_time_series
from app.services.analyzers.base import Analyzer, import_class, wrap_if_flat
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class TimeSeriesAnalyzer(Analyzer):
    analysis_type = "time_series_analysis"

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
            return {"error": "TimeSeriesAnalyzer requires target_columns=[temporal_col, metric_col]"}
        temporal_col, metric_col = target_columns[0], target_columns[1]
        if temporal_col not in df.columns or metric_col not in df.columns:
            return {"error": f"Time series target columns not found in data: {temporal_col!r}, {metric_col!r}"}

        ts_df = build_time_series(df, temporal_col, metric_col)
        strategy_cls = import_class(selected_model.implementation_class)
        result = strategy_cls().compute(ts_df)
        return wrap_if_flat(result)
