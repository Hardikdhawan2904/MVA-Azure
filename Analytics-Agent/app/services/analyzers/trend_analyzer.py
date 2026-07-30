"""app/services/analyzers/trend_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): TrendAnalyzer.

target_columns convention: [temporal_col, metric_col]. Always deterministic
by design (no capability gate — see ANALYSIS_TYPE_TO_CAPABILITY's
docstring); the sole registry entry wraps AnalyticsTool.trend() unchanged.

Feeds AnalyticsTool.trend() a per-date aggregate (build_time_series(),
the same grouped-sum helper ForecastAnalyzer/TimeSeriesAnalyzer already
use), not the raw multi-row-per-date df -- trend()'s first/last-value
comparison only means "how did this metric move over time" once there is
exactly one row per date behind it. Feeding it raw rows for a dataset
with several rows per date (multiple stores/regions/categories per day,
for example) would make first_value/last_value/direction/
overall_change_pct compare two effectively random individual rows instead
of anything resembling a trend -- confirmed on a real 20-store dataset
where trend() reported a "-89.78%% decreasing" headline while actual
per-date total revenue was roughly flat/cyclical in the 276k-518k range.
A no-op for the common case (a dataset that already has one row per date).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers._dataset_helpers import build_time_series
from app.services.analyzers.base import Analyzer, import_class
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class TrendAnalyzer(Analyzer):
    analysis_type = "trend"

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
            return {"error": "TrendAnalyzer requires target_columns=[temporal_col, metric_col]"}
        temporal_col, metric_col = target_columns[0], target_columns[1]
        if temporal_col not in df.columns or metric_col not in df.columns:
            return {"error": f"Trend target columns not found in data: {temporal_col!r}, {metric_col!r}"}

        ts_df = build_time_series(df, temporal_col, metric_col)
        tool_cls = import_class(selected_model.implementation_class)
        result = tool_cls().trend(ts_df, "ds", "y")
        if "error" in result:
            return {"error": result["error"]}
        return {"evidence": result}
