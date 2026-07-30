"""app/services/analyzers/anomaly_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): AnomalyAnalyzer.

target_columns convention: numeric feature columns to score for anomalies;
falls back to every "metric"-role column in DatasetContext when the
Planner didn't narrow it. Categorical columns are used as row-identifying
label columns in the output, never as scoring features.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers._dataset_helpers import categorical_columns, numeric_columns
from app.services.analyzers.base import Analyzer, import_class, wrap_if_flat
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class AnomalyAnalyzer(Analyzer):
    analysis_type = "anomaly_detection"

    def _run(
        self,
        *,
        df: pd.DataFrame,
        dataset_context: DatasetContext,
        target_columns: list[str],
        selected_model: SelectedModel,
        extra_context: dict[str, Any],
    ) -> dict[str, Any]:
        feature_cols = [c for c in (target_columns or numeric_columns(dataset_context)) if c in df.columns]
        if not feature_cols:
            return {"error": "AnomalyAnalyzer found no usable numeric feature columns"}
        label_cols = [c for c in categorical_columns(dataset_context) if c in df.columns]

        strategy_cls = import_class(selected_model.implementation_class)
        result = strategy_cls().detect(df, feature_cols=feature_cols, label_cols=label_cols)
        return wrap_if_flat(result)
