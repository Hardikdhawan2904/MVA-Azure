"""app/services/analyzers/outlier_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): OutlierAnalyzer.

A distinct analysis_type from "anomaly_detection" per the Planner's rule
table (≥2 numeric columns, no clear target → correlation, clustering,
outlier_detection) even though today's only registered algorithm
(Z-Score) is the same strategy class AnomalyAnalyzer uses — kept as its
own Analyzer rather than aliased so a future outlier-specific algorithm
can be registered without touching AnomalyAnalyzer.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers._dataset_helpers import categorical_columns, numeric_columns
from app.services.analyzers.base import Analyzer, import_class, wrap_if_flat
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class OutlierAnalyzer(Analyzer):
    analysis_type = "outlier_detection"

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
            return {"error": "OutlierAnalyzer found no usable numeric feature columns"}
        label_cols = [c for c in categorical_columns(dataset_context) if c in df.columns]

        strategy_cls = import_class(selected_model.implementation_class)
        result = strategy_cls().detect(df, feature_cols=feature_cols, label_cols=label_cols)
        return wrap_if_flat(result)
