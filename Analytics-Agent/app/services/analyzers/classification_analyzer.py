"""app/services/analyzers/classification_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): ClassificationAnalyzer.

target_columns convention: [target_col, feature_col1, feature_col2, ...].
Every registered classification algorithm (VarianceClassifier/XGBoost,
Random Forest, Logistic Regression, Majority Class Baseline) shares the
identical `.fit_and_evaluate(df, target_col, feature_cols)` interface, so
no per-algorithm dispatch is needed here — unlike RegressionAnalyzer or
SegmentationAnalyzer, which do.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers.base import Analyzer, import_class, wrap_if_flat
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class ClassificationAnalyzer(Analyzer):
    analysis_type = "classification"

    def _run(
        self,
        *,
        df: pd.DataFrame,
        dataset_context: DatasetContext,
        target_columns: list[str],
        selected_model: SelectedModel,
        extra_context: dict[str, Any],
    ) -> dict[str, Any]:
        if not target_columns:
            return {"error": "ClassificationAnalyzer requires target_columns=[target_col, feature_col, ...]"}
        target_col, feature_cols = target_columns[0], target_columns[1:]
        if target_col not in df.columns:
            return {"error": f"Classification target column not found in data: {target_col!r}"}

        strategy_cls = import_class(selected_model.implementation_class)
        result = strategy_cls().fit_and_evaluate(df, target_col=target_col, feature_cols=list(feature_cols))
        return wrap_if_flat(result)
