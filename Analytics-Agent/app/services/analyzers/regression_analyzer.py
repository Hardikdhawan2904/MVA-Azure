"""app/services/analyzers/regression_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): RegressionAnalyzer.

target_columns convention: [target_col, feature_col1, feature_col2, ...] —
populated from DatasetContext.feature_recommendation by the Planner (Stage
4's rule table: "feature_recommendation.target_column present ->
classification or regression").
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers._dataset_helpers import categorical_columns
from app.services.analyzers.base import Analyzer, import_class, wrap_if_flat
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class RegressionAnalyzer(Analyzer):
    analysis_type = "regression"

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
            return {"error": "RegressionAnalyzer requires target_columns=[target_col, feature_col, ...]"}
        target_col, feature_cols = target_columns[0], target_columns[1:]
        if target_col not in df.columns:
            return {"error": f"Regression target column not found in data: {target_col!r}"}

        strategy_cls = import_class(selected_model.implementation_class)
        algorithm = selected_model.algorithm

        if algorithm == "Linear Regression":
            result = strategy_cls().fit_and_predict(df, target_col=target_col, feature_cols=feature_cols)
            return wrap_if_flat(result)

        cat_cols = [c for c in categorical_columns(dataset_context) if c in feature_cols]
        result = strategy_cls().fit_and_predict(
            df, target_col=target_col, feature_cols=feature_cols, categorical_cols=cat_cols,
        )
        return wrap_if_flat(result)
