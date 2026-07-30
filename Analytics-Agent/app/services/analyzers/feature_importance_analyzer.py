"""app/services/analyzers/feature_importance_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): FeatureImportanceAnalyzer.

target_columns convention: [target_col, feature_col1, feature_col2, ...].
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers.base import Analyzer, import_class, wrap_if_flat
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class FeatureImportanceAnalyzer(Analyzer):
    analysis_type = "feature_importance"

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
            return {"error": "FeatureImportanceAnalyzer requires target_columns=[target_col, feature_col, ...]"}
        target_col, feature_cols = target_columns[0], list(target_columns[1:])
        if target_col not in df.columns:
            return {"error": f"Feature importance target column not found in data: {target_col!r}"}

        strategy_cls = import_class(selected_model.implementation_class)

        if selected_model.algorithm == "SHAP":
            # VarianceClassifier is classification-shaped (XGBoost multi-
            # class); its own model.feature_importances_ is what
            # fit_and_evaluate() already reports as "feature_importances" —
            # reshaped to this purpose's feature_contributions key. Deeper
            # per-record SHAP attribution (predict_batch_with_explanation)
            # is what RootCauseAnalyzer's ML-corroboration path already
            # uses; not duplicated here.
            result = strategy_cls().fit_and_evaluate(df, target_col=target_col, feature_cols=feature_cols)
            if "error" in result:
                return {"error": result["error"]}
            result = {**result, "feature_contributions": result.pop("feature_importances", [])}
            return {"evidence": result}

        result = strategy_cls().compute(df, target_col=target_col, feature_cols=feature_cols)
        return wrap_if_flat(result)
