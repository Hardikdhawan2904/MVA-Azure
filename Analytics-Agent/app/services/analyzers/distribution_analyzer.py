"""app/services/analyzers/distribution_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): DistributionAnalyzer.

target_columns convention: [metric_col] or [metric_col, dimension_col].
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers.base import Analyzer, import_class, wrap_if_flat
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class DistributionAnalyzer(Analyzer):
    analysis_type = "distribution_analysis"

    def _run(
        self,
        *,
        df: pd.DataFrame,
        dataset_context: DatasetContext,
        target_columns: list[str],
        selected_model: SelectedModel,
        extra_context: dict[str, Any],
    ) -> dict[str, Any]:
        if not target_columns or target_columns[0] not in df.columns:
            return {"error": "DistributionAnalyzer requires target_columns=[metric_col, ...]"}
        metric_col = target_columns[0]
        dimension_col = target_columns[1] if len(target_columns) > 1 else None

        strategy_cls = import_class(selected_model.implementation_class)
        result = strategy_cls().compute(df, metric_col=metric_col, dimension_col=dimension_col)
        return wrap_if_flat(result)
