"""app/services/analyzers/comparative_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): ComparativeAnalyzer.

target_columns convention: [metric_col, dimension_col]. Always
deterministic by design; wraps AnalyticsTool.compare_groups().
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers.base import Analyzer, import_class
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class ComparativeAnalyzer(Analyzer):
    analysis_type = "comparative_analysis"

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
            return {"error": "ComparativeAnalyzer requires target_columns=[metric_col, dimension_col]"}
        metric_col, dimension_col = target_columns[0], target_columns[1]

        tool_cls = import_class(selected_model.implementation_class)
        result = tool_cls().compare_groups(df, metric_col, dimension_col)
        if "error" in result:
            return {"error": result["error"]}
        return {"evidence": result}
