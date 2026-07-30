"""app/services/analyzers/ranking_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): RankingAnalyzer.

target_columns convention: [value_col, label_col]. Always deterministic by
design; wraps AnalyticsTool.rank() (returns a DataFrame) into evidence's
dict shape.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers.base import Analyzer, import_class
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class RankingAnalyzer(Analyzer):
    analysis_type = "ranking"

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
            return {"error": "RankingAnalyzer requires target_columns=[value_col, label_col]"}
        value_col, label_col = target_columns[0], target_columns[1]
        if value_col not in df.columns or label_col not in df.columns:
            return {"error": f"Ranking target columns not found in data: {value_col!r}, {label_col!r}"}

        tool_cls = import_class(selected_model.implementation_class)
        ranked_df = tool_cls().rank(df, value_col, label_col, top_n=10)
        if ranked_df.empty:
            return {"error": f"No numeric data in column '{value_col}' to rank."}

        return {
            "evidence": {
                "ranked_records": ranked_df.to_dict(orient="records"),
                "value_column": value_col,
                "label_column": label_col,
                "records_returned": len(ranked_df),
            }
        }
