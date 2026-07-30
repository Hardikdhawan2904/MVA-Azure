"""app/services/analyzers/clustering_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): ClusteringAnalyzer.

A lighter-weight sibling of SegmentationAnalyzer (Stage 4's Planner
proposes "clustering" for ≥2 numeric columns with no clear target — see
the Planner rule table) that reuses the same registry algorithms
(K-Means, Quantile Binning, Categorical Grouping) rather than duplicating
the model_registry.yml entries.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers._dataset_helpers import categorical_columns
from app.services.analyzers.base import Analyzer, import_class, wrap_if_flat
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class ClusteringAnalyzer(Analyzer):
    analysis_type = "clustering"

    def _run(
        self,
        *,
        df: pd.DataFrame,
        dataset_context: DatasetContext,
        target_columns: list[str],
        selected_model: SelectedModel,
        extra_context: dict[str, Any],
    ) -> dict[str, Any]:
        algorithm = selected_model.algorithm
        strategy_cls = import_class(selected_model.implementation_class)

        if "K-Means" in algorithm:
            feature_cols = [c for c in target_columns if c in df.columns]
            if len(feature_cols) < 2:
                return {"error": "ClusteringAnalyzer needs >= 2 numeric target_columns to cluster on"}
            label_cols = [c for c in categorical_columns(dataset_context) if c in df.columns]
            return wrap_if_flat(strategy_cls().segment(df, feature_cols=feature_cols, label_cols=label_cols))

        if "Quantile Binning" in algorithm:
            if not target_columns or target_columns[0] not in df.columns:
                return {"error": "ClusteringAnalyzer needs target_columns=[metric_col] to bin"}
            return wrap_if_flat(strategy_cls().segment(df, target_columns[0]))

        if "Categorical Grouping" in algorithm:
            if not target_columns or target_columns[0] not in df.columns:
                return {"error": "ClusteringAnalyzer needs target_columns=[dimension_col] to group"}
            return wrap_if_flat(strategy_cls().segment(df, target_columns[0]))

        return {"error": f"ClusteringAnalyzer has no dispatch rule for algorithm '{algorithm}'"}
