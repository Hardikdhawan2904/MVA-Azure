"""app/services/analyzers/segmentation_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): SegmentationAnalyzer.

Dispatches on `selected_model.algorithm` because this purpose spans two
genuinely different calling conventions: cluster-on-N-numeric-features
(K-Means/DBSCAN/Hierarchical Clustering — target_columns = feature list)
and bin-a-single-metric (Quantile/Equal-Width/Equal-Frequency Binning,
Insurance Combined Ratio Buckets — target_columns = [metric_col]) or
group-a-single-dimension (Categorical Grouping — target_columns =
[dimension_col]).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers._dataset_helpers import categorical_columns
from app.services.analyzers.base import Analyzer, import_class, wrap_if_flat
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel

_CLUSTER_ON_FEATURES = {"K-Means", "DBSCAN", "Hierarchical Clustering"}
_BIN_SINGLE_METRIC = {"Quantile Binning", "Equal-Width Binning", "Equal-Frequency Binning", "Insurance Combined Ratio Buckets"}


class SegmentationAnalyzer(Analyzer):
    analysis_type = "segmentation"

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

        if algorithm in _CLUSTER_ON_FEATURES:
            feature_cols = [c for c in target_columns if c in df.columns]
            if len(feature_cols) < 2:
                return {"error": "SegmentationAnalyzer needs >= 2 numeric target_columns to cluster on"}
            label_cols = [c for c in categorical_columns(dataset_context) if c in df.columns]
            method = "segment" if algorithm == "K-Means" else "cluster"
            result = getattr(strategy_cls(), method)(df, feature_cols=feature_cols, label_cols=label_cols)
            return wrap_if_flat(result)

        if algorithm in _BIN_SINGLE_METRIC:
            if not target_columns or target_columns[0] not in df.columns:
                return {"error": "SegmentationAnalyzer needs target_columns=[metric_col] to bin"}
            if algorithm == "Insurance Combined Ratio Buckets":
                # This strategy is only ever selected via a domain plugin's
                # get_preferred_deterministic_strategy() (Phase 2) — its
                # natural column is always "combined_ratio_actual". When
                # target_columns carries the full multi-column ML feature
                # set (enhance_plan()'s anomaly/segmentation override,
                # Phase 4) that column may not be target_columns[0] —
                # search for it explicitly rather than assume position 0.
                ratio_col = "combined_ratio_actual" if "combined_ratio_actual" in target_columns and "combined_ratio_actual" in df.columns else target_columns[0]
                return wrap_if_flat(strategy_cls().segment(df, ratio_column=ratio_col))
            return wrap_if_flat(strategy_cls().segment(df, target_columns[0]))

        if algorithm == "Categorical Grouping":
            if not target_columns or target_columns[0] not in df.columns:
                return {"error": "SegmentationAnalyzer needs target_columns=[dimension_col] to group"}
            return wrap_if_flat(strategy_cls().segment(df, target_columns[0]))

        return {"error": f"SegmentationAnalyzer has no dispatch rule for algorithm '{algorithm}'"}
