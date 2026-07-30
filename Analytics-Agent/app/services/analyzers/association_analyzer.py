"""app/services/analyzers/association_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): AssociationAnalyzer.

target_columns convention: 2+ categorical dimension column names.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers._dataset_helpers import categorical_columns
from app.services.analyzers.base import Analyzer, import_class, wrap_if_flat
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class AssociationAnalyzer(Analyzer):
    analysis_type = "association_rules"

    def _run(
        self,
        *,
        df: pd.DataFrame,
        dataset_context: DatasetContext,
        target_columns: list[str],
        selected_model: SelectedModel,
        extra_context: dict[str, Any],
    ) -> dict[str, Any]:
        columns = target_columns or [c for c in categorical_columns(dataset_context) if c in df.columns]
        strategy_cls = import_class(selected_model.implementation_class)
        result = strategy_cls().compute(df, dimension_cols=columns)
        return wrap_if_flat(result)
