"""app/services/analyzers/correlation_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): CorrelationAnalyzer.

target_columns convention: 2+ numeric column names to correlate pairwise.
Always deterministic by design (no ML variant registered for this
purpose — SciPy correlation is already the cheapest possible answer).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers.base import Analyzer, import_class
from app.services.analyzers._dataset_helpers import numeric_columns
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class CorrelationAnalyzer(Analyzer):
    analysis_type = "correlation"

    def _run(
        self,
        *,
        df: pd.DataFrame,
        dataset_context: DatasetContext,
        target_columns: list[str],
        selected_model: SelectedModel,
        extra_context: dict[str, Any],
    ) -> dict[str, Any]:
        columns = target_columns or [c for c in numeric_columns(dataset_context) if c in df.columns]
        strategy_cls = import_class(selected_model.implementation_class)
        return strategy_cls().compute(df, columns)
