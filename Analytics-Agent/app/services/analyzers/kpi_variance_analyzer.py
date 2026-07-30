"""app/services/analyzers/kpi_variance_analyzer.py — Stage 7, Phase 4
(plan "zany-giggling-crayon"): KPIVarianceAnalyzer.

Same extra_context["kpi"] contract as KPISummaryAnalyzer — see that
module's docstring. A distinct analysis_type/analyzer, not a mode flag on
KPISummaryAnalyzer, because the two produce genuinely different evidence
key shapes (see kpi_strategies.py's module docstring).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers.base import Analyzer, import_class
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class KPIVarianceAnalyzer(Analyzer):
    analysis_type = "kpi_variance"

    def _run(
        self,
        *,
        df: pd.DataFrame,
        dataset_context: DatasetContext,
        target_columns: list[str],
        selected_model: SelectedModel,
        extra_context: dict[str, Any],
    ) -> dict[str, Any]:
        kpi = extra_context.get("kpi")
        if not kpi:
            return {"error": "KPIVarianceAnalyzer requires extra_context['kpi'] (a resolved RuleEngine KPI dict)"}

        strategy_cls = import_class(selected_model.implementation_class)
        result = strategy_cls().compute(df, kpi)
        if "error" in result:
            return {"error": result["error"]}
        result["evidence"]["filters"] = extra_context.get("filters", {})
        return result
