"""app/services/analyzers/kpi_summary_analyzer.py — Stage 7, Phase 4 (plan
"zany-giggling-crayon"): KPISummaryAnalyzer.

Requires extra_context["kpi"] — the resolved RuleEngine.get_kpi() dict
(label/actual_column/budget_column/unit/higher_is_better/...), supplied
by the orchestration node (interpret_question's KPI resolution + a domain
plugin's enhance_plan()). target_columns carries the same columns for
traceability but isn't itself read here — the strategy reads column names
off the kpi dict directly, exactly like the old handler did.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers.base import Analyzer, import_class
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class KPISummaryAnalyzer(Analyzer):
    analysis_type = "kpi_summary"

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
            return {"error": "KPISummaryAnalyzer requires extra_context['kpi'] (a resolved RuleEngine KPI dict)"}

        strategy_cls = import_class(selected_model.implementation_class)
        result = strategy_cls().compute(df, kpi)
        if "error" in result:
            return {"error": result["error"]}
        result["evidence"]["filters"] = extra_context.get("filters", {})
        return result
