"""app/services/analyzers/root_cause_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): RootCauseAnalyzer.

Root cause is kept exactly as the plan specifies: primary answer always
deterministic, with two mutually exclusive modes —

1. Labeled-driver mode (today's exact behavior, byte-identical) — used
   whenever known pre-computed variance-driver columns are available
   (populated into target_columns[1:] by the Planner from a domain
   plugin's get_driver_columns(), e.g. InsurancePlugin's 14 columns).
   Delegates straight to RootCauseTool, which only ever implements this
   mode (see its own docstring).

2. Correlation-based mode (new, generic default when no plugin/labels
   exist) — |Pearson correlation| of every other available numeric column
   against the outcome metric, ranked the same way, reusing the exact
   same driver_breakdown/primary_driver field names so ExplanationTool's
   existing rendering code needs no branch between the two modes.

   Candidate pool: if any numeric columns look driver-shaped by name
   (ending in/containing "variance"/"driver"/"contribution"/"impact" —
   e.g. Banking's nim_rate_variance, headcount_volume_variance), only
   those are correlated — a much more targeted answer than every numeric
   column in the dataset. Still never labeled mode: correlation strength
   is still discovered statistically, not assumed from the name alone;
   the heuristic only narrows *which* columns get that statistical
   treatment. Falls back to every numeric column when no name matches.

target_columns convention: [metric_col, *known_driver_columns]. Only
target_columns[0] is required; an empty tail always means "no known
drivers — use correlation-based mode", never an error.

Only the registry's single root_cause algorithm (RootCauseTool,
"Deterministic Driver Decomposition") is ever selected by Stage 6 for
this purpose — this analyzer, not the registry, decides which of the two
modes to run, exactly as the plan describes.

extra_context["total_variance"] (Phase 4): when the orchestration node
has already computed a KPI-grounded actual-vs-budget variance (today's
handle_root_cause does this via a separate SQL query before calling
RootCauseTool), it's passed straight through to RootCauseTool.analyse()
so unexplained_variance is populated exactly as today. Absent for the
generic/non-KPI-grounded case — RootCauseTool already defaults it to
None, unchanged from Phase 3.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from app.services.analyzers._dataset_helpers import numeric_columns
from app.services.analyzers.base import Analyzer, import_class
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel

# Heuristic candidate-narrowing for correlation-based mode — column names
# that look driver-shaped even without a domain plugin declaring them
# (e.g. Banking's nim_rate_variance, headcount_volume_variance). Never
# triggers labeled mode by itself; see the module docstring.
_HEURISTIC_DRIVER_NAME_RE = re.compile(r"variance|driver|contribution|impact", re.IGNORECASE)

# Same qualifier suffixes BusinessQuestionInterpreter._base_phrase strips for
# question-to-column matching -- reused here to recognize a candidate driver
# column as the *same* metric's own budget/prior-year/forecast counterpart
# (e.g. revenue_budget for revenue_actual) rather than an independent
# driver. Correlating a metric against its own budget baseline is nearly
# tautological (they're highly correlated by construction, not because the
# budget "caused" the variance) -- confirmed live where a correlation-based
# root cause named "revenue_budget" the #1 driver of "why is revenue actual
# below budget", at correlation 0.9977.
_QUALIFIER_SUFFIX_RE = re.compile(
    r"\b(actual|budget|forecast|prior[_ ]year|ytd|ratio|rate|amount|pct|percent)\b", re.IGNORECASE,
)


def _base_metric_name(column_name: str) -> str:
    normalized = column_name.replace("_", " ").lower()
    stripped = _QUALIFIER_SUFFIX_RE.sub("", normalized)
    return re.sub(r"\s+", " ", stripped).strip()


class RootCauseAnalyzer(Analyzer):
    analysis_type = "root_cause"

    def _run(
        self,
        *,
        df: pd.DataFrame,
        dataset_context: DatasetContext,
        target_columns: list[str],
        selected_model: SelectedModel,
        extra_context: dict[str, Any],
    ) -> dict[str, Any]:
        if not target_columns:
            return {"error": "RootCauseAnalyzer requires target_columns=[metric_col, ...]"}
        metric_col = target_columns[0]
        # Labeled mode sums each driver column as a variance amount — a
        # non-numeric column here (e.g. a DiscoveredKPI's grouping
        # dimension, not an actual pre-computed driver) can't be summed;
        # treat it as "no known drivers" rather than attempt labeled mode
        # and silently produce zero drivers.
        driver_columns = [
            c for c in target_columns[1:]
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]

        if driver_columns:
            tool_cls = import_class(selected_model.implementation_class)
            total_variance = extra_context.get("total_variance")
            result = tool_cls(driver_columns=driver_columns).analyse(df, total_variance=total_variance)
            if "error" in result:
                return {"error": result["error"]}
            result["mode"] = "labeled_driver"
            return {"evidence": result}

        return self._correlation_based(df, dataset_context, metric_col)

    @staticmethod
    def _correlation_based(df: pd.DataFrame, dataset_context: DatasetContext, metric_col: str) -> dict[str, Any]:
        if metric_col not in df.columns:
            return {"error": f"Root cause metric column not found in data: {metric_col!r}"}

        all_candidates = [c for c in numeric_columns(dataset_context, exclude={metric_col}) if c in df.columns]

        # Drop the metric's own budget/prior-year/forecast counterpart(s),
        # if any -- e.g. revenue_budget for revenue_actual. Only applied
        # when the metric column actually has a meaningful base name left
        # after stripping qualifiers (guards against every column
        # collapsing to "" and excluding each other).
        metric_base = _base_metric_name(metric_col)
        if metric_base:
            all_candidates = [c for c in all_candidates if _base_metric_name(c) != metric_base]

        if not all_candidates:
            return {"error": "No candidate driver columns available for correlation-based root cause analysis."}

        heuristic_candidates = [c for c in all_candidates if _HEURISTIC_DRIVER_NAME_RE.search(c)]
        candidate_cols = heuristic_candidates or all_candidates

        data = df[[metric_col] + candidate_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(data) < 3:
            return {"error": f"Insufficient data for correlation-based root cause analysis: {len(data)} rows (need >= 3)"}

        correlations = data[candidate_cols].corrwith(data[metric_col]).dropna()
        total_abs = correlations.abs().sum()

        driver_results = []
        for col, corr in correlations.items():
            corr = round(float(corr), 4)
            driver_results.append({
                "driver": col,
                "label": col.replace("_", " ").title(),
                "amount": corr,  # same key ExplanationTool's driver_breakdown renderer already reads
                "correlation": corr,
                "contribution_pct": round(abs(corr) / total_abs * 100, 2) if total_abs else 0.0,
                "direction": "favorable" if corr > 0 else ("unfavorable" if corr < 0 else "neutral"),
            })
        driver_results.sort(key=lambda d: abs(d["correlation"]), reverse=True)

        primary = driver_results[0] if driver_results else None
        secondary = driver_results[1] if len(driver_results) >= 2 else None

        return {
            "evidence": {
                "mode": "correlation_based",
                "metric_column": metric_col,
                "driver_breakdown": driver_results,
                "primary_driver": primary,
                "secondary_driver": secondary,
                "drivers_analysed": len(driver_results),
                "candidate_pool": "heuristic_driver_named_columns" if heuristic_candidates else "all_numeric_columns",
            }
        }
