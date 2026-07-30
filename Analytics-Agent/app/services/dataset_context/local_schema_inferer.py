"""app/services/dataset_context/local_schema_inferer.py — lightweight
fallback column classifier, used only when Agent 2's rich column_profiles
weren't forwarded (a direct /analyze call, or an un-updated Orchestrator).

Deliberately far simpler than Agent 2's TypeRefiner/SemanticCandidateGenerator
(Data-Profiling-Agent/app/services/profiling/) — this is a safety net,
not a second implementation of Agent 2's classification pipeline. Pure
pandas dtype checks plus a small name-pattern regex set; good enough to
let the generic planner (Stages 1-3) produce a sensible plan on an
unclassified dataset, not good enough to replace what Agent 2 already does
well when it's available.
"""

from __future__ import annotations

import re
import warnings

import pandas as pd

from app.services.dataset_context.models import ColumnContext, DatasetContext

_IDENTIFIER_NAME_RE = re.compile(r"(^id$|_id$|^id_|uuid|guid)", re.IGNORECASE)
_TEMPORAL_NAME_RE = re.compile(r"(date|_at$|_on$|timestamp|^time$|_time$|day|month|year)", re.IGNORECASE)
_FLAG_VALUE_SETS = (
    {"y", "n"}, {"yes", "no"}, {"true", "false"}, {"0", "1"}, {"t", "f"},
)

# Small, deliberately non-exhaustive semantic-type hints from column name —
# a best-effort bonus for Stage 2 (Semantic KPI Discovery) to have
# something to work with even without Agent 2, not a full classifier.
_NAME_SEMANTIC_HINTS = [
    (re.compile(r"revenue|sales_amount", re.IGNORECASE), "revenue_amount"),
    (re.compile(r"cost|expense", re.IGNORECASE), "expense_amount"),
    (re.compile(r"salary|compensation|wage", re.IGNORECASE), "salary_amount"),
    (re.compile(r"claim", re.IGNORECASE), "claims_amount"),
    (re.compile(r"budget", re.IGNORECASE), "budget_amount"),
    (re.compile(r"forecast", re.IGNORECASE), "forecast_amount"),
    (re.compile(r"actual", re.IGNORECASE), "actual_amount"),
    (re.compile(r"amount|price|value", re.IGNORECASE), "monetary_amount"),
    (re.compile(r"status|state", re.IGNORECASE), "status"),
]

_IDENTIFIER_CARDINALITY_THRESHOLD = 0.95
_DIMENSION_CARDINALITY_THRESHOLD = 0.5


class LocalSchemaInferer:
    def infer(self, df: pd.DataFrame, detected_domain: str | None = None) -> DatasetContext:
        columns = [self._infer_column(df, col) for col in df.columns]
        return DatasetContext(
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
            context_source="local_fallback",
            detected_domain=detected_domain,
        )

    def _infer_column(self, df: pd.DataFrame, col: str) -> ColumnContext:
        series = df[col]
        non_null = series.dropna()
        row_count = len(series)
        null_ratio = 1.0 - (len(non_null) / row_count) if row_count else 0.0
        cardinality_ratio = (non_null.nunique() / len(non_null)) if len(non_null) else 0.0
        sample_values = non_null.head(5).tolist()

        is_temporal = self._is_temporal(series, non_null, col)
        is_identifier = (
            not is_temporal
            and cardinality_ratio >= _IDENTIFIER_CARDINALITY_THRESHOLD
            and bool(_IDENTIFIER_NAME_RE.search(col))
        )
        is_flag = self._is_flag(non_null)
        is_numeric = pd.api.types.is_numeric_dtype(series) and not is_flag

        if is_temporal:
            role = "temporal_dimension"
        elif is_identifier:
            role = "identifier"
        elif is_flag:
            role = "flag"
        elif is_numeric:
            role = "metric"
        elif cardinality_ratio < _DIMENSION_CARDINALITY_THRESHOLD:
            role = "dimension"
        else:
            role = "text_field"

        semantic_type = self._semantic_type_hint(col, role)
        physical_type = self._physical_type(series, is_temporal)

        return ColumnContext(
            name=col,
            physical_type=physical_type,
            semantic_type=semantic_type,
            semantic_role=role,
            cardinality_ratio=round(cardinality_ratio, 4),
            null_ratio=round(null_ratio, 4),
            is_identifier=is_identifier,
            is_temporal=is_temporal,
            sample_values=sample_values,
            confidence=None,  # heuristic-only — never claim a confidence score we didn't earn
        )

    @staticmethod
    def _is_temporal(series: pd.Series, non_null: pd.Series, col: str) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if len(non_null) == 0 or not _TEMPORAL_NAME_RE.search(col):
            return False
        sample = non_null.head(20)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # expected on genuinely non-date columns we're probing
                parsed = pd.to_datetime(sample, errors="coerce")
        except (ValueError, TypeError):
            return False
        return parsed.notna().mean() >= 0.8

    @staticmethod
    def _is_flag(non_null: pd.Series) -> bool:
        if len(non_null) == 0:
            return False
        if pd.api.types.is_bool_dtype(non_null):
            return True
        distinct = {str(v).strip().lower() for v in non_null.unique()[:10]}
        return any(distinct <= flag_set for flag_set in _FLAG_VALUE_SETS)

    @staticmethod
    def _physical_type(series: pd.Series, is_temporal: bool) -> str:
        if is_temporal:
            return "datetime"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_integer_dtype(series):
            return "integer"
        if pd.api.types.is_float_dtype(series):
            return "decimal"
        return "text"

    @staticmethod
    def _semantic_type_hint(col: str, role: str) -> str | None:
        for pattern, semantic_type in _NAME_SEMANTIC_HINTS:
            if pattern.search(col):
                return semantic_type
        if role == "metric":
            return "numeric_measure"
        return None
