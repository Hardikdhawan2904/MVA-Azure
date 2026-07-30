"""app/services/capability_resolution/analytics_capability_resolver.py —
Stage 1 of the Agent 3 redesign (plan "zany-giggling-crayon").

Answers "which analytics are possible on this dataset" — split into two
independently-reasoned questions per capability:

  1. Structural — can this analysis mechanically run at all, given the
     dataset's shape (DatasetContext)? Pure shape question, e.g. "forecast
     needs a temporal column" — Agent 3's own legitimate concern, zero
     overlap with Agent 2's readiness scoring.
  2. Execution — if structurally possible, does Agent 2's OWN
     ml_readiness_score clear Agent 3's OWN existing execution threshold
     (ML_READINESS_THRESHOLD, unchanged, already used by MLTool today),
     i.e. is a real trained model allowed to run?

Hard constraint (do not weaken): this class receives only the two
already-computed readiness scores and a DatasetContext — never the raw
DataFrame, never Agent 2's evidence/strengths/blocking_issues. There is
nothing here it could use to recompute a readiness score even by
accident; Agent 2 owns that scoring, this class only ever reads the
final number.
"""

from __future__ import annotations

from app.config import CAPABILITY_THRESHOLDS, ML_READINESS_THRESHOLD
from app.services.capability_resolution.models import (
    ALL_CAPABILITIES,
    ANOMALY_DETECTION,
    ASSOCIATION_RULES,
    CLASSIFICATION,
    CLUSTERING,
    CORRELATION,
    FORECASTING,
    REGRESSION,
    SEGMENTATION,
    TIME_SERIES,
    CapabilityProfile,
    CapabilityResult,
    ExecutionResult,
    StructuralResult,
)
from app.services.dataset_context.models import DatasetContext, SEMANTIC_ROLE_DIMENSION, SEMANTIC_ROLE_METRIC

_MIN_ROWS_FOR_TIME_SERIES = CAPABILITY_THRESHOLDS.get("min_rows_for_time_series", 10)
_MIN_METRICS_FOR_CLUSTERING = CAPABILITY_THRESHOLDS.get("min_metrics_for_clustering", 2)
_MIN_METRICS_FOR_CORRELATION = CAPABILITY_THRESHOLDS.get("min_metrics_for_correlation", 2)
_MIN_METRICS_FOR_ANOMALY = CAPABILITY_THRESHOLDS.get("min_metrics_for_anomaly_detection", 1)
_CARDINALITY_RATIO_MIN = CAPABILITY_THRESHOLDS.get("categorical_cardinality_ratio_min", 0.01)
_CARDINALITY_RATIO_MAX = CAPABILITY_THRESHOLDS.get("categorical_cardinality_ratio_max", 0.5)
_MIN_CATEGORICALS_FOR_ASSOCIATION = CAPABILITY_THRESHOLDS.get("min_categoricals_for_association_rules", 2)


def _workable_categoricals(dataset_context: DatasetContext) -> list:
    """Dimension columns whose cardinality is neither near-unique (really
    an identifier) nor single-valued (no signal)."""
    dims = dataset_context.columns_with_role(SEMANTIC_ROLE_DIMENSION)
    return [
        c for c in dims
        if c.cardinality_ratio is not None and _CARDINALITY_RATIO_MIN <= c.cardinality_ratio <= _CARDINALITY_RATIO_MAX
    ]


class AnalyticsCapabilityResolver:
    def resolve(
        self,
        dataset_context: DatasetContext,
        ml_readiness_score: float,
        llm_readiness_score: float,  # accepted for interface symmetry; unused here — narration-only gate (Stage 9), not an analysis capability
    ) -> CapabilityProfile:
        capabilities: dict[str, CapabilityResult] = {}
        for name in ALL_CAPABILITIES:
            structural = self._check_structural(name, dataset_context)
            if not structural.supported:
                capabilities[name] = CapabilityResult(structural=structural, execution=None)
                continue
            capabilities[name] = CapabilityResult(
                structural=structural,
                execution=self._evaluate_execution(ml_readiness_score),
            )
        return CapabilityProfile(capabilities=capabilities)

    @staticmethod
    def _evaluate_execution(ml_readiness_score: float) -> ExecutionResult:
        if ml_readiness_score >= ML_READINESS_THRESHOLD:
            return ExecutionResult(supported=True, confidence=round(ml_readiness_score / 100.0, 4))
        return ExecutionResult(
            supported=False,
            reason=(
                f"ML readiness score ({ml_readiness_score:.1f}%) below the {ML_READINESS_THRESHOLD}% "
                f"threshold Agent 3 requires to run its own models"
            ),
        )

    def _check_structural(self, capability: str, ctx: DatasetContext) -> StructuralResult:
        if capability in (FORECASTING, TIME_SERIES):
            return self._check_temporal_and_metric(ctx)
        if capability == CLUSTERING:
            return self._check_min_metrics(ctx, _MIN_METRICS_FOR_CLUSTERING, "clustering")
        if capability == CORRELATION:
            return self._check_min_metrics(ctx, _MIN_METRICS_FOR_CORRELATION, "correlation")
        if capability == ANOMALY_DETECTION:
            return self._check_min_metrics(ctx, _MIN_METRICS_FOR_ANOMALY, "anomaly detection")
        if capability == CLASSIFICATION:
            return self._check_target(ctx, "classification")
        if capability == REGRESSION:
            return self._check_target(ctx, "regression")
        if capability == SEGMENTATION:
            return self._check_segmentation(ctx)
        if capability == ASSOCIATION_RULES:
            return self._check_association_rules(ctx)
        raise ValueError(f"No structural check registered for capability '{capability}'")

    @staticmethod
    def _check_temporal_and_metric(ctx: DatasetContext) -> StructuralResult:
        temporal_cols = [c for c in ctx.columns if c.is_temporal]
        metric_cols = ctx.columns_with_role(SEMANTIC_ROLE_METRIC)
        if not temporal_cols:
            return StructuralResult(False, "No temporal (date/datetime) column detected — cannot build a time axis")
        if not metric_cols:
            return StructuralResult(False, "No numeric metric column detected to analyze over time")
        if ctx.row_count < _MIN_ROWS_FOR_TIME_SERIES:
            return StructuralResult(
                False, f"Only {ctx.row_count} rows — below the {_MIN_ROWS_FOR_TIME_SERIES}-row minimum for a meaningful time series",
            )
        return StructuralResult(
            True, f"Temporal column '{temporal_cols[0].name}' and metric column '{metric_cols[0].name}' present",
        )

    @staticmethod
    def _check_min_metrics(ctx: DatasetContext, minimum: int, label: str) -> StructuralResult:
        metric_cols = ctx.columns_with_role(SEMANTIC_ROLE_METRIC)
        if len(metric_cols) < minimum:
            return StructuralResult(
                False, f"Only {len(metric_cols)} numeric metric column(s) — {label} needs at least {minimum}",
            )
        return StructuralResult(True, f"{len(metric_cols)} numeric metric columns available for {label}")

    @staticmethod
    def _check_target(ctx: DatasetContext, problem_type: str) -> StructuralResult:
        rec = ctx.feature_recommendation or {}
        target = rec.get("target_column")
        if not target:
            return StructuralResult(False, "No target column identified (feature_recommendation.target_column is empty)")
        actual_problem_type = rec.get("problem_type")
        if actual_problem_type != problem_type:
            return StructuralResult(
                False,
                f"Target column '{target}' has problem_type='{actual_problem_type}', not '{problem_type}'",
            )
        return StructuralResult(True, f"Target column '{target}' identified with problem_type='{problem_type}'")

    @staticmethod
    def _check_segmentation(ctx: DatasetContext) -> StructuralResult:
        metric_cols = ctx.columns_with_role(SEMANTIC_ROLE_METRIC)
        workable_dims = _workable_categoricals(ctx)
        if metric_cols:
            return StructuralResult(True, f"Metric column '{metric_cols[0].name}' available to bin into segments")
        if workable_dims:
            return StructuralResult(True, f"Categorical column '{workable_dims[0].name}' available to group by")
        return StructuralResult(False, "No metric column to bin and no workable categorical column to group by")

    @staticmethod
    def _check_association_rules(ctx: DatasetContext) -> StructuralResult:
        workable_dims = _workable_categoricals(ctx)
        if len(workable_dims) < _MIN_CATEGORICALS_FOR_ASSOCIATION:
            return StructuralResult(
                False,
                f"Only {len(workable_dims)} categorical column(s) with workable cardinality — "
                f"association rules need at least {_MIN_CATEGORICALS_FOR_ASSOCIATION}",
            )
        return StructuralResult(True, f"{len(workable_dims)} workable categorical columns available")
