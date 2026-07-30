"""app/services/analyzers/base.py — Stage 7 of the Agent 3 redesign (plan
"zany-giggling-crayon"): the Analyzer interface (Template Method).

Each concrete Analyzer receives a SelectedModel from Stage 6 (already
resolved — the analyzer doesn't re-decide ML vs. deterministic, or which
algorithm) and executes it, producing one Evidence object (Stage 8).
`execute()` is the fixed skeleton; `_run()` is what each analysis-type
family overrides to know how to call its own family's implementation
classes (import_class() resolves `SelectedModel.implementation_class`,
a dotted path straight out of config/model_registry.yml, exactly as
`ModelRegistry`/`ModelSelector` already treat it as data, not code).
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from app.services.dataset_context.models import DatasetContext
from app.services.evidence.evidence_builder import Evidence
from app.services.models_registry.model_selector import SelectedModel
from app.services.scheduling.models import ScheduledAnalysis

logger = logging.getLogger(__name__)


def import_class(dotted_path: str) -> type:
    module_path, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def wrap_if_flat(result: dict) -> dict[str, Any]:
    """Some implementation classes (the pre-existing ml/*.py facades —
    AnomalyDetector, VarianceClassifier, RiskSegmenter, ProphetForecaster,
    LightGBMForecaster) already return their payload as a flat dict rather
    than this package's {"evidence": {...}} shape; the new strategy
    classes written for this phase return the wrapped shape directly.
    Callers pass either through this so `_run()` doesn't need an
    if/else at every call site."""
    if "error" in result or "evidence" in result:
        return result
    return {"evidence": result}


class Analyzer(ABC):
    """One concrete subclass per analysis-type family (Stage 7's own
    folder listing: trend_analyzer.py, forecast_analyzer.py, ...)."""

    analysis_type: str = ""

    def execute(
        self,
        df: pd.DataFrame,
        dataset_context: DatasetContext,
        scheduled_analysis: ScheduledAnalysis,
        selected_model: SelectedModel,
        ml_confidence: float | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> Evidence:
        reasons = list(selected_model.reasons)

        if selected_model.algorithm is None or not selected_model.implementation_class:
            reasons.append(f"No algorithm available for '{self.analysis_type}' given this dataset — no evidence produced")
            return Evidence(evidence={}, reasons=reasons)

        try:
            payload = self._run(
                df=df,
                dataset_context=dataset_context,
                target_columns=list(scheduled_analysis.planned_analysis.target_columns),
                selected_model=selected_model,
                extra_context=extra_context or {},
            )
        except Exception as e:
            logger.error(f"{type(self).__name__}._run failed for '{self.analysis_type}': {e}", exc_info=True)
            reasons.append(f"Analyzer execution failed: {e}")
            return Evidence(evidence={}, reasons=reasons)

        if payload.get("error"):
            reasons.append(str(payload["error"]))
            return Evidence(evidence={}, reasons=reasons)

        model_metadata = None
        fallback_metadata = None
        confidence = None

        if selected_model.requires_ml:
            model_metadata = {
                "algorithm": selected_model.algorithm,
                "cost_tier": selected_model.cost_tier,
            }
            # capability_profile's execution.confidence (== ml_readiness_score
            # / 100, Stage 1) — passed in by the caller, since Analyzers don't
            # import capability_resolution themselves (same boundary
            # discipline as ModelSelector consuming it, not recomputing it).
            confidence = ml_confidence
        else:
            # Mirrors today's flat evidence dict's
            # ml_readiness_blocked/fallback_reason/fallback_applied shape —
            # see Evidence.flatten()'s docstring. ml_readiness_blocked is
            # derived from ModelSelector's own recorded reason (Stage 6),
            # never re-derived from a readiness score Analyzers never see.
            ml_readiness_blocked = any(
                "execution gate did not" in r for r in reasons
            )
            fallback_metadata = {
                "ml_readiness_blocked": ml_readiness_blocked,
                "fallback_reason": "; ".join(reasons),
                "fallback_applied": selected_model.algorithm,
            }

        return Evidence(
            evidence=payload.get("evidence", {}),
            confidence=confidence,
            metrics=payload.get("metrics", {}),
            charts=payload.get("charts"),
            model_metadata=model_metadata,
            fallback_metadata=fallback_metadata,
            reasons=reasons,
        )

    @abstractmethod
    def _run(
        self,
        *,
        df: pd.DataFrame,
        dataset_context: DatasetContext,
        target_columns: list[str],
        selected_model: SelectedModel,
        extra_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Returns {"evidence": {...}, "metrics": {...}, "charts": [...]}
        — or {"error": "..."} to signal no evidence could be produced (e.g.
        target columns missing from df).

        extra_context (Phase 4, plan "zany-giggling-crayon"): orchestration-
        supplied, non-column context an analyzer can't derive from df/
        target_columns alone — e.g. RootCauseAnalyzer's pre-computed
        total_variance, KPISummaryStrategy/KPIVarianceStrategy's resolved
        KPI dict. Always a dict, never None (execute() defaults it); empty
        for every analyzer that doesn't need it — most just ignore the
        parameter."""
        raise NotImplementedError


class AnalyzerRegistry:
    """Maps analysis_type -> Analyzer instance. A small explicit dict, the
    same deliberate non-auto-discovery choice as PluginRegistry (Phase 2)
    — one new analysis type is one new entry here, not filesystem
    globbing magic."""

    def __init__(self, analyzers: dict[str, Analyzer] | None = None):
        self._analyzers = dict(analyzers) if analyzers is not None else _default_analyzers()

    def get(self, analysis_type: str) -> Analyzer | None:
        return self._analyzers.get(analysis_type)

    def all(self) -> dict[str, Analyzer]:
        return dict(self._analyzers)


def _default_analyzers() -> dict[str, Analyzer]:
    from app.services.analyzers.forecast_analyzer import ForecastAnalyzer
    from app.services.analyzers.trend_analyzer import TrendAnalyzer
    from app.services.analyzers.correlation_analyzer import CorrelationAnalyzer
    from app.services.analyzers.clustering_analyzer import ClusteringAnalyzer
    from app.services.analyzers.segmentation_analyzer import SegmentationAnalyzer
    from app.services.analyzers.anomaly_analyzer import AnomalyAnalyzer
    from app.services.analyzers.outlier_analyzer import OutlierAnalyzer
    from app.services.analyzers.classification_analyzer import ClassificationAnalyzer
    from app.services.analyzers.regression_analyzer import RegressionAnalyzer
    from app.services.analyzers.feature_importance_analyzer import FeatureImportanceAnalyzer
    from app.services.analyzers.distribution_analyzer import DistributionAnalyzer
    from app.services.analyzers.ranking_analyzer import RankingAnalyzer
    from app.services.analyzers.comparative_analyzer import ComparativeAnalyzer
    from app.services.analyzers.root_cause_analyzer import RootCauseAnalyzer
    from app.services.analyzers.time_series_analyzer import TimeSeriesAnalyzer
    from app.services.analyzers.association_analyzer import AssociationAnalyzer
    from app.services.analyzers.kpi_summary_analyzer import KPISummaryAnalyzer
    from app.services.analyzers.kpi_variance_analyzer import KPIVarianceAnalyzer

    analyzers: list[Analyzer] = [
        ForecastAnalyzer(), TrendAnalyzer(), CorrelationAnalyzer(),
        ClusteringAnalyzer(), SegmentationAnalyzer(), AnomalyAnalyzer(),
        OutlierAnalyzer(), ClassificationAnalyzer(), RegressionAnalyzer(),
        FeatureImportanceAnalyzer(), DistributionAnalyzer(), RankingAnalyzer(),
        ComparativeAnalyzer(), RootCauseAnalyzer(), TimeSeriesAnalyzer(),
        AssociationAnalyzer(), KPISummaryAnalyzer(), KPIVarianceAnalyzer(),
    ]
    return {a.analysis_type: a for a in analyzers}
