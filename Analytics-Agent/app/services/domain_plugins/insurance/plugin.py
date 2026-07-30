"""app/services/domain_plugins/insurance/plugin.py — the reference
DomainPlugin implementation for the Agent 3 redesign (plan
"zany-giggling-crayon"), Phase 2.

Contains the *exact current content* that used to be hardcoded directly in
`nodes/pipeline.py` (`_KPI_ALIASES`, `_DIMENSION_KEYWORDS`) — moved here
verbatim, not rewritten. `kpi_definitions.json`/`business_rules.yml`/
`drill_down_hierarchy.json` in this same directory are byte-identical
copies of `config/rules/kpi_definitions.json` etc. (verified via `diff`
at copy time) — RuleEngine's own defaults still point at the original
`config/` files (nothing there was moved or deleted), so any caller that
doesn't go through this plugin keeps behaving exactly as before; this
plugin's paths are what `nodes/pipeline.py` is wired to use explicitly
(Phase 2's "wire pipeline.py to source values from the plugin" step).

This is "100% of today's Insurance behavior preserved" in concrete form:
if this plugin were deleted, Insurance would fall back to whatever the
generic engine does by default — a real, testable property once the new
graph topology (Phase 4) exists. For now, in Phase 2, it exists and is
already the thing nodes/pipeline.py's classes are constructed from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.domain_plugins.base import DomainPlugin
from app.services.planning.models import PlannedAnalysis
from app.services.question_interpreter.models import QuestionIntent
from app.services.tools.root_cause_tool import DEFAULT_DRIVER_LABELS, DEFAULT_VARIANCE_DRIVER_COLUMNS

_PLUGIN_DIR = Path(__file__).parent

# Moved verbatim from nodes/pipeline.py's _KPI_ALIASES.
_KPI_ALIASES = {
    "gross written premium": "gross_written_premium",
    "gwp": "gross_written_premium",
    "net written premium": "net_written_premium",
    "nwp": "net_written_premium",
    "loss ratio": "loss_ratio",
    "combined ratio": "combined_ratio",
    "expense ratio": "expense_ratio",
    "underwriting result": "underwriting_result",
    "uwr": "underwriting_result",
    "earned premium": "earned_premium",
    "net earned premium": "net_earned_premium",
    "nep": "net_earned_premium",
    "claim frequency": "claim_frequency",
    "claims frequency": "claim_frequency",
    "claim severity": "average_claim_severity",
    "renewal rate": "renewal_rate",
    "lapse rate": "policy_lapse_rate",
    "reserve adequacy": "reserve_adequacy_ratio",
    "ibnr": "ibnr_reserve",
    "premium collection": "premium_collection_rate",
}

# Moved verbatim from nodes/pipeline.py's _DIMENSION_KEYWORDS.
_DIMENSION_KEYWORDS = {
    "region": {
        "emea": "EMEA", "apac": "APAC", "africa": "Africa", "middle east": "Middle East",
    },
    "insurance_segment": {
        "general insurance": "General Insurance", "health": "Health",
        "specialty": "Commercial Specialty", "commercial": "Commercial Insurance", "life": "Life",
    },
    "country_name": {
        "uk": "United Kingdom", "united kingdom": "United Kingdom",
        "singapore": "Singapore", "india": "India",
        "australia": "Australia", "uae": "United Arab Emirates",
        "south africa": "South Africa",
    },
    "line_of_business": {
        "motor": "Motor Insurance", "property": "Property Insurance",
        "travel": "Travel Insurance",
    },
}

_PREFERRED_DETERMINISTIC_STRATEGIES = {
    # Matches today's exact fallback engine names/behavior (see
    # nodes/pipeline.py's handle_forecast/handle_anomaly/handle_segment) —
    # a later phase's generic registry defaults (Moving Average, IQR, ...)
    # never change Insurance's own narrated output.
    "forecast": "Linear Trend",       # closest registry match to today's analytics_tool.trend()
    "anomaly_detection": "Z-Score",   # already an exact name match
    "segmentation": "Insurance Combined Ratio Buckets",
}


class InsuranceCombinedRatioBucketsStrategy:
    """Reproduces today's exact deterministic segmentation fallback
    (nodes/pipeline.py's handle_segment): fixed Combined Ratio buckets,
    not a generic quantile/equal-width bin. Registered in
    config/model_registry.yml but only ever selected via
    InsurancePlugin.get_preferred_deterministic_strategy() — not a
    generic default for other domains."""

    @staticmethod
    def segment(df, ratio_column: str = "combined_ratio_actual") -> dict[str, Any]:
        low = int((df[ratio_column] < 90).sum())
        medium = int(((df[ratio_column] >= 90) & (df[ratio_column] <= 100)).sum())
        high = int((df[ratio_column] > 100).sum())
        return {
            "evidence": {
                # Nested under "segments"/"total_records" — matches
                # handle_segment's exact deterministic-fallback evidence
                # shape (and every other segmentation strategy's own
                # evidence["segments"] convention), not a flat top-level
                # dict — see Evidence.flatten()/wrap_if_flat()'s handling.
                "segments": {
                    "Low Risk (<90% Combined Ratio)": low,
                    "Medium Risk (90-100% Combined Ratio)": medium,
                    "High Risk (>100% Combined Ratio)": high,
                },
                "total_records": low + medium + high,
            }
        }


class InsurancePlugin(DomainPlugin):
    def applies_to(self, detected_domain: str) -> bool:
        return (detected_domain or "").strip().lower() == "insurance"

    def get_kpi_definitions(self) -> dict[str, Any]:
        import json
        with open(_PLUGIN_DIR / "kpi_definitions.json", encoding="utf-8") as f:
            return json.load(f).get("kpis", {})

    def get_intent_vocabulary(self) -> dict[str, Any]:
        return {"kpi_aliases": dict(_KPI_ALIASES), "dimension_keywords": dict(_DIMENSION_KEYWORDS)}

    def get_driver_columns(self) -> list[str] | None:
        return list(DEFAULT_VARIANCE_DRIVER_COLUMNS)

    def get_driver_labels(self) -> dict[str, str] | None:
        return dict(DEFAULT_DRIVER_LABELS)

    def get_preferred_deterministic_strategy(self, analysis_type: str) -> str | None:
        return _PREFERRED_DETERMINISTIC_STRATEGIES.get(analysis_type)

    def get_rule_engine_paths(self) -> dict[str, str]:
        return {
            "kpi_definitions_path": str(_PLUGIN_DIR / "kpi_definitions.json"),
            "hierarchy_path": str(_PLUGIN_DIR / "drill_down_hierarchy.json"),
            "business_rules_path": str(_PLUGIN_DIR / "business_rules.yml"),
        }

    def get_view_name(self) -> str:
        return "insurance"

    def get_default_kpi_name(self) -> str | None:
        # The named home for what used to be nodes/pipeline.py's hardcoded
        # `kpi_name = self._detect_kpi(query_lower) or self.memory.get_last_kpi()
        # or "underwriting_result"` fallback.
        return "underwriting_result"

    def enhance_plan(self, plan: list, dataset_context: Any, question_intent: QuestionIntent | None = None) -> list:
        """Injects kpi_summary/kpi_variance/KPI-grounded root_cause
        PlannedAnalysis entries using the resolved curated KPI's actual/
        budget/prior_year/driver columns — the only place this Insurance-
        specific curated-KPI-query behavior lives; the generic
        AnalyticsPlanner never proposes these itself (see DomainPlugin.
        enhance_plan()'s docstring).

        Any generic pattern-matched PlannedAnalysis already in `plan` for
        the same analysis_type (e.g. rule_kpi_grounded_root_cause's
        Stage-2-KPI-sourced root_cause) is replaced, not duplicated —
        AnalyticsPlanner dedupes by analysis_type, but only within its own
        rule pass; enhance_plan() runs after that dedup, so it must do its
        own to avoid the Scheduler seeing two root_cause entries.
        """
        plan = self._override_ml_feature_columns(plan)
        plan = self._inject_kpi_grounded_analyses(plan, dataset_context, question_intent)
        return plan

    def _override_ml_feature_columns(self, plan: list) -> list:
        """anomaly_detection/segmentation/clustering never depend on which
        KPI (if any) the question is about — today's handle_anomaly/
        handle_segment always score the same fixed ratio-column set
        (ml/anomaly_detector.py's RATIO_FEATURE_COLUMNS,
        ml/classifier.py's SEGMENT_FEATURE_COLS), unconditionally.
        Overrides whatever generic target_columns the Planner's structural
        rules picked — those rules have no way to know this domain-
        specific column set exists."""
        from app.services.ml.anomaly_detector import RATIO_FEATURE_COLUMNS
        from app.services.ml.classifier import SEGMENT_FEATURE_COLS

        overrides = {
            "anomaly_detection": list(RATIO_FEATURE_COLUMNS),
            "segmentation": list(SEGMENT_FEATURE_COLS),
            "clustering": list(SEGMENT_FEATURE_COLS),
        }
        result = []
        for p in plan:
            cols = overrides.get(p.analysis_type)
            if cols:
                result.append(PlannedAnalysis(
                    analysis_type=p.analysis_type, target_columns=cols,
                    rationale=p.rationale + " (Insurance's fixed ratio-feature columns)",
                    priority=p.priority, is_kpi_grounded=p.is_kpi_grounded,
                ))
            else:
                result.append(p)
        return result

    def _inject_kpi_grounded_analyses(self, plan: list, dataset_context: Any, question_intent: QuestionIntent | None) -> list:
        """Injects kpi_summary/kpi_variance/KPI-grounded root_cause/
        forecast/trend/time_series_analysis PlannedAnalysis entries using
        the resolved curated KPI's actual/budget/prior_year/driver
        columns — the only place this Insurance-specific curated-KPI-query
        behavior lives; the generic AnalyticsPlanner never proposes these
        itself (see DomainPlugin.enhance_plan()'s docstring). Today's
        handle_forecast/handle_trend are just as KPI-driven as
        handle_show_kpi/handle_variance/handle_root_cause — every one of
        them resolves the curated KPI and reads its actual_column, unlike
        handle_anomaly/handle_segment above.

        Any generic pattern-matched PlannedAnalysis already in `plan` for
        the same analysis_type (e.g. rule_kpi_grounded_root_cause's
        Stage-2-KPI-sourced root_cause, or rule_temporal_metric's generic
        forecast/trend) is replaced, not duplicated — AnalyticsPlanner
        dedupes by analysis_type, but only within its own rule pass;
        enhance_plan() runs after that dedup, so it must do its own to
        avoid the Scheduler seeing two entries of the same type.
        """
        if question_intent is None or not question_intent.resolved_kpi_name:
            return plan

        kpi = self.get_kpi_definitions().get(question_intent.resolved_kpi_name)
        if not kpi:
            return plan

        actual_col = kpi.get("actual_column")
        budget_col = kpi.get("budget_column")
        prior_col = kpi.get("prior_year_column")
        summary_cols = [c for c in [actual_col, budget_col, prior_col] if c]
        temporal_col = next((c.name for c in dataset_context.columns if c.is_temporal), None) if dataset_context else None
        candidates = question_intent.candidate_analysis_types

        injected: list[PlannedAnalysis] = []
        if "kpi_summary" in candidates:
            injected.append(PlannedAnalysis(
                analysis_type="kpi_summary", target_columns=summary_cols,
                rationale=f"Curated KPI '{question_intent.resolved_kpi_name}' resolved from the question",
                priority=0, is_kpi_grounded=True,
            ))
        if "kpi_variance" in candidates:
            injected.append(PlannedAnalysis(
                analysis_type="kpi_variance", target_columns=summary_cols,
                rationale=f"Curated KPI '{question_intent.resolved_kpi_name}' variance requested",
                priority=0, is_kpi_grounded=True,
            ))
        driver_cols = self.get_driver_columns()
        if "root_cause" in candidates and driver_cols and actual_col:
            injected.append(PlannedAnalysis(
                analysis_type="root_cause", target_columns=[actual_col, *driver_cols],
                rationale=f"Root-cause decomposition for curated KPI '{question_intent.resolved_kpi_name}'",
                priority=0, is_kpi_grounded=True,
            ))
        if actual_col and temporal_col:
            for analysis_type in ("forecast", "trend", "time_series_analysis"):
                if analysis_type in candidates:
                    injected.append(PlannedAnalysis(
                        analysis_type=analysis_type, target_columns=[temporal_col, actual_col],
                        rationale=f"Curated KPI '{question_intent.resolved_kpi_name}' — {analysis_type} requested",
                        priority=0, is_kpi_grounded=True,
                    ))

        if not injected:
            return plan
        injected_types = {p.analysis_type for p in injected}
        return [p for p in plan if p.analysis_type not in injected_types] + injected
