"""app/services/domain_plugins/thin_kpi_plugin.py — shared enhance_plan()
logic for the Finance/HR/Payments/Customer starter plugins (Agent 3
redesign plan "zany-giggling-crayon", "Ship Finance/HR/Payments/Customer
starter plugins").

These four plugins are deliberately thin — a curated KPI catalog only, no
driver columns (no labeled-mode root_cause), no ML-feature-column
overrides, no forecast/trend/time_series injection — unlike InsurancePlugin,
which is the fully-built reference implementation. The one piece of
curated-KPI *behavior* they still need, to avoid regressing relative to
GenericDomainPlugin, is kpi_summary/kpi_variance injection when a question
resolves to one of their own curated KPIs — factored here once rather than
copy-pasted four times, since all four plugins need the exact same narrow
logic.

Correctness note: interpret_question's KPI/filter resolution (Phase 4,
Decision 4) narrows question_intent.candidate_analysis_types to
{"kpi_summary"} or {"kpi_variance"} whenever a plugin's own
get_default_kpi_name()/get_intent_vocabulary() resolves a real KPI and no
stronger keyword group already matched. The generic AnalyticsPlanner never
proposes kpi_summary/kpi_variance itself (that's exclusively DomainPlugin.
enhance_plan()'s job) — so a plugin that populates get_intent_vocabulary()
KPI aliases or get_default_kpi_name() without also injecting the matching
PlannedAnalysis would make some questions produce strictly less than
GenericDomainPlugin's own generic report (an empty, unplannable candidate
set instead of a real multi-analysis fallback). This mixin exists so that
never happens.
"""

from __future__ import annotations

from typing import Any

from app.services.domain_plugins.base import DomainPlugin
from app.services.planning.models import PlannedAnalysis
from app.services.question_interpreter.models import QuestionIntent


class ThinKPIDomainPlugin(DomainPlugin):
    """Concrete base for a curated-KPI-catalog-only domain plugin.
    Subclasses still implement applies_to()/get_kpi_definitions()/
    get_rule_engine_paths() (DomainPlugin's own contract) — this class adds
    only the kpi_summary/kpi_variance injection that makes
    get_default_kpi_name()/get_intent_vocabulary()'s KPI aliases actually
    answerable, kept intentionally narrower than InsurancePlugin's own
    enhance_plan() (no forecast/trend/root_cause/time_series injection, no
    ML-feature-column overrides — those require domain-specific ML/driver
    knowledge this thin tier doesn't have)."""

    def enhance_plan(self, plan: list, dataset_context: Any, question_intent: QuestionIntent | None = None) -> list:
        if question_intent is None or not question_intent.resolved_kpi_name:
            return plan

        kpi = self.get_kpi_definitions().get(question_intent.resolved_kpi_name)
        if not kpi:
            return plan

        actual_col = kpi.get("actual_column")
        budget_col = kpi.get("budget_column")
        prior_col = kpi.get("prior_year_column")
        summary_cols = [c for c in [actual_col, budget_col, prior_col] if c]
        if not summary_cols:
            return plan
        candidates = question_intent.candidate_analysis_types

        injected: list[PlannedAnalysis] = []
        if "kpi_summary" in candidates:
            injected.append(PlannedAnalysis(
                analysis_type="kpi_summary", target_columns=summary_cols,
                rationale=f"Curated KPI '{question_intent.resolved_kpi_name}' resolved from the question",
                priority=0, is_kpi_grounded=True,
            ))
        if "kpi_variance" in candidates and budget_col:
            injected.append(PlannedAnalysis(
                analysis_type="kpi_variance", target_columns=summary_cols,
                rationale=f"Curated KPI '{question_intent.resolved_kpi_name}' variance requested",
                priority=0, is_kpi_grounded=True,
            ))

        if not injected:
            return plan
        injected_types = {p.analysis_type for p in injected}
        return [p for p in plan if p.analysis_type not in injected_types] + injected
