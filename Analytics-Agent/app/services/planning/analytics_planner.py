"""app/services/planning/analytics_planner.py — Stage 4 of the Agent 3
redesign (plan "zany-giggling-crayon").

Decides WHAT is analytically relevant. Deliberately does not decide how
many analyses run or in what order — that's the Scheduler's job (Stage 5)
— so adding parallel execution or new prioritization rules later never
requires touching planning logic.
"""

from __future__ import annotations

from app.services.capability_resolution.models import CapabilityProfile
from app.services.dataset_context.models import DatasetContext
from app.services.kpi_discovery.models import DiscoveredKPI
from app.services.planning.models import AnalyticsPlan, PlannedAnalysis
from app.services.planning.planning_rules import ALL_RULES
from app.services.question_interpreter.models import QuestionIntent


class AnalyticsPlanner:
    def plan(
        self,
        dataset_context: DatasetContext,
        capability_profile: CapabilityProfile,
        discovered_kpis: list[DiscoveredKPI],
        question_intent: QuestionIntent | None = None,
    ) -> AnalyticsPlan:
        raw: list[PlannedAnalysis] = []
        for rule in ALL_RULES:
            raw.extend(rule(dataset_context, capability_profile, discovered_kpis, question_intent))

        # Dedup by analysis_type — rules are ordered most-specific-first,
        # so the first occurrence is kept.
        deduped: dict[str, PlannedAnalysis] = {}
        for analysis in raw:
            if analysis.analysis_type not in deduped:
                deduped[analysis.analysis_type] = analysis

        if question_intent is not None:
            deduped = {
                analysis_type: analysis for analysis_type, analysis in deduped.items()
                if analysis_type in question_intent.candidate_analysis_types
            }

        return sorted(deduped.values(), key=lambda a: a.priority)
