"""app/services/scheduling/analytics_scheduler.py — Stage 5 of the Agent 3
redesign (plan "zany-giggling-crayon").

Decides WHEN/HOW MANY, never WHAT — that's the Planner's job (Stage 4).
Without this stage, a wide dataset (many numeric/categorical columns, a
hierarchy, several date columns) could make the Planner's unbounded output
translate into dozens of executed analyses in one request: slow,
expensive, and mostly irrelevant to what anyone actually asked. Keeping
budget/ordering here — not in the Planner — means adding parallel
execution or new priority rules later is a Scheduler-only change; the
Planner never needs to know.

Prioritization, in order, before trimming to max_parallel_analyses:
  1. Analyses matching question_intent.candidate_analysis_types —
     explicitly requested, always kept, never trimmed by budget.
  2. is_kpi_grounded analyses — tied to a real named business metric.
  3. Everything else, ranked by PlannedAnalysis.priority.

Among the survivors, only the top max_ml_analyses (same order) get
ml_execution_allowed=True — the rest are forced toward deterministic
strategies in Stage 6 regardless of what CapabilityProfile.execution says.
max_expensive_operations is carried in BudgetConfig for Stage 6's
ModelSelector to enforce (it isn't known here which algorithm — and
therefore which cost_tier — a given analysis will end up selecting).
"""

from __future__ import annotations

from app.services.planning.models import AnalyticsPlan
from app.services.question_interpreter.models import QuestionIntent
from app.services.scheduling.budget_config import BudgetConfig
from app.services.scheduling.models import ScheduledAnalysis, ScheduledPlan


class AnalyticsScheduler:
    def schedule(
        self,
        analytics_plan: AnalyticsPlan,
        budget: BudgetConfig,
        question_intent: QuestionIntent | None = None,
    ) -> ScheduledPlan:
        requested_types = question_intent.candidate_analysis_types if question_intent else set()

        def sort_key(pa):
            return (pa.analysis_type not in requested_types, not pa.is_kpi_grounded, pa.priority)

        ordered = sorted(analytics_plan, key=sort_key)

        requested = [pa for pa in ordered if pa.analysis_type in requested_types]
        others = [pa for pa in ordered if pa.analysis_type not in requested_types]
        remaining_budget = max(0, budget.max_parallel_analyses - len(requested))
        final = requested + others[:remaining_budget]

        scheduled: ScheduledPlan = []
        ml_allowed_count = 0
        for sequence, pa in enumerate(final, start=1):
            allow_ml = ml_allowed_count < budget.max_ml_analyses
            if allow_ml:
                ml_allowed_count += 1
            scheduled.append(ScheduledAnalysis(sequence=sequence, planned_analysis=pa, ml_execution_allowed=allow_ml))
        return scheduled
