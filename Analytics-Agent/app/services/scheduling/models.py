"""app/services/scheduling/models.py — Stage 5 of the Agent 3 redesign
(plan "zany-giggling-crayon"): the ScheduledAnalysis shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.planning.models import PlannedAnalysis


@dataclass
class ScheduledAnalysis:
    sequence: int
    planned_analysis: PlannedAnalysis
    # Can only ever downgrade toward cheaper execution, never upgrade past
    # what Stage 1 (AnalyticsCapabilityResolver) already determined is
    # genuinely viable — see AnalyticsScheduler.schedule()'s docstring.
    ml_execution_allowed: bool


ScheduledPlan = list[ScheduledAnalysis]
