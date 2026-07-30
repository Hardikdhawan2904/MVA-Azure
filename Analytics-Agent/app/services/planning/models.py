"""app/services/planning/models.py — Stage 4 of the Agent 3 redesign (plan
"zany-giggling-crayon"): the PlannedAnalysis shape.

AnalyticsPlan is just `list[PlannedAnalysis]` — no wrapper class, matching
the plain-list pattern already used for `list[DiscoveredKPI]` (Stage 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlannedAnalysis:
    analysis_type: str
    target_columns: list[str] = field(default_factory=list)
    rationale: str = ""
    priority: int = 5             # lower = higher priority; consumed by the Scheduler (Stage 5), not decided here
    is_kpi_grounded: bool = False  # True when this analysis targets a DiscoveredKPI/curated KPI, not a bare structural pattern


AnalyticsPlan = list[PlannedAnalysis]
