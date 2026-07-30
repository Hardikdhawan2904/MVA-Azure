"""app/services/kpi_discovery/semantic_kpi_discovery.py — Stage 2 of the
Agent 3 redesign (plan "zany-giggling-crayon").

Synthesizes named business KPIs from semantic-role *combinations* instead
of hardcoded column names — "Revenue → Financial Measure → generate KPIs
dynamically" instead of "Revenue, Budget, Actual hardcoded". Runs every
rule in `kpi_discovery_rules.ALL_RULES` and concatenates whatever each one
contributes; a dataset with no matching semantic vocabulary at all simply
yields an empty list rather than erroring — this stage is additive, never
required for the rest of the pipeline to function.
"""

from __future__ import annotations

from app.services.dataset_context.models import DatasetContext
from app.services.kpi_discovery.kpi_discovery_rules import ALL_RULES
from app.services.kpi_discovery.models import DiscoveredKPI


class SemanticKPIDiscovery:
    def discover(self, dataset_context: DatasetContext) -> list[DiscoveredKPI]:
        discovered: list[DiscoveredKPI] = []
        for rule in ALL_RULES:
            discovered.extend(rule(dataset_context))
        return discovered
