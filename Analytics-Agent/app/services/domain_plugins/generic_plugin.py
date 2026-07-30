"""app/services/domain_plugins/generic_plugin.py — GenericDomainPlugin,
the true "no plugin matched" fallback (Agent 3 redesign, Phase 4 — plan
"zany-giggling-crayon").

Before Phase 4, nodes/pipeline.py's `PluginRegistry().find_plugin(domain)
or InsurancePlugin()` fell back to InsurancePlugin() unconditionally
whenever no registered plugin matched (e.g. detected_domain not
forwarded, or a genuinely unrelated dataset) — harmless while Phase 2/3
only ever exercised it against the Insurance dataset itself, but a real
bug once Stage 1-8's generic engine is actually live: an HR/Payments/
whatever dataset with no matching plugin would silently inherit
Insurance's curated KPI catalog (defaulting every unresolved query to
"underwriting_result") instead of getting the fully generic, dataset-
column-driven pipeline. Confirmed live during Phase 4 verification.

GenericDomainPlugin is almost entirely DomainPlugin's own base-class
defaults (empty intent vocabulary, no driver columns, no preferred
strategy, no default KPI) — the two things it must override are
get_kpi_definitions() (the abstractmethod; empty catalog) and
get_rule_engine_paths(), since RuleEngine's own no-args default is
Insurance's real files (Phase 2's backward-compat requirement) — pointing
at deliberately nonexistent paths makes RuleEngine load empty dicts via
its own already-tested "file not found -> {}" graceful degradation,
rather than silently inheriting Insurance's KPI catalog through the back
door.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.domain_plugins.base import DomainPlugin

_NO_CONFIG_DIR = Path(__file__).parent / "generic" / "_no_config_here"


class GenericDomainPlugin(DomainPlugin):
    def applies_to(self, detected_domain: str) -> bool:
        # Never matched via PluginRegistry.find_plugin() — this is the
        # explicit fallback constructed directly when no real plugin
        # applies, not something to compete for the "Insurance"/"HR"/...
        # string.
        return False

    def get_kpi_definitions(self) -> dict[str, Any]:
        return {}

    def get_rule_engine_paths(self) -> dict[str, str]:
        return {
            "kpi_definitions_path": str(_NO_CONFIG_DIR / "kpi_definitions.json"),
            "hierarchy_path": str(_NO_CONFIG_DIR / "drill_down_hierarchy.json"),
            "business_rules_path": str(_NO_CONFIG_DIR / "business_rules.yml"),
        }

    def get_view_name(self) -> str:
        return "dataset"
