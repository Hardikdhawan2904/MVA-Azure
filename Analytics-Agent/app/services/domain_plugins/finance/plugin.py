"""app/services/domain_plugins/finance/plugin.py — thin starter plugin for
Agent 2's "Finance" domain (Agent 3 redesign plan "zany-giggling-crayon",
"Ship Finance/HR/Payments/Customer starter plugins"). Curated KPI catalog
only — no driver columns, no ML-feature-column overrides. See
ThinKPIDomainPlugin's docstring for why enhance_plan() is still required
(not a no-op) despite this being "thin".

hierarchy_path/business_rules_path deliberately point at nonexistent
files — RuleEngine's own "file not found -> {}" graceful degradation (see
rule_engine.py's _load_json/_load_yaml_file) means this plugin never
silently inherits Insurance's hierarchy/business rules, the same
correctness property GenericDomainPlugin was built to guarantee.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.domain_plugins.thin_kpi_plugin import ThinKPIDomainPlugin

_PLUGIN_DIR = Path(__file__).parent
_NO_CONFIG_DIR = _PLUGIN_DIR / "_no_config_here"

_KPI_ALIASES = {
    "net profit": "net_profit",
    "profit": "net_profit",
    "revenue": "revenue",
    "sales": "revenue",
    "operating cost": "operating_cost",
    "operating costs": "operating_cost",
    "costs": "operating_cost",
}


class FinancePlugin(ThinKPIDomainPlugin):
    def applies_to(self, detected_domain: str) -> bool:
        return (detected_domain or "").strip().lower() == "finance"

    def get_kpi_definitions(self) -> dict[str, Any]:
        with open(_PLUGIN_DIR / "kpi_definitions.json", encoding="utf-8") as f:
            return json.load(f).get("kpis", {})

    def get_intent_vocabulary(self) -> dict[str, Any]:
        return {"kpi_aliases": dict(_KPI_ALIASES), "dimension_keywords": {}}

    def get_rule_engine_paths(self) -> dict[str, str]:
        return {
            "kpi_definitions_path": str(_PLUGIN_DIR / "kpi_definitions.json"),
            "hierarchy_path": str(_NO_CONFIG_DIR / "drill_down_hierarchy.json"),
            "business_rules_path": str(_NO_CONFIG_DIR / "business_rules.yml"),
        }

    def get_view_name(self) -> str:
        return "finance"

    def get_default_kpi_name(self) -> str | None:
        return "net_profit"
