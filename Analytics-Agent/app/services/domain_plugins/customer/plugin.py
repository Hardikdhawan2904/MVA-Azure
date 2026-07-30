"""app/services/domain_plugins/customer/plugin.py — thin starter plugin
for Agent 2's "Customer" domain. See finance/plugin.py's docstring for the
shared design rationale (ThinKPIDomainPlugin, nonexistent hierarchy/
business_rules paths)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.domain_plugins.thin_kpi_plugin import ThinKPIDomainPlugin

_PLUGIN_DIR = Path(__file__).parent
_NO_CONFIG_DIR = _PLUGIN_DIR / "_no_config_here"

_KPI_ALIASES = {
    "customer satisfaction": "customer_satisfaction_score",
    "csat": "customer_satisfaction_score",
    "satisfaction score": "customer_satisfaction_score",
    "churn": "churn_rate",
    "churn rate": "churn_rate",
    "lifetime value": "customer_lifetime_value",
    "clv": "customer_lifetime_value",
    "ltv": "customer_lifetime_value",
}


class CustomerPlugin(ThinKPIDomainPlugin):
    def applies_to(self, detected_domain: str) -> bool:
        return (detected_domain or "").strip().lower() == "customer"

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
        return "customer"

    def get_default_kpi_name(self) -> str | None:
        return "customer_satisfaction_score"
