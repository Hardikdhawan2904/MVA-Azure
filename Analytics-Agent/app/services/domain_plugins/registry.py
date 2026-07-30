"""app/services/domain_plugins/registry.py — PluginRegistry for the Agent
3 redesign (plan "zany-giggling-crayon").

Deliberately a small, explicit list rather than filesystem auto-discovery
(no `config/domains/*/plugin.py` globbing) — auto-discovery adds real
complexity for zero benefit at this scale (5 plugins total). Adding a
plugin later is still a one-line change here, not a redesign.

Finance/HR/Payments/Customer are thin starters (ThinKPIDomainPlugin —
curated KPI catalog + kpi_summary/kpi_variance only, no driver columns, no
ML-feature-column overrides) — Insurance remains the only fully-built
reference implementation. Any of these four falling out of the registry
(e.g. a bug) still degrades safely to GenericDomainPlugin's real generic
report, never to Insurance's catalog — see generic_plugin.py's docstring.
"""

from __future__ import annotations

from app.services.domain_plugins.base import DomainPlugin
from app.services.domain_plugins.customer.plugin import CustomerPlugin
from app.services.domain_plugins.finance.plugin import FinancePlugin
from app.services.domain_plugins.hr.plugin import HRPlugin
from app.services.domain_plugins.insurance.plugin import InsurancePlugin
from app.services.domain_plugins.payments.plugin import PaymentsPlugin

_PLUGINS: list[DomainPlugin] = [
    InsurancePlugin(),
    FinancePlugin(),
    HRPlugin(),
    PaymentsPlugin(),
    CustomerPlugin(),
]


class PluginRegistry:
    def __init__(self, plugins: list[DomainPlugin] | None = None):
        self._plugins = list(plugins) if plugins is not None else list(_PLUGINS)

    def find_plugin(self, detected_domain: str | None) -> DomainPlugin | None:
        if not detected_domain:
            return None
        for plugin in self._plugins:
            if plugin.applies_to(detected_domain):
                return plugin
        return None
