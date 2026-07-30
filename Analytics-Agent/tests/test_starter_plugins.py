"""tests/test_starter_plugins.py — Finance/HR/Payments/Customer thin
starter plugins (Agent 3 redesign plan "zany-giggling-crayon", "Ship
Finance/HR/Payments/Customer starter plugins").

Covers the property that actually matters for a "thin" plugin: it must
never do *worse* than GenericDomainPlugin's own generic report — see
ThinKPIDomainPlugin's docstring for exactly why that's a real risk (a
plugin that resolves a curated KPI without injecting the matching
kpi_summary/kpi_variance analysis would narrow the plan to something
unplannable).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.services.domain_plugins.customer.plugin import CustomerPlugin
from app.services.domain_plugins.finance.plugin import FinancePlugin
from app.services.domain_plugins.hr.plugin import HRPlugin
from app.services.domain_plugins.payments.plugin import PaymentsPlugin
from app.services.planning.models import PlannedAnalysis
from app.services.question_interpreter.models import QuestionIntent
from app.services.tools.rule_engine import RuleEngine

_PLUGINS = [
    ("Finance", FinancePlugin(), "net_profit"),
    ("HR", HRPlugin(), "headcount"),
    ("Payments", PaymentsPlugin(), "transaction_volume"),
    ("Customer", CustomerPlugin(), "customer_satisfaction_score"),
]


@pytest.mark.parametrize("domain,plugin,default_kpi", _PLUGINS)
def test_applies_to_matches_own_canonicalized_domain_only(domain, plugin, default_kpi):
    assert plugin.applies_to(domain) is True
    assert plugin.applies_to(domain.lower()) is True
    assert plugin.applies_to(f"  {domain}  ") is True
    assert plugin.applies_to("SomeOtherDomain") is False
    assert plugin.applies_to("Insurance") is False


@pytest.mark.parametrize("domain,plugin,default_kpi", _PLUGINS)
def test_kpi_definitions_load_and_include_the_default_kpi(domain, plugin, default_kpi):
    kpis = plugin.get_kpi_definitions()
    assert default_kpi in kpis
    for name, kpi in kpis.items():
        assert "label" in kpi and "unit" in kpi and "higher_is_better" in kpi and "category" in kpi
    assert plugin.get_default_kpi_name() == default_kpi


@pytest.mark.parametrize("domain,plugin,default_kpi", _PLUGINS)
def test_rule_engine_paths_never_silently_inherit_insurance_config(domain, plugin, default_kpi):
    """RuleEngine.__init__ defaults hierarchy_path/business_rules_path to
    Insurance's real files whenever *any* path is overridden and the
    others are left None (see rule_engine.py's constructor) — so this
    plugin must point hierarchy_path/business_rules_path at genuinely
    nonexistent files, not omit them, or Finance/HR/Payments/Customer
    would silently load Insurance's hierarchy/business rules."""
    paths = plugin.get_rule_engine_paths()
    assert Path(paths["kpi_definitions_path"]).exists()
    assert not Path(paths["hierarchy_path"]).exists()
    assert not Path(paths["business_rules_path"]).exists()

    engine = RuleEngine(**paths)
    assert engine.get_kpi(default_kpi) is not None
    assert engine.get_kpi("underwriting_result") is None  # never inherited from Insurance


@pytest.mark.parametrize("domain,plugin,default_kpi", _PLUGINS)
def test_view_name_is_domain_specific_not_the_insurance_default(domain, plugin, default_kpi):
    assert plugin.get_view_name() not in ("insurance", "")


@pytest.mark.parametrize("domain,plugin,default_kpi", _PLUGINS)
def test_enhance_plan_is_noop_without_a_resolved_kpi(domain, plugin, default_kpi):
    plan = [PlannedAnalysis(analysis_type="correlation", target_columns=["a", "b"])]
    assert plugin.enhance_plan(plan, dataset_context=None) == plan
    assert plugin.enhance_plan(plan, dataset_context=None, question_intent=None) == plan

    intent_no_kpi = QuestionIntent(candidate_analysis_types={"kpi_summary"}, resolved_kpi_name=None)
    assert plugin.enhance_plan(plan, dataset_context=None, question_intent=intent_no_kpi) == plan


@pytest.mark.parametrize("domain,plugin,default_kpi", _PLUGINS)
def test_enhance_plan_injects_kpi_summary_for_the_resolved_kpi(domain, plugin, default_kpi):
    """The correctness property ThinKPIDomainPlugin exists to guarantee:
    a resolved KPI must actually become a plannable kpi_summary analysis,
    not just sit in the catalog unused."""
    intent = QuestionIntent(candidate_analysis_types={"kpi_summary"}, resolved_kpi_name=default_kpi)
    result = plugin.enhance_plan([], dataset_context=None, question_intent=intent)
    kpi_summary_entries = [p for p in result if p.analysis_type == "kpi_summary"]
    assert len(kpi_summary_entries) == 1
    assert kpi_summary_entries[0].is_kpi_grounded is True
    assert kpi_summary_entries[0].target_columns  # at least the actual_column


@pytest.mark.parametrize("domain,plugin,default_kpi", _PLUGINS)
def test_enhance_plan_injects_kpi_variance_only_when_a_budget_column_exists(domain, plugin, default_kpi):
    kpi = plugin.get_kpi_definitions()[default_kpi]
    intent = QuestionIntent(candidate_analysis_types={"kpi_variance"}, resolved_kpi_name=default_kpi)
    result = plugin.enhance_plan([], dataset_context=None, question_intent=intent)
    variance_entries = [p for p in result if p.analysis_type == "kpi_variance"]
    if kpi.get("budget_column"):
        assert len(variance_entries) == 1
    else:
        assert variance_entries == []


@pytest.mark.parametrize("domain,plugin,default_kpi", _PLUGINS)
def test_enhance_plan_ignores_unknown_kpi_names(domain, plugin, default_kpi):
    plan = [PlannedAnalysis(analysis_type="correlation", target_columns=["a", "b"])]
    intent = QuestionIntent(candidate_analysis_types={"kpi_summary"}, resolved_kpi_name="not_a_real_kpi")
    assert plugin.enhance_plan(plan, dataset_context=None, question_intent=intent) == plan


@pytest.mark.parametrize("domain,plugin,default_kpi", _PLUGINS)
def test_intent_vocabulary_default_kpi_alias_resolves_to_default_kpi_name(domain, plugin, default_kpi):
    vocab = plugin.get_intent_vocabulary()
    aliases = vocab["kpi_aliases"]
    assert default_kpi in aliases.values()
