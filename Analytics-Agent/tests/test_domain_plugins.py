"""tests/test_domain_plugins.py — Agent 3 redesign, Phase 2 (plan
"zany-giggling-crayon"): DomainPlugin / PluginRegistry / InsurancePlugin.

The core property under test: the Insurance plugin's content is
byte-identical to what RuleEngine/RootCauseTool/SQLTool already default to
today — proving the plugin genuinely drives behavior rather than being
unused scaffolding, without changing what Insurance actually does.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import yaml

from app.services.dataset_context.models import ColumnContext, DatasetContext
from app.services.domain_plugins.customer.plugin import CustomerPlugin
from app.services.domain_plugins.finance.plugin import FinancePlugin
from app.services.domain_plugins.generic_plugin import GenericDomainPlugin
from app.services.domain_plugins.hr.plugin import HRPlugin
from app.services.domain_plugins.insurance.plugin import InsurancePlugin
from app.services.domain_plugins.payments.plugin import PaymentsPlugin
from app.services.domain_plugins.registry import PluginRegistry
from app.services.planning.models import PlannedAnalysis
from app.services.question_interpreter.models import QuestionIntent
from app.services.tools.root_cause_tool import DEFAULT_DRIVER_LABELS, DEFAULT_VARIANCE_DRIVER_COLUMNS, RootCauseTool
from app.services.tools.rule_engine import RuleEngine

_INSURANCE_PLUGIN_DIR = Path(__file__).parent.parent / "app/services/domain_plugins/insurance"
_ORIGINAL_KPI_DEFINITIONS = Path(__file__).parent.parent / "config/rules/kpi_definitions.json"
_ORIGINAL_BUSINESS_RULES = Path(__file__).parent.parent / "config/business_rules.yml"
_ORIGINAL_HIERARCHY = Path(__file__).parent.parent / "config/rules/drill_down_hierarchy.json"


# ── Copied files are byte-identical to the originals ────────────────────────

def test_kpi_definitions_copy_is_byte_identical_to_original():
    with open(_INSURANCE_PLUGIN_DIR / "kpi_definitions.json") as f:
        copy = f.read()
    with open(_ORIGINAL_KPI_DEFINITIONS) as f:
        original = f.read()
    assert copy == original


def test_business_rules_copy_is_byte_identical_to_original():
    with open(_INSURANCE_PLUGIN_DIR / "business_rules.yml") as f:
        copy = f.read()
    with open(_ORIGINAL_BUSINESS_RULES) as f:
        original = f.read()
    assert copy == original


def test_hierarchy_copy_is_byte_identical_to_original():
    with open(_INSURANCE_PLUGIN_DIR / "drill_down_hierarchy.json") as f:
        copy = f.read()
    with open(_ORIGINAL_HIERARCHY) as f:
        original = f.read()
    assert copy == original


# ── PluginRegistry ────────────────────────────────────────────────────────────

def test_registry_finds_insurance_plugin_case_insensitively():
    reg = PluginRegistry()
    assert isinstance(reg.find_plugin("Insurance"), InsurancePlugin)
    assert isinstance(reg.find_plugin("insurance"), InsurancePlugin)
    assert isinstance(reg.find_plugin("  Insurance  "), InsurancePlugin)


def test_registry_returns_none_for_unmatched_or_missing_domain():
    reg = PluginRegistry()
    assert reg.find_plugin("SomeFutureUnrecognizedDomain") is None
    assert reg.find_plugin(None) is None
    assert reg.find_plugin("") is None


# ── Phase 4.5 / starter plugins: explicit domain -> plugin resolution ───────
#
# Agent-Orchestrator's gate no longer restricts which of Agent 2's 5
# canonicalized domains (Finance/Payments/Customer/HR/Insurance) reach
# Agent 3 — this makes the "what actually runs for each domain" contract
# explicit and regression-proof. Finance/Payments/Customer/HR now resolve
# to their own thin starter plugins (ThinKPIDomainPlugin — curated KPI
# catalog + kpi_summary/kpi_variance only); a genuinely unrecognized domain
# still falls through to nothing here, which is what AnalyticsGraphNodes.
# __init__'s own `PluginRegistry().find_plugin(domain) or
# GenericDomainPlugin()` line (Phase 4) then turns into the fully generic
# engine.
@pytest.mark.parametrize("domain,expected_plugin_type", [
    ("Insurance", InsurancePlugin),
    ("Finance", FinancePlugin),
    ("Payments", PaymentsPlugin),
    ("Customer", CustomerPlugin),
    ("HR", HRPlugin),
    ("SomeFutureUnrecognizedDomain", None),
])
def test_registry_resolution_for_every_canonicalized_domain(domain, expected_plugin_type):
    plugin = PluginRegistry().find_plugin(domain)
    if expected_plugin_type is None:
        assert plugin is None
    else:
        assert isinstance(plugin, expected_plugin_type)


@pytest.mark.parametrize("domain,expected_plugin_type", [
    ("Insurance", InsurancePlugin),
    ("Finance", FinancePlugin),
    ("Payments", PaymentsPlugin),
    ("Customer", CustomerPlugin),
    ("HR", HRPlugin),
    (None, GenericDomainPlugin),
    ("SomeFutureUnrecognizedDomain", GenericDomainPlugin),
])
def test_analytics_graph_nodes_resolves_expected_plugin_per_domain(domain, expected_plugin_type, tmp_path):
    """The actual construction-time fallback AnalyticsGraphNodes.__init__
    uses (nodes/pipeline.py) — not just PluginRegistry in isolation. This
    is what an incoming request with each detected_domain value actually
    gets, end to end."""
    from app.agents.analytics_agent.nodes.pipeline import AnalyticsGraphNodes

    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    nodes = AnalyticsGraphNodes(
        dataset_path=str(csv_path), conversation_id="test-plugin-resolution", detected_domain=domain,
    )
    assert isinstance(nodes._domain_plugin, expected_plugin_type)


# ── InsurancePlugin content matches today's exact hardcoded values ─────────

def test_insurance_plugin_driver_columns_match_root_cause_tool_defaults():
    plugin = InsurancePlugin()
    assert plugin.get_driver_columns() == DEFAULT_VARIANCE_DRIVER_COLUMNS
    assert plugin.get_driver_labels() == DEFAULT_DRIVER_LABELS


def test_insurance_plugin_kpi_definitions_match_original_json():
    plugin = InsurancePlugin()
    with open(_ORIGINAL_KPI_DEFINITIONS) as f:
        expected = json.load(f)["kpis"]
    assert plugin.get_kpi_definitions() == expected


def test_insurance_plugin_intent_vocabulary_matches_pipeline_constants():
    """The _KPI_ALIASES/_DIMENSION_KEYWORDS moved out of nodes/pipeline.py
    verbatim — spot-check a few well-known entries survived the move
    exactly, not just that *some* dict came back."""
    plugin = InsurancePlugin()
    vocab = plugin.get_intent_vocabulary()
    assert vocab["kpi_aliases"]["gwp"] == "gross_written_premium"
    assert vocab["kpi_aliases"]["combined ratio"] == "combined_ratio"
    assert vocab["dimension_keywords"]["region"]["emea"] == "EMEA"
    assert vocab["dimension_keywords"]["insurance_segment"]["health"] == "Health"


def test_insurance_plugin_preferred_deterministic_strategies():
    plugin = InsurancePlugin()
    assert plugin.get_preferred_deterministic_strategy("forecast") == "Linear Trend"
    assert plugin.get_preferred_deterministic_strategy("anomaly_detection") == "Z-Score"
    assert plugin.get_preferred_deterministic_strategy("segmentation") == "Insurance Combined Ratio Buckets"
    assert plugin.get_preferred_deterministic_strategy("some_unrelated_type") is None


# ── End-to-end: RuleEngine/RootCauseTool constructed via the plugin ────────

def test_rule_engine_via_plugin_paths_matches_default_construction():
    plugin = InsurancePlugin()
    default = RuleEngine()
    via_plugin = RuleEngine(**plugin.get_rule_engine_paths())
    assert default._kpis == via_plugin._kpis
    assert default._predefined_rules == via_plugin._predefined_rules
    assert default._thresholds == via_plugin._thresholds
    assert default._variance_drivers == via_plugin._variance_drivers
    assert default._flag_decoding == via_plugin._flag_decoding
    assert default._hierarchy == via_plugin._hierarchy
    # And the lookup behavior itself matches, not just the raw dicts
    assert default.get_kpi("loss_ratio") == via_plugin.get_kpi("loss_ratio")


def test_root_cause_tool_via_plugin_matches_default_construction():
    import pandas as pd
    plugin = InsurancePlugin()
    default = RootCauseTool()
    via_plugin = RootCauseTool(driver_columns=plugin.get_driver_columns(), driver_labels=plugin.get_driver_labels())

    df = pd.DataFrame({
        "exposure_growth_variance": [100.0, -50.0],
        "premium_rate_variance": [200.0, 30.0],
    })
    result_default = default.analyse(df)
    result_plugin = via_plugin.analyse(df)
    assert result_default["driver_breakdown"] == result_plugin["driver_breakdown"]
    assert result_default["primary_driver"] == result_plugin["primary_driver"]


def test_insurance_plugin_view_name():
    assert InsurancePlugin().get_view_name() == "insurance"


# ── Phase 4: get_default_kpi_name() / enhance_plan() ────────────────────────

def test_insurance_plugin_default_kpi_name():
    assert InsurancePlugin().get_default_kpi_name() == "underwriting_result"


def test_enhance_plan_returns_plan_unchanged_with_no_question_intent():
    plugin = InsurancePlugin()
    plan = [PlannedAnalysis(analysis_type="trend", target_columns=["x"], rationale="r")]
    assert plugin.enhance_plan(plan, dataset_context=None) == plan
    assert plugin.enhance_plan(plan, dataset_context=None, question_intent=None) == plan


def test_enhance_plan_returns_plan_unchanged_when_no_kpi_resolved():
    plugin = InsurancePlugin()
    plan = [PlannedAnalysis(analysis_type="trend", target_columns=["x"], rationale="r")]
    intent = QuestionIntent(candidate_analysis_types={"kpi_summary"}, resolved_kpi_name=None)
    assert plugin.enhance_plan(plan, dataset_context=None, question_intent=intent) == plan


def test_enhance_plan_injects_kpi_summary_with_actual_budget_prior_year_columns():
    plugin = InsurancePlugin()
    intent = QuestionIntent(candidate_analysis_types={"kpi_summary"}, resolved_kpi_name="gross_written_premium")
    result = plugin.enhance_plan([], dataset_context=None, question_intent=intent)
    assert len(result) == 1
    injected = result[0]
    assert injected.analysis_type == "kpi_summary"
    assert injected.is_kpi_grounded is True
    kpi = plugin.get_kpi_definitions()["gross_written_premium"]
    assert injected.target_columns == [c for c in [kpi.get("actual_column"), kpi.get("budget_column"), kpi.get("prior_year_column")] if c]


def test_enhance_plan_injects_kpi_variance_distinctly_from_kpi_summary():
    plugin = InsurancePlugin()
    intent = QuestionIntent(candidate_analysis_types={"kpi_variance"}, resolved_kpi_name="gross_written_premium")
    result = plugin.enhance_plan([], dataset_context=None, question_intent=intent)
    assert [p.analysis_type for p in result] == ["kpi_variance"]


def test_enhance_plan_replaces_generic_root_cause_not_duplicates_it():
    plugin = InsurancePlugin()
    generic_root_cause = PlannedAnalysis(analysis_type="root_cause", target_columns=["some_generic_col"], rationale="generic Stage-2 KPI match")
    unrelated = PlannedAnalysis(analysis_type="trend", target_columns=["y"], rationale="r")
    intent = QuestionIntent(candidate_analysis_types={"root_cause"}, resolved_kpi_name="underwriting_result")
    result = plugin.enhance_plan([generic_root_cause, unrelated], dataset_context=None, question_intent=intent)

    root_cause_entries = [p for p in result if p.analysis_type == "root_cause"]
    assert len(root_cause_entries) == 1
    assert root_cause_entries[0].is_kpi_grounded is True
    assert root_cause_entries[0].target_columns[0] == plugin.get_kpi_definitions()["underwriting_result"]["actual_column"]
    assert root_cause_entries[0].target_columns[1:] == plugin.get_driver_columns()
    assert unrelated in result  # unrelated analyses untouched


def test_enhance_plan_injects_forecast_and_trend_using_kpi_actual_column_and_temporal_column():
    plugin = InsurancePlugin()
    ctx = DatasetContext(
        row_count=10, column_count=1,
        columns=[ColumnContext(name="reporting_date", is_temporal=True)],
        context_source="local_fallback",
    )
    intent = QuestionIntent(candidate_analysis_types={"forecast", "trend"}, resolved_kpi_name="underwriting_result")
    result = plugin.enhance_plan([], dataset_context=ctx, question_intent=intent)
    by_type = {p.analysis_type: p for p in result}
    assert set(by_type) == {"forecast", "trend"}
    actual_col = plugin.get_kpi_definitions()["underwriting_result"]["actual_column"]
    assert by_type["forecast"].target_columns == ["reporting_date", actual_col]
    assert by_type["trend"].target_columns == ["reporting_date", actual_col]


def test_enhance_plan_overrides_anomaly_and_segmentation_feature_columns_unconditionally():
    """No resolved KPI needed — handle_anomaly/handle_segment never
    depended on the current KPI in the old system either."""
    from app.services.ml.anomaly_detector import RATIO_FEATURE_COLUMNS
    from app.services.ml.classifier import SEGMENT_FEATURE_COLS

    plugin = InsurancePlugin()
    plan = [
        PlannedAnalysis(analysis_type="anomaly_detection", target_columns=["some_generic_col"], rationale="generic"),
        PlannedAnalysis(analysis_type="segmentation", target_columns=["some_generic_col"], rationale="generic"),
        PlannedAnalysis(analysis_type="trend", target_columns=["y"], rationale="untouched"),
    ]
    result = plugin.enhance_plan(plan, dataset_context=None, question_intent=None)
    by_type = {p.analysis_type: p for p in result}
    assert by_type["anomaly_detection"].target_columns == list(RATIO_FEATURE_COLUMNS)
    assert by_type["segmentation"].target_columns == list(SEGMENT_FEATURE_COLS)
    assert by_type["trend"].target_columns == ["y"]  # unrelated, untouched


# ── Phase 4: GenericDomainPlugin (the true "no plugin matched" fallback) ────

def test_generic_plugin_has_empty_kpi_catalog():
    plugin = GenericDomainPlugin()
    assert plugin.get_kpi_definitions() == {}
    assert plugin.get_default_kpi_name() is None
    assert plugin.get_driver_columns() is None
    assert plugin.get_intent_vocabulary() == {}
    assert plugin.get_preferred_deterministic_strategy("segmentation") is None


def test_generic_plugin_never_applies_via_registry_matching():
    plugin = GenericDomainPlugin()
    assert plugin.applies_to("Insurance") is False
    assert plugin.applies_to("HR") is False
    assert plugin.applies_to("") is False


def test_generic_plugin_rule_engine_paths_point_at_nonexistent_files_and_load_empty():
    plugin = GenericDomainPlugin()
    engine = RuleEngine(**plugin.get_rule_engine_paths())
    assert engine.list_kpis() == []
    assert engine.get_predefined_rules() == []
    assert engine.get_kpi("underwriting_result") is None


def test_generic_plugin_view_name_is_not_insurance():
    assert GenericDomainPlugin().get_view_name() == "dataset"


def test_generic_plugin_enhance_plan_is_a_pure_noop():
    plugin = GenericDomainPlugin()
    plan = [PlannedAnalysis(analysis_type="trend", target_columns=["x"], rationale="r")]
    intent = QuestionIntent(candidate_analysis_types={"kpi_summary"}, resolved_kpi_name="anything")
    assert plugin.enhance_plan(plan, dataset_context=None, question_intent=intent) == plan
