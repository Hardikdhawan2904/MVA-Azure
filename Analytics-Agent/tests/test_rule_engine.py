"""tests/test_rule_engine.py — Unit tests for the Rule Engine Tool"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.tools.rule_engine import RuleEngine


def test_rule_engine_loads():
    engine = RuleEngine()
    assert len(engine.list_kpis()) > 0, "Rule Engine should load KPIs"


def test_known_kpi_lookup():
    engine = RuleEngine()
    kpi = engine.get_kpi("loss_ratio")
    assert kpi is not None, "loss_ratio should be in the Rule Engine"
    assert kpi["label"] == "Loss Ratio"
    assert kpi["actual_column"] == "loss_ratio_actual"


def test_abbreviation_lookup():
    engine = RuleEngine()
    kpi = engine.get_kpi("GWP")
    assert kpi is not None, "GWP abbreviation should resolve"
    assert kpi["key"] == "gross_written_premium"


def test_missing_kpi_returns_none():
    engine = RuleEngine()
    result = engine.get_kpi("nonexistent_metric_xyz")
    assert result is None


def test_threshold_lookup():
    engine = RuleEngine()
    thresholds = engine.get_kpi_threshold("combined_ratio")
    assert thresholds.get("breakeven") == 100.0 or "warning" in thresholds


def test_variance_driver_descriptions():
    engine = RuleEngine()
    desc = engine.get_variance_driver_description("claim_frequency_variance")
    assert len(desc) > 0


def test_hierarchy_lookup():
    engine = RuleEngine()
    geo = engine.get_hierarchy("geographic")
    assert "level_1" in geo
    assert geo["level_1"] == "region"


if __name__ == "__main__":
    test_rule_engine_loads()
    test_known_kpi_lookup()
    test_abbreviation_lookup()
    test_missing_kpi_returns_none()
    test_variance_driver_descriptions()
    test_hierarchy_lookup()
    print("✅ All Rule Engine tests passed")
