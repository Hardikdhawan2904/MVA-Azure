"""Coverage hardening — additional tests from Section 26 spec requirements."""

import pytest
import pandas as pd
from pathlib import Path

from app.core.config import Settings
from app.core.enums import (
    RuleType, RuleSource, RefinedDataType,
    HierarchyEdgeStatus, ReadinessStatus,
)
from app.services.profiling.column_profiler import ColumnProfiler
from app.services.profiling.type_refiner import TypeRefiner
from app.services.profiling.identifier_detector import IdentifierDetector
from app.services.hierarchy.functional_dependency import FunctionalDependencyValidator
from app.services.rules.rule_engine import RuleEngine
from app.services.rules.rule_loader import RuleDefinition
from app.services.quality.overall_score import calculate_overall_score
from app.services.readiness.readiness_engine import _determine_status


@pytest.fixture
def profiler():
    return ColumnProfiler(Settings(DATABASE_URL="postgresql://x:x@localhost/test", MAX_SAMPLE_VALUES=10))


@pytest.fixture
def refiner():
    return TypeRefiner()


@pytest.fixture
def fd_validator():
    return FunctionalDependencyValidator()


# --- Type refinement edge cases ---

class TestTypeRefinementEdgeCases:
    """Additional type refinement tests."""

    def test_phone_number_detection(self, profiler, refiner):
        values = ["+1-555-123-4567", "+44 20 7123 4567", "+49 30 12345678"] * 10
        series = pd.Series(values)
        profile = profiler.profile_column(series, "phone", "phone")
        result = refiner.refine(profile)
        assert result == RefinedDataType.PHONE

    def test_mixed_type_stays_text(self, profiler, refiner):
        """Column with mixed types that can't be resolved should be text or categorical."""
        values = ["hello", "123", "2024-01-01", "true", "USD"] * 10
        series = pd.Series(values)
        profile = profiler.profile_column(series, "mixed", "mixed")
        result = refiner.refine(profile)
        assert result in (RefinedDataType.TEXT, RefinedDataType.CATEGORICAL)

    def test_country_code_3_letter(self, profiler, refiner):
        values = ["USA", "GBR", "DEU", "FRA", "JPN", "IND", "AUS"] * 8
        series = pd.Series(values)
        profile = profiler.profile_column(series, "country", "country")
        result = refiner.refine(profile)
        assert result == RefinedDataType.COUNTRY_CODE


# --- Identifier detection ---

class TestIdentifierEdgeCases:
    """Additional identifier detection tests."""

    def test_low_cardinality_code_retained_as_dimension(self, profiler, refiner):
        """department_code with 5 values over 200 rows = dimension, not identifier."""
        values = ["DEPT-A", "DEPT-B", "DEPT-C", "DEPT-D", "DEPT-E"] * 40
        series = pd.Series(values)
        profile = profiler.profile_column(series, "department_code", "department_code")
        refined_type = refiner.refine(profile)
        detector = IdentifierDetector()
        result = detector.detect([profile], [refined_type])
        assert "department_code" not in result.grain_columns
        assert refined_type == RefinedDataType.CATEGORICAL


# --- Hierarchy with fixtures ---

class TestHierarchyWithFixtures:
    """Test hierarchy with fixture files."""

    def test_dirty_hierarchy_5pct(self, fd_validator):
        """Dirty hierarchy fixture — US appears in multiple regions."""
        fixture = Path(__file__).parent.parent / "fixtures" / "dirty_hierarchy_5pct.csv"
        df = pd.read_csv(fixture)
        # Country → City is cleaner (cities don't repeat across countries)
        result = fd_validator.validate_edge(df, "country", "city")
        # Most cities map to exactly one country
        assert result.fd_consistency >= 0.90

    def test_invalid_hierarchy_40pct(self, fd_validator):
        """40% conflict hierarchy should be rejected."""
        fixture = Path(__file__).parent.parent / "fixtures" / "invalid_hierarchy_40pct.csv"
        df = pd.read_csv(fixture)
        result = fd_validator.validate_edge(df, "country", "city")
        # Many cities appear under multiple countries → rejected
        assert result.status == HierarchyEdgeStatus.REJECTED
        assert result.fd_consistency < 0.90


# --- Rule engine edge cases ---

class TestRuleEngineEdgeCases:
    """Additional rule engine tests."""

    def test_date_range_not_implemented_gracefully(self):
        """date_range type should handle gracefully (returns error or skips)."""
        engine = RuleEngine()
        df = pd.DataFrame({"dt": ["2024-01-01", "2024-06-15", "2024-12-31"]})
        rule = RuleDefinition(
            rule_key="date_check", rule_type=RuleType.DATE_RANGE,
            source=RuleSource.REQUEST, domain="Payments", secondary_domain=None,
            parameters={"target_column": "dt", "min_date": "2024-01-01", "max_date": "2024-12-31"},
        )
        results = engine.evaluate(df, [rule])
        # Should either evaluate or return unsupported gracefully
        assert len(results) == 1

    def test_cross_field_equality_not_implemented_gracefully(self):
        """cross_field_equality should handle gracefully."""
        engine = RuleEngine()
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["x", "y", "w"]})
        rule = RuleDefinition(
            rule_key="eq_check", rule_type=RuleType.CROSS_FIELD_EQUALITY,
            source=RuleSource.REQUEST, domain="Payments", secondary_domain=None,
            parameters={"left_column": "a", "right_column": "b"},
        )
        results = engine.evaluate(df, [rule])
        assert len(results) == 1


# --- Quality overall score ---

class TestOverallScoreEdgeCases:
    """Additional overall score tests."""

    def test_single_assessed_dimension(self):
        """One assessed dimension should still produce a score."""
        dims = [
            {"dimension": "completeness", "score": 0.95, "status": "assessed"},
            {"dimension": "timeliness", "score": None, "status": "not_assessable"},
            {"dimension": "accuracy", "score": None, "status": "not_assessable"},
        ]
        weights = {"completeness": 0.2, "timeliness": 0.1, "accuracy": 0.1}
        result = calculate_overall_score(dims, weights)
        assert result["overall_score"] == 0.95
        assert len(result["assessed_dimensions"]) == 1

    def test_display_score_is_percentage(self):
        dims = [{"dimension": "completeness", "score": 0.82, "status": "assessed"}]
        weights = {"completeness": 0.3}
        result = calculate_overall_score(dims, weights)
        assert result["display_score"] == 82.0


# --- Readiness thresholds ---

class TestReadinessThresholdBoundaries:
    """Test exact boundary values for readiness thresholds."""

    def test_exactly_80_is_ready(self):
        assert _determine_status(80.0) == ReadinessStatus.READY

    def test_79_99_is_partially_ready(self):
        assert _determine_status(79.99) == ReadinessStatus.PARTIALLY_READY

    def test_exactly_60_is_partially_ready(self):
        assert _determine_status(60.0) == ReadinessStatus.PARTIALLY_READY

    def test_59_99_is_not_ready(self):
        assert _determine_status(59.99) == ReadinessStatus.NOT_READY


# --- Fixture-based integration ---

class TestFixtureDatasets:
    """Test that fixture datasets can be profiled without errors."""

    @pytest.fixture
    def fixtures_dir(self):
        return Path(__file__).parent.parent / "fixtures"

    def test_payments_auth_fixture(self, fixtures_dir, profiler, refiner):
        df = pd.read_csv(fixtures_dir / "payments_authorization.csv", dtype=str)
        assert len(df) == 30
        assert "auth_status" in df.columns
        profiles = profiler.profile_all(df, list(df.columns))
        assert len(profiles) == 10

    def test_settlement_fraud_fixture(self, fixtures_dir, profiler):
        df = pd.read_csv(fixtures_dir / "payments_settlement_fraud.csv", dtype=str)
        assert len(df) == 20
        assert "fraud_flag" in df.columns
        assert "settlement_date" in df.columns

    def test_customer_crm_fixture(self, fixtures_dir, profiler):
        df = pd.read_csv(fixtures_dir / "customer_crm_loyalty.csv", dtype=str)
        assert len(df) == 15
        assert "loyalty_tier" in df.columns

    def test_hr_employee_fixture(self, fixtures_dir, profiler):
        df = pd.read_csv(fixtures_dir / "hr_employee_payroll.csv", dtype=str)
        assert len(df) == 15
        assert "salary" in df.columns

    def test_finance_revenue_fixture(self, fixtures_dir, profiler):
        df = pd.read_csv(fixtures_dir / "finance_revenue_forecasting.csv", dtype=str)
        assert len(df) == 15
        assert "actual_amount" in df.columns
        assert "forecast_amount" in df.columns

    def test_missing_mandatory_fixture(self, fixtures_dir, profiler):
        df = pd.read_csv(fixtures_dir / "missing_mandatory_values.csv", dtype=str)
        assert len(df) == 15
        # Verify nulls exist
        assert df["amount"].isna().sum() >= 4

    def test_expected_unique_dupes_fixture(self, fixtures_dir, profiler):
        df = pd.read_csv(fixtures_dir / "expected_unique_duplicates.csv", dtype=str)
        assert len(df) == 15
        # Verify duplicates in customer_id
        assert df["customer_id"].duplicated().sum() >= 3
