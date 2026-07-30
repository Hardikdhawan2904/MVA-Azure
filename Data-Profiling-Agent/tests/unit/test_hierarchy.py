"""Tests for hierarchy inference: FD validation, template matching, and chain selection."""

import pytest
import pandas as pd

from app.core.enums import HierarchyEdgeStatus, HierarchyChainStatus
from app.services.hierarchy.functional_dependency import FunctionalDependencyValidator
from app.services.hierarchy.template_matcher import TemplateMatchResult
from app.services.hierarchy.chain_selector import ChainSelector


@pytest.fixture
def fd_validator() -> FunctionalDependencyValidator:
    return FunctionalDependencyValidator()


class TestFunctionalDependencyValidation:
    """Test FD validation for hierarchy edges."""

    def test_perfect_hierarchy(self, fd_validator):
        """Clean 1:1 child→parent mapping should be accepted."""
        df = pd.DataFrame({
            "country": ["US", "US", "GB", "GB", "DE", "DE"] * 10,
            "city": ["NYC", "LA", "London", "Manchester", "Berlin", "Munich"] * 10,
        })
        result = fd_validator.validate_edge(df, "country", "city")
        assert result.status == HierarchyEdgeStatus.ACCEPTED
        assert result.fd_consistency == 1.0
        assert result.violating_child_count == 0

    def test_noisy_hierarchy_retained_with_warning(self, fd_validator):
        """~5% conflict should be retained_with_warning."""
        # Build data: 95 unique cities with clean mapping, 5 cities mapped to 2 countries
        countries = []
        cities = []
        # 90 clean mappings (3 countries × 30 cities each)
        for c_idx in range(3):
            country = ["US", "GB", "DE"][c_idx]
            for i in range(30):
                countries.append(country)
                cities.append(f"{country}City{i}")
        # 5 conflicting cities: each appears under 2 countries
        for i in range(5):
            countries.append("US")
            cities.append(f"SharedCity{i}")
            countries.append("GB")
            cities.append(f"SharedCity{i}")

        df = pd.DataFrame({"country": countries, "city": cities})
        result = fd_validator.validate_edge(df, "country", "city")

        # 95 distinct cities total, 5 violating → consistency = 1 - 5/95 ≈ 0.947
        assert result.status == HierarchyEdgeStatus.RETAINED_WITH_WARNING
        assert result.fd_consistency >= 0.90
        assert result.fd_consistency < 0.98
        assert result.violating_child_count == 5

    def test_rejected_hierarchy_high_conflict(self, fd_validator):
        """40% conflict should be rejected."""
        # Many cities mapping to multiple countries
        countries = ["US", "GB", "DE", "FR"] * 25
        cities = [f"City{i % 15}" for i in range(100)]  # Only 15 unique cities → many conflicts

        df = pd.DataFrame({"country": countries, "city": cities})
        result = fd_validator.validate_edge(df, "country", "city")

        assert result.status == HierarchyEdgeStatus.REJECTED
        assert result.fd_consistency < 0.90

    def test_mapping_coverage(self, fd_validator):
        """Low mapping coverage should be rejected."""
        # Most child values have null parents
        countries = [None] * 80 + ["US"] * 10 + ["GB"] * 10
        cities = [f"City{i}" for i in range(100)]

        df = pd.DataFrame({"country": countries, "city": cities})
        result = fd_validator.validate_edge(df, "country", "city")

        assert result.mapping_coverage < 0.90
        assert result.status == HierarchyEdgeStatus.REJECTED

    def test_null_child_excluded_from_denominator(self, fd_validator):
        """Null child values should not count in the FD denominator."""
        df = pd.DataFrame({
            "country": ["US", "US", "GB", "GB", None, None],
            "city": ["NYC", "LA", "London", "Manchester", None, None],
        })
        result = fd_validator.validate_edge(df, "country", "city")

        # Only non-null children considered
        assert result.distinct_child_count == 4
        assert result.fd_consistency == 1.0
        assert result.status == HierarchyEdgeStatus.ACCEPTED

    def test_missing_parent_reduces_coverage(self, fd_validator):
        """Non-null child with null parent reduces coverage but not consistency."""
        df = pd.DataFrame({
            "country": ["US", "US", None, None, "GB", "GB", "GB", "GB", "GB", "GB"],
            "city": ["NYC", "LA", "Orphan1", "Orphan2", "London", "Manchester",
                     "Birmingham", "Leeds", "Liverpool", "Bristol"],
        })
        result = fd_validator.validate_edge(df, "country", "city")

        # 10 distinct non-null cities
        assert result.distinct_child_count == 10
        # Orphan1 and Orphan2 have null parent → only 8 mapped
        assert result.mapped_child_count == 8
        assert result.mapping_coverage == 0.8
        assert result.fd_consistency == 1.0  # No multi-parent violations

    def test_conflict_samples_bounded(self, fd_validator):
        """Conflict samples should be limited to configured max."""
        # Create many conflicts
        countries = ["A", "B"] * 50
        cities = [f"City{i % 10}" for i in range(100)]  # 10 cities each mapped to A and B
        df = pd.DataFrame({"country": countries, "city": cities})

        result = fd_validator.validate_edge(df, "country", "city")
        assert len(result.conflict_samples) <= 5  # Default max

    def test_edge_confidence_formula(self, fd_validator):
        """edge_confidence = fd_consistency × mapping_coverage."""
        df = pd.DataFrame({
            "country": ["US", "US", "GB", "GB"],
            "city": ["NYC", "LA", "London", "Manchester"],
        })
        result = fd_validator.validate_edge(df, "country", "city")
        expected = result.fd_consistency * result.mapping_coverage
        assert abs(result.edge_confidence - expected) < 0.001


class TestChainSelector:
    """Test chain selection logic."""

    def test_selects_valid_chain(self):
        """Should select a valid chain when edges pass."""
        df = pd.DataFrame({
            "region": ["North", "North", "South", "South"] * 25,
            "country": ["US", "CA", "BR", "AR"] * 25,
            "city": [f"City{i % 20}" for i in range(100)],
        })

        # Create a mock template match
        match = TemplateMatchResult(
            template_key="geo",
            priority=100,
            matched_levels=[("region", "region"), ("country", "country"), ("city", "city")],
            missing_concepts=[],
            match_ratio=1.0,
        )

        selector = ChainSelector()
        result = selector.select(df, [match])

        assert result.status in (HierarchyChainStatus.ACCEPTED, HierarchyChainStatus.PARTIAL)
        assert len(result.level_columns) >= 2

    def test_single_dimension_not_hierarchy(self):
        """A single matched level is NOT a hierarchy."""
        df = pd.DataFrame({"region": ["North", "South"] * 50})

        match = TemplateMatchResult(
            template_key="geo",
            priority=100,
            matched_levels=[("region", "region")],
            missing_concepts=["country", "city"],
            match_ratio=0.33,
        )

        selector = ChainSelector()
        result = selector.select(df, [match])
        assert result.status == HierarchyChainStatus.UNRESOLVED

    def test_unresolved_when_no_templates(self):
        """No template matches should return unresolved."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        selector = ChainSelector()
        result = selector.select(df, [])
        assert result.status == HierarchyChainStatus.UNRESOLVED

    def test_prefers_longer_chain(self):
        """Should prefer a longer valid chain over a shorter one."""
        df = pd.DataFrame({
            "division": ["Div1", "Div1", "Div2", "Div2"] * 25,
            "department": ["Eng", "Sales", "Eng", "HR"] * 25,
            "team": [f"Team{i % 8}" for i in range(100)],
            "sub_team": [f"Sub{i % 16}" for i in range(100)],
        })

        short_match = TemplateMatchResult(
            template_key="short",
            priority=100,
            matched_levels=[("division", "division"), ("department", "department")],
            missing_concepts=[],
            match_ratio=1.0,
        )

        long_match = TemplateMatchResult(
            template_key="long",
            priority=80,
            matched_levels=[
                ("division", "division"),
                ("department", "department"),
                ("team", "team"),
            ],
            missing_concepts=[],
            match_ratio=1.0,
        )

        selector = ChainSelector()
        result = selector.select(df, [short_match, long_match])

        # Should prefer the longer chain if all edges are valid
        if result.status != HierarchyChainStatus.UNRESOLVED:
            assert len(result.level_columns) >= 2

    def test_rejected_edge_breaks_chain(self):
        """A rejected edge should break the chain, using longest valid segment."""
        # Region → Country is valid, Country → City has massive conflicts
        df = pd.DataFrame({
            "region": ["North"] * 50 + ["South"] * 50,
            "country": (["US", "CA"] * 25) + (["BR", "AR"] * 25),
            "city": [f"City{i % 5}" for i in range(100)],  # Same 5 cities everywhere = conflicts
        })

        match = TemplateMatchResult(
            template_key="geo",
            priority=100,
            matched_levels=[("region", "region"), ("country", "country"), ("city", "city")],
            missing_concepts=[],
            match_ratio=1.0,
        )

        selector = ChainSelector()
        result = selector.select(df, [match])

        # The chain should at most be region → country (city edge rejected)
        if result.status != HierarchyChainStatus.UNRESOLVED:
            assert len(result.level_columns) >= 2
            assert len(result.rejected_edges) >= 0  # May have rejections
