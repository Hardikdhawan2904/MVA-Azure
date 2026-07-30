"""Tests for semantic candidate generation."""

import pytest
import pandas as pd

from app.core.config import Settings
from app.core.enums import ColumnRole
from app.services.profiling.column_profiler import ColumnProfiler
from app.services.profiling.type_refiner import TypeRefiner
from app.services.profiling.semantic_candidate_generator import SemanticCandidateGenerator


@pytest.fixture
def profiler() -> ColumnProfiler:
    settings = Settings(DATABASE_URL="postgresql://x:x@localhost/test", MAX_SAMPLE_VALUES=10)
    return ColumnProfiler(settings)


@pytest.fixture
def refiner() -> TypeRefiner:
    return TypeRefiner()


@pytest.fixture
def generator() -> SemanticCandidateGenerator:
    return SemanticCandidateGenerator()


def _profile_and_refine(profiler, refiner, values, name="col"):
    series = pd.Series(values)
    profile = profiler.profile_column(series, name, name)
    refined_type = refiner.refine(profile)
    return profile, refined_type


class TestSemanticCandidateGenerator:
    """Test deterministic semantic candidate generation."""

    def test_identifier_role(self, profiler, refiner, generator):
        """Identifier columns should get role=identifier."""
        values = [f"ID-{i:06d}" for i in range(100)]
        profile, rt = _profile_and_refine(profiler, refiner, values, "record_id")
        result = generator.generate(profile, rt, is_identifier=True)
        assert result.candidate_column_role == ColumnRole.IDENTIFIER
        assert result.candidate_semantic_type == "identifier"
        assert result.candidate_confidence >= 0.85

    def test_transaction_identifier_role(self, profiler, refiner, generator):
        """Identifier columns whose name signals a specific entity get a refined
        semantic_type, so chart templates requiring e.g. transaction_identifier
        can resolve against them."""
        values = [f"TXN-{i:06d}" for i in range(100)]
        profile, rt = _profile_and_refine(profiler, refiner, values, "transaction_id")
        result = generator.generate(profile, rt, is_identifier=True)
        assert result.candidate_column_role == ColumnRole.IDENTIFIER
        assert result.candidate_semantic_type == "transaction_identifier"

    def test_monetary_amount_metric(self, profiler, refiner, generator):
        """Numeric column with 'amount' in name should be metric."""
        values = [str(i * 100.50) for i in range(1, 51)]
        profile, rt = _profile_and_refine(profiler, refiner, values, "transaction_amount")
        result = generator.generate(profile, rt, is_identifier=False)
        assert result.candidate_column_role == ColumnRole.METRIC
        assert result.candidate_semantic_type == "monetary_amount"

    def test_temporal_dimension(self, profiler, refiner, generator):
        """Date columns should get temporal_dimension role."""
        values = ["2024-01-01", "2024-02-15", "2024-03-20", "2024-04-10", "2024-05-05"]
        profile, rt = _profile_and_refine(profiler, refiner, values, "created_date")
        result = generator.generate(profile, rt, is_identifier=False)
        assert result.candidate_column_role == ColumnRole.TEMPORAL_DIMENSION
        assert "creation_date" in result.candidate_semantic_type

    def test_categorical_dimension(self, profiler, refiner, generator):
        """Low-cardinality string column should be dimension."""
        values = ["approved", "declined", "pending"] * 50
        profile, rt = _profile_and_refine(profiler, refiner, values, "payment_status")
        result = generator.generate(profile, rt, is_identifier=False)
        assert result.candidate_column_role == ColumnRole.DIMENSION
        assert result.candidate_semantic_type == "status"

    def test_flag_role(self, profiler, refiner, generator):
        """Boolean columns should get flag role."""
        values = ["true", "false"] * 50
        profile, rt = _profile_and_refine(profiler, refiner, values, "is_active")
        result = generator.generate(profile, rt, is_identifier=False)
        assert result.candidate_column_role == ColumnRole.FLAG

    def test_currency_code_dimension(self, profiler, refiner, generator):
        """Currency code columns should be dimension."""
        values = ["USD", "EUR", "GBP", "JPY", "INR"] * 20
        profile, rt = _profile_and_refine(profiler, refiner, values, "currency")
        result = generator.generate(profile, rt, is_identifier=False)
        assert result.candidate_column_role == ColumnRole.DIMENSION
        assert result.candidate_semantic_type == "currency_code"

    def test_description_text_field(self, profiler, refiner, generator):
        """Text columns with 'description' in name should be text_field."""
        base_descriptions = [
            "Payment processed successfully",
            "Transaction declined due to insufficient funds",
            "Refund issued to customer account",
            "Chargeback initiated by cardholder",
            "Settlement completed for merchant batch",
        ]
        values = base_descriptions * 20
        profile, rt = _profile_and_refine(profiler, refiner, values, "transaction_description")
        result = generator.generate(profile, rt, is_identifier=False)
        assert result.candidate_column_role == ColumnRole.TEXT_FIELD
        assert result.candidate_semantic_type == "description"

    def test_geographic_dimension(self, profiler, refiner, generator):
        """Geographic categorical columns should be dimension."""
        values = ["North", "South", "East", "West"] * 40
        profile, rt = _profile_and_refine(profiler, refiner, values, "region")
        result = generator.generate(profile, rt, is_identifier=False)
        assert result.candidate_column_role == ColumnRole.DIMENSION
        assert result.candidate_semantic_type == "region"

    def test_percentage_metric(self, profiler, refiner, generator):
        """Percentage columns should be metric."""
        values = [str(i) for i in range(0, 101, 5)]
        profile, rt = _profile_and_refine(profiler, refiner, values, "success_rate")
        result = generator.generate(profile, rt, is_identifier=False)
        assert result.candidate_column_role == ColumnRole.METRIC
        assert result.candidate_semantic_type == "percentage"

    def test_low_cardinality_numeric_is_dimension(self, profiler, refiner, generator):
        """Numeric with very low cardinality should be dimension not metric."""
        values = ["1", "2", "3", "4", "5"] * 40
        profile, rt = _profile_and_refine(profiler, refiner, values, "priority_level")
        result = generator.generate(profile, rt, is_identifier=False)
        # Should recognize low cardinality numeric as dimension
        assert result.candidate_column_role == ColumnRole.DIMENSION

    def test_generate_all(self, profiler, refiner, generator):
        """generate_all should handle multiple columns."""
        df = pd.DataFrame({
            "id": [f"R{i}" for i in range(100)],
            "amount": [str(i * 10) for i in range(100)],
            "status": ["active", "inactive"] * 50,
        })
        profiles = []
        types = []
        id_flags = []
        for col in df.columns:
            p = profiler.profile_column(df[col], col, col)
            profiles.append(p)
            rt = refiner.refine(p)
            types.append(rt)
            id_flags.append(col == "id")

        results = generator.generate_all(profiles, types, id_flags)
        assert len(results) == 3
        assert results[0].candidate_column_role == ColumnRole.IDENTIFIER
        assert results[2].candidate_column_role in (ColumnRole.DIMENSION, ColumnRole.FLAG)

    def test_confidence_always_between_0_and_1(self, profiler, refiner, generator):
        """All confidence scores should be in [0, 1]."""
        test_values = [
            (["abc"] * 100, "generic"),
            ([str(i) for i in range(100)], "numbers"),
            (["2024-01-01"] * 50, "dates"),
        ]
        for values, name in test_values:
            profile, rt = _profile_and_refine(profiler, refiner, values, name)
            result = generator.generate(profile, rt, is_identifier=False)
            assert 0.0 <= result.candidate_confidence <= 1.0

    def test_evidence_always_provided(self, profiler, refiner, generator):
        """Every candidate should have at least one evidence entry."""
        values = ["hello", "world"] * 50
        profile, rt = _profile_and_refine(profiler, refiner, values, "text_col")
        result = generator.generate(profile, rt, is_identifier=False)
        assert len(result.evidence) >= 1
