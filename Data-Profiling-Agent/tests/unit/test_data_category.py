"""Tests for column data-category classification."""

import pytest
import pandas as pd

from app.core.config import Settings
from app.core.enums import DataCategory
from app.services.classification.data_category_classifier import DataCategoryClassifier
from app.services.profiling.column_profiler import ColumnProfiler
from app.services.profiling.type_refiner import TypeRefiner
from app.services.profiling.semantic_candidate_generator import SemanticCandidateGenerator


@pytest.fixture
def classifier() -> DataCategoryClassifier:
    return DataCategoryClassifier()


@pytest.fixture
def profiler() -> ColumnProfiler:
    settings = Settings(DATABASE_URL="postgresql://x:x@localhost/test", MAX_SAMPLE_VALUES=10)
    return ColumnProfiler(settings)


@pytest.fixture
def refiner() -> TypeRefiner:
    return TypeRefiner()


@pytest.fixture
def sem_gen() -> SemanticCandidateGenerator:
    return SemanticCandidateGenerator()


def _make_candidate(profiler, refiner, sem_gen, values, name):
    """Helper to create a semantic candidate from values."""
    series = pd.Series(values)
    profile = profiler.profile_column(series, name, name)
    rt = refiner.refine(profile)
    candidate = sem_gen.generate(profile, rt, is_identifier=False)
    return profile, candidate


class TestDataCategoryClassifier:
    """Test column data-category classification."""

    def test_transaction_amount_classified(self, classifier, profiler, refiner, sem_gen):
        """Transaction amount should be Transaction Data."""
        profile, candidate = _make_candidate(
            profiler, refiner, sem_gen,
            [str(i * 100) for i in range(50)],
            "transaction_amount"
        )
        result = classifier.classify(profile, candidate, "Payments")
        assert result.primary_category == DataCategory.TRANSACTION

    def test_financial_column(self, classifier, profiler, refiner, sem_gen):
        """Revenue column should be Financial Data."""
        profile, candidate = _make_candidate(
            profiler, refiner, sem_gen,
            [str(i * 1000) for i in range(50)],
            "revenue_amount"
        )
        result = classifier.classify(profile, candidate, "Finance")
        assert result.primary_category == DataCategory.FINANCIAL

    def test_geographic_column(self, classifier, profiler, refiner, sem_gen):
        """Country column should be Geographic Data."""
        profile, candidate = _make_candidate(
            profiler, refiner, sem_gen,
            ["US", "GB", "DE", "FR", "JP"] * 20,
            "country_code"
        )
        result = classifier.classify(profile, candidate, "Payments")
        assert result.primary_category == DataCategory.GEOGRAPHIC

    def test_temporal_column(self, classifier, profiler, refiner, sem_gen):
        """Date column should be Time Series Data."""
        profile, candidate = _make_candidate(
            profiler, refiner, sem_gen,
            ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"] * 10,
            "created_date"
        )
        result = classifier.classify(profile, candidate, "Payments")
        assert result.primary_category == DataCategory.TIME_SERIES

    def test_risk_column(self, classifier, profiler, refiner, sem_gen):
        """Risk/fraud columns should be Risk Data."""
        profile, candidate = _make_candidate(
            profiler, refiner, sem_gen,
            [str(i % 100) for i in range(50)],
            "fraud_risk_score"
        )
        result = classifier.classify(profile, candidate, "Payments")
        assert result.primary_category == DataCategory.RISK

    def test_secondary_categories(self, classifier, profiler, refiner, sem_gen):
        """Column may have secondary categories."""
        # Transaction amount could be both Transaction and Financial
        profile, candidate = _make_candidate(
            profiler, refiner, sem_gen,
            [str(i * 50) for i in range(50)],
            "payment_amount"
        )
        result = classifier.classify(profile, candidate, "Payments")
        # Should have at least primary category
        assert result.primary_category is not None
        # Transaction + Financial overlap is expected
        all_cats = [result.primary_category] + result.secondary_categories
        assert len(all_cats) >= 1

    def test_exactly_one_primary_category(self, classifier, profiler, refiner, sem_gen):
        """Each column must have exactly one primary category."""
        test_cases = [
            (["alice@test.com"] * 50, "customer_email"),
            (["approved", "declined"] * 25, "auth_status"),
            ([str(i) for i in range(50)], "record_count"),
        ]
        for values, name in test_cases:
            profile, candidate = _make_candidate(profiler, refiner, sem_gen, values, name)
            result = classifier.classify(profile, candidate, "Payments")
            assert isinstance(result.primary_category, DataCategory)

    def test_confidence_range(self, classifier, profiler, refiner, sem_gen):
        """Confidence must be between 0 and 1."""
        profile, candidate = _make_candidate(
            profiler, refiner, sem_gen,
            ["val"] * 50, "generic_col"
        )
        result = classifier.classify(profile, candidate, "Payments")
        assert 0.0 <= result.confidence <= 1.0

    def test_evidence_present(self, classifier, profiler, refiner, sem_gen):
        """Results should contain evidence."""
        profile, candidate = _make_candidate(
            profiler, refiner, sem_gen,
            ["2024-01-01"] * 50, "transaction_date"
        )
        result = classifier.classify(profile, candidate, "Payments")
        assert len(result.evidence) >= 1

    def test_classify_all(self, classifier, profiler, refiner, sem_gen):
        """classify_all should handle multiple columns."""
        df = pd.DataFrame({
            "amount": [str(i) for i in range(50)],
            "status": ["active", "inactive"] * 25,
            "country": ["US", "GB"] * 25,
        })
        profiles = []
        candidates = []
        for col in df.columns:
            p = profiler.profile_column(df[col], col, col)
            profiles.append(p)
            rt = refiner.refine(p)
            candidates.append(sem_gen.generate(p, rt, is_identifier=False))

        results = classifier.classify_all(profiles, candidates, "Payments")
        assert len(results) == 3
        assert all(isinstance(r.primary_category, DataCategory) for r in results)

    def test_to_dict_format(self, classifier, profiler, refiner, sem_gen):
        """to_dict should produce proper structure."""
        profile, candidate = _make_candidate(
            profiler, refiner, sem_gen,
            [str(i) for i in range(50)], "transaction_amount"
        )
        result = classifier.classify(profile, candidate, "Payments")
        d = result.to_dict()
        assert "column_name" in d
        assert "primary_category" in d
        assert "secondary_categories" in d
        assert "confidence" in d
        assert "evidence" in d
        assert isinstance(d["secondary_categories"], list)
