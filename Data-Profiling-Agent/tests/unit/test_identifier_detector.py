"""Tests for identifier and grain detection."""

import pytest
import pandas as pd

from app.core.config import Settings
from app.services.profiling.column_profiler import ColumnProfiler
from app.services.profiling.type_refiner import TypeRefiner
from app.services.profiling.identifier_detector import IdentifierDetector


@pytest.fixture
def profiler() -> ColumnProfiler:
    settings = Settings(DATABASE_URL="postgresql://x:x@localhost/test", MAX_SAMPLE_VALUES=10)
    return ColumnProfiler(settings)


@pytest.fixture
def refiner() -> TypeRefiner:
    return TypeRefiner()


@pytest.fixture
def detector() -> IdentifierDetector:
    return IdentifierDetector()


class TestIdentifierDetector:
    """Test identifier detection based on behavioral analysis."""

    def test_unique_column_detected_as_identifier(
        self, profiler: ColumnProfiler, refiner: TypeRefiner, detector: IdentifierDetector
    ):
        """A column that is unique per row should be detected as identifier."""
        values = [f"EMP-{i:05d}" for i in range(100)]
        series = pd.Series(values)
        profile = profiler.profile_column(series, "employee_id", "employee_id")
        refined_type = refiner.refine(profile)

        result = detector.detect([profile], [refined_type])
        assert len(result.grain_columns) >= 1
        assert "employee_id" in result.grain_columns

    def test_low_cardinality_not_identifier(
        self, profiler: ColumnProfiler, refiner: TypeRefiner, detector: IdentifierDetector
    ):
        """A repeated categorical column should NOT be identifier."""
        values = ["DEPT-A", "DEPT-B", "DEPT-C"] * 50
        series = pd.Series(values)
        profile = profiler.profile_column(series, "department_code", "department_code")
        refined_type = refiner.refine(profile)

        result = detector.detect([profile], [refined_type])
        assert "department_code" not in result.grain_columns

    def test_cardinality_based_not_name_based(
        self, profiler: ColumnProfiler, refiner: TypeRefiner, detector: IdentifierDetector
    ):
        """Column named 'code' but with low cardinality should NOT be identifier."""
        values = ["US", "GB", "DE", "FR", "JP"] * 40
        series = pd.Series(values)
        profile = profiler.profile_column(series, "country_code", "country_code")
        refined_type = refiner.refine(profile)

        result = detector.detect([profile], [refined_type])
        candidates = result.identifier_candidates
        country_candidate = next(c for c in candidates if c.column_name == "country_code")
        assert country_candidate.is_identifier is False

    def test_multiple_columns_grain_detection(
        self, profiler: ColumnProfiler, refiner: TypeRefiner, detector: IdentifierDetector
    ):
        """Detect grain across multiple columns."""
        df = pd.DataFrame({
            "txn_id": [f"T{i:06d}" for i in range(100)],
            "amount": [str((i % 20) * 10.5) for i in range(100)],
            "status": ["approved", "declined", "pending"] * 33 + ["approved"],
            "merchant": [f"M{i % 10}" for i in range(100)],
        })

        profiles = []
        refined_types = []
        for col in df.columns:
            p = profiler.profile_column(df[col], col, col)
            profiles.append(p)
            refined_types.append(refiner.refine(p))

        result = detector.detect(profiles, refined_types)

        # txn_id should be the grain
        assert "txn_id" in result.grain_columns
        # Others should not
        assert "amount" not in result.grain_columns
        assert "status" not in result.grain_columns
        assert "merchant" not in result.grain_columns

    def test_identifier_score_provided(
        self, profiler: ColumnProfiler, refiner: TypeRefiner, detector: IdentifierDetector
    ):
        """Each candidate should have a score and evidence."""
        values = [f"ID-{i}" for i in range(50)]
        series = pd.Series(values)
        profile = profiler.profile_column(series, "record_id", "record_id")
        refined_type = refiner.refine(profile)

        result = detector.detect([profile], [refined_type])
        candidate = result.identifier_candidates[0]
        assert 0 <= candidate.identifier_score <= 1.0
        assert len(candidate.evidence) > 0

    def test_inferred_grain_string(
        self, profiler: ColumnProfiler, refiner: TypeRefiner, detector: IdentifierDetector
    ):
        """inferred_grain should return a concatenated string of grain columns."""
        values = [f"KEY-{i}" for i in range(100)]
        series = pd.Series(values)
        profile = profiler.profile_column(series, "primary_key", "primary_key")
        refined_type = refiner.refine(profile)

        result = detector.detect([profile], [refined_type])
        assert result.inferred_grain is not None
        assert "primary_key" in result.inferred_grain

    def test_unique_long_commentary_not_identifier(
        self, profiler: ColumnProfiler, refiner: TypeRefiner, detector: IdentifierDetector
    ):
        """A free-text commentary column where every row happens to be a unique
        sentence should NOT become a grain key — high cardinality alone isn't
        enough when the values are long narrative text, not short codes."""
        values = [
            f"Total Operating Income of ${17000 + i}K was ${1000 + i}K below budget, "
            f"primarily driven by volume and rate variance in the reporting period."
            for i in range(60)
        ]
        series = pd.Series(values)
        profile = profiler.profile_column(series, "commentary_headline", "commentary_headline")
        refined_type = refiner.refine(profile)

        result = detector.detect([profile], [refined_type])
        assert "commentary_headline" not in result.grain_columns

    def test_high_null_reduces_identifier_confidence(
        self, profiler: ColumnProfiler, refiner: TypeRefiner, detector: IdentifierDetector
    ):
        """Columns with high null ratio should have reduced identifier score."""
        values = [f"ID-{i}" for i in range(50)] + [None] * 50  # type: ignore
        series = pd.Series(values)
        profile = profiler.profile_column(series, "maybe_id", "maybe_id")
        refined_type = refiner.refine(profile)

        result = detector.detect([profile], [refined_type])
        candidate = result.identifier_candidates[0]
        # High null means it may still have high cardinality among non-nulls
        # but the null penalty should reduce confidence
        assert any("null" in str(e.get("signal", "")) for e in candidate.evidence)
