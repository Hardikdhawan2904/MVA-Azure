"""Tests for column-level profiling."""

import pytest
import pandas as pd

from app.core.config import Settings
from app.services.profiling.column_profiler import ColumnProfiler


@pytest.fixture
def profiler() -> ColumnProfiler:
    settings = Settings(
        DATABASE_URL="postgresql://x:x@localhost/test",
        MAX_SAMPLE_VALUES=5,
    )
    return ColumnProfiler(settings)


class TestColumnProfiler:
    """Test column-level profiling statistics."""

    def test_basic_counts(self, profiler: ColumnProfiler):
        series = pd.Series(["a", "b", None, "a", "c"])
        result = profiler.profile_column(series, "test_col", "test_col")
        assert result.row_count == 5
        assert result.non_null_count == 4
        assert result.null_count == 1
        assert abs(result.null_ratio - 0.2) < 0.001

    def test_distinct_count(self, profiler: ColumnProfiler):
        series = pd.Series(["a", "b", "a", "c", "b"])
        result = profiler.profile_column(series, "col", "col")
        assert result.distinct_count == 3
        assert result.duplicate_count == 2

    def test_cardinality_ratio_unique(self, profiler: ColumnProfiler):
        series = pd.Series(["a", "b", "c", "d", "e"])
        result = profiler.profile_column(series, "col", "col")
        assert result.cardinality_ratio == 1.0

    def test_cardinality_ratio_low(self, profiler: ColumnProfiler):
        series = pd.Series(["x"] * 100)
        result = profiler.profile_column(series, "col", "col")
        assert result.cardinality_ratio == 1 / 100

    def test_numeric_statistics(self, profiler: ColumnProfiler):
        series = pd.Series(["10", "20", "30", "40", "50"])
        result = profiler.profile_column(series, "amount", "amount")
        assert result.numeric_parse_ratio == 1.0
        assert result.mean == 30.0
        assert result.min_value == 10.0
        assert result.max_value == 50.0
        assert result.median == 30.0
        assert result.quantiles is not None
        assert result.quantiles["q50"] == 30.0

    def test_numeric_with_non_numeric(self, profiler: ColumnProfiler):
        series = pd.Series(["10", "20", "abc", "40", "50"])
        result = profiler.profile_column(series, "mixed", "mixed")
        assert result.numeric_parse_ratio == 0.8

    def test_string_lengths(self, profiler: ColumnProfiler):
        series = pd.Series(["a", "abc", "abcde"])
        result = profiler.profile_column(series, "text", "text")
        assert result.min_string_length == 1
        assert result.max_string_length == 5
        assert abs(result.avg_string_length - 3.0) < 0.001

    def test_datetime_parse_ratio(self, profiler: ColumnProfiler):
        series = pd.Series(["2024-01-01", "2024-02-15", "2024-03-20"])
        result = profiler.profile_column(series, "date_col", "date_col")
        assert result.datetime_parse_ratio == 1.0

    def test_datetime_parse_ratio_mixed(self, profiler: ColumnProfiler):
        series = pd.Series(["2024-01-01", "not_a_date", "2024-03-20"])
        result = profiler.profile_column(series, "date_col", "date_col")
        assert 0.5 <= result.datetime_parse_ratio <= 0.75

    def test_boolean_parse_ratio(self, profiler: ColumnProfiler):
        series = pd.Series(["true", "false", "true", "yes", "no"])
        result = profiler.profile_column(series, "flag", "flag")
        assert result.boolean_parse_ratio == 1.0

    def test_boolean_parse_ratio_numeric(self, profiler: ColumnProfiler):
        series = pd.Series(["0", "1", "1", "0", "1"])
        result = profiler.profile_column(series, "binary", "binary")
        assert result.boolean_parse_ratio == 1.0

    def test_dominant_patterns_date(self, profiler: ColumnProfiler):
        series = pd.Series(["2024-01-01", "2024-02-15", "2024-03-20", "2024-04-10"])
        result = profiler.profile_column(series, "dt", "dt")
        assert "YYYY-MM-DD" in result.dominant_patterns

    def test_dominant_patterns_integer(self, profiler: ColumnProfiler):
        series = pd.Series(["100", "200", "300", "400"])
        result = profiler.profile_column(series, "num", "num")
        assert "INTEGER" in result.dominant_patterns or "NUMERIC" in result.dominant_patterns

    def test_sample_values_bounded(self, profiler: ColumnProfiler):
        series = pd.Series([str(i) for i in range(100)])
        result = profiler.profile_column(series, "big", "big")
        assert len(result.sample_values) <= 5

    def test_representative_values(self, profiler: ColumnProfiler):
        series = pd.Series(["cat", "cat", "cat", "dog", "bird"])
        result = profiler.profile_column(series, "animal", "animal")
        # Most frequent should be first
        assert result.representative_values[0] == "cat"

    def test_all_null_column(self, profiler: ColumnProfiler):
        series = pd.Series([None, None, None])
        result = profiler.profile_column(series, "empty", "empty")
        assert result.null_ratio == 1.0
        assert result.distinct_count == 0
        assert result.numeric_parse_ratio == 0.0

    def test_warnings_high_null(self, profiler: ColumnProfiler):
        values = [None] * 8 + ["a", "b"]
        series = pd.Series(values)
        result = profiler.profile_column(series, "sparse", "sparse")
        assert any("null" in w.lower() for w in result.warnings)

    def test_warnings_near_unique(self, profiler: ColumnProfiler):
        series = pd.Series([f"id_{i}" for i in range(100)])
        result = profiler.profile_column(series, "uid", "uid")
        assert any("identifier" in w.lower() for w in result.warnings)

    def test_profile_all(self, profiler: ColumnProfiler):
        df = pd.DataFrame({
            "col_a": ["1", "2", "3"],
            "col_b": ["x", "y", "z"],
        })
        results = profiler.profile_all(df, ["col_a", "col_b"])
        assert len(results) == 2
        assert results[0].physical_name == "col_a"
        assert results[1].physical_name == "col_b"

    def test_to_statistics_dict(self, profiler: ColumnProfiler):
        series = pd.Series(["10", "20", "30"])
        result = profiler.profile_column(series, "x", "x")
        d = result.to_statistics_dict()
        assert "row_count" in d
        assert "null_ratio" in d
        assert "numeric_parse_ratio" in d
        assert "sample_values" in d
