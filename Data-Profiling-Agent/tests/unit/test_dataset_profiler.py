"""Tests for dataset-level profiling."""

import pytest
import pandas as pd

from app.services.profiling.dataset_profiler import DatasetProfiler, normalize_column_name


class TestNormalizeColumnName:
    """Test column name normalization."""

    def test_lowercase(self):
        assert normalize_column_name("MyColumn") == "mycolumn"

    def test_spaces_to_underscore(self):
        assert normalize_column_name("First Name") == "first_name"

    def test_special_chars(self):
        assert normalize_column_name("Amount ($)") == "amount"

    def test_multiple_specials(self):
        assert normalize_column_name("col---name!!!") == "col_name"

    def test_already_normalized(self):
        assert normalize_column_name("simple_col") == "simple_col"

    def test_numeric_prefix(self):
        assert normalize_column_name("123_col") == "123_col"

    def test_empty_after_strip(self):
        assert normalize_column_name("!!!") == "column"


class TestDatasetProfiler:
    """Test dataset-level profiling."""

    @pytest.fixture
    def profiler(self) -> DatasetProfiler:
        return DatasetProfiler()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Transaction ID": ["T001", "T002", "T003", "T001"],
            "Amount": ["100.50", "200.75", "50.00", "100.50"],
            "Status": ["approved", "declined", "approved", "approved"],
        })

    def test_row_count(self, profiler: DatasetProfiler, sample_df: pd.DataFrame):
        result = profiler.profile(sample_df)
        assert result.row_count == 4

    def test_column_count(self, profiler: DatasetProfiler, sample_df: pd.DataFrame):
        result = profiler.profile(sample_df)
        assert result.column_count == 3

    def test_duplicate_row_count(self, profiler: DatasetProfiler, sample_df: pd.DataFrame):
        result = profiler.profile(sample_df)
        # Row 0 and row 3 are identical
        assert result.duplicate_row_count == 1

    def test_memory_estimate(self, profiler: DatasetProfiler, sample_df: pd.DataFrame):
        result = profiler.profile(sample_df)
        assert result.memory_estimate_bytes > 0

    def test_column_names_preserved(self, profiler: DatasetProfiler, sample_df: pd.DataFrame):
        result = profiler.profile(sample_df)
        assert result.column_names == ["Transaction ID", "Amount", "Status"]

    def test_normalized_keys(self, profiler: DatasetProfiler, sample_df: pd.DataFrame):
        result = profiler.profile(sample_df)
        assert result.normalized_keys == ["transaction_id", "amount", "status"]

    def test_duplicate_column_names_detected(self, profiler: DatasetProfiler):
        df = pd.DataFrame([[1, 2, 3]], columns=["col_a", "Col A", "col_b"])
        result = profiler.profile(df)
        assert "col_a" in result.duplicate_column_names

    def test_duplicate_column_keys_made_unique(self, profiler: DatasetProfiler):
        df = pd.DataFrame([[1, 2, 3]], columns=["col_a", "Col A", "col_b"])
        result = profiler.profile(df)
        # Should have unique keys like col_a, col_a_1
        assert len(set(result.normalized_keys)) == 3

    def test_to_dict(self, profiler: DatasetProfiler, sample_df: pd.DataFrame):
        result = profiler.profile(sample_df)
        d = result.to_dict()
        assert d["row_count"] == 4
        assert d["column_count"] == 3
