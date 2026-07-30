"""Dataset-level profiling — computes aggregate dataset statistics."""

import re
from typing import Any

import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)

# Separator for normalized column keys
_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")


def normalize_column_name(name: str) -> str:
    """
    Generate a normalized internal key from a column name.

    - Lowercase
    - Replace non-alphanumeric sequences with underscore
    - Strip leading/trailing underscores
    """
    normalized = _NORMALIZE_PATTERN.sub("_", name.lower()).strip("_")
    return normalized or "column"


class DatasetProfileResult:
    """Typed result of dataset-level profiling."""

    def __init__(
        self,
        row_count: int,
        column_count: int,
        duplicate_row_count: int,
        memory_estimate_bytes: int,
        column_names: list[str],
        normalized_keys: list[str],
        duplicate_column_names: list[str],
    ):
        self.row_count = row_count
        self.column_count = column_count
        self.duplicate_row_count = duplicate_row_count
        self.memory_estimate_bytes = memory_estimate_bytes
        self.column_names = column_names
        self.normalized_keys = normalized_keys
        self.duplicate_column_names = duplicate_column_names

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "duplicate_row_count": self.duplicate_row_count,
            "memory_estimate_bytes": self.memory_estimate_bytes,
            "duplicate_column_names": self.duplicate_column_names,
        }


class DatasetProfiler:
    """Computes dataset-level statistics from a DataFrame."""

    def profile(self, df: pd.DataFrame) -> DatasetProfileResult:
        """
        Compute dataset-level profile.

        - Row/column counts
        - Duplicate row detection
        - Memory estimation
        - Column name normalization
        - Duplicate column name detection
        """
        row_count = len(df)
        column_count = len(df.columns)

        # Duplicate rows
        duplicate_row_count = int(df.duplicated().sum())

        # Memory estimation
        memory_estimate_bytes = int(df.memory_usage(deep=True).sum())

        # Column names and normalization
        column_names = list(df.columns)
        normalized_keys = [normalize_column_name(name) for name in column_names]

        # Detect duplicate column names (case-insensitive)
        seen: dict[str, int] = {}
        duplicate_column_names: list[str] = []
        for key in normalized_keys:
            seen[key] = seen.get(key, 0) + 1
        for key, count in seen.items():
            if count > 1:
                duplicate_column_names.append(key)

        # Make normalized keys unique by appending index for duplicates
        if duplicate_column_names:
            counters: dict[str, int] = {}
            unique_keys: list[str] = []
            for key in normalized_keys:
                if seen[key] > 1:
                    idx = counters.get(key, 0)
                    counters[key] = idx + 1
                    unique_keys.append(f"{key}_{idx}" if idx > 0 else key)
                else:
                    unique_keys.append(key)
            normalized_keys = unique_keys

        logger.info(
            "dataset_profiled",
            row_count=row_count,
            column_count=column_count,
            duplicate_rows=duplicate_row_count,
            memory_bytes=memory_estimate_bytes,
        )

        return DatasetProfileResult(
            row_count=row_count,
            column_count=column_count,
            duplicate_row_count=duplicate_row_count,
            memory_estimate_bytes=memory_estimate_bytes,
            column_names=column_names,
            normalized_keys=normalized_keys,
            duplicate_column_names=duplicate_column_names,
        )
