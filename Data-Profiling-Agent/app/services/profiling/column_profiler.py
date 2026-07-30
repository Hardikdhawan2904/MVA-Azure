"""Column-level profiling — computes per-column statistics deterministically."""

from typing import Any

import numpy as np
import pandas as pd

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ColumnProfileResult:
    """Typed result of profiling a single column."""

    def __init__(
        self,
        physical_name: str,
        normalized_key: str,
        pandas_dtype: str,
        row_count: int,
        non_null_count: int,
        null_count: int,
        null_ratio: float,
        distinct_count: int,
        cardinality_ratio: float,
        duplicate_count: int,
        sample_values: list[Any],
        representative_values: list[Any],
        min_value: Any | None,
        max_value: Any | None,
        mean: float | None,
        median: float | None,
        std_dev: float | None,
        quantiles: dict[str, float] | None,
        min_string_length: int | None,
        max_string_length: int | None,
        avg_string_length: float | None,
        datetime_parse_ratio: float,
        numeric_parse_ratio: float,
        boolean_parse_ratio: float,
        dominant_patterns: list[str],
        warnings: list[str],
        all_integer_valued: bool | None = None,
    ):
        self.physical_name = physical_name
        self.normalized_key = normalized_key
        self.pandas_dtype = pandas_dtype
        self.row_count = row_count
        self.non_null_count = non_null_count
        self.null_count = null_count
        self.null_ratio = null_ratio
        self.distinct_count = distinct_count
        self.cardinality_ratio = cardinality_ratio
        self.duplicate_count = duplicate_count
        self.sample_values = sample_values
        self.representative_values = representative_values
        self.min_value = min_value
        self.max_value = max_value
        self.mean = mean
        self.median = median
        self.std_dev = std_dev
        self.quantiles = quantiles
        self.min_string_length = min_string_length
        self.max_string_length = max_string_length
        self.avg_string_length = avg_string_length
        self.datetime_parse_ratio = datetime_parse_ratio
        self.numeric_parse_ratio = numeric_parse_ratio
        self.boolean_parse_ratio = boolean_parse_ratio
        self.dominant_patterns = dominant_patterns
        self.warnings = warnings
        # Whether every non-null numeric value in the FULL column (not a
        # sample, not just min/max) has zero fractional part. None when the
        # column isn't predominantly numeric. This is the authoritative
        # integer/decimal signal — computed once, here, where the complete
        # parsed series is available; see TypeRefiner._is_integer.
        self.all_integer_valued = all_integer_valued

    def to_statistics_dict(self) -> dict[str, Any]:
        """Serialize statistics for JSONB storage."""
        return {
            "row_count": self.row_count,
            "non_null_count": self.non_null_count,
            "null_count": self.null_count,
            "null_ratio": self.null_ratio,
            "distinct_count": self.distinct_count,
            "cardinality_ratio": self.cardinality_ratio,
            "duplicate_count": self.duplicate_count,
            "sample_values": self.sample_values,
            "representative_values": self.representative_values,
            "min_value": _serialize_value(self.min_value),
            "max_value": _serialize_value(self.max_value),
            "mean": self.mean,
            "median": self.median,
            "std_dev": self.std_dev,
            "quantiles": self.quantiles,
            "min_string_length": self.min_string_length,
            "max_string_length": self.max_string_length,
            "avg_string_length": self.avg_string_length,
            "datetime_parse_ratio": self.datetime_parse_ratio,
            "numeric_parse_ratio": self.numeric_parse_ratio,
            "boolean_parse_ratio": self.boolean_parse_ratio,
            "dominant_patterns": self.dominant_patterns,
            "warnings": self.warnings,
            "all_integer_valued": self.all_integer_valued,
        }


def _serialize_value(val: Any) -> Any:
    """Make a value JSON-serializable."""
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return str(val)


class ColumnProfiler:
    """Computes deterministic statistics for each column in a DataFrame."""

    def __init__(self, settings: Settings):
        self._max_samples = settings.max_sample_values

    def profile_column(
        self, series: pd.Series, physical_name: str, normalized_key: str
    ) -> ColumnProfileResult:
        """Profile a single column/series."""
        row_count = len(series)
        null_count = int(series.isna().sum())
        non_null_count = row_count - null_count
        null_ratio = null_count / row_count if row_count > 0 else 0.0

        # Work with non-null values
        non_null = series.dropna()

        distinct_count = int(non_null.nunique())
        cardinality_ratio = distinct_count / non_null_count if non_null_count > 0 else 0.0
        duplicate_count = max(0, non_null_count - distinct_count)

        # Sample and representative values
        sample_values = self._get_sample_values(non_null)
        representative_values = self._get_representative_values(non_null)

        # Numeric statistics
        numeric_series = pd.to_numeric(non_null, errors="coerce")
        numeric_valid = numeric_series.dropna()
        numeric_parse_ratio = len(numeric_valid) / non_null_count if non_null_count > 0 else 0.0

        min_value: Any = None
        max_value: Any = None
        mean: float | None = None
        median: float | None = None
        std_dev: float | None = None
        quantiles: dict[str, float] | None = None
        all_integer_valued: bool | None = None

        if len(numeric_valid) > 0 and numeric_parse_ratio > 0.5:
            min_value = float(numeric_valid.min())
            max_value = float(numeric_valid.max())
            mean = float(numeric_valid.mean())
            median = float(numeric_valid.median())
            std_dev = float(numeric_valid.std()) if len(numeric_valid) > 1 else 0.0
            q = numeric_valid.quantile([0.25, 0.50, 0.75])
            quantiles = {"q25": float(q.iloc[0]), "q50": float(q.iloc[1]), "q75": float(q.iloc[2])}
            # Checked against every parsed value, not a sample or just the
            # extremes — a frequency-dominant mode (e.g. mostly "0") can't
            # mask a real fractional tail elsewhere in the column.
            all_integer_valued = bool((numeric_valid % 1 == 0).all())

        # String statistics
        str_values = non_null.astype(str)
        str_lengths = str_values.str.len()
        min_string_length: int | None = None
        max_string_length: int | None = None
        avg_string_length: float | None = None

        if non_null_count > 0:
            min_string_length = int(str_lengths.min())
            max_string_length = int(str_lengths.max())
            avg_string_length = float(str_lengths.mean())

        # Datetime parse ratio
        datetime_parse_ratio = self._compute_datetime_parse_ratio(non_null)

        # Boolean parse ratio
        boolean_parse_ratio = self._compute_boolean_parse_ratio(non_null)

        # Dominant patterns
        dominant_patterns = self._detect_dominant_patterns(non_null)

        # Warnings
        warnings: list[str] = []
        if null_ratio > 0.5:
            warnings.append(f"High null ratio: {null_ratio:.2%}")
        if cardinality_ratio > 0.98 and non_null_count > 10:
            warnings.append("Near-unique column — possible identifier")
        if duplicate_count == 0 and non_null_count > 1:
            pass  # Not a warning, just noting uniqueness

        return ColumnProfileResult(
            physical_name=physical_name,
            normalized_key=normalized_key,
            pandas_dtype=str(series.dtype),
            row_count=row_count,
            non_null_count=non_null_count,
            null_count=null_count,
            null_ratio=null_ratio,
            distinct_count=distinct_count,
            cardinality_ratio=cardinality_ratio,
            duplicate_count=duplicate_count,
            sample_values=sample_values,
            representative_values=representative_values,
            min_value=min_value,
            max_value=max_value,
            mean=mean,
            median=median,
            std_dev=std_dev,
            quantiles=quantiles,
            min_string_length=min_string_length,
            max_string_length=max_string_length,
            avg_string_length=avg_string_length,
            datetime_parse_ratio=datetime_parse_ratio,
            numeric_parse_ratio=numeric_parse_ratio,
            boolean_parse_ratio=boolean_parse_ratio,
            dominant_patterns=dominant_patterns,
            warnings=warnings,
            all_integer_valued=all_integer_valued,
        )

    def profile_all(
        self, df: pd.DataFrame, normalized_keys: list[str]
    ) -> list[ColumnProfileResult]:
        """Profile all columns in the DataFrame."""
        results: list[ColumnProfileResult] = []
        for i, col in enumerate(df.columns):
            key = normalized_keys[i] if i < len(normalized_keys) else col
            result = self.profile_column(df[col], physical_name=str(col), normalized_key=key)
            results.append(result)
        return results

    def _get_sample_values(self, series: pd.Series) -> list[Any]:
        """Get a bounded sample of actual values."""
        if len(series) == 0:
            return []
        sample_size = min(self._max_samples, len(series))
        return [_serialize_value(v) for v in series.head(sample_size).tolist()]

    def _get_representative_values(self, series: pd.Series) -> list[Any]:
        """Get representative values (most frequent)."""
        if len(series) == 0:
            return []
        value_counts = series.value_counts().head(self._max_samples)
        return [_serialize_value(v) for v in value_counts.index.tolist()]

    def _compute_datetime_parse_ratio(self, series: pd.Series) -> float:
        """Compute ratio of values that successfully parse as datetime."""
        if len(series) == 0:
            return 0.0

        sample = series.head(min(1000, len(series)))
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            valid_count = parsed.notna().sum()
            return float(valid_count / len(sample))
        except Exception:
            return 0.0

    def _compute_boolean_parse_ratio(self, series: pd.Series) -> float:
        """Compute ratio of values interpretable as boolean."""
        if len(series) == 0:
            return 0.0

        boolean_values = {
            "true", "false", "yes", "no", "y", "n",
            "1", "0", "t", "f", "on", "off",
        }
        str_values = series.astype(str).str.lower().str.strip()
        matches = str_values.isin(boolean_values).sum()
        return float(matches / len(series))

    def _detect_dominant_patterns(self, series: pd.Series) -> list[str]:
        """Detect dominant patterns in string values."""
        if len(series) == 0:
            return []

        sample = series.head(min(500, len(series))).astype(str)
        patterns: dict[str, int] = {}

        pattern_checks = [
            (r"^\d{4}-\d{2}-\d{2}$", "YYYY-MM-DD"),
            (r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "ISO_DATETIME"),
            (r"^\d{2}/\d{2}/\d{4}$", "MM/DD/YYYY"),
            (r"^\d{1,2}/\d{1,2}/\d{2,4}$", "DATE_SLASH"),
            (r"^[A-Z]{3}$", "THREE_LETTER_CODE"),
            (r"^[A-Z]{2}$", "TWO_LETTER_CODE"),
            (r"^[a-f0-9\-]{36}$", "UUID"),
            (r"^[\w.+-]+@[\w-]+\.[\w.]+$", "EMAIL"),
            # Requires an actual separator or leading "+" — a bare unbroken
            # digit run (e.g. an 8-digit BIN/account code) must not match;
            # it's indistinguishable from any other numeric identifier.
            (r"^(?=.*[\s\-\(\)+])\+?\d[\d\s\-\(\)]{7,}$", "PHONE"),
            (r"^\d+\.\d{2}$", "DECIMAL_2DP"),
            (r"^\d+$", "INTEGER"),
            (r"^-?\d+\.?\d*$", "NUMERIC"),
        ]

        for pattern, label in pattern_checks:
            match_count = sample.str.match(pattern, na=False).sum()
            ratio = match_count / len(sample)
            if ratio > 0.5:
                patterns[label] = int(match_count)

        # Sort by frequency and return top patterns
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        return [p[0] for p in sorted_patterns[:5]]
