"""
Data Quality Validation Service.
Executes configured quality checks on the full dataset using vectorized pandas operations.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

# Resolve config path relative to service location
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "quality_threshold.json",
)


class BaseQualityCheck:
    """Interface for modular dataset quality checks."""

    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        """
        Execute check on sample DataFrame.
        
        Args:
            df_sample: The sampled Pandas DataFrame to inspect.
            weight: Max weight/score allocated for this check in configuration.
            limits: Dictionary containing limit rules, thresholds, and parameters.
            
        Returns:
            A tuple of (achieved_score, optional_warning_string).
        """
        raise NotImplementedError("Check must implement validate method")


# 1. Column Count Check
class ColumnCountCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        min_cols = limits.get("min_columns", 1)
        col_count = len(df_sample.columns)
        if col_count < min_cols:
            warning = f"Dataset column count {col_count} is below the minimum limit of {min_cols}."
            return 0.0, warning
        return weight, None


# 2. Missing Values Check
class MissingValuesCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if df_sample.size == 0:
            return weight, None
        total_missing = df_sample.isna().sum().sum()
        pct_missing = (total_missing / df_sample.size) * 100

        # A dataset-wide average can hide a single near-empty column (e.g. 95% missing)
        # when the rest of the dataset is clean — EmptyColumnsCheck only catches columns
        # that are 100% missing. Score off whichever is worse: the overall average, or
        # the single worst column.
        per_column_pct_missing = df_sample.isna().mean() * 100
        worst_col_pct = per_column_pct_missing.max() if len(per_column_pct_missing) else 0.0
        effective_pct = max(pct_missing, worst_col_pct)

        # Linear score deduction based on missing cells
        achieved_score = weight * (1.0 - (effective_pct / 100.0))

        warnings = []
        max_allowed = limits.get("max_missing_values_pct", 20.0)
        if pct_missing > max_allowed:
            warnings.append(f"Missing values density ({pct_missing:.1f}%) exceeds threshold of {max_allowed}%.")

        max_single_col_allowed = limits.get("max_single_column_missing_pct", 50.0)
        if worst_col_pct > max_single_col_allowed:
            worst_col_name = per_column_pct_missing.idxmax()
            warnings.append(
                f"Column '{worst_col_name}' is {worst_col_pct:.1f}% missing, "
                f"exceeding the per-column threshold of {max_single_col_allowed}%."
            )

        warning = " ".join(warnings) if warnings else None
        return max(0.0, achieved_score), warning


# 3. Duplicate Columns Check
class DuplicateColumnsCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if len(df_sample.columns) == 0:
            return weight, None
        
        # Check duplicate column names
        dup_cols = df_sample.columns.duplicated().sum()
        pct_dup = (dup_cols / len(df_sample.columns)) * 100
        
        achieved_score = weight * (1.0 - (dup_cols / len(df_sample.columns)))
        
        warning = None
        if dup_cols > 0:
            warning = f"Detected {dup_cols} duplicate column names ({pct_dup:.1f}% duplicate rate)."
            
        return max(0.0, achieved_score), warning


# 4. Duplicate Rows Check
class DuplicateRowsCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if len(df_sample) == 0:
            return weight, None
        
        # Count duplicate rows
        dup_rows = df_sample.duplicated().sum()
        pct_dup = (dup_rows / len(df_sample)) * 100
        
        achieved_score = weight * (1.0 - (dup_rows / len(df_sample)))
        
        warning = None
        max_allowed = limits.get("max_duplicate_rows_pct", 10.0)
        if pct_dup > max_allowed:
            warning = f"Duplicate rows ({pct_dup:.1f}%) exceeds threshold of {max_allowed}%."
            
        return max(0.0, achieved_score), warning


# 5. Empty Columns Check
class EmptyColumnsCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if len(df_sample.columns) == 0:
            return weight, None
        
        empty_cols = sum(df_sample[col].isna().all() for col in df_sample.columns)
        pct_empty = (empty_cols / len(df_sample.columns)) * 100
        
        achieved_score = weight * (1.0 - (empty_cols / len(df_sample.columns)))
        
        warning = None
        max_allowed = limits.get("max_empty_columns_pct", 10.0)
        if pct_empty > max_allowed:
            warning = f"Entirely empty columns ({pct_empty:.1f}%) exceeds threshold of {max_allowed}%."
            
        return max(0.0, achieved_score), warning


# 6. Datatype Consistency Check
class DatatypeConsistencyCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if len(df_sample.columns) == 0:
            return weight, None

        col_consistencies = []
        for col in df_sample.columns:
            non_null = df_sample[col].dropna()
            n = len(non_null)
            if n == 0:
                col_consistencies.append(1.0)
                continue

            # Vectorized type-group masks (bool checked first — bool is a subtype of int,
            # so it must be excluded from the numeric mask explicitly)
            is_bool = non_null.map(lambda v: isinstance(v, bool))
            is_numeric = non_null.map(lambda v: isinstance(v, (int, float))) & ~is_bool
            is_datetime = non_null.map(lambda v: isinstance(v, (pd.Timestamp, datetime)))
            is_native_typed = is_bool | is_numeric | is_datetime

            # CSV-sourced mixed columns never carry native int/float/bool/datetime values —
            # pandas keeps every cell in an object-dtype CSV column as plain str, even
            # numeric-looking ones ("123"), so without this pass they'd all fall into
            # is_string and a column mixing "123"/"456" with "abc"/"xyz" would score as
            # 100% consistent. Classify the remaining plain strings by apparent type instead.
            remaining = non_null[~is_native_typed]
            is_numeric_str = pd.Series(False, index=non_null.index)
            is_bool_str = pd.Series(False, index=non_null.index)
            is_datetime_str = pd.Series(False, index=non_null.index)
            if len(remaining) > 0:
                str_vals = remaining.astype(str).str.strip()

                numeric_coerced = pd.to_numeric(str_vals, errors="coerce")
                is_numeric_str.loc[remaining.index] = numeric_coerced.notna()

                not_numeric_idx = remaining.index[~is_numeric_str.loc[remaining.index]]
                if len(not_numeric_idx) > 0:
                    bool_match = str_vals.loc[not_numeric_idx].str.lower().isin(["true", "false"])
                    is_bool_str.loc[not_numeric_idx] = bool_match

                    leftover_idx = not_numeric_idx[~bool_match]
                    if len(leftover_idx) > 0:
                        datetime_coerced = pd.to_datetime(
                            str_vals.loc[leftover_idx], errors="coerce", format="mixed"
                        )
                        is_datetime_str.loc[leftover_idx] = datetime_coerced.notna()

            is_bool = is_bool | is_bool_str
            is_numeric = is_numeric | is_numeric_str
            is_datetime = is_datetime | is_datetime_str
            is_string = ~(is_bool | is_numeric | is_datetime)

            dominant_count = max(is_bool.sum(), is_numeric.sum(), is_datetime.sum(), is_string.sum())
            col_consistencies.append(dominant_count / n)

        overall_consistency = np.mean(col_consistencies) if col_consistencies else 1.0
        achieved_score = weight * overall_consistency
        
        warning = None
        min_allowed = limits.get("min_datatype_consistency_pct", 80.0)
        if (overall_consistency * 100.0) < min_allowed:
            warning = f"Datatype consistency ({overall_consistency*100.0:.1f}%) is below minimum requirements of {min_allowed}%."
            
        return max(0.0, achieved_score), warning


# 7. Corrupted Values Check
class CorruptedValuesCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if df_sample.size == 0:
            return weight, None
        
        placeholders = set(p.lower() for p in limits.get("corrupted_placeholders", []))
        corrupted_count = 0
        total_valid = 0

        for col in df_sample.columns:
            non_null = df_sample[col].dropna()
            total_valid += len(non_null)
            if len(non_null) == 0:
                continue
            normalized = non_null.astype(str).str.strip().str.lower()
            corrupted_count += int(normalized.isin(placeholders).sum())

        if total_valid == 0:
            return weight, None
            
        pct_corrupted = (corrupted_count / total_valid) * 100
        achieved_score = weight * (1.0 - (corrupted_count / total_valid))
        
        warning = None
        max_allowed = limits.get("max_corrupted_values_pct", 5.0)
        if pct_corrupted > max_allowed:
            warning = f"Corrupted placeholders ({pct_corrupted:.1f}%) exceeds threshold of {max_allowed}%."
            
        return max(0.0, achieved_score), warning


# 8. Null-heavy Rows Check
class NullHeavyRowsCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if len(df_sample) == 0:
            return weight, None
            
        threshold = limits.get("null_heavy_row_pct_threshold", 50.0)

        # Vectorized: null percentage for every row in one call, instead of iterrows()
        row_null_pcts = df_sample.isna().mean(axis=1) * 100
        null_heavy_count = int((row_null_pcts > threshold).sum())

        pct_null_heavy = (null_heavy_count / len(df_sample)) * 100
        achieved_score = weight * (1.0 - (null_heavy_count / len(df_sample)))
        
        warning = None
        max_allowed = limits.get("max_null_heavy_rows_pct", 10.0)
        if pct_null_heavy > max_allowed:
            warning = f"Null-heavy rows ({pct_null_heavy:.1f}%) exceeds threshold of {max_allowed}%."
            
        return max(0.0, achieved_score), warning


def _string_values(non_null: pd.Series, exclude_blank: bool = False) -> pd.Series:
    """Return only the string-typed values from non_null. Shared by
    CellLengthOutliersCheck and MixedFormatsCheck, which both start from
    "just the actual text values" before computing their own separate
    statistic on top. exclude_blank additionally drops whitespace-only
    strings (MixedFormatsCheck needs this for casing checks; CellLengthOutliersCheck
    doesn't, since an empty string is still a valid — if extreme — length)."""
    if exclude_blank:
        mask = non_null.map(lambda v: isinstance(v, str) and v.strip() != "")
    else:
        mask = non_null.map(lambda v: isinstance(v, str))
    return non_null[mask]


def _is_free_text_column(strings: pd.Series, limits: Dict[str, Any]) -> bool:
    """Detects genuine free-text columns (notes, descriptions, addresses) where
    length variance and inconsistent casing are normal, not a data-quality defect —
    so CellLengthOutliersCheck / MixedFormatsCheck don't penalize them."""
    if len(strings) == 0:
        return False
    avg_words = strings.str.split().str.len().mean()
    avg_length = strings.str.len().mean()
    min_avg_words = limits.get("free_text_min_avg_words", 4)
    min_avg_length = limits.get("free_text_min_avg_length", 40)
    return avg_words >= min_avg_words or avg_length >= min_avg_length


# 9. Cell Length Outliers Check
class CellLengthOutliersCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if df_sample.size == 0:
            return weight, None
            
        outliers_count = 0
        total_strings = 0

        for col in df_sample.columns:
            non_null = df_sample[col].dropna()
            if len(non_null) == 0:
                continue
            strings = _string_values(non_null)
            if len(strings) < 3:
                continue

            if _is_free_text_column(strings, limits):
                continue

            lengths = strings.astype(str).str.len()
            total_strings += len(lengths)
            mean_len = lengths.mean()
            std_len = lengths.std(ddof=0)  # match np.std()'s default (population, not sample)

            if std_len > 0:
                # Count lengths outside 3 standard deviations — fully vectorized
                outliers_count += int(((lengths - mean_len).abs() > 3 * std_len).sum())

        if total_strings == 0:
            return weight, None
            
        pct_outliers = (outliers_count / total_strings) * 100
        achieved_score = weight * (1.0 - (outliers_count / total_strings))
        
        warning = None
        max_allowed = limits.get("max_cell_length_outliers_pct", 5.0)
        if pct_outliers > max_allowed:
            warning = f"Cell length outliers ({pct_outliers:.1f}%) exceeds threshold of {max_allowed}%."
            
        return max(0.0, achieved_score), warning


# 10. Mixed Data Formats Check
class MixedFormatsCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if len(df_sample.columns) == 0:
            return weight, None
            
        casing_consistencies = []
        for col in df_sample.columns:
            non_null = df_sample[col].dropna()
            strings = _string_values(non_null, exclude_blank=True).astype(str)
            if len(strings) == 0:
                continue

            if _is_free_text_column(strings, limits):
                continue

            # Vectorized pandas string-accessor methods, applied with the same priority
            # order as the original if/elif chain so categories stay mutually exclusive
            # (e.g. a single letter like "A" is both .isupper() and .istitle() — it must
            # only ever count once, under whichever category is checked first).
            is_upper = strings.str.isupper()
            is_lower = ~is_upper & strings.str.islower()
            is_title = ~is_upper & ~is_lower & strings.str.istitle()
            is_mixed = ~(is_upper | is_lower | is_title)

            majority_count = max(is_upper.sum(), is_lower.sum(), is_title.sum(), is_mixed.sum())
            casing_consistencies.append(majority_count / len(strings))
            
        overall_consistency = np.mean(casing_consistencies) if casing_consistencies else 1.0
        achieved_score = weight * overall_consistency
        
        pct_inconsistent = (1.0 - overall_consistency) * 100
        warning = None
        max_allowed = limits.get("max_mixed_formats_pct", 10.0)
        if pct_inconsistent > max_allowed:
            warning = f"Casing format inconsistency ({pct_inconsistent:.1f}%) exceeds threshold of {max_allowed}%."
            
        return max(0.0, achieved_score), warning


# Registry containing all modular validator checks
QUALITY_CHECKS: Dict[str, BaseQualityCheck] = {
    "column_count": ColumnCountCheck(),
    "missing_values": MissingValuesCheck(),
    "duplicate_columns": DuplicateColumnsCheck(),
    "duplicate_rows": DuplicateRowsCheck(),
    "empty_columns": EmptyColumnsCheck(),
    "datatype_consistency": DatatypeConsistencyCheck(),
    "corrupted_values": CorruptedValuesCheck(),
    "null_heavy_rows": NullHeavyRowsCheck(),
    "cell_length_outliers": CellLengthOutliersCheck(),
    "mixed_formats": MixedFormatsCheck(),
}


class DataQualityValidator:
    """Configuration-driven data quality validator runner."""

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse the JSON configuration file."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found at {self.config_path}. Using fallback defaults.")
            return self._get_fallback_config()
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading quality config: {e}. Using fallback defaults.")
            return self._get_fallback_config()

    def _get_fallback_config(self) -> Dict[str, Any]:
        """Provide fallback configuration if config file reading fails."""
        return {
            "passing_score": 75,
            "weights": {
                "column_count": 15,
                "missing_values": 20,
                "duplicate_columns": 5,
                "duplicate_rows": 5,
                "empty_columns": 10,
                "datatype_consistency": 15,
                "corrupted_values": 10,
                "null_heavy_rows": 10,
                "cell_length_outliers": 5,
                "mixed_formats": 5,
            },
            "limits": {
                "min_columns": 1,
                "max_missing_values_pct": 20.0,
                "max_single_column_missing_pct": 50.0,
                "max_duplicate_rows_pct": 10.0,
                "max_empty_columns_pct": 10.0,
                "min_datatype_consistency_pct": 80.0,
                "max_corrupted_values_pct": 5.0,
                "max_null_heavy_rows_pct": 10.0,
                "max_cell_length_outliers_pct": 5.0,
                "max_mixed_formats_pct": 10.0,
                "null_heavy_row_pct_threshold": 50.0,
                "corrupted_placeholders": ["?", "n/a", "na", "null", "none", "undefined", "NaN"],
                "free_text_min_avg_words": 4,
                "free_text_min_avg_length": 40,
            },
        }

    def run_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all configured validation checks on the full dataset.

        Args:
            df: The complete input DataFrame to validate.
            
        Returns:
            A structured dict matching the QualityReport output schema.
        """
        # All 10 checks are vectorized pandas operations, so scanning the full dataset
        # is cheap regardless of size — no need to sample only the head/tail anymore.
        df_sample = df

        # Execute checks
        weights = self.config.get("weights", {})
        limits = self.config.get("limits", {})
        passing_score = self.config.get("passing_score", 75)
        
        check_scores: Dict[str, float] = {}
        warnings: List[str] = []
        total_score = 0.0
        
        for check_name, weight in weights.items():
            check_obj = QUALITY_CHECKS.get(check_name)
            if check_obj:
                try:
                    score, warning = check_obj.validate(df_sample, float(weight), limits)
                    check_scores[check_name] = round(score, 2)
                    total_score += score
                    if warning:
                        warnings.append(warning)
                except Exception as e:
                    logger.error(f"Error executing validation check '{check_name}': {e}")
                    check_scores[check_name] = 0.0
                    warnings.append(f"Validation check '{check_name}' failed to execute: {str(e)}")
            else:
                logger.warning(f"Configured check '{check_name}' not found in registry.")
                check_scores[check_name] = 0.0

        decision = "PASS" if total_score >= passing_score else "FAIL"
        
        return {
            "dataset_score": round(total_score, 2),
            "passing_score": passing_score,
            "decision": decision,
            "summary": {
                "rows_analyzed": len(df_sample),
                "columns": len(df.columns),
            },
            "checks": check_scores,
            "warnings": warnings,
        }
