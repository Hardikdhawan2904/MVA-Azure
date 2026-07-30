"""Uniqueness quality dimension — only assesses expected-unique columns."""

from typing import Any

from app.core.enums import QualityDimension, QualityStatus
from app.services.profiling.column_profiler import ColumnProfileResult


def assess_uniqueness(
    profiles: list[ColumnProfileResult],
    expected_unique_columns: list[str],
) -> dict[str, Any]:
    """
    Assess uniqueness only for columns flagged expected_unique.

    Per column: 1 - duplicate_count / total_count
    duplicate_count = rows beyond the first occurrence.
    """
    if not expected_unique_columns:
        return {
            "dimension": QualityDimension.UNIQUENESS.value,
            "score": None,
            "status": QualityStatus.NOT_ASSESSABLE.value,
            "assessed_count": 0,
            "violation_count": 0,
            "evidence": [],
            "reason": "No expected-unique columns defined.",
        }

    unique_profiles = [
        p for p in profiles
        if p.physical_name in expected_unique_columns or p.normalized_key in expected_unique_columns
    ]

    if not unique_profiles:
        return {
            "dimension": QualityDimension.UNIQUENESS.value,
            "score": None,
            "status": QualityStatus.NOT_ASSESSABLE.value,
            "assessed_count": 0,
            "violation_count": 0,
            "evidence": [],
            "reason": "Expected-unique columns not found in dataset.",
        }

    total = 0
    violations = 0
    evidence: list[dict[str, Any]] = []

    for p in unique_profiles:
        col_total = p.row_count
        col_dupes = p.duplicate_count
        total += col_total
        violations += col_dupes
        col_score = 1.0 - (col_dupes / col_total) if col_total > 0 else 1.0
        evidence.append({
            "column": p.physical_name,
            "duplicate_count": col_dupes,
            "score": round(col_score, 4),
        })

    score = 1.0 - (violations / total) if total > 0 else 1.0

    return {
        "dimension": QualityDimension.UNIQUENESS.value,
        "score": score,
        "status": QualityStatus.ASSESSED.value,
        "assessed_count": total,
        "violation_count": violations,
        "evidence": evidence,
    }
