"""Completeness quality dimension."""

from typing import Any

from app.core.enums import QualityDimension, QualityStatus
from app.services.profiling.column_profiler import ColumnProfileResult


class CompletenessResult:
    """Result of completeness assessment."""

    def __init__(self, dimension: str, score: float | None, status: QualityStatus,
                 assessed_count: int, violation_count: int, evidence: list[dict[str, Any]],
                 reason: str | None = None):
        self.dimension = dimension
        self.score = score
        self.status = status
        self.assessed_count = assessed_count
        self.violation_count = violation_count
        self.evidence = evidence
        self.reason = reason


def assess_completeness(
    profiles: list[ColumnProfileResult],
    mandatory_columns: list[str],
) -> CompletenessResult:
    """
    Assess completeness dimension.

    If mandatory columns defined: weighted average of null ratios for mandatory columns.
    If no mandatory columns: return not_assessable.
    """
    if not mandatory_columns:
        return CompletenessResult(
            dimension=QualityDimension.COMPLETENESS.value,
            score=None,
            status=QualityStatus.NOT_ASSESSABLE,
            assessed_count=0,
            violation_count=0,
            evidence=[],
            reason="No mandatory columns were defined by request, domain configuration, or Schema Intelligence.",
        )

    mandatory_profiles = [p for p in profiles if p.physical_name in mandatory_columns or p.normalized_key in mandatory_columns]

    if not mandatory_profiles:
        return CompletenessResult(
            dimension=QualityDimension.COMPLETENESS.value,
            score=None,
            status=QualityStatus.NOT_ASSESSABLE,
            assessed_count=0,
            violation_count=0,
            evidence=[],
            reason="Mandatory columns not found in dataset.",
        )

    total_cells = 0
    null_cells = 0
    evidence: list[dict[str, Any]] = []

    for p in mandatory_profiles:
        total_cells += p.row_count
        null_cells += p.null_count
        evidence.append({
            "column": p.physical_name,
            "null_ratio": round(p.null_ratio, 4),
            "null_count": p.null_count,
        })

    score = 1.0 - (null_cells / total_cells) if total_cells > 0 else 1.0

    return CompletenessResult(
        dimension=QualityDimension.COMPLETENESS.value,
        score=score,
        status=QualityStatus.ASSESSED,
        assessed_count=total_cells,
        violation_count=null_cells,
        evidence=evidence,
    )
