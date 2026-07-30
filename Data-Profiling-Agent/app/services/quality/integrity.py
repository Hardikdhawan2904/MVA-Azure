"""Integrity quality dimension."""

from typing import Any
from app.core.enums import QualityDimension, QualityStatus


def assess_integrity() -> dict[str, Any]:
    """Integrity requires reference data. Returns not_assessable without it."""
    return {
        "dimension": QualityDimension.INTEGRITY.value,
        "score": None,
        "status": QualityStatus.NOT_ASSESSABLE.value,
        "assessed_count": 0,
        "violation_count": 0,
        "evidence": [],
        "reason": "No reference table or reference dataset was supplied.",
    }
