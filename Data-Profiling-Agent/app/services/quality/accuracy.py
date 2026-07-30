"""Accuracy quality dimension."""

from typing import Any
from app.core.enums import QualityDimension, QualityStatus


def assess_accuracy() -> dict[str, Any]:
    """Accuracy requires trusted reference dataset. Returns not_assessable without it."""
    return {
        "dimension": QualityDimension.ACCURACY.value,
        "score": None,
        "status": QualityStatus.NOT_ASSESSABLE.value,
        "assessed_count": 0,
        "violation_count": 0,
        "evidence": [],
        "reason": "No trusted reference dataset was supplied.",
    }
