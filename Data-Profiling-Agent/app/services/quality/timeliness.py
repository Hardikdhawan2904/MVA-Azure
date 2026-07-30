"""Timeliness quality dimension."""

from typing import Any
from app.core.enums import QualityDimension, QualityStatus


def assess_timeliness() -> dict[str, Any]:
    """Timeliness requires SLA/recency config. Returns not_assessable without it."""
    return {
        "dimension": QualityDimension.TIMELINESS.value,
        "score": None,
        "status": QualityStatus.NOT_ASSESSABLE.value,
        "assessed_count": 0,
        "violation_count": 0,
        "evidence": [],
        "reason": "No expected-recency or SLA configuration was supplied.",
    }
