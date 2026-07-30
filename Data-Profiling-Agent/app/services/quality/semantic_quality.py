"""Semantic quality dimension — weighted average of intelligence confidences."""

from typing import Any
from app.core.enums import QualityDimension, QualityStatus


def assess_semantic_quality(
    schema_confidence_avg: float | None,
    classification_confidence_avg: float | None,
    secondary_domain_confidence: float | None,
    hierarchy_confidence: float | None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Semantic quality as weighted average of:
    - Schema Intelligence confidence
    - Column classification confidence
    - Secondary domain confidence
    - Hierarchy confidence
    """
    default_weights = {
        "schema_intelligence_confidence": 0.30,
        "column_classification_confidence": 0.25,
        "secondary_domain_confidence": 0.25,
        "hierarchy_confidence": 0.20,
    }
    w = weights or default_weights

    components = {
        "schema_intelligence_confidence": schema_confidence_avg,
        "column_classification_confidence": classification_confidence_avg,
        "secondary_domain_confidence": secondary_domain_confidence,
        "hierarchy_confidence": hierarchy_confidence,
    }

    weighted_sum = 0.0
    weight_sum = 0.0
    evidence: list[dict[str, Any]] = []

    for key, value in components.items():
        component_weight = w.get(key, 0.25)
        if value is not None:
            weighted_sum += value * component_weight
            weight_sum += component_weight
            evidence.append({"component": key, "value": round(value, 4), "weight": component_weight})

    if weight_sum == 0:
        return {
            "dimension": QualityDimension.SEMANTIC_QUALITY.value,
            "score": None,
            "status": QualityStatus.NOT_ASSESSABLE.value,
            "assessed_count": 0,
            "violation_count": 0,
            "evidence": [],
            "reason": "No semantic confidence scores available.",
        }

    score = weighted_sum / weight_sum

    return {
        "dimension": QualityDimension.SEMANTIC_QUALITY.value,
        "score": score,
        "status": QualityStatus.ASSESSED.value,
        "assessed_count": len([v for v in components.values() if v is not None]),
        "violation_count": 0,
        "evidence": evidence,
    }
