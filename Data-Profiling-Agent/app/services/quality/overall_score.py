"""Overall quality score — weighted average excluding not_assessable dimensions."""

from typing import Any
from app.core.enums import QualityStatus


def calculate_overall_score(
    dimension_results: list[dict[str, Any]],
    weights: dict[str, float],
    formula_version: str = "quality-v1",
) -> dict[str, Any]:
    """
    Calculate overall quality score.

    Formula: sum(weight_i * score_i) / sum(weight_i for assessed dimensions)
    not_assessable dimensions excluded from both numerator and denominator.
    Never treat not_assessable as zero.
    """
    weighted_sum = 0.0
    weight_sum = 0.0
    assessed_dimensions: list[str] = []
    excluded_dimensions: list[str] = []

    for result in dimension_results:
        dimension = result.get("dimension", "")
        status = result.get("status", "")
        score = result.get("score")
        dim_weight = weights.get(dimension, 0.0)

        if status == QualityStatus.NOT_ASSESSABLE.value or score is None:
            excluded_dimensions.append(dimension)
            continue

        weighted_sum += dim_weight * score
        weight_sum += dim_weight
        assessed_dimensions.append(dimension)

    overall_score = weighted_sum / weight_sum if weight_sum > 0 else None
    display_score = round(overall_score * 100, 2) if overall_score is not None else None

    return {
        "overall_score": round(overall_score, 4) if overall_score is not None else None,
        "display_score": display_score,
        "assessed_dimensions": assessed_dimensions,
        "excluded_dimensions": excluded_dimensions,
        "weight_version": formula_version,
        "formula_version": formula_version,
    }
