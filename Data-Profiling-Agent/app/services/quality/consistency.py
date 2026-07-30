"""Consistency quality dimension."""

from typing import Any
from app.core.enums import QualityDimension, QualityStatus
from app.services.rules.rule_engine import RuleEvaluationResult


def assess_consistency(rule_results: list[RuleEvaluationResult]) -> dict[str, Any]:
    """
    Assess consistency: 1 - cross-field contradiction count / records checked.
    Uses column_comparison, cross_field_equality, cross_field_inequality rules.
    """
    consistency_types = {"column_comparison", "cross_field_equality", "cross_field_inequality"}
    relevant = [r for r in rule_results if r.rule_type.value in consistency_types and r.error is None]

    if not relevant:
        return {
            "dimension": QualityDimension.CONSISTENCY.value,
            "score": None,
            "status": QualityStatus.NOT_ASSESSABLE.value,
            "assessed_count": 0,
            "violation_count": 0,
            "evidence": [],
            "reason": "No consistency rules (column_comparison, cross_field) evaluated.",
        }

    total_checked = sum(r.records_checked for r in relevant)
    total_fail = sum(r.fail_count for r in relevant)
    score = 1.0 - (total_fail / total_checked) if total_checked > 0 else 1.0

    evidence = [{"rule": r.rule_key, "score": round(r.score, 4)} for r in relevant]

    return {
        "dimension": QualityDimension.CONSISTENCY.value,
        "score": score,
        "status": QualityStatus.ASSESSED.value,
        "assessed_count": total_checked,
        "violation_count": total_fail,
        "evidence": evidence,
    }
