"""Validity quality dimension."""

from typing import Any
from app.core.enums import QualityDimension, QualityStatus
from app.services.rules.rule_engine import RuleEvaluationResult


def assess_validity(rule_results: list[RuleEvaluationResult]) -> dict[str, Any]:
    """
    Assess validity: matching non-null values / non-null values checked.
    Uses business-defined ranges, allowed values, logical constraints.
    """
    validity_types = {"numeric_range", "allowed_values", "date_range"}
    relevant = [r for r in rule_results if r.rule_type.value in validity_types and r.error is None]

    if not relevant:
        return {
            "dimension": QualityDimension.VALIDITY.value,
            "score": None,
            "status": QualityStatus.NOT_ASSESSABLE.value,
            "assessed_count": 0,
            "violation_count": 0,
            "evidence": [],
            "reason": "No validity rules (numeric_range, allowed_values, date_range) evaluated.",
        }

    total_checked = sum(r.records_checked for r in relevant)
    total_pass = sum(r.pass_count for r in relevant)
    total_fail = sum(r.fail_count for r in relevant)
    score = total_pass / total_checked if total_checked > 0 else 1.0

    evidence = [{"rule": r.rule_key, "score": round(r.score, 4)} for r in relevant]

    return {
        "dimension": QualityDimension.VALIDITY.value,
        "score": score,
        "status": QualityStatus.ASSESSED.value,
        "assessed_count": total_checked,
        "violation_count": total_fail,
        "evidence": evidence,
    }
