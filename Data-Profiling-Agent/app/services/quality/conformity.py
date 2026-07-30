"""Conformity quality dimension."""

from typing import Any
from app.core.enums import QualityDimension, QualityStatus
from app.services.rules.rule_engine import RuleEvaluationResult


def assess_conformity(rule_results: list[RuleEvaluationResult]) -> dict[str, Any]:
    """
    Assess conformity: matching against standard/format specifications.
    Uses regex_match rules (ISO dates, currency codes, etc).
    """
    relevant = [r for r in rule_results if r.rule_type.value == "regex_match" and r.error is None]

    if not relevant:
        return {
            "dimension": QualityDimension.CONFORMITY.value,
            "score": None,
            "status": QualityStatus.NOT_ASSESSABLE.value,
            "assessed_count": 0,
            "violation_count": 0,
            "evidence": [],
            "reason": "No conformity rules (regex_match) evaluated.",
        }

    total_checked = sum(r.records_checked for r in relevant)
    total_pass = sum(r.pass_count for r in relevant)
    total_fail = sum(r.fail_count for r in relevant)
    score = total_pass / total_checked if total_checked > 0 else 1.0

    evidence = [{"rule": r.rule_key, "score": round(r.score, 4)} for r in relevant]

    return {
        "dimension": QualityDimension.CONFORMITY.value,
        "score": score,
        "status": QualityStatus.ASSESSED.value,
        "assessed_count": total_checked,
        "violation_count": total_fail,
        "evidence": evidence,
    }
