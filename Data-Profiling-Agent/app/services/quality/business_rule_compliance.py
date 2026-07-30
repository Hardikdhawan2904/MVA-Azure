"""Business rule compliance quality dimension."""

from typing import Any
from app.core.enums import QualityDimension, QualityStatus
from app.services.rules.rule_engine import RuleEvaluationResult


def assess_business_rule_compliance(rule_results: list[RuleEvaluationResult]) -> dict[str, Any]:
    """
    Business rule compliance: pass_count / total_checked across all active rules.
    Only configured + request + approved rules included.
    """
    valid_results = [r for r in rule_results if r.error is None and r.records_checked > 0]

    if not valid_results:
        return {
            "dimension": QualityDimension.BUSINESS_RULE_COMPLIANCE.value,
            "score": None,
            "status": QualityStatus.NOT_ASSESSABLE.value,
            "assessed_count": 0,
            "violation_count": 0,
            "evidence": [],
            "reason": "No business rules were evaluated.",
        }

    total_checked = sum(r.records_checked for r in valid_results)
    total_pass = sum(r.pass_count for r in valid_results)
    total_fail = sum(r.fail_count for r in valid_results)
    score = total_pass / total_checked if total_checked > 0 else 1.0

    evidence = [
        {"rule": r.rule_key, "score": round(r.score, 4), "fail_count": r.fail_count}
        for r in valid_results
    ]

    return {
        "dimension": QualityDimension.BUSINESS_RULE_COMPLIANCE.value,
        "score": score,
        "status": QualityStatus.ASSESSED.value,
        "assessed_count": total_checked,
        "violation_count": total_fail,
        "evidence": evidence,
    }
