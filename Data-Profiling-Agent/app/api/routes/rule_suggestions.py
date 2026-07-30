"""Rule suggestions API routes — DB-backed approve/reject workflow for AI-proposed rules."""

import uuid
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.enums import RuleSuggestionStatus
from app.core.exceptions import RuleSuggestionNotFoundError, RunNotFoundError
from app.api.dependencies import get_db
from app.models.rule_suggestion import RuleSuggestion
from app.repositories.rule_repository import RuleRepository
from app.repositories.profile_run_repository import ProfileRunRepository
from app.services.rules.approval_service import ApprovalService
from app.schemas.requests import RuleApprovalRequest, RuleRejectionRequest

router = APIRouter(prefix="/rule-suggestions", tags=["rule-suggestions"])

approval_service = ApprovalService()


def _parse_suggestion_id(suggestion_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(suggestion_id)
    except ValueError:
        raise RuleSuggestionNotFoundError(suggestion_id)


def _suggestion_to_dict(s: RuleSuggestion) -> dict[str, Any]:
    return {
        "suggestion_id": str(s.id),
        "run_id": str(s.run_id),
        "definition": s.suggested_definition_json,
        "confidence": s.confidence,
        "status": s.status,
        "evidence": s.evidence_json,
        "comment": s.comment,
        "rejection_reason": s.rejection_reason,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "reviewed_at": s.reviewed_at.isoformat() if s.reviewed_at else None,
    }


@router.get("")
def list_suggestions(status: str | None = None, db: Session = Depends(get_db)) -> dict[str, Any]:
    """List all rule suggestions, optionally filtered by status."""
    repo = RuleRepository(db)
    suggestions = repo.list_suggestions(status)
    return {"suggestions": [_suggestion_to_dict(s) for s in suggestions]}


@router.get("/{suggestion_id}")
def get_suggestion(suggestion_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get a single rule suggestion."""
    repo = RuleRepository(db)
    suggestion = repo.get_suggestion(_parse_suggestion_id(suggestion_id))
    if not suggestion:
        raise RuleSuggestionNotFoundError(suggestion_id)
    return _suggestion_to_dict(suggestion)


@router.post("/{suggestion_id}/approve")
def approve_suggestion(
    suggestion_id: str, body: RuleApprovalRequest, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Approve a proposed rule suggestion. Materializes it into an active rule definition."""
    rule_repo = RuleRepository(db)
    suggestion = rule_repo.get_suggestion(_parse_suggestion_id(suggestion_id))
    if not suggestion:
        raise RuleSuggestionNotFoundError(suggestion_id)

    run_repo = ProfileRunRepository(db)
    run = run_repo.get_run(suggestion.run_id)
    if not run:
        raise RunNotFoundError(str(suggestion.run_id))

    current_status = RuleSuggestionStatus(suggestion.status)
    result = approval_service.approve(suggestion_id, current_status, body.comment)

    rule_repo.update_suggestion_status(suggestion.id, result["new_status"], comment=result.get("comment"))
    rule_def = rule_repo.create_rule_definition_from_suggestion(suggestion, domain=run.primary_domain)
    rule_repo.commit()

    return {
        "suggestion_id": suggestion_id,
        "rule_definition_id": str(rule_def.id),
        **result,
    }


@router.post("/{suggestion_id}/reject")
def reject_suggestion(
    suggestion_id: str, body: RuleRejectionRequest, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Reject a proposed rule suggestion. No rule definition is created."""
    rule_repo = RuleRepository(db)
    suggestion = rule_repo.get_suggestion(_parse_suggestion_id(suggestion_id))
    if not suggestion:
        raise RuleSuggestionNotFoundError(suggestion_id)

    current_status = RuleSuggestionStatus(suggestion.status)
    result = approval_service.reject(suggestion_id, current_status, body.reason)

    rule_repo.update_suggestion_status(
        suggestion.id, result["new_status"], rejection_reason=result.get("rejection_reason")
    )
    rule_repo.commit()

    return {"suggestion_id": suggestion_id, **result}
