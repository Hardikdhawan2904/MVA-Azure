"""Rule suggestion approval/rejection service."""

from datetime import datetime, timezone
from typing import Any

from app.core.enums import RuleSuggestionStatus
from app.core.exceptions import InvalidRuleTransitionError
from app.core.logging import get_logger

logger = get_logger(__name__)


class ApprovalService:
    """
    Manages approval and rejection of AI-proposed rule suggestions.

    Only 'proposed' rules can be approved or rejected.
    Approved rules do NOT modify original domain configuration.
    """

    def approve(
        self,
        suggestion_id: str,
        current_status: RuleSuggestionStatus,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """
        Approve a proposed rule suggestion.

        Returns metadata to persist.
        """
        if current_status != RuleSuggestionStatus.PROPOSED:
            raise InvalidRuleTransitionError(
                suggestion_id=suggestion_id,
                current_status=current_status.value,
                requested_action="approve",
            )

        return {
            "new_status": RuleSuggestionStatus.APPROVED.value,
            "comment": comment,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }

    def reject(
        self,
        suggestion_id: str,
        current_status: RuleSuggestionStatus,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Reject a proposed rule suggestion.

        Returns metadata to persist.
        """
        if current_status != RuleSuggestionStatus.PROPOSED:
            raise InvalidRuleTransitionError(
                suggestion_id=suggestion_id,
                current_status=current_status.value,
                requested_action="reject",
            )

        return {
            "new_status": RuleSuggestionStatus.REJECTED.value,
            "rejection_reason": reason,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
