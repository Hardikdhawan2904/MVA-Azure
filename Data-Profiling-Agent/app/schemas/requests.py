"""API request schemas."""

from pydantic import BaseModel


class DrillDownRequest(BaseModel):
    """Drill-down request body."""
    selected_path: dict[str, str]


class RuleApprovalRequest(BaseModel):
    """Rule approval request."""
    comment: str | None = None


class RuleRejectionRequest(BaseModel):
    """Rule rejection request."""
    reason: str | None = None
