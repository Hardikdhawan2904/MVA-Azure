"""API response schemas."""

from pydantic import BaseModel


class RunCreatedResponse(BaseModel):
    """Response after creating a profile run."""
    run_id: str
    status: str
