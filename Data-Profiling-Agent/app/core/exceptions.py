"""Custom exceptions for the MVA Data Profiling Engine."""

from typing import Any


class MVABaseException(Exception):
    """Base exception for all MVA errors."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)


class FileValidationError(MVABaseException):
    """Raised when file validation fails."""
    pass


class DatasetLimitError(MVABaseException):
    """Raised when dataset exceeds configured limits."""
    pass


class UnsupportedDomainError(MVABaseException):
    """Raised when an unsupported primary domain is supplied."""
    pass


class ProcessingError(MVABaseException):
    """Raised when pipeline processing fails."""
    pass


class ConfigurationError(MVABaseException):
    """Raised when configuration is invalid or missing."""
    pass


class RunNotFoundError(MVABaseException):
    """Raised when a profile run is not found."""

    def __init__(self, run_id: str):
        super().__init__(
            code="RUN_NOT_FOUND",
            message=f"Profile run '{run_id}' not found.",
            details={"run_id": run_id},
        )


class RuleSuggestionNotFoundError(MVABaseException):
    """Raised when a rule suggestion is not found."""

    def __init__(self, suggestion_id: str):
        super().__init__(
            code="RULE_SUGGESTION_NOT_FOUND",
            message=f"Rule suggestion '{suggestion_id}' not found.",
            details={"suggestion_id": suggestion_id},
        )


class InvalidRuleTransitionError(MVABaseException):
    """Raised when a rule suggestion status transition is invalid."""

    def __init__(self, suggestion_id: str, current_status: str, requested_action: str):
        super().__init__(
            code="INVALID_RULE_TRANSITION",
            message=f"Cannot {requested_action} suggestion '{suggestion_id}' with status '{current_status}'.",
            details={
                "suggestion_id": suggestion_id,
                "current_status": current_status,
                "requested_action": requested_action,
            },
        )
