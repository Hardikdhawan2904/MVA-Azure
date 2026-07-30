"""Centralized error handlers for FastAPI."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    MVABaseException,
    FileValidationError,
    DatasetLimitError,
    UnsupportedDomainError,
    RunNotFoundError,
    RuleSuggestionNotFoundError,
    InvalidRuleTransitionError,
)

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    """Register all custom error handlers on the FastAPI app."""

    @app.exception_handler(FileValidationError)
    async def file_validation_handler(request: Request, exc: FileValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )

    @app.exception_handler(DatasetLimitError)
    async def dataset_limit_handler(request: Request, exc: DatasetLimitError) -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )

    @app.exception_handler(UnsupportedDomainError)
    async def unsupported_domain_handler(request: Request, exc: UnsupportedDomainError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )

    @app.exception_handler(RunNotFoundError)
    async def run_not_found_handler(request: Request, exc: RunNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )

    @app.exception_handler(RuleSuggestionNotFoundError)
    async def rule_suggestion_not_found_handler(
        request: Request, exc: RuleSuggestionNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )

    @app.exception_handler(InvalidRuleTransitionError)
    async def invalid_rule_transition_handler(
        request: Request, exc: InvalidRuleTransitionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )

    @app.exception_handler(MVABaseException)
    async def generic_mva_handler(request: Request, exc: MVABaseException) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Every other handler above covers a known MVABaseException subtype
        # with a structured {"error": {...}} envelope — this is the only
        # thing standing between an unexpected exception type (a bare
        # KeyError, a raw SQLAlchemy IntegrityError, ...) and Starlette's
        # unstructured default 500, which broke the documented API
        # contract every other endpoint honors. The real exception is
        # logged server-side with a traceback; the response never echoes
        # internal exception text back to the caller.
        logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please try again or contact support.",
                "details": {},
            }},
        )
