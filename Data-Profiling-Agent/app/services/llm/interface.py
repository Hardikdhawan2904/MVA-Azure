"""LLM provider interface — abstracts the underlying model API."""

from typing import Any, Protocol

from pydantic import BaseModel


class LLMRequest(BaseModel):
    """Structured request to the LLM."""
    prompt: str
    system_message: str = ""
    model: str = ""
    temperature: float = 0.1
    max_tokens: int = 2000
    response_schema: dict[str, Any] | None = None


class LLMResponse(BaseModel):
    """Structured response from the LLM."""
    content: str
    parsed: dict[str, Any] | None = None
    model: str = ""
    prompt_version: str = ""
    usage_tokens: int = 0
    success: bool = True
    error: str | None = None


class LLMProvider(Protocol):
    """Protocol for LLM provider implementations."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request and return structured response."""
        ...

    def complete_structured(
        self, request: LLMRequest, response_model: type[BaseModel]
    ) -> tuple[BaseModel | None, LLMResponse]:
        """
        Send a completion request expecting a structured JSON response.

        Returns a tuple of (parsed_model_or_None, raw_response).
        """
        ...
