"""LLM provider implementation — mock provider for testing without API calls.
(The real backend is AzureOpenAIProvider, app/services/llm/azure_provider.py.)
"""

from pydantic import BaseModel, ValidationError

from app.services.llm.interface import LLMRequest, LLMResponse


class MockLLMProvider:
    """Mock LLM provider for testing and demo without API keys."""

    def __init__(self):
        self._call_count = 0
        self._responses: list[LLMResponse] = []

    def set_responses(self, responses: list[LLMResponse]) -> None:
        """Pre-configure responses for testing."""
        self._responses = responses
        self._call_count = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return pre-configured or default mock response."""
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp

        self._call_count += 1
        return LLMResponse(
            content="{}",
            parsed={},
            model="mock",
            prompt_version="mock-v1",
            success=True,
        )

    def complete_structured(
        self, request: LLMRequest, response_model: type[BaseModel]
    ) -> tuple[BaseModel | None, LLMResponse]:
        """Return mock structured response."""
        response = self.complete(request)
        if not response.success or not response.parsed:
            return None, response

        try:
            parsed = response_model.model_validate(response.parsed)
            return parsed, response
        except ValidationError:
            return None, response
