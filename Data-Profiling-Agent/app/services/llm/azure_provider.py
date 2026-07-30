"""Azure OpenAI LLM provider — this project's LLM backend.

Uses the OpenAI-compatible /openai/v1/chat/completions surface Azure OpenAI
resources expose (no api-version query param needed, unlike the older
/openai/deployments/{name}/chat/completions surface), authenticated via the
api-key header (Azure's own scheme, not Authorization: Bearer).
"""

import json
import time
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.llm.interface import LLMRequest, LLMResponse

logger = get_logger(__name__)


class AzureOpenAIProvider:
    """
    LLM provider using Azure OpenAI.

    Endpoint: {AZURE_OPENAI_ENDPOINT}/openai/v1/chat/completions
    Model: the deployment name configured in Azure AI Studio (AZURE_OPENAI_DEPLOYMENT).
    """

    def __init__(self, settings: Settings):
        self._model = settings.azure_openai_deployment
        self._api_key = settings.azure_openai_api_key
        self._timeout = settings.llm_timeout_seconds
        self._max_retries = settings.llm_max_retries
        self._base_url = f"{settings.azure_openai_endpoint.rstrip('/')}/openai/v1"

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request to Azure OpenAI."""
        if not self._api_key or not self._model or not self._base_url.startswith("http"):
            return LLMResponse(
                content="",
                model=self._model,
                success=False,
                error="AZURE_OPENAI_API_KEY/AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_DEPLOYMENT not configured",
            )

        model = request.model or self._model
        messages: list[dict[str, str]] = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }

        if request.response_schema:
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(self._max_retries + 1):
            try:
                response = httpx.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "api-key": self._api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                data = response.json()

                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {}).get("total_tokens", 0)

                # Parse JSON response
                parsed = None
                try:
                    parsed = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    # Try to extract JSON from markdown code blocks
                    parsed = self._extract_json(content)

                return LLMResponse(
                    content=content,
                    parsed=parsed,
                    model=model,
                    prompt_version=request.system_message[:20] if request.system_message else "",
                    usage_tokens=usage,
                    success=True,
                )

            except httpx.TimeoutException:
                logger.warning("azure_openai_timeout", attempt=attempt, model=model)
                if attempt == self._max_retries:
                    return LLMResponse(
                        content="", model=model, success=False,
                        error="Azure OpenAI request timed out",
                    )
                # Same backoff as the rate-limit and generic-exception
                # branches below — a timeout is at least as likely to
                # benefit from backoff before retrying as a 429 is;
                # retrying instantly into an already-slow/overloaded
                # endpoint doesn't give it a chance to recover.
                time.sleep(min(2 ** attempt, 10))

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                logger.warning("azure_openai_http_error", status=status_code, attempt=attempt)

                # Rate limit — don't retry immediately
                if status_code == 429:
                    time.sleep(min(2 ** attempt, 10))

                if attempt == self._max_retries:
                    return LLMResponse(
                        content="", model=model, success=False,
                        error=f"Azure OpenAI HTTP {status_code}",
                    )

            except Exception as e:
                logger.error("azure_openai_unexpected_error", error=str(e), attempt=attempt)
                # Connection-level failures (resets, refused connections) benefit from
                # the same backoff as rate limits — retrying immediately can worsen a
                # network/throttling issue instead of giving it a chance to clear.
                if attempt < self._max_retries:
                    time.sleep(min(2 ** attempt, 10))
                if attempt == self._max_retries:
                    return LLMResponse(
                        content="", model=model, success=False,
                        error=str(e),
                    )

        return LLMResponse(content="", model=model, success=False, error="Max retries exceeded")

    def complete_structured(
        self, request: LLMRequest, response_model: type[BaseModel]
    ) -> tuple[BaseModel | None, LLMResponse]:
        """Send request and validate response against a Pydantic model."""
        request_with_schema = LLMRequest(
            prompt=request.prompt,
            system_message=request.system_message,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            response_schema=response_model.model_json_schema(),
        )

        response = self.complete(request_with_schema)
        if not response.success or not response.parsed:
            return None, response

        try:
            parsed_model = response_model.model_validate(response.parsed)
            return parsed_model, response
        except ValidationError as e:
            logger.warning("azure_openai_validation_error", error=str(e)[:200])
            response.error = f"Response validation failed: {str(e)[:200]}"
            return None, response

    def _extract_json(self, content: str) -> dict[str, Any] | None:
        """Try to extract JSON from content that may have markdown wrapping."""
        content = content.strip()

        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
