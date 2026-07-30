"""LLM provider factory — creates the appropriate provider based on settings."""

from app.core.config import Settings
from app.services.llm.interface import LLMProvider
from app.services.llm.provider import MockLLMProvider
from app.services.llm.azure_provider import AzureOpenAIProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    """
    Create the appropriate LLM provider based on configuration.

    Providers:
    - azure: Azure OpenAI (default — this project's only real backend)
    - mock: Mock provider for testing (no API calls)
    """
    provider_type = settings.llm_provider.lower()

    if provider_type == "mock":
        return MockLLMProvider()
    else:
        # "azure" and any unrecognized value both default to Azure OpenAI.
        return AzureOpenAIProvider(settings)
