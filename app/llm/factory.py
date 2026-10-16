"""Factory for resolving and instantiating LLM providers."""

from typing import Any, Optional

from app.config import settings
from app.llm.base import BaseLLMProvider, LLMConfigurationError
from app.llm.groq_provider import GroqProvider
from app.llm.mock_provider import MockLLMProvider


def get_llm_provider(
    provider_name: Optional[str] = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    """Resolve and instantiate an LLM provider based on name or settings.

    Args:
        provider_name: Name of the provider ('mock' or 'groq'). If None,
            falls back to settings.LLM_PROVIDER.
        **kwargs: Provider-specific configuration options passed to constructor.

    Returns:
        Configured BaseLLMProvider instance.

    Raises:
        LLMConfigurationError: If an unknown provider name is supplied or required
            credentials are missing.
    """
    target = (provider_name or settings.LLM_PROVIDER).strip().lower()

    if target == "mock":
        return MockLLMProvider(**kwargs)
    elif target == "groq":
        return GroqProvider(**kwargs)
    else:
        raise LLMConfigurationError(
            f"Unsupported LLM provider: '{target}'. Supported providers are 'mock' and 'groq'."
        )
