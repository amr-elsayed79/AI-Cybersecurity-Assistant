"""Base interface, response schemas, and exception hierarchy for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------

class LLMProviderError(Exception):
    """Base exception for all LLM provider errors."""
    pass


class LLMConfigurationError(LLMProviderError):
    """Raised when provider configuration or credentials are missing or invalid."""
    pass


class LLMTransientError(LLMProviderError):
    """Raised when a transient error (e.g., rate limit, timeout, 503) exhausts all retries."""
    pass


class LLMNonTransientError(LLMProviderError):
    """Raised when a non-transient error occurs (e.g., 401 auth, 400 bad request) and must not be retried."""
    pass


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

class LLMResponse(BaseModel):
    """Raw response DTO returned by an LLM provider.
    
    Contains generated text narrative and execution metadata.
    Does NOT contain citation payloads or knowledge base metadata.
    """
    content: str = Field(..., description="Raw generated narrative from the LLM.")
    model: str = Field(..., description="Identifier of the model used for generation.")
    latency_ms: float = Field(0.0, description="Generation latency in milliseconds.")
    usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage counts (prompt_tokens, completion_tokens, total_tokens).",
    )
    finish_reason: Optional[str] = Field(
        default=None,
        description="Reason generation terminated (e.g., 'stop', 'length', 'content_filter', or None if unavailable).",
    )


# ---------------------------------------------------------------------------
# Base Provider Interface
# ---------------------------------------------------------------------------

class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion from an input message sequence.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            **kwargs: Provider-specific overrides (temperature, max_tokens, etc.).

        Returns:
            LLMResponse containing generated content and metadata.

        Raises:
            LLMConfigurationError: If provider configuration is invalid.
            LLMTransientError: If a transient failure exhausts retries.
            LLMNonTransientError: If a non-transient error occurs.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the provider is configured and ready for generation."""
        pass
