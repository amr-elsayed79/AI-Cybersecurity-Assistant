"""Groq LLM provider implementation using official Groq SDK with bounded retries and safe error handling."""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import groq
from groq import Groq

from app.config import settings
from app.llm.base import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMNonTransientError,
    LLMResponse,
    LLMTransientError,
)

logger = logging.getLogger(__name__)


class GroqProvider(BaseLLMProvider):
    """Production LLM provider integrating Groq API with robust error handling.

    Features:
    - Bounded exponential backoff for documented transient errors (RateLimitError, Timeout, Connection, 503).
    - Immediate fail-fast without retry for non-transient errors (Authentication, BadRequest, NotFound).
    - Injectable sleep function for deterministic, non-blocking unit testing.
    - Strict secret redaction: API keys are never printed, logged, or exposed in repr/str.
    """

    TRANSIENT_EXCEPTIONS = (
        groq.RateLimitError,
        groq.APITimeoutError,
        groq.APIConnectionError,
        groq.InternalServerError,
    )

    NON_TRANSIENT_EXCEPTIONS = (
        groq.AuthenticationError,
        groq.BadRequestError,
        groq.PermissionDeniedError,
        groq.NotFoundError,
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        client: Optional[Any] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ):
        """Initialize GroqProvider.

        Args:
            api_key: Groq API key. If omitted, loaded from settings.GROQ_API_KEY.
            model: Model identifier (default: settings.GROQ_MODEL).
            temperature: Sampling temperature (default: settings.GROQ_TEMPERATURE).
            max_tokens: Maximum tokens in response (default: settings.GROQ_MAX_TOKENS).
            timeout: Request timeout in seconds (default: settings.GROQ_TIMEOUT_SECONDS).
            max_retries: Maximum retries for transient errors (default: settings.GROQ_MAX_RETRIES).
            client: Optional pre-configured client (used for mocking in tests).
            sleep_fn: Optional sleep callable for tests (default: time.sleep).

        Raises:
            LLMConfigurationError: If no API key is provided or configured and client is not supplied.
        """
        self._api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self.temperature = temperature if temperature is not None else settings.GROQ_TEMPERATURE
        self.max_tokens = max_tokens or settings.GROQ_MAX_TOKENS
        self.timeout = timeout or settings.GROQ_TIMEOUT_SECONDS
        self.max_retries = max_retries if max_retries is not None else settings.GROQ_MAX_RETRIES
        self.sleep_fn = sleep_fn or time.sleep

        if client is not None:
            self._client = client
        elif self._api_key and self._api_key.strip():
            self._client = Groq(
                api_key=self._api_key.strip(),
                timeout=self.timeout,
            )
        else:
            raise LLMConfigurationError(
                "Groq API key not configured. Please set GROQ_API_KEY in your environment or local .env file."
            )

    def is_available(self) -> bool:
        """Check if Groq provider is configured with a valid client."""
        return self._client is not None

    def __repr__(self) -> str:
        """Safe representation without exposing secrets."""
        return (
            f"GroqProvider(model='{self.model}', "
            f"configured={bool(self._api_key or self._client)}, "
            f"max_retries={self.max_retries})"
        )

    def __str__(self) -> str:
        """Safe string conversion without exposing secrets."""
        return self.__repr__()

    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response via Groq API with bounded transient error retries.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            **kwargs: Overrides for temperature, max_tokens, or model.

        Returns:
            LLMResponse with generation narrative and latency.

        Raises:
            LLMNonTransientError: On non-transient errors (401, 400, etc.) without retrying.
            LLMTransientError: When transient errors exhaust the configured retry limit.
            LLMProviderError: On unexpected exceptions.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty.")

        target_model = kwargs.get("model", self.model)
        temp = kwargs.get("temperature", self.temperature)
        tokens = kwargs.get("max_tokens", self.max_tokens)

        last_transient_error: Optional[Exception] = None
        attempt = 0

        while attempt <= self.max_retries:
            t0 = time.perf_counter()
            try:
                completion = self._client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                )
                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0

                content = ""
                finish_reason = None
                if completion.choices and len(completion.choices) > 0:
                    choice = completion.choices[0]
                    content = choice.message.content or ""
                    raw_finish_reason = getattr(choice, "finish_reason", None)
                    if isinstance(raw_finish_reason, str) and raw_finish_reason.strip():
                        finish_reason = raw_finish_reason.strip()

                usage_dict = {}
                if hasattr(completion, "usage") and completion.usage:
                    usage_dict = {
                        "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(completion.usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(completion.usage, "total_tokens", 0) or 0,
                    }

                return LLMResponse(
                    content=content,
                    model=target_model,
                    latency_ms=round(latency_ms, 2),
                    usage=usage_dict,
                    finish_reason=finish_reason,
                )

            except self.NON_TRANSIENT_EXCEPTIONS as e:
                # Non-transient failure: do NOT retry, fail immediately
                sanitized_msg = f"Groq non-transient error ({e.__class__.__name__}): {getattr(e, 'message', str(e))}"
                logger.error("Non-transient Groq error encountered: %s", sanitized_msg)
                raise LLMNonTransientError(sanitized_msg) from e

            except self.TRANSIENT_EXCEPTIONS as e:
                # Transient failure: retry with backoff
                last_transient_error = e
                attempt += 1
                if attempt <= self.max_retries:
                    backoff = min(2.0 ** (attempt - 1), 8.0)
                    logger.warning(
                        "Transient Groq error (%s) on attempt %d/%d. Retrying in %.2fs...",
                        e.__class__.__name__,
                        attempt,
                        self.max_retries + 1,
                        backoff,
                    )
                    self.sleep_fn(backoff)
                else:
                    break

            except Exception as e:
                sanitized_msg = f"Unexpected Groq client failure ({e.__class__.__name__}): {str(e)}"
                logger.error("Unexpected error in Groq provider: %s", sanitized_msg)
                raise LLMNonTransientError(sanitized_msg) from e

        # Retries exhausted
        err_msg = (
            f"Groq transient failure exhausted {self.max_retries} retries. "
            f"Last error: {last_transient_error.__class__.__name__}: {getattr(last_transient_error, 'message', str(last_transient_error))}"
        )
        logger.error(err_msg)
        raise LLMTransientError(err_msg) from last_transient_error
