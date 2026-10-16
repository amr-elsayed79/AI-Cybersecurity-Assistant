"""Unit tests for LLM provider abstractions, MockLLMProvider, and GroqProvider.

All tests are 100% offline with zero live network calls and no real API keys.
"""

from unittest.mock import MagicMock
import httpx
import pytest

from app.llm.base import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMNonTransientError,
    LLMResponse,
    LLMTransientError,
)
from app.llm.factory import get_llm_provider
from app.llm.groq_provider import GroqProvider
from app.llm.mock_provider import MockLLMProvider
import groq


def _create_mock_groq_exception(exc_class, status_code: int, message: str = "Test error"):
    """Helper to instantiate Groq SDK exceptions with dummy httpx request/response."""
    dummy_req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    dummy_resp = httpx.Response(status_code, request=dummy_req)
    return exc_class(message=message, response=dummy_resp, body={"error": {"message": message}})


# ===========================================================================
# 1. MockLLMProvider Tests
# ===========================================================================

def test_mock_provider_scope_documentation():
    """Verify that MockLLMProvider explicitly documents its flow-only testing scope."""
    doc = MockLLMProvider.__doc__ or ""
    assert "validates software execution flow" in doc
    assert "DOES NOT assess or validate answer quality" in doc


def test_mock_provider_deterministic_response():
    """Verify deterministic generation and message parsing in MockLLMProvider."""
    provider = MockLLMProvider()
    assert provider.is_available() is True

    messages = [
        {"role": "system", "content": "You are a cybersecurity assistant."},
        {"role": "user", "content": "What is SQL Injection?"},
    ]
    response = provider.generate(messages)

    assert isinstance(response, LLMResponse)
    assert "SQL Injection" in response.content
    assert response.model == "mock-cybersecurity-llm"
    assert response.latency_ms >= 0.0
    assert response.usage["prompt_tokens"] > 0
    assert response.usage["completion_tokens"] > 0


def test_mock_provider_custom_response_and_override():
    """Verify canned response and per-call override behavior in MockLLMProvider."""
    provider = MockLLMProvider(default_response="Default security guidance.")
    response1 = provider.generate([{"role": "user", "content": "Hi"}])
    assert response1.content == "Default security guidance."

    response2 = provider.generate(
        [{"role": "user", "content": "Hi"}],
        mock_response="Explicit override text.",
    )
    assert response2.content == "Explicit override text."


# ===========================================================================
# 2. GroqProvider Configuration & Secret Redaction Tests
# ===========================================================================

def test_groq_provider_missing_api_key_raises_config_error(monkeypatch):
    """Verify that GroqProvider raises LLMConfigurationError if no key is supplied."""
    from app.config import settings
    monkeypatch.setattr(settings, "GROQ_API_KEY", None)

    with pytest.raises(LLMConfigurationError) as exc_info:
        GroqProvider(api_key=None, client=None)

    assert "Groq API key not configured" in str(exc_info.value)


def test_groq_provider_secret_redaction():
    """Verify that string representations and logs NEVER expose the raw API key."""
    fake_key = "gsk_SUPER_SECRET_VALUE_1234567890"
    mock_client = MagicMock()
    provider = GroqProvider(api_key=fake_key, client=mock_client)

    repr_str = repr(provider)
    str_str = str(provider)

    assert fake_key not in repr_str
    assert fake_key not in str_str
    assert "configured=True" in repr_str
    assert "configured=True" in str_str


def test_groq_provider_rejects_empty_messages():
    """Verify ValueError when empty messages list is passed to generate."""
    mock_client = MagicMock()
    provider = GroqProvider(api_key="fake_key", client=mock_client)

    with pytest.raises(ValueError) as exc_info:
        provider.generate([])
    assert "Messages list cannot be empty" in str(exc_info.value)


# ===========================================================================
# 3. Non-Transient Error Handling (Zero Retries)
# ===========================================================================

def test_groq_provider_authentication_error_fails_fast_no_retry():
    """Verify 401 AuthenticationError is NEVER retried and raises LLMNonTransientError."""
    mock_client = MagicMock()
    auth_err = _create_mock_groq_exception(groq.AuthenticationError, 401, "Invalid API Key")
    mock_client.chat.completions.create.side_effect = auth_err

    sleep_calls = []
    provider = GroqProvider(
        api_key="fake_key",
        client=mock_client,
        max_retries=3,
        sleep_fn=sleep_calls.append,
    )

    with pytest.raises(LLMNonTransientError) as exc_info:
        provider.generate([{"role": "user", "content": "Test prompt"}])

    assert "AuthenticationError" in str(exc_info.value)
    assert mock_client.chat.completions.create.call_count == 1
    assert len(sleep_calls) == 0  # Zero retries


def test_groq_provider_bad_request_error_fails_fast_no_retry():
    """Verify 400 BadRequestError is NEVER retried and raises LLMNonTransientError."""
    mock_client = MagicMock()
    bad_req_err = _create_mock_groq_exception(groq.BadRequestError, 400, "Malformed request body")
    mock_client.chat.completions.create.side_effect = bad_req_err

    sleep_calls = []
    provider = GroqProvider(
        api_key="fake_key",
        client=mock_client,
        max_retries=2,
        sleep_fn=sleep_calls.append,
    )

    with pytest.raises(LLMNonTransientError) as exc_info:
        provider.generate([{"role": "user", "content": "Test prompt"}])

    assert "BadRequestError" in str(exc_info.value)
    assert mock_client.chat.completions.create.call_count == 1
    assert len(sleep_calls) == 0


# ===========================================================================
# 4. Transient Error Handling & Retries (Mock Sleep)
# ===========================================================================

def test_groq_provider_rate_limit_retry_success():
    """Verify 429 RateLimitError triggers backoff retry and succeeds on attempt 2."""
    mock_client = MagicMock()
    rate_err = _create_mock_groq_exception(groq.RateLimitError, 429, "Rate limit reached")

    # Mock successful response
    mock_choice = MagicMock()
    mock_choice.message.content = "Grounded response after rate limit."
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage.prompt_tokens = 15
    mock_completion.usage.completion_tokens = 25
    mock_completion.usage.total_tokens = 40

    mock_client.chat.completions.create.side_effect = [rate_err, mock_completion]

    sleep_calls = []
    provider = GroqProvider(
        api_key="fake_key",
        client=mock_client,
        max_retries=2,
        sleep_fn=sleep_calls.append,
    )

    response = provider.generate([{"role": "user", "content": "Prompt"}])

    assert response.content == "Grounded response after rate limit."
    assert mock_client.chat.completions.create.call_count == 2
    assert len(sleep_calls) == 1
    assert sleep_calls[0] > 0.0  # Backoff duration applied


def test_groq_provider_timeout_and_connection_retries():
    """Verify APITimeoutError and APIConnectionError are retried with backoff."""
    mock_client = MagicMock()
    timeout_err = groq.APITimeoutError(httpx.Request("POST", "https://api.groq.com"))
    conn_err = groq.APIConnectionError(message="Connection refused", request=httpx.Request("POST", "https://api.groq.com"))

    mock_choice = MagicMock()
    mock_choice.message.content = "Success after network recovery."
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = None

    mock_client.chat.completions.create.side_effect = [timeout_err, conn_err, mock_completion]

    sleep_calls = []
    provider = GroqProvider(
        api_key="fake_key",
        client=mock_client,
        max_retries=3,
        sleep_fn=sleep_calls.append,
    )

    response = provider.generate([{"role": "user", "content": "Prompt"}])

    assert response.content == "Success after network recovery."
    assert mock_client.chat.completions.create.call_count == 3
    assert len(sleep_calls) == 2


def test_groq_provider_retry_exhaustion_raises_transient_error():
    """Verify exhausting retries on 503 InternalServerError raises LLMTransientError."""
    mock_client = MagicMock()
    server_err = _create_mock_groq_exception(groq.InternalServerError, 503, "Service Unavailable")
    mock_client.chat.completions.create.side_effect = server_err

    sleep_calls = []
    max_retries = 2
    provider = GroqProvider(
        api_key="fake_key",
        client=mock_client,
        max_retries=max_retries,
        sleep_fn=sleep_calls.append,
    )

    with pytest.raises(LLMTransientError) as exc_info:
        provider.generate([{"role": "user", "content": "Prompt"}])

    assert "transient failure exhausted 2 retries" in str(exc_info.value)
    # Total attempts = 1 initial + 2 retries = 3
    assert mock_client.chat.completions.create.call_count == 3
    assert len(sleep_calls) == max_retries


# ===========================================================================
# 5. Factory Selection Tests
# ===========================================================================

def test_factory_returns_mock_provider():
    """Verify get_llm_provider('mock') returns MockLLMProvider."""
    provider = get_llm_provider("mock", default_response="Factory mock test.")
    assert isinstance(provider, MockLLMProvider)
    assert provider.is_available() is True


def test_factory_returns_groq_provider_with_mock_client():
    """Verify get_llm_provider('groq') returns GroqProvider using mocked client."""
    mock_client = MagicMock()
    provider = get_llm_provider("groq", api_key="fake_key", client=mock_client)
    assert isinstance(provider, GroqProvider)
    assert provider.is_available() is True


def test_factory_rejects_unsupported_provider():
    """Verify get_llm_provider raises LLMConfigurationError for unknown providers."""
    with pytest.raises(LLMConfigurationError) as exc_info:
        get_llm_provider("unsupported_provider")

    assert "Unsupported LLM provider" in str(exc_info.value)


# ===========================================================================
# 6. Finish Reason Handling Tests (Step 2)
# ===========================================================================

def test_mock_provider_finish_reason_handling():
    """Verify MockLLMProvider returns default finish_reason='stop', supports overrides and None."""
    # 1. Default finish_reason is 'stop'
    p1 = MockLLMProvider()
    resp1 = p1.generate([{"role": "user", "content": "Hello"}])
    assert resp1.finish_reason == "stop"

    # 2. Configured default_finish_reason=None
    p2 = MockLLMProvider(default_finish_reason=None)
    resp2 = p2.generate([{"role": "user", "content": "Hello"}])
    assert resp2.finish_reason is None

    # 3. Per-call override to 'length'
    resp3 = p1.generate([{"role": "user", "content": "Hello"}], finish_reason="length")
    assert resp3.finish_reason == "length"

    # 4. Per-call override to 'content_filter'
    resp4 = p1.generate([{"role": "user", "content": "Hello"}], finish_reason="content_filter")
    assert resp4.finish_reason == "content_filter"


def test_groq_provider_carries_finish_reasons():
    """Verify GroqProvider accurately extracts stop, length, content_filter, or None."""
    mock_client = MagicMock()
    provider = GroqProvider(api_key="fake_key", client=mock_client)

    # 1. Normal 'stop'
    choice_stop = MagicMock()
    choice_stop.message.content = "Answer text."
    choice_stop.finish_reason = "stop"
    completion_stop = MagicMock(choices=[choice_stop], usage=None)
    mock_client.chat.completions.create.return_value = completion_stop

    r1 = provider.generate([{"role": "user", "content": "Prompt"}])
    assert r1.finish_reason == "stop"
    assert r1.content == "Answer text."

    # 2. Truncation 'length'
    choice_len = MagicMock()
    choice_len.message.content = "Truncated..."
    choice_len.finish_reason = "length"
    completion_len = MagicMock(choices=[choice_len], usage=None)
    mock_client.chat.completions.create.return_value = completion_len

    r2 = provider.generate([{"role": "user", "content": "Prompt"}])
    assert r2.finish_reason == "length"
    assert r2.content == "Truncated..."

    # 3. Content filter 'content_filter'
    choice_filter = MagicMock()
    choice_filter.message.content = ""
    choice_filter.finish_reason = "content_filter"
    completion_filter = MagicMock(choices=[choice_filter], usage=None)
    mock_client.chat.completions.create.return_value = completion_filter

    r3 = provider.generate([{"role": "user", "content": "Prompt"}])
    assert r3.finish_reason == "content_filter"

    # 4. Missing finish_reason attribute (None, not inferred as stop)
    choice_none = MagicMock(spec=["message"])
    choice_none.message = MagicMock(content="No finish reason provided.")
    completion_none = MagicMock(choices=[choice_none], usage=None)
    mock_client.chat.completions.create.return_value = completion_none

    r4 = provider.generate([{"role": "user", "content": "Prompt"}])
    assert r4.finish_reason is None
    assert r4.content == "No finish reason provided."

