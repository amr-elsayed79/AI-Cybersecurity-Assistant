"""Deterministic mock LLM provider for offline testing and software flow verification."""

import time
from typing import Any, Dict, List, Optional

from app.llm.base import BaseLLMProvider, LLMResponse


class MockLLMProvider(BaseLLMProvider):
    """Deterministic mock provider for offline testing.

    IMPORTANT ARCHITECTURAL SCOPE:
    This mock validates software execution flow, interface contracts, and
    pipeline plumbing only. It DOES NOT assess or validate answer quality,
    factual correctness, or RAG grounding.
    """

    def __init__(
        self,
        default_response: Optional[str] = None,
        model_name: str = "mock-cybersecurity-llm",
        simulated_latency_ms: float = 0.0,
        default_finish_reason: Optional[str] = "stop",
    ):
        """Initialize mock provider.

        Args:
            default_response: Text content to return. If None, synthesizes a deterministic response.
            model_name: Mock model identifier string.
            simulated_latency_ms: Optional artificial latency to simulate in milliseconds.
            default_finish_reason: Simulated termination reason (defaults to 'stop').
        """
        self.default_response = default_response
        self.model_name = model_name
        self.simulated_latency_ms = simulated_latency_ms
        self.default_finish_reason = default_finish_reason

    def is_available(self) -> bool:
        """Mock provider is always available without network or credentials."""
        return True

    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a deterministic completion from input messages.

        Args:
            messages: Input chat messages.
            **kwargs: Extra parameters (e.g. override response).

        Returns:
            LLMResponse with deterministic content and latency.
        """
        t0 = time.perf_counter()

        if self.simulated_latency_ms > 0:
            time.sleep(self.simulated_latency_ms / 1000.0)

        # Check for per-call override
        override_response = kwargs.get("mock_response")
        if override_response is not None:
            content = override_response
        elif self.default_response is not None:
            content = self.default_response
        else:
            # Deterministic echo of the last user query
            last_user_msg = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user_msg = m.get("content", "")
                    break
            content = f"Mock response for query: '{last_user_msg}'"

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0

        finish_reason = kwargs.get("finish_reason", self.default_finish_reason)
        return LLMResponse(
            content=content,
            model=self.model_name,
            latency_ms=round(elapsed_ms, 2),
            usage={
                "prompt_tokens": sum(len(m.get("content", "").split()) for m in messages),
                "completion_tokens": len(content.split()),
                "total_tokens": sum(len(m.get("content", "").split()) for m in messages) + len(content.split()),
            },
            finish_reason=finish_reason,
        )
