"""Lightweight in-memory IP rate limiter and payload size guard for chat endpoints.

Designed for single-process ASGI deployment (e.g. Uvicorn with 1 worker on Render).
Zero external dependencies, thread-safe, sliding-window algorithm.
"""

from collections import defaultdict, deque
import threading
import time
from typing import Deque, Dict, Tuple

from fastapi import HTTPException, Request, status

from app.config import settings


class InMemoryRateLimiter:
    """Thread-safe sliding-window IP rate limiter."""

    def __init__(self, requests_per_minute: int = 15, window_seconds: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self._requests: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def get_client_ip(self, request: Request) -> str:
        """Extract client IP from X-Forwarded-For (reverse proxy) or socket host."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Client IP is the first entry in comma-separated chain
            return forwarded_for.split(",")[0].strip()
        if request.client and request.client.host:
            return request.client.host
        return "127.0.0.1"

    def is_rate_limited(self, ip: str) -> Tuple[bool, int]:
        """Check if client IP has exceeded the allowed rate within the sliding window.

        Returns:
            Tuple of (is_limited, retry_after_seconds).
        """
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            # Periodic cleanup of idle IPs every 5 minutes to prevent memory leaks
            if now - self._last_cleanup > 300:
                self._cleanup_stale(window_start)
                self._last_cleanup = now

            timestamps = self._requests[ip]
            # Prune timestamps outside the current window
            while timestamps and timestamps[0] <= window_start:
                timestamps.popleft()

            limit = settings.RATE_LIMIT_REQUESTS_PER_MINUTE
            if len(timestamps) >= limit:
                oldest = timestamps[0]
                retry_after = max(1, int(oldest + self.window_seconds - now))
                return True, retry_after

            timestamps.append(now)
            return False, 0

    def _cleanup_stale(self, window_start: float) -> None:
        stale_keys = [ip for ip, ts in self._requests.items() if not ts or ts[-1] <= window_start]
        for ip in stale_keys:
            del self._requests[ip]

    def reset(self) -> None:
        """Reset all tracked rate limits (useful for testing)."""
        with self._lock:
            self._requests.clear()
            self._last_cleanup = time.time()


# Shared singleton rate limiter instance
rate_limiter = InMemoryRateLimiter()


def check_chat_rate_limit(request: Request) -> None:
    """FastAPI dependency enforcing sliding-window rate limit and payload size on chat endpoints."""
    # 1. Payload size check
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > settings.MAX_REQUEST_BODY_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Request payload exceeds maximum allowed size ({settings.MAX_REQUEST_BODY_BYTES} bytes).",
                )
        except ValueError:
            pass

    # 2. Rate limiting check
    if not settings.RATE_LIMIT_ENABLED:
        return

    client_ip = rate_limiter.get_client_ip(request)

    # Exempt localhost/testclient unless explicitly testing rate limits
    if client_ip in ("testclient", "127.0.0.1", "localhost") and not settings.RATE_LIMIT_TESTING:
        return

    is_limited, retry_after = rate_limiter.is_rate_limited(client_ip)
    if is_limited:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_REQUESTS_PER_MINUTE} "
                f"requests per minute allowed. Please try again in {retry_after} seconds."
            ),
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(settings.RATE_LIMIT_REQUESTS_PER_MINUTE),
                "X-RateLimit-Remaining": "0",
            },
        )
