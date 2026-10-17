"""Automated integration tests for FastAPI chat and health endpoints (Phase 4.3).

All tests run 100% offline with zero network calls and zero Groq API calls.
"""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.api.rate_limiter import rate_limiter
from app.api.v1.chat import get_conversation_store, get_pipeline, set_pipeline
from app.api.v1.health import get_health_vector_store, set_health_vector_store
from app.config import Settings, settings
from app.llm.base import LLMProviderError, LLMResponse
from app.main import app, create_app
from app.memory.conversation_store import InMemoryConversationStore, conversation_store
from app.models.schemas import (
    GroundingStatus,
    RAGResponse,
    RetrievalResult,
    RetrievedChunk,
    SourceCitation,
)
from app.rag.pipeline import RAGChatPipeline


@pytest.fixture(autouse=True)
def reset_dependencies():
    """Reset global dependencies, conversation store, and rate limiter before and after each test."""
    conversation_store.clear()
    rate_limiter.reset()
    set_pipeline(None)
    set_health_vector_store(None)
    yield
    conversation_store.clear()
    rate_limiter.reset()
    set_pipeline(None)
    set_health_vector_store(None)


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def _make_sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="sql-injection_prevention",
        vulnerability_id="sql-injection",
        name="SQL Injection",
        category="Injection",
        section="prevention_and_remediation",
        severity_level="Critical",
        content="Use parameterized queries and prepared statements [CWE-89].",
        score=0.92,
        cwe=["CWE-89"],
        owasp=["A03:2021-Injection"],
        metadata={"verified": True},
    )


def test_root_informational_endpoint(client):
    """Verify GET / returns application status and links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "version" in data
    assert data["health_endpoint"] == "/health"
    assert data["chat_endpoint"] == "/api/v1/chat"


def test_health_check_healthy(client):
    """Verify GET /health returns 200 and healthy metrics when store is ready."""
    mock_vs = MagicMock()
    mock_vs.count_points.return_value = 588
    set_health_vector_store(mock_vs)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["master_kb_checksum_verified"] is True
    assert data["vector_store_status"] == "ready"
    assert data["indexed_point_count"] == 588
    assert data["llm_configured"] is True
    # Verify no secret key is exposed
    assert "GROQ_API_KEY" not in str(data)
    assert "api_key" not in str(data)


def test_health_check_api_v1_route(client):
    """Verify GET /api/v1/health functions identically to root /health."""
    mock_vs = MagicMock()
    mock_vs.count_points.return_value = 588
    set_health_vector_store(mock_vs)

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_check_degraded_when_vector_store_empty(client):
    """Verify GET /health returns 503 and degraded status when vector store is empty."""
    mock_vs = MagicMock()
    mock_vs.count_points.return_value = 0
    set_health_vector_store(mock_vs)

    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["vector_store_status"] == "empty"
    assert data["indexed_point_count"] == 0


def test_health_check_degraded_when_vector_store_unreachable(client):
    """Verify GET /health handles store connection exceptions gracefully without 500 crash."""
    mock_vs = MagicMock()
    mock_vs.count_points.side_effect = RuntimeError("Connection refused to Qdrant")
    set_health_vector_store(mock_vs)

    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["vector_store_status"] == "unreachable"


def test_chat_endpoint_successful_grounded_response(client):
    """Verify POST /api/v1/chat processes query and returns complete ChatAPIResponse."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        query="What is SQL injection?",
        top_k=1,
        latency_ms=1.5,
        matches=[_make_sample_chunk()],
    )

    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content="SQL injection can be mitigated using parameterized queries [E1].",
        model="mock-cyber-v1",
    )

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    set_pipeline(pipeline)

    payload = {"message": "What is SQL injection?"}
    response = client.post("/api/v1/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "SQL injection can be mitigated" in data["answer"]
    assert data["grounded"] is True
    assert data["grounding_status"] == "grounded"
    assert len(data["retrieved_sources"]) == 1
    assert len(data["cited_sources"]) == 1
    assert data["cited_sources"][0]["chunk_id"] == "sql-injection_prevention"
    assert "conversation_id" in data
    assert len(data["conversation_id"]) > 0


def test_chat_endpoint_query_alias(client):
    """Verify POST /api/v1/chat accepts 'query' field as an alias for 'message'."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        query="Explain XSS",
        top_k=1,
        latency_ms=1.0,
        matches=[_make_sample_chunk()],
    )

    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content="Cross-site scripting guidance [E1].",
        model="mock-cyber-v1",
    )

    set_pipeline(RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm))

    payload = {"query": "Explain XSS"}
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    assert "Cross-site scripting guidance" in response.json()["answer"]


def test_chat_endpoint_alias_route(client):
    """Verify POST /api/chat alias functions identically to /api/v1/chat."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        query="Test query",
        top_k=1,
        latency_ms=1.0,
        matches=[_make_sample_chunk()],
    )

    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content="Response from alias route [E1].",
        model="mock-cyber-v1",
    )

    set_pipeline(RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm))

    payload = {"message": "Test query"}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    assert "Response from alias route" in response.json()["answer"]


def test_chat_multi_turn_session_preservation(client):
    """Verify conversation context is retained across successive turns with same conversation_id."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        query="Turn query",
        top_k=1,
        latency_ms=1.0,
        matches=[_make_sample_chunk()],
    )

    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content="Safe answer [E1].",
        model="mock-cyber-v1",
    )

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    set_pipeline(pipeline)

    # Turn 1
    resp1 = client.post("/api/v1/chat", json={"message": "First message"})
    assert resp1.status_code == 200
    conv_id = resp1.json()["conversation_id"]

    # Verify turn 1 was saved in memory
    history = conversation_store.get_history(conv_id)
    assert len(history) == 2
    assert history[0].content == "First message"
    assert history[1].content == "Safe answer [E1]."

    # Turn 2 with same conversation_id
    resp2 = client.post("/api/v1/chat", json={"message": "Follow-up message", "conversation_id": conv_id})
    assert resp2.status_code == 200
    assert resp2.json()["conversation_id"] == conv_id

    # Verify history now has 4 turns
    history_after = conversation_store.get_history(conv_id)
    assert len(history_after) == 4
    assert history_after[2].content == "Follow-up message"


def test_chat_input_validation_empty_and_whitespace(client):
    """Verify empty or whitespace queries return HTTP 422 with structured errors."""
    resp_empty = client.post("/api/v1/chat", json={"message": ""})
    assert resp_empty.status_code == 422

    resp_ws = client.post("/api/v1/chat", json={"message": "   "})
    assert resp_ws.status_code == 422

    resp_none = client.post("/api/v1/chat", json={})
    assert resp_none.status_code == 422


def test_chat_input_validation_oversized_query(client):
    """Verify queries exceeding 2000 characters return HTTP 422."""
    oversized = "A" * 2001
    response = client.post("/api/v1/chat", json={"message": oversized})
    assert response.status_code == 422


def test_chat_input_validation_invalid_conversation_id(client):
    """Verify malformed conversation IDs return HTTP 422."""
    bad_ids = [
        "../../../etc/passwd",
        "session;DROP TABLE",
        "id with spaces",
        "a" * 65,  # Exceeds max length
    ]
    for bad_id in bad_ids:
        response = client.post("/api/v1/chat", json={"message": "hello", "conversation_id": bad_id})
        assert response.status_code == 422, f"Expected 422 for bad conversation_id: {bad_id}"


def test_chat_error_handling_llm_provider_error_masks_details(client):
    """Verify LLMProviderError returns 503 without leaking secrets or stack traces."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        query="Error test",
        top_k=1,
        latency_ms=1.0,
        matches=[_make_sample_chunk()],
    )

    mock_llm = MagicMock()
    mock_llm.generate.side_effect = LLMProviderError("Secret groq api error with internal token gsk_secret_token_12345")

    set_pipeline(RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm))

    response = client.post("/api/v1/chat", json={"message": "Error test"})
    assert response.status_code == 503
    data = response.json()
    assert "gsk_secret_token_12345" not in str(data)
    assert "LLM service is temporarily unavailable" in data["detail"]


def test_chat_error_handling_unhandled_exception_masks_details(client):
    """Verify unhandled server exceptions return 500 without leaking filesystem paths or traces."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = RuntimeError("Fatal disk read error at D:/secret/internal/database.bin")

    set_pipeline(RAGChatPipeline(retriever=mock_retriever, llm_provider=MagicMock()))

    response = client.post("/api/v1/chat", json={"message": "Unhandled error test"})
    assert response.status_code == 500
    data = response.json()
    assert "database.bin" not in str(data)
    assert "D:/" not in str(data)
    assert "unexpected error" in data["detail"].lower()


def test_cors_headers_handling(client):
    """Verify CORS preflight and origin headers are respected."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }
    response = client.options("/api/v1/chat", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_scanner_custom_origin_cors_support():
    """Verify Settings parses comma-separated and JSON list CORS origins for scanner integration."""
    # 1. Comma-separated format in .env
    s_comma = Settings(CORS_ALLOW_ORIGINS="http://localhost:3000,http://scanner.local:5173")
    assert s_comma.CORS_ALLOW_ORIGINS == ["http://localhost:3000", "http://scanner.local:5173"]

    # 2. JSON array format in .env
    s_json = Settings(CORS_ALLOW_ORIGINS='["http://localhost:3000", "http://scanner.local:5173"]')
    assert s_json.CORS_ALLOW_ORIGINS == ["http://localhost:3000", "http://scanner.local:5173"]

    # 3. Native list
    s_list = Settings(CORS_ALLOW_ORIGINS=["http://scanner.local:5173"])
    assert s_list.CORS_ALLOW_ORIGINS == ["http://scanner.local:5173"]


def test_scanner_ask_ai_api_contract_and_multi_turn_flow(client):
    """Verify Ask AI integration contract for scanner website: predictable schema, session continuity, error masking."""
    # Mock pipeline to guarantee zero live Groq calls and fast deterministic testing
    mock_retriever = MagicMock()
    chunk = _make_sample_chunk()
    mock_retriever.retrieve.return_value = RetrievalResult(
        query="How do I prevent SQL Injection?",
        top_k=1,
        latency_ms=12.5,
        matches=[chunk],
    )

    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content="To prevent SQL injection, use parameterized queries [E1].",
        raw_content="To prevent SQL injection, use parameterized queries [E1].",
        model="mock-cyber-v1",
        finish_reason="stop",
        latency_ms=85.0,
        usage={"prompt_tokens": 120, "completion_tokens": 25, "total_tokens": 145},
    )

    set_pipeline(RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm))

    # Turn 1: First user query from scanner website without conversation_id
    resp_turn1 = client.post(
        "/api/v1/chat",
        json={"message": "How do I prevent SQL Injection?"},
    )
    assert resp_turn1.status_code == 200
    data1 = resp_turn1.json()

    # Verify complete API contract fields
    assert "answer" in data1
    assert "grounded" in data1
    assert "grounding_status" in data1
    assert "is_site_specific" in data1
    assert "language" in data1
    assert "retrieved_sources" in data1
    assert "cited_sources" in data1
    assert "conversation_id" in data1
    assert "retrieval_latency_ms" in data1
    assert "llm_latency_ms" in data1
    assert "total_latency_ms" in data1

    assert data1["grounded"] is True
    assert data1["grounding_status"] == "grounded"
    assert data1["is_site_specific"] is False
    assert len(data1["cited_sources"]) == 1
    assert data1["cited_sources"][0]["chunk_id"] == "sql-injection_prevention"

    conv_id = data1["conversation_id"]
    assert conv_id is not None
    assert len(conv_id) > 0

    # Turn 2: Follow-up question passing existing conversation_id
    mock_retriever.retrieve.return_value = RetrievalResult(
        query="What about ORM frameworks?",
        top_k=1,
        latency_ms=10.0,
        matches=[chunk],
    )
    mock_llm.generate.return_value = LLMResponse(
        content="Modern ORMs use parameterized queries under the hood [E1].",
        raw_content="Modern ORMs use parameterized queries under the hood [E1].",
        model="mock-cyber-v1",
        finish_reason="stop",
        latency_ms=75.0,
        usage={"prompt_tokens": 180, "completion_tokens": 20, "total_tokens": 200},
    )

    resp_turn2 = client.post(
        "/api/v1/chat",
        json={"message": "What about ORM frameworks?", "conversation_id": conv_id},
    )
    assert resp_turn2.status_code == 200
    data2 = resp_turn2.json()
    assert data2["conversation_id"] == conv_id

    # Verify memory contains the multi-turn exchange
    store = get_conversation_store()
    history = store.get_history(conv_id)
    assert len(history) == 4  # user1, assistant1, user2, assistant2

    # Validation tests for invalid inputs from scanner frontend
    # Empty message
    resp_empty = client.post("/api/v1/chat", json={"message": ""})
    assert resp_empty.status_code == 422
    assert "errors" in resp_empty.json()

    # Whitespace-only message
    resp_ws = client.post("/api/v1/chat", json={"message": "    \n   "})
    assert resp_ws.status_code == 422

    # Message exceeding 2000 chars
    resp_oversize = client.post("/api/v1/chat", json={"message": "A" * 2001})
    assert resp_oversize.status_code == 422

    # Invalid conversation_id format (spaces / special symbols)
    resp_bad_conv = client.post(
        "/api/v1/chat",
        json={"message": "Test valid message", "conversation_id": "invalid session id with spaces!"},
    )
    assert resp_bad_conv.status_code == 422

    # Security check: zero credentials exposed
    assert "GROQ_API_KEY" not in str(data1)
    assert "gsk_" not in str(data1)
    assert "GROQ_API_KEY" not in str(data2)
    assert "gsk_" not in str(data2)


def test_health_then_chat_unmocked_lifecycle_preservation(client):
    """Verify GET /health followed by POST /chat shares the vector store without lock collision.

    Uses realistic QdrantVectorStore and pipeline dependencies with safe offline execution.
    Verifies that neither endpoint throws RuntimeError due to multiple client lock attempts.
    """
    # 1. First call: health check (initializes shared vector store if not yet loaded)
    resp_health_1 = client.get("/api/v1/health")
    assert resp_health_1.status_code == 200
    health_data_1 = resp_health_1.json()
    assert health_data_1["status"] == "healthy"
    assert health_data_1["vector_store_status"] == "ready"
    assert health_data_1["indexed_point_count"] >= 588

    # 2. Second call: chat endpoint using realistic retrieval over indexed KB
    resp_chat_1 = client.post(
        "/api/v1/chat",
        json={"message": "What is SQL injection, and what are two defensive ways to prevent it?"},
    )
    assert resp_chat_1.status_code == 200
    chat_data_1 = resp_chat_1.json()
    assert chat_data_1["grounding_status"] in (
        GroundingStatus.GROUNDED.value,
        GroundingStatus.SITE_SPECIFIC_QUALIFIED.value,
    )
    assert len(chat_data_1["retrieved_sources"]) > 0

    # 3. Third call: health check again (must still succeed without lock collision or degraded status)
    resp_health_2 = client.get("/health")
    assert resp_health_2.status_code == 200
    assert resp_health_2.json()["status"] == "healthy"

    # 4. Fourth call: subsequent chat turn in same session
    resp_chat_2 = client.post(
        "/api/v1/chat",
        json={
            "message": "Can parameterized queries prevent blind SQL injection?",
            "conversation_id": chat_data_1["conversation_id"],
        },
    )
    assert resp_chat_2.status_code == 200
    assert resp_chat_2.json()["grounding_status"] in (
        GroundingStatus.GROUNDED.value,
        GroundingStatus.SITE_SPECIFIC_QUALIFIED.value,
    )
    assert resp_chat_2.json()["conversation_id"] == chat_data_1["conversation_id"]


def test_chat_then_health_unmocked_reverse_order(client):
    """Verify POST /chat followed by GET /health works when chat is invoked first.

    Ensures that initializing the pipeline first does not block subsequent health checks.
    """
    # 1. Chat first
    resp_chat = client.post(
        "/api/v1/chat",
        json={"message": "What is SQL injection, and what are two defensive ways to prevent it?"},
    )
    assert resp_chat.status_code == 200
    assert resp_chat.json()["grounding_status"] in (
        GroundingStatus.GROUNDED.value,
        GroundingStatus.SITE_SPECIFIC_QUALIFIED.value,
    )

    # 2. Health second
    resp_health = client.get("/api/v1/health")
    assert resp_health.status_code == 200
    health_data = resp_health.json()
    assert health_data["status"] == "healthy"
    assert health_data["vector_store_status"] == "ready"
    assert health_data["indexed_point_count"] >= 588


def test_root_browser_redirects_to_dashboard(client):
    """Verify GET / with text/html Accept header redirects to /dashboard/."""
    response = client.get("/", headers={"accept": "text/html,application/xhtml+xml"}, follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/dashboard/"


def test_chat_rate_limiting_enforcement(client, monkeypatch):
    """Verify chat endpoint rejects requests with HTTP 429 when rate limit is exceeded."""
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_TESTING", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 2)

    # 1. First request succeeds
    resp1 = client.post(
        "/api/v1/chat",
        json={"message": "What is SQL injection?"},
        headers={"x-forwarded-for": "203.0.113.195"},
    )
    assert resp1.status_code == 200

    # 2. Second request succeeds
    resp2 = client.post(
        "/api/v1/chat",
        json={"message": "What is XSS?"},
        headers={"x-forwarded-for": "203.0.113.195"},
    )
    assert resp2.status_code == 200

    # 3. Third request within same minute is blocked
    resp3 = client.post(
        "/api/v1/chat",
        json={"message": "What is CSRF?"},
        headers={"x-forwarded-for": "203.0.113.195"},
    )
    assert resp3.status_code == 429
    assert "Rate limit exceeded" in resp3.json()["detail"]
    assert "Retry-After" in resp3.headers
    assert resp3.headers["X-RateLimit-Remaining"] == "0"

    # Different IP should succeed
    resp_other_ip = client.post(
        "/api/v1/chat",
        json={"message": "What is SSRF?"},
        headers={"x-forwarded-for": "198.51.100.42"},
    )
    assert resp_other_ip.status_code == 200


def test_chat_payload_size_limit(client):
    """Verify chat endpoint rejects requests exceeding MAX_REQUEST_BODY_BYTES with HTTP 413."""
    oversized_length = str(settings.MAX_REQUEST_BODY_BYTES + 500)
    resp = client.post(
        "/api/v1/chat",
        json={"message": "Testing payload size limit."},
        headers={"content-length": oversized_length},
    )
    assert resp.status_code == 413
    assert "exceeds maximum allowed size" in resp.json()["detail"]
