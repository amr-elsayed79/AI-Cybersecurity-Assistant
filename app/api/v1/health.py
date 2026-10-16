"""Health check endpoint for the AI Cybersecurity Assistant."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Response, status

from app.config import settings
from app.models.schemas import HealthResponse
from app.rag.ingestion import compute_file_sha256
from app.rag.retriever import QdrantVectorStore, get_shared_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

_vector_store: Optional[QdrantVectorStore] = None


def get_health_vector_store() -> QdrantVectorStore:
    """Obtain or initialize vector store reference for health checks."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store
    return get_shared_vector_store()


def set_health_vector_store(store: Optional[QdrantVectorStore]) -> None:
    """Inject a vector store instance (useful for testing)."""
    global _vector_store
    _vector_store = store


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check system health, vector store indexing, Master KB integrity, and LLM configuration without billable API calls.",
)
def get_health(
    response: Response,
    vector_store: QdrantVectorStore = Depends(get_health_vector_store),
) -> HealthResponse:
    """Perform non-billable, local health inspection."""
    # 1. Master KB File & Checksum Integrity
    kb_verified = False
    try:
        if settings.MASTER_KB_PATH.exists() and settings.MASTER_KB_PATH.is_file():
            current_hash = compute_file_sha256(settings.MASTER_KB_PATH)
            kb_verified = (current_hash.lower() == settings.BASELINE_KB_SHA256.lower())
    except Exception as e:
        logger.warning("Health check: failed to verify Master KB checksum: %s", type(e).__name__)
        kb_verified = False

    # 2. Vector Store Connectivity and Point Count
    vector_status = "unavailable"
    point_count = 0
    try:
        point_count = vector_store.count_points()
        if point_count >= 588:
            vector_status = "ready"
        elif point_count > 0:
            vector_status = "partial"
        else:
            vector_status = "empty"
    except Exception as e:
        logger.warning("Health check: vector store unavailable: %s", type(e).__name__)
        vector_status = "unreachable"

    # 3. LLM Provider Configuration (Zero external calls)
    provider_name = settings.LLM_PROVIDER.lower()
    if provider_name == "mock":
        llm_configured = True
    elif provider_name == "groq":
        llm_configured = bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip())
    else:
        llm_configured = False

    # 4. Overall Health Determination
    is_healthy = (
        kb_verified
        and vector_status == "ready"
        and point_count >= 588
        and llm_configured
    )
    overall_status = "healthy" if is_healthy else "degraded"

    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status=overall_status,
        version=settings.VERSION,
        master_kb_checksum_verified=kb_verified,
        vector_store_status=vector_status,
        indexed_point_count=point_count,
        llm_provider=provider_name,
        llm_configured=llm_configured,
    )
