"""Chat API endpoint orchestrating multi-turn conversation and RAG pipeline execution."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.rate_limiter import check_chat_rate_limit
from app.api.v1.health import get_health_vector_store
from app.llm.base import LLMProviderError
from app.memory.conversation_store import InMemoryConversationStore, conversation_store as default_store
from app.models.schemas import ChatAPIResponse, ChatRequest
from app.rag.pipeline import RAGChatPipeline
from app.rag.retriever import QdrantVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

# Module-level pipeline dependency (can be overridden in tests)
_pipeline: Optional[RAGChatPipeline] = None


def get_pipeline(
    vector_store: Optional[QdrantVectorStore] = Depends(get_health_vector_store),
) -> RAGChatPipeline:
    """Obtain or initialize RAGChatPipeline instance with shared vector store."""
    global _pipeline
    if _pipeline is None:
        resolved_vs = (
            vector_store
            if isinstance(vector_store, QdrantVectorStore) or hasattr(vector_store, "retrieve")
            else get_health_vector_store()
        )
        _pipeline = RAGChatPipeline(retriever=resolved_vs)
    return _pipeline


def set_pipeline(pipeline: Optional[RAGChatPipeline]) -> None:
    """Inject a pipeline instance (useful for testing with mocks)."""
    global _pipeline
    _pipeline = pipeline


def get_conversation_store() -> InMemoryConversationStore:
    """Obtain conversation store dependency."""
    return default_store


@router.post(
    "/chat",
    response_model=ChatAPIResponse,
    summary="Chat with AI Cybersecurity Assistant",
    description=(
        "Submit a cybersecurity question with optional conversation session ID for bounded multi-turn context. "
        "Strictly intended for local execution (127.0.0.1). Omitted conversation_id generates a new session; "
        "invalid format returns HTTP 422. Memory is in-process and volatile."
    ),
    status_code=status.HTTP_200_OK,
)
def chat_endpoint(
    request: ChatRequest,
    pipeline: RAGChatPipeline = Depends(get_pipeline),
    store: InMemoryConversationStore = Depends(get_conversation_store),
    _rate_limit: None = Depends(check_chat_rate_limit),
) -> ChatAPIResponse:
    """Process a user query through the RAG pipeline with session isolation and attribution checks."""
    user_message = request.message.strip()

    # 1. Session Isolation and Turn Management
    conv_id = store.validate_or_create_conversation_id(request.conversation_id)

    # 2. Retrieve Bounded Conversation Context
    conversation_context = store.get_formatted_context(conv_id)

    # 3. Pipeline Execution with Strict Attribution Guardrails
    try:
        rag_response = pipeline.process_query(
            query=user_message,
            conversation_context=conversation_context,
        )
    except ValueError as e:
        logger.warning("Query validation error: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from None
    except LLMProviderError as e:
        logger.error("LLM Provider error: %s", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is temporarily unavailable. Please try again later.",
        ) from None
    except Exception as e:
        # Secure error masking: log error class without sensitive payload or stack trace leakage
        logger.error("Unhandled pipeline exception: %s", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request.",
        ) from None

    # 4. Update Conversation Memory (FIFO bounded turns)
    store.add_turn(conversation_id=conv_id, role="user", content=user_message)
    store.add_turn(conversation_id=conv_id, role="assistant", content=rag_response.answer)

    # 5. Return Complete ChatAPIResponse
    return ChatAPIResponse(
        answer=rag_response.answer,
        grounded=rag_response.grounded,
        grounding_status=rag_response.grounding_status,
        is_site_specific=rag_response.is_site_specific,
        language=rag_response.language,
        retrieved_sources=rag_response.retrieved_sources,
        cited_sources=rag_response.cited_sources,
        retrieval_latency_ms=rag_response.retrieval_latency_ms,
        llm_latency_ms=rag_response.llm_latency_ms,
        total_latency_ms=rag_response.total_latency_ms,
        conversation_id=conv_id,
    )
