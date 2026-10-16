"""Pydantic data schemas for Master Knowledge Base records and semantic chunks."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class SeverityInfo(BaseModel):
    """Vulnerability severity rating and rationale."""
    level: str
    rationale: str


class DetectionInfo(BaseModel):
    """Detection mechanisms and code review guidelines."""
    overview: str
    indicators: List[str] = Field(default_factory=list)
    code_patterns: List[str] = Field(default_factory=list)
    testing_approaches: List[str] = Field(default_factory=list)


class SecureCodingInfo(BaseModel):
    """Secure coding principles."""
    principles: List[str] = Field(default_factory=list)


class KnowledgeQualityInfo(BaseModel):
    """Knowledge quality assurance metadata."""
    status: str
    cwe_confidence: str
    owasp_mapping_type: str
    site_specific_claims: bool
    normalization_notes: List[str] = Field(default_factory=list)


class VulnerabilityRecord(BaseModel):
    """Canonical representation of a vulnerability in the Master KB."""
    id: str
    name: str
    aliases: List[str] = Field(default_factory=list)
    category: str
    description: str
    how_it_works: str
    root_cause: List[str] = Field(default_factory=list)
    variants: List[str] = Field(default_factory=list)
    severity: SeverityInfo
    impact: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    affected_technologies: List[str] = Field(default_factory=list)
    affected_languages: List[str] = Field(default_factory=list)
    affected_frameworks: List[str] = Field(default_factory=list)
    common_locations: List[str] = Field(default_factory=list)
    detection: DetectionInfo
    prevention: List[str] = Field(default_factory=list)
    remediation: List[str] = Field(default_factory=list)
    secure_coding: SecureCodingInfo
    common_mistakes: List[str] = Field(default_factory=list)
    cwe: List[str] = Field(default_factory=list)
    owasp: List[str] = Field(default_factory=list)
    related_vulnerabilities: List[Dict[str, Any]] = Field(default_factory=list)
    comparisons: List[Dict[str, Any]] = Field(default_factory=list)
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    user_questions: List[str] = Field(default_factory=list)
    knowledge_quality: KnowledgeQualityInfo


class SemanticChunk(BaseModel):
    """A semantic chunk derived from a canonical vulnerability record."""
    chunk_id: str
    vulnerability_id: str
    name: str
    category: str
    section: str
    severity_level: str
    cwe: List[str] = Field(default_factory=list)
    owasp: List[str] = Field(default_factory=list)
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A retrieved chunk with similarity score and provenance metadata."""
    chunk_id: str
    vulnerability_id: str
    name: str
    category: str
    section: str
    severity_level: str
    content: str
    score: float
    cwe: List[str] = Field(default_factory=list)
    owasp: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Result of a top-k vector similarity search."""
    query: str
    top_k: int
    latency_ms: float
    embed_latency_ms: float = 0.0
    search_latency_ms: float = 0.0
    matches: List[RetrievedChunk] = Field(default_factory=list)


class GroundingStatus(str, Enum):
    """Granular operational status of RAG response grounding and attribution checks."""
    GROUNDED = "grounded"
    REFUSED_EMPTY_RETRIEVAL = "empty_retrieval"
    REFUSED_INSUFFICIENT = "insufficient_evidence"
    SITE_SPECIFIC_QUALIFIED = "site_qualified"
    ATTRIBUTION_FAILED = "attribution_failed"
    INCOMPLETE_TRUNCATED = "incomplete_truncated"


class SourceCitation(BaseModel):
    """Verified citation payload derived strictly from retrieved Master KB chunks.

    The LLM never generates or modifies this metadata; it is populated solely by
    backend code from verified Qdrant/KB payloads.
    """
    chunk_id: str = Field(..., description="Deterministic chunk identifier.")
    vulnerability_id: str = Field(..., description="Canonical vulnerability ID in Master KB.")
    name: str = Field(..., description="Human-readable vulnerability title.")
    section: str = Field(..., description="Vulnerability knowledge section.")
    cwe: List[str] = Field(default_factory=list, description="Associated CWE identifiers from KB.")
    owasp: List[str] = Field(default_factory=list, description="Associated OWASP category tags from KB.")
    score: float = Field(..., description="Hybrid retrieval ranking score.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional verified chunk metadata.")


class RAGResponse(BaseModel):
    """Synthesized RAG response payload with strict provenance and attribution tracking.

    IMPORTANT GROUNDING DEFINITION:
    'grounded' is a conservative, structural guardrail status indicating that:
    (1) verified KB evidence was retrieved and injected,
    (2) the response did not trigger an insufficient-evidence refusal,
    (3) all cited sources correspond to verified retrieved chunks, and
    (4) no unverified identifiers (CWE/CVE/OWASP) were detected.

    Passing citation and identifier checks DOES NOT constitute absolute proof of
    semantic or factual correctness, nor does it guarantee the absence of subtle hallucinations.
    """
    answer: str = Field(..., description="Synthesized answer narrative.")
    grounded: bool = Field(
        ...,
        description="Structural attribution flag. True only if all attribution guardrails passed.",
    )
    grounding_status: GroundingStatus = Field(
        ...,
        description="Granular operational status of grounding and fallback processing.",
    )
    is_site_specific: bool = Field(
        default=False,
        description="True if query targeted a specific system/website rather than general knowledge.",
    )
    language: str = Field(
        default="en",
        description="Detected query language ('en' or 'ar').",
    )
    retrieved_sources: List[SourceCitation] = Field(
        default_factory=list,
        description="All valid chunks injected into prompt context.",
    )
    cited_sources: List[SourceCitation] = Field(
        default_factory=list,
        description="Subset of retrieved chunks actually cited in the narrative.",
    )
    retrieval_latency_ms: float = Field(
        default=0.0,
        description="Embedding and search latency in milliseconds.",
    )
    llm_latency_ms: float = Field(
        default=0.0,
        description="LLM inference latency in milliseconds.",
    )
    total_latency_ms: float = Field(
        default=0.0,
        description="Total end-to-end wall-clock latency in milliseconds.",
    )


class ChatRequest(BaseModel):
    r"""User request payload for chat interaction.

    CONVERSATION ID BEHAVIOR & SECURITY:
    - Omitted or None: Automatically generates a fresh, random UUID4 session identifier.
    - Valid String: Must match ^[a-zA-Z0-9_\-]{1,64}$ to continue an existing session.
    - Invalid Format: Strictly rejected with HTTP 422 Unprocessable Content.
    - Access Control Notice: Conversation IDs rely on 128-bit unguessability for local demos,
      not authenticated ownership or multi-tenant authorization.
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question or security prompt (max 2000 characters).",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_\-]{1,64}$",
        description="Optional session ID (1-64 alphanumeric/dash/underscore chars). Omitted generates fresh UUID4; invalid format returns HTTP 422.",
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_and_clean_input(cls, values: Any) -> Any:
        if isinstance(values, dict):
            raw_text = values.get("message") or values.get("query")
            if raw_text is None or not str(raw_text).strip():
                raise ValueError("Query string cannot be empty or whitespace-only.")
            values["message"] = str(raw_text).strip()
            cid = values.get("conversation_id")
            if cid is not None:
                cleaned_cid = str(cid).strip()
                values["conversation_id"] = cleaned_cid if cleaned_cid else None
        return values


class ChatAPIResponse(RAGResponse):
    """API response payload combining RAG response with conversation session tracking."""
    conversation_id: str = Field(..., description="Active conversation session identifier.")


class HealthResponse(BaseModel):
    """Health check response detailing service and component operational status."""
    status: str = Field(..., description="Overall service status ('healthy' or 'degraded').")
    version: str = Field(..., description="Application semantic version.")
    master_kb_checksum_verified: bool = Field(..., description="Integrity verification of Master KB file.")
    vector_store_status: str = Field(..., description="Vector store connectivity/index status.")
    indexed_point_count: int = Field(..., description="Number of indexed chunks in Qdrant.")
    llm_provider: str = Field(..., description="Active configured LLM provider name.")
    llm_configured: bool = Field(..., description="True if LLM provider has required credentials configured.")



