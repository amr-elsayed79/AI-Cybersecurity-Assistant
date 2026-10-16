"""Comprehensive unit and integration tests for Stage D: RAGChatPipeline.

All tests run 100% offline using synthetic in-memory chunks and mock providers/retrievers.
Zero live network calls, zero Groq API calls, zero database downloads.
"""

from unittest.mock import MagicMock
import pytest

from app.llm.base import LLMProviderError, LLMResponse
from app.llm.mock_provider import MockLLMProvider
from app.models.schemas import GroundingStatus, RAGResponse, RetrievedChunk, RetrievalResult
from app.prompts.system_prompt import (
    SITE_SPECIFIC_DISCLAIMER_AR,
    SITE_SPECIFIC_DISCLAIMER_EN,
    STANDARD_REFUSAL_AR,
    STANDARD_REFUSAL_EN,
)
from app.rag.pipeline import (
    ATTRIBUTION_FAILURE_AR,
    ATTRIBUTION_FAILURE_EN,
    CONTENT_FILTER_NOTICE_AR,
    CONTENT_FILTER_NOTICE_EN,
    RAGChatPipeline,
    TRUNCATION_NOTICE_AR,
    TRUNCATION_NOTICE_EN,
)


def _make_test_chunk(
    chunk_id: str = "sql-injection_prevention",
    vulnerability_id: str = "sql-injection",
    name: str = "SQL Injection",
    section: str = "prevention_and_remediation",
    content: str = "Use parameterized queries and prepared statements to prevent SQL injection.",
    score: float = 0.88,
    cwe: list = None,
    owasp: list = None,
) -> RetrievedChunk:
    """Helper to synthesize test chunks."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        vulnerability_id=vulnerability_id,
        name=name,
        category="Injection",
        section=section,
        severity_level="Critical",
        content=content,
        score=score,
        cwe=cwe or ["CWE-89"],
        owasp=owasp or ["A03:2021-Injection"],
        metadata={"verified": True},
    )


def _create_mock_retriever(chunks: list) -> MagicMock:
    """Helper to construct a mock retriever returning given chunks."""
    retriever = MagicMock()
    retriever.retrieve.return_value = RetrievalResult(
        query="mock query",
        top_k=len(chunks),
        latency_ms=5.0,
        matches=chunks,
    )
    return retriever


# ===========================================================================
# 1. Successful Grounded Answers & Source Separation Tests
# ===========================================================================

def test_pipeline_successful_grounded_answer():
    """Verify end-to-end grounded answer correctly parses citations and updates metrics."""
    c1 = _make_test_chunk(chunk_id="chk_sql_prev", cwe=["CWE-89"])
    c2 = _make_test_chunk(chunk_id="chk_sql_over", section="overview", content="Overview text.")

    mock_retriever = _create_mock_retriever([c1, c2])
    mock_llm = MockLLMProvider(
        default_response="Parameterized queries prevent SQL injection [E1]. Refer to CWE-89."
    )

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    response = pipeline.process_query("How can I prevent SQL injection?")

    assert isinstance(response, RAGResponse)
    assert response.grounded is True
    assert response.grounding_status == GroundingStatus.GROUNDED
    assert response.language == "en"
    assert response.is_site_specific is False

    # Verify separation of retrieved sources vs cited sources
    assert len(response.retrieved_sources) == 2
    assert len(response.cited_sources) == 1
    assert response.cited_sources[0].chunk_id == "chk_sql_prev"

    # Latencies
    assert response.retrieval_latency_ms >= 0.0
    assert response.llm_latency_ms >= 0.0
    assert response.total_latency_ms >= 0.0


# ===========================================================================
# 2. Empty Retrieval & All-Invalid Fallbacks (Zero LLM Calls)
# ===========================================================================

def test_pipeline_empty_retrieval_zero_llm_calls():
    """Verify empty retrieval triggers deterministic refusal with ZERO LLM calls."""
    mock_retriever = _create_mock_retriever([])
    mock_llm = MagicMock()

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    response = pipeline.process_query("Explain zero-day vulnerability in custom system")

    # LLM must NEVER be called
    assert mock_llm.generate.call_count == 0

    assert response.grounded is False
    assert response.grounding_status == GroundingStatus.REFUSED_EMPTY_RETRIEVAL
    assert response.answer == STANDARD_REFUSAL_EN
    assert response.retrieved_sources == []
    assert response.cited_sources == []
    assert response.llm_latency_ms == 0.0


def test_pipeline_all_invalid_chunks_zero_llm_calls():
    """Verify all-malformed chunks safely trigger deterministic refusal without calling LLM."""
    c_malformed = _make_test_chunk(chunk_id="")  # Missing chunk_id
    mock_retriever = _create_mock_retriever([c_malformed])
    mock_llm = MagicMock()

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    response = pipeline.process_query("What is SQLi?")

    assert mock_llm.generate.call_count == 0
    assert response.grounded is False
    assert response.grounding_status == GroundingStatus.REFUSED_EMPTY_RETRIEVAL
    assert response.answer == STANDARD_REFUSAL_EN


# ===========================================================================
# 3. Insufficient Evidence Refusal Detection
# ===========================================================================

def test_pipeline_insufficient_evidence_clears_cited_sources():
    """Verify that when the LLM emits a refusal, cited sources are cleared."""
    c1 = _make_test_chunk()
    mock_retriever = _create_mock_retriever([c1])
    mock_llm = MockLLMProvider(default_response=STANDARD_REFUSAL_EN)

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    response = pipeline.process_query("Tell me about a vulnerability not in the KB")

    assert response.grounded is False
    assert response.grounding_status == GroundingStatus.REFUSED_INSUFFICIENT
    assert response.answer == STANDARD_REFUSAL_EN
    assert len(response.retrieved_sources) == 1
    assert response.cited_sources == []


# ===========================================================================
# 4. Attribution Safeguards: Fabricated Labels & Unsupported Identifiers
# ===========================================================================

def test_pipeline_fabricated_evidence_tag_suppresses_response():
    """Verify that emitting unverified evidence tags (e.g. [E99]) triggers suppression."""
    c1 = _make_test_chunk()
    mock_retriever = _create_mock_retriever([c1])
    # LLM hallucinates an evidence tag E99 not in the registry
    mock_llm = MockLLMProvider(
        default_response="Fabricated claim with fake citation [E99]."
    )

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    response = pipeline.process_query("What is SQL Injection?")

    assert response.grounded is False
    assert response.grounding_status == GroundingStatus.ATTRIBUTION_FAILED
    assert response.answer == ATTRIBUTION_FAILURE_EN
    assert response.cited_sources == []
    assert len(response.retrieved_sources) == 1


def test_pipeline_unsupported_cwe_identifier_suppresses_response():
    """Verify that emitting an unsupported CWE identifier triggers narrative suppression."""
    c1 = _make_test_chunk(cwe=["CWE-89"])
    mock_retriever = _create_mock_retriever([c1])
    # LLM emits CWE-99999 which is not in any injected chunk
    mock_llm = MockLLMProvider(
        default_response="SQL injection is classified under CWE-99999 [E1]."
    )

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    response = pipeline.process_query("What is SQLi CWE?")

    assert response.grounded is False
    assert response.grounding_status == GroundingStatus.ATTRIBUTION_FAILED
    assert response.answer == ATTRIBUTION_FAILURE_EN
    assert response.cited_sources == []


def test_pipeline_cwe_supported_by_another_chunk_is_allowed():
    """Verify identifier present in chunk 2 (even if absent in chunk 1) is NOT falsely flagged."""
    c1 = _make_test_chunk(chunk_id="chk_1", cwe=["CWE-89"])
    c2 = _make_test_chunk(chunk_id="chk_2", cwe=["CWE-20"], content="Improper input validation CWE-20.")

    mock_retriever = _create_mock_retriever([c1, c2])
    # LLM references CWE-20 while citing [E1]
    mock_llm = MockLLMProvider(
        default_response="SQL injection relates to improper validation CWE-20 [E1]."
    )

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    response = pipeline.process_query("What are the CWEs for SQLi?")

    assert response.grounded is True
    assert response.grounding_status == GroundingStatus.GROUNDED
    assert "CWE-20" in response.answer


# ===========================================================================
# 5. Site-Specific Diagnosis Safeguards & Claim Suppression
# ===========================================================================

def test_pipeline_site_specific_prepends_disclaimer():
    """Verify site-specific query has mandatory disclaimer prepended."""
    c1 = _make_test_chunk()
    mock_retriever = _create_mock_retriever([c1])
    mock_llm = MockLLMProvider(
        default_response="SQL Injection occurs when untrusted input is concatenated into queries [E1]."
    )

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    response = pipeline.process_query("Is https://mysite.com/login vulnerable to SQL injection?")

    assert response.is_site_specific is True
    assert response.grounding_status == GroundingStatus.SITE_SPECIFIC_QUALIFIED
    assert response.answer.startswith(SITE_SPECIFIC_DISCLAIMER_EN)
    assert len(response.cited_sources) == 1


def test_pipeline_site_specific_diagnostic_affirmation_suppressed():
    """Verify affirmative diagnostic claim about target site is replaced with safe guidance."""
    c1 = _make_test_chunk()
    mock_retriever = _create_mock_retriever([c1])
    # LLM makes an unsupported claim diagnosing user's site
    mock_llm = MockLLMProvider(
        default_response="Yes, your website is vulnerable to SQL injection [E1]."
    )

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    response = pipeline.process_query("Is https://company.com vulnerable?")

    assert response.is_site_specific is True
    assert response.grounding_status == GroundingStatus.SITE_SPECIFIC_QUALIFIED
    # Unsupported diagnostic claim must be stripped
    assert "Yes, your website is vulnerable" not in response.answer
    assert response.answer.startswith(SITE_SPECIFIC_DISCLAIMER_EN)
    assert "cannot verify whether your specific application or infrastructure is vulnerable" in response.answer


# ===========================================================================
# 6. Bilingual & Mixed Language Tests
# ===========================================================================

def test_pipeline_bilingual_arabic_refusal():
    """Verify Arabic empty retrieval returns Arabic refusal."""
    mock_retriever = _create_mock_retriever([])
    mock_llm = MagicMock()

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    response = pipeline.process_query("ما هي آلية ثغرة غير موجودة في قاعدة البيانات؟")

    assert response.language == "ar"
    assert response.answer == STANDARD_REFUSAL_AR
    assert response.grounded is False
    assert response.grounding_status == GroundingStatus.REFUSED_EMPTY_RETRIEVAL


def test_pipeline_bilingual_arabic_attribution_failure():
    """Verify Arabic query with hallucinated tag returns Arabic attribution failure message."""
    c1 = _make_test_chunk()
    mock_retriever = _create_mock_retriever([c1])
    mock_llm = MockLLMProvider(
        default_response="إجابة تحتوي على مرجع وهمي [E99]."
    )

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    response = pipeline.process_query("كيف أحمي تطبيقي من ثغرة حقن الأوامر؟")

    assert response.language == "ar"
    assert response.grounded is False
    assert response.grounding_status == GroundingStatus.ATTRIBUTION_FAILED
    assert response.answer == ATTRIBUTION_FAILURE_AR


# ===========================================================================
# 7. Error Handling & Input Validation Tests
# ===========================================================================

def test_pipeline_rejects_empty_or_whitespace_query():
    """Verify ValueError is raised on empty or whitespace queries."""
    pipeline = RAGChatPipeline()
    with pytest.raises(ValueError) as exc:
        pipeline.process_query("")
    assert "empty or whitespace" in str(exc.value)

    with pytest.raises(ValueError):
        pipeline.process_query("   ")


def test_pipeline_rejects_excessively_long_query():
    """Verify ValueError is raised on queries exceeding MAX_QUERY_CHARS."""
    pipeline = RAGChatPipeline()
    long_query = "A" * 2001
    with pytest.raises(ValueError) as exc:
        pipeline.process_query(long_query)
    assert "exceeds maximum allowed length" in str(exc.value)


def test_pipeline_masks_provider_exception():
    """Verify provider exceptions are masked safely without exposing internal details."""
    c1 = _make_test_chunk()
    mock_retriever = _create_mock_retriever([c1])
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = RuntimeError("Internal secret connection string leak")

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)

    with pytest.raises(LLMProviderError) as exc:
        pipeline.process_query("What is SQL injection?")

    # Internal secret must NOT be in the exception message
    assert "Internal secret connection string leak" not in str(exc.value)
    assert "LLM generation failed" in str(exc.value)


# ===========================================================================
# 8. Whitespace-Tolerant Citation & Label Normalization Tests (Step 1)
# ===========================================================================

def test_pipeline_citation_whitespace_tolerance():
    """Verify [E1], [ E1 ], [  E1  ] all produce the expected cited source identically."""
    c1 = _make_test_chunk(chunk_id="chunk_1", name="SQL Injection")
    c2 = _make_test_chunk(chunk_id="chunk_2", name="XSS", cwe=["CWE-79"])
    mock_retriever = _create_mock_retriever([c1, c2])

    # 1. Standard unspaced citation [E1]
    mock_llm_std = MockLLMProvider(default_response="Standard citation [E1]. Refer to CWE-89.")
    p1 = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm_std)
    r1 = p1.process_query("What is SQL injection?")
    assert r1.grounding_status == GroundingStatus.GROUNDED
    assert len(r1.cited_sources) == 1
    assert r1.cited_sources[0].chunk_id == "chunk_1"

    # 2. Single internal space [ E1 ]
    mock_llm_single_space = MockLLMProvider(default_response="Single space citation [ E1 ]. Refer to CWE-89.")
    p2 = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm_single_space)
    r2 = p2.process_query("What is SQL injection?")
    assert r2.grounding_status == GroundingStatus.GROUNDED
    assert len(r2.cited_sources) == 1
    assert r2.cited_sources[0].chunk_id == "chunk_1"

    # 3. Multiple internal spaces [  E1  ]
    mock_llm_multi_space = MockLLMProvider(default_response="Multi space citation [  E1  ]. Refer to CWE-89.")
    p3 = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm_multi_space)
    r3 = p3.process_query("What is SQL injection?")
    assert r3.grounding_status == GroundingStatus.GROUNDED
    assert len(r3.cited_sources) == 1
    assert r3.cited_sources[0].chunk_id == "chunk_1"

    # 4. Multiple distinct labels with varying whitespace [ E1 ] and [   E2   ]
    mock_llm_multi_labels = MockLLMProvider(
        default_response="Claim 1 [ E1 ] and Claim 2 [   E2   ]. Mentions CWE-89 and CWE-79."
    )
    p4 = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm_multi_labels)
    r4 = p4.process_query("Compare SQLi and XSS")
    assert r4.grounding_status == GroundingStatus.GROUNDED
    assert len(r4.cited_sources) == 2
    assert {s.chunk_id for s in r4.cited_sources} == {"chunk_1", "chunk_2"}


def test_pipeline_citation_whitespace_tolerant_fabricated_tag_triggers_failure():
    """Verify that a fake tag with internal whitespace [ E99 ] triggers ATTRIBUTION_FAILED."""
    c1 = _make_test_chunk()
    mock_retriever = _create_mock_retriever([c1])
    mock_llm = MockLLMProvider(default_response="Claim with spaced fake citation [ E99 ].")

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    response = pipeline.process_query("What is SQL injection?")

    assert response.grounded is False
    assert response.grounding_status == GroundingStatus.ATTRIBUTION_FAILED
    assert response.answer == ATTRIBUTION_FAILURE_EN
    assert response.cited_sources == []


def test_pipeline_citation_case_insensitivity_and_fabricated_lowercase_tags():
    """Verify lowercase citations [e1], [ e1 ] are accepted and [e99], [ e99 ] trigger attribution failure."""
    c1 = _make_test_chunk(chunk_id="chunk_1", name="SQL Injection")
    mock_retriever = _create_mock_retriever([c1])

    # 1. Lowercase unspaced citation [e1]
    mock_llm_lower = MockLLMProvider(default_response="Lowercase citation [e1]. Refer to CWE-89.")
    p1 = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm_lower)
    r1 = p1.process_query("What is SQL injection?")
    assert r1.grounding_status == GroundingStatus.GROUNDED
    assert len(r1.cited_sources) == 1
    assert r1.cited_sources[0].chunk_id == "chunk_1"

    # 2. Lowercase spaced citation [ e1 ]
    mock_llm_lower_space = MockLLMProvider(default_response="Lowercase spaced citation [ e1 ]. Refer to CWE-89.")
    p2 = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm_lower_space)
    r2 = p2.process_query("What is SQL injection?")
    assert r2.grounding_status == GroundingStatus.GROUNDED
    assert len(r2.cited_sources) == 1
    assert r2.cited_sources[0].chunk_id == "chunk_1"

    # 3. Lowercase fabricated citation [e99]
    mock_llm_fake_lower = MockLLMProvider(default_response="Fabricated lowercase citation [e99].")
    p3 = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm_fake_lower)
    r3 = p3.process_query("What is SQL injection?")
    assert r3.grounded is False
    assert r3.grounding_status == GroundingStatus.ATTRIBUTION_FAILED
    assert r3.answer == ATTRIBUTION_FAILURE_EN
    assert r3.cited_sources == []

    # 4. Lowercase spaced fabricated citation [ e99 ]
    mock_llm_fake_lower_space = MockLLMProvider(default_response="Fabricated lowercase spaced citation [ e99 ].")
    p4 = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm_fake_lower_space)
    r4 = p4.process_query("What is SQL injection?")
    assert r4.grounded is False
    assert r4.grounding_status == GroundingStatus.ATTRIBUTION_FAILED
    assert r4.answer == ATTRIBUTION_FAILURE_EN
    assert r4.cited_sources == []


def test_evidence_registry_label_normalization_direct_calls():
    """Verify EvidenceRegistry directly handles raw, bracketed, whitespace-padded, and lowercase labels."""
    from app.rag.context_builder import EvidenceRegistry

    chunk = _make_test_chunk(chunk_id="chunk_test", name="Test Vulnerability")
    registry = EvidenceRegistry(
        label_to_chunk={"E1": chunk},
        chunk_id_to_label={"chunk_test": "E1"},
    )

    # is_valid_label tests (uppercase and lowercase)
    assert registry.is_valid_label("E1") is True
    assert registry.is_valid_label("e1") is True
    assert registry.is_valid_label("[E1]") is True
    assert registry.is_valid_label("[e1]") is True
    assert registry.is_valid_label("[ E1 ]") is True
    assert registry.is_valid_label("[ e1 ]") is True
    assert registry.is_valid_label("[  E1  ]") is True
    assert registry.is_valid_label("[  e1  ]") is True
    assert registry.is_valid_label("  E1  ") is True
    assert registry.is_valid_label("  e1  ") is True
    assert registry.is_valid_label("[ E99 ]") is False
    assert registry.is_valid_label("[ e99 ]") is False
    assert registry.is_valid_label("[e99]") is False
    assert registry.is_valid_label("E2") is False
    assert registry.is_valid_label("e2") is False

    # get_citation_by_label tests
    cit1 = registry.get_citation_by_label("E1")
    assert cit1 is not None
    assert cit1.chunk_id == "chunk_test"

    cit2 = registry.get_citation_by_label("[ E1 ]")
    assert cit2 is not None
    assert cit2.chunk_id == "chunk_test"

    cit3 = registry.get_citation_by_label("[   E1   ]")
    assert cit3 is not None
    assert cit3.chunk_id == "chunk_test"

    cit_lower = registry.get_citation_by_label("[e1]")
    assert cit_lower is not None
    assert cit_lower.chunk_id == "chunk_test"

    cit_lower_space = registry.get_citation_by_label("[  e1  ]")
    assert cit_lower_space is not None
    assert cit_lower_space.chunk_id == "chunk_test"

    assert registry.get_citation_by_label("[ E99 ]") is None
    assert registry.get_citation_by_label("[ e99 ]") is None
    assert registry.get_citation_by_label("[e99]") is None


# ===========================================================================
# 9. Provider Finish Reason Guardrail Tests (Step 2)
# ===========================================================================

def test_pipeline_token_truncation_suppression_english():
    """Verify that finish_reason='length' suppresses partial response and returns INCOMPLETE_TRUNCATED."""
    c1 = _make_test_chunk()
    mock_retriever = _create_mock_retriever([c1])
    # Model output cut off mid-sentence with valid citation tag [E1]
    mock_llm = MockLLMProvider(
        default_response="SQL injection alters database commands [E1]. To defend your app, use parameter",
        default_finish_reason="length",
    )

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    response = pipeline.process_query("What is SQL injection?")

    assert response.grounded is False
    assert response.grounding_status == GroundingStatus.INCOMPLETE_TRUNCATED
    assert response.answer == TRUNCATION_NOTICE_EN
    assert response.cited_sources == []  # Not credited as grounded
    assert len(response.retrieved_sources) == 1
    assert response.retrieval_latency_ms >= 0.0
    assert response.llm_latency_ms >= 0.0


def test_pipeline_token_truncation_suppression_arabic():
    """Verify that finish_reason='length' returns Arabic truncation notice on Arabic query."""
    c1 = _make_test_chunk()
    mock_retriever = _create_mock_retriever([c1])
    mock_llm = MockLLMProvider(
        default_response="حقن الأوامر هو هجوم خطير [E1] ويجب استخدام مصفوفة",
        default_finish_reason="length",
    )

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    response = pipeline.process_query("كيف يعمل هجوم حقن الأوامر؟")

    assert response.grounded is False
    assert response.grounding_status == GroundingStatus.INCOMPLETE_TRUNCATED
    assert response.answer == TRUNCATION_NOTICE_AR
    assert response.cited_sources == []
    assert response.language == "ar"


def test_pipeline_content_filter_suppression_bilingual():
    """Verify finish_reason='content_filter' returns safe refusal without leaking provider details."""
    c1 = _make_test_chunk()
    mock_retriever = _create_mock_retriever([c1])
    mock_llm = MockLLMProvider(
        default_response="Blocked content",
        default_finish_reason="content_filter",
    )

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)

    # English query
    resp_en = pipeline.process_query("How to exploit XSS?")
    assert resp_en.grounded is False
    assert resp_en.grounding_status == GroundingStatus.REFUSED_INSUFFICIENT
    assert resp_en.answer == CONTENT_FILTER_NOTICE_EN
    assert resp_en.cited_sources == []

    # Arabic query
    resp_ar = pipeline.process_query("كيف استغل ثغرة في موقع؟")
    assert resp_ar.grounded is False
    assert resp_ar.grounding_status == GroundingStatus.REFUSED_INSUFFICIENT
    assert resp_ar.answer == CONTENT_FILTER_NOTICE_AR
    assert resp_ar.cited_sources == []


def test_pipeline_missing_finish_reason_proceeds_to_attribution():
    """Verify that finish_reason=None (missing) does NOT suppress valid grounded responses."""
    c1 = _make_test_chunk()
    mock_retriever = _create_mock_retriever([c1])
    mock_llm = MockLLMProvider(
        default_response="Parameterized queries prevent SQL injection [E1]. Refer to CWE-89.",
        default_finish_reason=None,
    )

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_llm)
    response = pipeline.process_query("What is SQL injection?")

    assert response.grounded is True
    assert response.grounding_status == GroundingStatus.GROUNDED
    assert len(response.cited_sources) == 1
    assert "Parameterized queries" in response.answer


