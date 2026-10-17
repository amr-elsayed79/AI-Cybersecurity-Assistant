"""Comprehensive test suite for Arabic RAG quality, intent classification, and bilingual handling."""

import pytest
from unittest.mock import MagicMock

from app.config import settings
from app.llm.base import LLMResponse
from app.models.schemas import GroundingStatus, RetrievedChunk
from app.prompts.system_prompt import build_system_prompt
from app.rag.intent import (
    QueryIntent,
    QueryLanguage,
    analyze_query_intent,
    detect_query_language,
)
from app.rag.pipeline import (
    RAGChatPipeline,
    TRUNCATION_NOTICE_AR,
    TRUNCATION_NOTICE_EN,
)
from app.rag.retriever import expand_query_aliases, get_shared_vector_store


# ===========================================================================
# 1. Intent & Language Classification Tests
# ===========================================================================

@pytest.mark.parametrize(
    "query, expected_lang, expected_intent",
    [
        # Egyptian colloquial Arabic questions
        ("يعني إيه SQL Injection؟", QueryLanguage.ARABIC, QueryIntent.GENERAL_KNOWLEDGE),
        ("إزاي أحمي الموقع من ثغرات SQL Injection؟", QueryLanguage.ARABIC, QueryIntent.PREVENTION_REMEDIATION),
        ("إيه الفرق بين SQLi وNoSQLi؟", QueryLanguage.ARABIC, QueryIntent.COMPARISON_EXPLANATION),
        ("اشرحلي XXE بطريقة بسيطة.", QueryLanguage.ARABIC, QueryIntent.GENERAL_KNOWLEDGE),
        ("إيه خطورة SSTI وإزاي أتجنبها؟", QueryLanguage.ARABIC, QueryIntent.GENERAL_KNOWLEDGE),
        ("اشرحلي الثغرة دي بالمصري وبمثال آمن.", QueryLanguage.ARABIC, QueryIntent.GENERAL_KNOWLEDGE),
        # Modern Standard Arabic questions
        ("ما هو حقن أوامر النظام؟", QueryLanguage.ARABIC, QueryIntent.GENERAL_KNOWLEDGE),
        ("ما هي ثغرة تزوير الطلبات عبر الخوادم؟", QueryLanguage.ARABIC, QueryIntent.GENERAL_KNOWLEDGE),
        ("كيف يمكن الوقاية من ثغرة البرمجة النصية عبر المواقع؟", QueryLanguage.ARABIC, QueryIntent.PREVENTION_REMEDIATION),
        ("ما الفرق بين Path Traversal وLFI؟", QueryLanguage.ARABIC, QueryIntent.COMPARISON_EXPLANATION),
        # Site-specific Arabic questions
        ("هل موقعي https://example.com مصاب بثغرة SSRF؟", QueryLanguage.ARABIC, QueryIntent.SITE_SPECIFIC_DIAGNOSIS),
        ("افحص موقعنا لمعرفة هل هو معرض للاختراق", QueryLanguage.ARABIC, QueryIntent.SITE_SPECIFIC_DIAGNOSIS),
    ],
)
def test_arabic_query_intent_and_language_detection(query, expected_lang, expected_intent):
    """Verify that Egyptian, MSA, and mixed Arabic queries are correctly classified."""
    result = analyze_query_intent(query)
    assert result.language == expected_lang
    assert result.intent == expected_intent


# ===========================================================================
# 2. Bilingual Query Expansion & Acronym Retrieval Tests
# ===========================================================================

def test_arabic_concept_expansion():
    """Verify deterministic Arabic-to-Canonical Technical Concept expansion."""
    q1 = "ما هو حقن أوامر النظام؟"
    expanded1 = expand_query_aliases(q1)
    assert "OS Command Injection" in expanded1

    q2 = "ما هي ثغرة تزوير الطلبات عبر الخوادم؟"
    expanded2 = expand_query_aliases(q2)
    assert "Server-Side Request Forgery" in expanded2
    assert "SSRF" in expanded2

    q3 = "كيف أتجنب حقن القوالب من جانب الخادم؟"
    expanded3 = expand_query_aliases(q3)
    assert "Server-Side Template Injection" in expanded3
    assert "SSTI" in expanded3

    q4 = "اشرحلي ثغرة XXE بالمصري"
    expanded4 = expand_query_aliases(q4)
    assert "XML External Entity Injection" in expanded4


def test_arabic_retrieval_accuracy_offline():
    """Verify top-ranked retrieval for Egyptian and MSA Arabic queries."""
    store = get_shared_vector_store()

    # Egyptian query with acronym
    res_xxe = store.retrieve("اشرحلي XXE بطريقة بسيطة.", top_k=3)
    assert len(res_xxe.matches) > 0
    assert res_xxe.matches[0].vulnerability_id == "xxe"

    # Egyptian remediation query
    res_sqli = store.retrieve("إزاي أحمي الموقع من ثغرات SQL Injection؟", top_k=3)
    assert len(res_sqli.matches) > 0
    assert res_sqli.matches[0].vulnerability_id == "sql-injection"

    # Pure MSA query (OS Command Injection)
    res_cmd = store.retrieve("ما هو حقن أوامر النظام؟", top_k=3)
    assert len(res_cmd.matches) > 0
    assert res_cmd.matches[0].vulnerability_id == "os-command-injection"

    # Pure MSA query (SSRF)
    res_ssrf = store.retrieve("ما هي ثغرة تزوير الطلبات عبر الخوادم؟", top_k=3)
    assert len(res_ssrf.matches) > 0
    assert res_ssrf.matches[0].vulnerability_id == "ssrf"


# ===========================================================================
# 3. Arabic End-to-End Pipeline Behavior & Grounding
# ===========================================================================

def test_arabic_pipeline_grounding_and_attribution():
    """Verify RAG pipeline sets Arabic communication language and properly validates citations."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = LLMResponse(
        content=(
            "حقن لغة الاستعلامات البنيوية (SQL Injection) هي ثغرة أمنية تحدث عند دمج مدخلات غير موثوقة في استعلام قاعدة البيانات [E1]. "
            "للوقاية منها، يجب دائماً استخدام الاستعلامات ذات المعاملات (Parameterized Queries) والتحقق من المدخلات [E2]."
        ),
        model="test-mock",
        prompt_tokens=100,
        completion_tokens=50,
        finish_reason="stop",
    )

    pipeline = RAGChatPipeline(llm_provider=mock_provider)
    response = pipeline.process_query("يعني إيه SQL Injection وإزاي أحمي موقعي منها؟")

    assert response.language == "ar"
    assert response.grounded is True
    assert response.grounding_status == GroundingStatus.GROUNDED
    assert len(response.cited_sources) > 0


def test_arabic_insufficient_evidence_refusal():
    """Verify out-of-scope query in Arabic triggers standard Arabic refusal without hallucinations."""
    store = get_shared_vector_store()
    pipeline = RAGChatPipeline()

    # Query about totally out-of-scope topic
    response = pipeline.process_query("ما هي خوارزمية شور في كسر تشفير RSA الكمومي؟")

    assert response.language == "ar"
    assert "لا تتوفر لدي معلومات كافية" in response.answer


def test_arabic_site_specific_disclaimer():
    """Verify site-specific question in Arabic triggers site disclaimer flag and educational response."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = LLMResponse(
        content=(
            "لا يمكنني تأكيد ما إذا كان موقع https://example.com مصاباً بثغرة SSRF دون فحص أمني مصرح به. "
            "لكن من الناحية النظرية، تحدث ثغرة Server-Side Request Forgery عندما يتمكن المهاجم من إجبار الخادم على إرسال طلبات غير مصرح بها [E1]."
        ),
        model="test-mock",
        finish_reason="stop",
    )

    pipeline = RAGChatPipeline(llm_provider=mock_provider)
    response = pipeline.process_query("هل موقع https://example.com مصاب بثغرة SSRF؟")

    assert response.is_site_specific is True
    assert response.language == "ar"
    assert response.grounding_status == GroundingStatus.SITE_SPECIFIC_QUALIFIED


# ===========================================================================
# 4. English Regression Tests
# ===========================================================================

def test_english_query_unaffected():
    """Verify that English queries continue to be classified and processed in English."""
    result = analyze_query_intent("How do I prevent SQL Injection in web apps?")
    assert result.language == QueryLanguage.ENGLISH
    assert result.intent == QueryIntent.PREVENTION_REMEDIATION

    expanded = expand_query_aliases("What is XXE and SSRF?")
    assert "XML External Entity Injection" in expanded
    assert "Server-Side Request Forgery" in expanded


# ===========================================================================
# 5. Conciseness, Proportionality & Token Budget Guardrail Tests
# ===========================================================================

def test_system_prompt_conciseness_and_proportionality_instructions():
    """Verify system prompt includes explicit rules for conciseness and proportionality."""
    prompt = build_system_prompt()
    assert "CONCISENESS, DIRECTNESS & PROPORTIONALITY" in prompt
    assert "DEFINITION & CONCEPT QUESTIONS" in prompt
    assert "PREVENTION & REMEDIATION QUESTIONS" in prompt
    assert "COMBINED QUESTIONS" in prompt
    assert "STRICTLY PROHIBITED FILLER & BOILERPLATE" in prompt


def test_groq_max_tokens_budget_configured_sufficiently():
    """Verify GROQ_MAX_TOKENS is configured to at least 2048 to prevent truncation on Arabic responses."""
    assert settings.GROQ_MAX_TOKENS >= 2048


def test_arabic_token_truncation_safety_notice():
    """Verify that when finish_reason is 'length', partial response is suppressed and notice is returned."""
    mock_provider = MagicMock()
    mock_provider.generate.return_value = LLMResponse(
        content="إجابة طويلة غير مكتملة عن حقن SQL...",
        model="test-mock",
        finish_reason="length",
    )

    mock_retriever = MagicMock()
    chunk = RetrievedChunk(
        chunk_id="sql-injection_overview",
        vulnerability_id="sql-injection",
        name="SQL Injection",
        category="Injection",
        section="overview",
        severity_level="High",
        content="SQL Injection overview content",
        score=0.9,
    )
    mock_retriever.retrieve.return_value = MagicMock(matches=[chunk])

    pipeline = RAGChatPipeline(retriever=mock_retriever, llm_provider=mock_provider)
    response = pipeline.process_query("يعني إيه SQL Injection وإزاي أحمي موقعي منها؟")

    assert response.language == "ar"
    assert response.grounded is False
    assert response.grounding_status == GroundingStatus.INCOMPLETE_TRUNCATED
    assert response.answer == TRUNCATION_NOTICE_AR
    assert len(response.cited_sources) == 0
