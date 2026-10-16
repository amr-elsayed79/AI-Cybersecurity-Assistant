"""Unit tests for Stage C: Query Intent and Language Classification.

All tests run 100% offline using deterministic rule checks without network, LLM, or embedding models.
"""

import pytest

from app.rag.intent import (
    IntentAnalysisResult,
    QueryIntent,
    QueryLanguage,
    analyze_query_intent,
    detect_query_language,
)


# ===========================================================================
# 1. Language Detection & Technical Identifiers Tests
# ===========================================================================

def test_detect_language_english_standard():
    """Verify standard English query is correctly detected."""
    lang, is_mixed, terms = detect_query_language("How does SQL Injection work in web applications?")
    assert lang == QueryLanguage.ENGLISH
    assert is_mixed is False
    assert any("sql injection" in t.lower() for t in terms)


def test_detect_language_arabic_standard():
    """Verify standard Arabic query is correctly detected."""
    lang, is_mixed, terms = detect_query_language("كيف يمكن للمطورين منع ثغرة حقن القوالب من جانب الخادم؟")
    assert lang == QueryLanguage.ARABIC
    assert is_mixed is False


def test_detect_language_mixed_arabic_with_english_tech_terms():
    """Verify Arabic query with English acronyms is classified as Arabic with mixed flag."""
    lang, is_mixed, terms = detect_query_language("كيف أمنع ثغرة SQLi و XSS في تطبيق Node.js؟")
    assert lang == QueryLanguage.ARABIC
    assert is_mixed is True
    assert "sqli" in [t.lower() for t in terms]
    assert "xss" in [t.lower() for t in terms]
    assert "node.js" in [t.lower() for t in terms]


def test_detect_language_empty_and_unknown():
    """Verify empty or numeric-only strings return UNKNOWN."""
    assert detect_query_language("")[0] == QueryLanguage.UNKNOWN
    assert detect_query_language("   ")[0] == QueryLanguage.UNKNOWN
    assert detect_query_language("12345 67890")[0] == QueryLanguage.UNKNOWN


# ===========================================================================
# 2. General Knowledge, Remediation, & Comparison Intents
# ===========================================================================

def test_intent_general_knowledge_english_and_arabic():
    """Verify definitions and mechanism questions map to GENERAL_KNOWLEDGE."""
    res_en = analyze_query_intent("What is Server-Side Request Forgery and what are its root causes?")
    assert res_en.intent == QueryIntent.GENERAL_KNOWLEDGE
    assert res_en.is_site_specific is False
    assert res_en.language == QueryLanguage.ENGLISH

    res_ar = analyze_query_intent("ما هي آلية عمل ثغرة تزوير الطلبات عبر المواقع (CSRF)؟")
    assert res_ar.intent == QueryIntent.GENERAL_KNOWLEDGE
    assert res_ar.is_site_specific is False
    assert res_ar.language == QueryLanguage.ARABIC


def test_intent_prevention_remediation():
    """Verify defensive guidance questions map to PREVENTION_REMEDIATION."""
    res_en = analyze_query_intent("How to prevent SQL Injection using parameterized queries?")
    assert res_en.intent == QueryIntent.PREVENTION_REMEDIATION
    assert res_en.is_site_specific is False

    res_ar = analyze_query_intent("كيف يمكن معالجة ثغرات التحكم بالوصول المكسور وتأمين الواجهات البرمجية؟")
    assert res_ar.intent == QueryIntent.PREVENTION_REMEDIATION
    assert res_ar.is_site_specific is False
    assert res_ar.language == QueryLanguage.ARABIC


def test_intent_comparison_explanation():
    """Verify comparative questions map to COMPARISON_EXPLANATION."""
    res_en = analyze_query_intent("What is the difference between SQLi vs NoSQLi injection?")
    assert res_en.intent == QueryIntent.COMPARISON_EXPLANATION
    assert res_en.is_site_specific is False

    res_ar = analyze_query_intent("ما هو الفرق بين ثغرة IDOR وثغرة Broken Access Control؟")
    assert res_ar.intent == QueryIntent.COMPARISON_EXPLANATION
    assert res_ar.is_site_specific is False
    assert res_ar.language == QueryLanguage.ARABIC


# ===========================================================================
# 3. Site-Specific Diagnosis vs Educational URL Distinction
# ===========================================================================

def test_site_specific_diagnosis_with_url_english():
    """Verify query with URL asking for diagnostic status is marked site-specific."""
    res = analyze_query_intent("Is https://production.shop.com/checkout vulnerable to SQL injection?")
    assert res.intent == QueryIntent.SITE_SPECIFIC_DIAGNOSIS
    assert res.is_site_specific is True
    assert any("https://production.shop.com/checkout" in s for s in res.target_signals)


def test_site_specific_diagnosis_possessive_without_url_english():
    """Verify query without URL using possessive asset target is marked site-specific."""
    res = analyze_query_intent("Can you check if our internal company database server is vulnerable to Log4Shell?")
    assert res.intent == QueryIntent.SITE_SPECIFIC_DIAGNOSIS
    assert res.is_site_specific is True
    assert any("our" in s.lower() for s in res.target_signals)


def test_site_specific_diagnosis_arabic():
    """Verify Arabic site-specific diagnostic inquiries are recognized."""
    # With URL
    res_url = analyze_query_intent("هل الرابط https://mybank.com/api مصاب بثغرة XSS؟")
    assert res_url.is_site_specific is True
    assert res_url.intent == QueryIntent.SITE_SPECIFIC_DIAGNOSIS
    assert res_url.language == QueryLanguage.ARABIC

    # Without URL (possessive target)
    res_possessive = analyze_query_intent("هل تطبيقنا وموقعنا الخاص بشركتنا معرض لثغرة حقن الأوامر؟")
    assert res_possessive.is_site_specific is True
    assert res_possessive.intent == QueryIntent.SITE_SPECIFIC_DIAGNOSIS
    assert res_possessive.language == QueryLanguage.ARABIC


def test_educational_question_with_dummy_url_preserved_as_educational():
    """Verify that an educational query mentioning an example URL is NOT marked site-specific."""
    query = "Explain how an attacker uses http://example.com/item?id=1 in an educational SQLi demonstration."
    res = analyze_query_intent(query)

    assert res.is_site_specific is False
    assert res.intent == QueryIntent.GENERAL_KNOWLEDGE
    assert res.language == QueryLanguage.ENGLISH


def test_educational_question_with_test_domain_arabic():
    """Verify Arabic educational question mentioning a test domain is not marked site-specific."""
    query = "اشرح كيف يستغل المهاجم رابط مثل http://example.com/api?user=admin في ثغرة IDOR"
    res = analyze_query_intent(query)

    assert res.is_site_specific is False
    assert res.intent == QueryIntent.GENERAL_KNOWLEDGE
    assert res.language == QueryLanguage.ARABIC


# ===========================================================================
# 4. Ambiguous and Edge Cases
# ===========================================================================

def test_ambiguous_or_unsupported_short_queries():
    """Verify brief, vague, or non-security queries map to AMBIGUOUS_UNSUPPORTED."""
    assert analyze_query_intent("").intent == QueryIntent.AMBIGUOUS_UNSUPPORTED
    assert analyze_query_intent("hello").intent == QueryIntent.AMBIGUOUS_UNSUPPORTED
    assert analyze_query_intent("test 123").intent == QueryIntent.AMBIGUOUS_UNSUPPORTED


def test_ip_address_target_diagnostic():
    """Verify direct IP target with audit intent triggers site-specific diagnosis."""
    res = analyze_query_intent("Can you scan 192.168.1.50 for command injection vulnerabilities?")
    assert res.is_site_specific is True
    assert res.intent == QueryIntent.SITE_SPECIFIC_DIAGNOSIS
    assert any("192.168.1.50" in s for s in res.target_signals)
