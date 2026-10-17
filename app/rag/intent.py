"""Deterministic query intent classification and language handling for security assistant.

All rules are 100% deterministic, explainable, and local.
Zero external network calls, zero LLMs, zero embedding models.
"""

from enum import Enum
import re
from typing import List, Optional, Set, Tuple
from pydantic import BaseModel, Field


class QueryLanguage(str, Enum):
    """Detected language classification for user queries."""
    ARABIC = "ar"
    ENGLISH = "en"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class QueryIntent(str, Enum):
    """Categorized intent of the user query."""
    GENERAL_KNOWLEDGE = "general_knowledge"
    PREVENTION_REMEDIATION = "prevention_remediation"
    COMPARISON_EXPLANATION = "comparison_explanation"
    SITE_SPECIFIC_DIAGNOSIS = "site_specific_diagnosis"
    AMBIGUOUS_UNSUPPORTED = "ambiguous_unsupported"


class IntentAnalysisResult(BaseModel):
    """Comprehensive linguistic and intent analysis output."""
    intent: QueryIntent = Field(..., description="Classified intent category.")
    is_site_specific: bool = Field(..., description="True if query seeks diagnosis of a specific target/website.")
    language: QueryLanguage = Field(..., description="Primary language classification.")
    is_mixed_language: bool = Field(default=False, description="True if query contains substantial secondary language tokens.")
    target_signals: List[str] = Field(default_factory=list, description="Target entities, URLs, or possessive phrases detected.")
    diagnostic_signals: List[str] = Field(default_factory=list, description="Diagnostic verbs or inspection requests detected.")
    educational_signals: List[str] = Field(default_factory=list, description="Educational or conceptual framing phrases detected.")
    technical_terms: List[str] = Field(default_factory=list, description="Known cybersecurity technical terms preserved.")
    explanation: str = Field(..., description="Deterministic rationale for classification.")


# ---------------------------------------------------------------------------
# Pre-compiled Patterns & Term Repositories
# ---------------------------------------------------------------------------

# Technical Cybersecurity Terms & Standards (Preserved across languages)
TECHNICAL_TERMS = [
    r"\bsqli\b", r"\bsql\s+injection\b", r"\bxss\b", r"\bcross[- ]site\s+scripting\b",
    r"\bcsrf\b", r"\bssrf\b", r"\bxxe\b", r"\bidor\b", r"\bbola\b", r"\bjwt\b",
    r"\bssti\b", r"\blfi\b", r"\brfi\b", r"\bcommand\s+injection\b", r"\brce\b",
    r"\bnosqli\b", r"\bnosql\s+injection\b", r"\bpath\s+traversal\b", r"\bdirectory\s+traversal\b",
    r"\bclickjacking\b", r"\bcors\b", r"\boauth\b", r"\blog4shell\b",
    r"\bopen\s+redirect\b", r"\bweb\s+cache\s+poisoning\b", r"\bredos\b", r"\bcrlf\b",
    r"\bxpathi\b", r"\bldapi\b", r"\bfile\s+upload\b", r"\brace\s+condition\b",
    r"\bcwe(?:-\d+)?\b", r"\bowasp(?:\s+top\s+10)?(?:\s+[a-z]\d{2}(?::\d{4})?)?\b",
    r"\bcve(?:-\d{4}-\d+)?\b", r"\bnode\.?js\b", r"\bpython\b", r"\bjavascript\b",
    r"\bapi\b", r"\bgraphql\b", r"\brest\b", r"\bcookie\b", r"\bsession\b",
]

# Educational Framing Signals (English & Arabic)
EDUCATIONAL_SIGNALS_EN = [
    r"\bwhat\s+is\b", r"\bwhat\s+are\b", r"\bhow\s+does\b", r"\bhow\s+do\b",
    r"\bexplain\b", r"\boverview\b", r"\bmechanism\b", r"\bdefinition\b",
    r"\btutorial\b", r"\bunderstand\b", r"\bwhy\s+does\b", r"\bexample\s+of\b",
    r"\bdemonstrate\b", r"\bwhat\s+causes\b", r"\broot\s+cause\b",
    r"\bhow\s+attackers?\s+(?:exploit|use|target)\b",
]
EDUCATIONAL_SIGNALS_AR = [
    r"ما\s+هو\b", r"ما\s+هي\b", r"كيف\s+يعمل\b", r"كيف\s+تعمل\b",
    r"اشرح\b", r"اشرحلي\b", r"اشرح\s+لي\b", r"وضح\b", r"فهمني\b",
    r"مفهوم\b", r"نظرة\s+عامة\b", r"آلية\b", r"ماذا\s+يعني\b",
    r"يعني\s+(?:إيه|ايه)\b", r"(?:إيه|ايه)\s+هو\b", r"(?:إيه|ايه)\s+هي\b",
    r"(?:إيه|ايه)\s+خطورة\b", r"ما\s+خطورة\b", r"خطورة\b",
    r"ما\s+سبب\b", r"أسباب\b", r"مثال\s+على\b", r"بمثال\s+آمن\b",
    r"بالمصري\b", r"بطريقة\s+بسيطة\b",
    r"كيف\s+يستغل\b", r"كيف\s+يقوم\s+المهاجم\b",
    r"(?:إزاي|ازاي)\s+(?:بيشتغل|بتشتغل|بيحصل|بتعمل)\b",
    r"(?:إزاي|ازاي)\s+(?:أكتشف|اكتشف|أعرف|اعرف)\b", r"دليل\s+كافي\b",
]

# Prevention & Remediation Signals (English & Arabic)
REMEDIATION_SIGNALS_EN = [
    r"\bhow\s+(?:can\s+i|do\s+i|to)\s+(?:prevent|fix|remediate|secure|mitigate|patch|defend)\b",
    r"\bprevention\b", r"\bremediation\b", r"\bmitigation\b", r"\bsecure\s+coding\b",
    r"\bbest\s+practices\b", r"\bprotect\s+against\b", r"\bhow\s+to\s+avoid\b",
]
REMEDIATION_SIGNALS_AR = [
    r"كيف\s+(?:أمنع|يمكن\s+منع|نمنع|أحمي|يمكن\s+حماية|نعالج|يمكن\s+معالجة|أصلح|يمكن\s+الوقاية)\b",
    r"(?:إزاي|ازاي)\s+(?:أحمي|احمي|أمنع|امنع|أتجنب|اتجنب|أوقف|اوقف|أصلح|اصلح|أعالج|اعالج|أحل|احل)\b",
    r"طريقة\s+(?:الحماية|الوقاية|تجنب|منع|حل|علاج)\b",
    r"الوقاية\s+من\b", r"حماية\s+من\b",
    r"طرق\s+الوقاية\b", r"إجراءات\s+الوقاية\b", r"كيفية\s+الحماية\b",
    r"أفضل\s+الممارسات\b", r"معالجة\b", r"تأمين\b", r"تصحيح\b",
]

# Comparison Signals (English & Arabic)
COMPARISON_SIGNALS_EN = [
    r"\bvs\.?\b", r"\bversus\b", r"\bdifference\s+between\b", r"\bcompare\b",
    r"\bcomparison\b", r"\bin\s+contrast\s+to\b",
]
COMPARISON_SIGNALS_AR = [
    r"الفرق\s+بين\b", r"مقارنة\s+بين\b", r"\bمقارنة\b", r"\bمقابل\b",
    r"(?:إيه|ايه)\s+الفرق\s+بين\b",
]

# Target Attachment Signals (URLs, IPs, Hostnames, Possessive Assets)
TARGET_NETWORK_REGEX = r"https?://[^\s]+|\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b|\b[a-zA-Z0-9_\-]+\.(?:com|org|net|io|edu|gov|local|internal)\b"
DUMMY_DOMAINS = ["example.com", "example.org", "test.com", "localhost", "dummy.local", "target.com"]

TARGET_POSSESSIVE_EN = [
    r"\b(?:my|our)\s+(?:[a-zA-Z0-9_\-]+\s+){0,4}(?:website|site|app|application|server|api|database|code|system|company|portal|service|network|infrastructure)\b",
]
TARGET_POSSESSIVE_AR = [
    r"\b(?:موقعي|موقعنا|تطبيقي|تطبيقنا|خادمي|خادمنا|سيرفرنا|نظامي|نظامنا|شركتي|شركتنا|قاعدتنا|كودنا|شفرتنا|بوابتنا)\b",
    r"(?:موقع|تطبيق|خادم|نظام|شفرة|بيانات|سيرفر)\s+(?:[^\s]+\s+){0,3}(?:شركتنا|مؤسستنا|فريقنا)",
]

# Diagnostic Request Signals (English & Arabic)
DIAGNOSTIC_SIGNALS_EN = [
    r"\bis\s+.*?\s+vulnerable\b", r"\bare\s+we\s+vulnerable\b",
    r"\bdoes\s+.*?\s+have\s+(?:a\s+)?vulnerability\b",
    r"\bcan\s+you\s+(?:check|scan|test|audit|assess|hack)\b",
    r"\bcheck\s+if\s+.*?\s+is\s+(?:vulnerable|safe|secure|compromised)\b",
    r"\btest\s+if\s+.*?\s+is\s+(?:vulnerable|safe|secure|compromised)\b",
    r"\baudit\s+(?:my|our)\b", r"\bscan\s+(?:my|our|https?://)\b",
]
DIAGNOSTIC_SIGNALS_AR = [
    r"هل\s+.*?\s+(?:مصاب|مخترق|آمن|سليم|معرض)\b",
    r"هل\s+يوجد\s+ثغرة\s+في\b",
    r"(?:افحص|اختبر|تحقق\s+من|دقق)\s+(?:موقع|تطبيق|خادم|سيرفر|https?://)",
    r"هل\s+نظامنا\s+(?:معرض|مصاب)\b",
    r"(?:إزاي|ازاي|كيف)\s+أعرف\s+لو\s+.*?\s+(?:مصاب|مخترق|فيه\s+ثغرة)",
]


# ---------------------------------------------------------------------------
# Core Analysis Functions
# ---------------------------------------------------------------------------

def detect_query_language(query: str) -> Tuple[QueryLanguage, bool, List[str]]:
    """Detect query language conservatively while preserving technical identifiers.

    Returns:
        Tuple of (QueryLanguage, is_mixed_bool, technical_terms_list).
    """
    if not query or not query.strip():
        return QueryLanguage.UNKNOWN, False, []

    # 1. Identify and extract technical terms
    extracted_terms = []
    text_without_terms = query
    for pattern in TECHNICAL_TERMS:
        matches = re.findall(pattern, query, flags=re.IGNORECASE)
        if matches:
            for m in matches:
                if m not in extracted_terms:
                    extracted_terms.append(m)
            text_without_terms = re.sub(pattern, " ", text_without_terms, flags=re.IGNORECASE)

    # 2. Count character sets
    arabic_chars = len(re.findall(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", text_without_terms))
    latin_chars = len(re.findall(r"[a-zA-Z]", text_without_terms))

    total_letters = arabic_chars + latin_chars

    if total_letters == 0:
        # Check if technical terms carried Latin letters
        if extracted_terms:
            return QueryLanguage.ENGLISH, False, extracted_terms
        return QueryLanguage.UNKNOWN, False, []

    # Predominantly Arabic
    if arabic_chars > 0 and latin_chars == 0:
        is_mixed = bool(extracted_terms)
        return QueryLanguage.ARABIC, is_mixed, extracted_terms

    # Predominantly English
    if latin_chars > 0 and arabic_chars == 0:
        return QueryLanguage.ENGLISH, False, extracted_terms

    # Both alphabets present outside technical terms
    ar_ratio = arabic_chars / total_letters
    if ar_ratio >= 0.40:
        return QueryLanguage.ARABIC, True, extracted_terms
    elif ar_ratio <= 0.15:
        return QueryLanguage.ENGLISH, True, extracted_terms
    else:
        return QueryLanguage.MIXED, True, extracted_terms


def analyze_query_intent(query: str) -> IntentAnalysisResult:
    """Classify user query intent, language, and site-specific diagnostic nature.

    Rules are purely deterministic, explainable, and offline.
    """
    raw_query = (query or "").strip()
    if not raw_query:
        return IntentAnalysisResult(
            intent=QueryIntent.AMBIGUOUS_UNSUPPORTED,
            is_site_specific=False,
            language=QueryLanguage.UNKNOWN,
            is_mixed_language=False,
            explanation="Empty or whitespace-only query.",
        )

    lang, is_mixed, tech_terms = detect_query_language(raw_query)

    # 1. Detect target signals
    target_signals: List[str] = []
    urls_found = re.findall(TARGET_NETWORK_REGEX, raw_query)
    for u in urls_found:
        target_signals.append(f"network_target:{u}")

    for pat in TARGET_POSSESSIVE_EN + TARGET_POSSESSIVE_AR:
        matches = re.findall(pat, raw_query, flags=re.IGNORECASE)
        for m in matches:
            target_signals.append(f"possessive_target:{m}")

    # 2. Detect diagnostic request signals
    diagnostic_signals: List[str] = []
    for pat in DIAGNOSTIC_SIGNALS_EN + DIAGNOSTIC_SIGNALS_AR:
        if re.search(pat, raw_query, flags=re.IGNORECASE):
            diagnostic_signals.append(pat)

    # 3. Detect educational framing signals
    educational_signals: List[str] = []
    for pat in EDUCATIONAL_SIGNALS_EN + EDUCATIONAL_SIGNALS_AR:
        if re.search(pat, raw_query, flags=re.IGNORECASE):
            educational_signals.append(pat)

    # 4. Check for dummy/example domains in educational context
    has_dummy_domain = any(dummy in raw_query.lower() for dummy in DUMMY_DOMAINS)

    # -----------------------------------------------------------------------
    # Decision Logic
    # -----------------------------------------------------------------------

    # A. Site-Specific Diagnosis Detection
    # Condition: Target signal exists AND diagnostic signal exists
    # OR: explicit request to diagnose an address, unless framed as pure educational demo
    is_site_specific = False
    if target_signals and diagnostic_signals:
        is_site_specific = True
    elif target_signals and not educational_signals and not has_dummy_domain:
        # User provides a target entity without educational framing (e.g., "scan http://mysite.com" or "our server")
        # Check if query asks about vulnerability status
        if any(w in raw_query.lower() for w in ["vulnerable", "safe", "secure", "hack", "مصاب", "آمن", "مخترق"]):
            is_site_specific = True

    # If dummy domain is present in an educational framing without diagnostic intent, preserve as educational
    if has_dummy_domain and educational_signals and not diagnostic_signals:
        is_site_specific = False

    if is_site_specific:
        return IntentAnalysisResult(
            intent=QueryIntent.SITE_SPECIFIC_DIAGNOSIS,
            is_site_specific=True,
            language=lang,
            is_mixed_language=is_mixed,
            target_signals=target_signals,
            diagnostic_signals=diagnostic_signals,
            educational_signals=educational_signals,
            technical_terms=tech_terms,
            explanation=(
                f"Query classified as site-specific diagnosis: detected target signals ({len(target_signals)}) "
                f"and diagnostic intent ({len(diagnostic_signals)})."
            ),
        )

    # B. Comparison Explanation Detection
    for pat in COMPARISON_SIGNALS_EN + COMPARISON_SIGNALS_AR:
        if re.search(pat, raw_query, flags=re.IGNORECASE):
            return IntentAnalysisResult(
                intent=QueryIntent.COMPARISON_EXPLANATION,
                is_site_specific=False,
                language=lang,
                is_mixed_language=is_mixed,
                target_signals=target_signals,
                diagnostic_signals=diagnostic_signals,
                educational_signals=educational_signals,
                technical_terms=tech_terms,
                explanation="Query explicitly requests a comparison or contrast between vulnerability concepts.",
            )

    # C. Prevention & Remediation Detection
    for pat in REMEDIATION_SIGNALS_EN + REMEDIATION_SIGNALS_AR:
        if re.search(pat, raw_query, flags=re.IGNORECASE):
            return IntentAnalysisResult(
                intent=QueryIntent.PREVENTION_REMEDIATION,
                is_site_specific=False,
                language=lang,
                is_mixed_language=is_mixed,
                target_signals=target_signals,
                diagnostic_signals=diagnostic_signals,
                educational_signals=educational_signals,
                technical_terms=tech_terms,
                explanation="Query requests preventive controls, secure coding guidelines, or remediation steps.",
            )

    # D. General Knowledge Detection
    if educational_signals or tech_terms:
        return IntentAnalysisResult(
            intent=QueryIntent.GENERAL_KNOWLEDGE,
            is_site_specific=False,
            language=lang,
            is_mixed_language=is_mixed,
            target_signals=target_signals,
            diagnostic_signals=diagnostic_signals,
            educational_signals=educational_signals,
            technical_terms=tech_terms,
            explanation="Query seeks general vulnerability definition, mechanism, or risk analysis.",
        )

    # E. Fallback: Short or Ambiguous Query
    words = raw_query.split()
    if len(words) <= 2 and not tech_terms:
        return IntentAnalysisResult(
            intent=QueryIntent.AMBIGUOUS_UNSUPPORTED,
            is_site_specific=False,
            language=lang,
            is_mixed_language=is_mixed,
            target_signals=target_signals,
            diagnostic_signals=diagnostic_signals,
            educational_signals=educational_signals,
            technical_terms=tech_terms,
            explanation="Query is too brief or ambiguous to determine a specific vulnerability inquiry.",
        )

    # Default to general knowledge if query has sufficient length
    return IntentAnalysisResult(
        intent=QueryIntent.GENERAL_KNOWLEDGE,
        is_site_specific=False,
        language=lang,
        is_mixed_language=is_mixed,
        target_signals=target_signals,
        diagnostic_signals=diagnostic_signals,
        educational_signals=educational_signals,
        technical_terms=tech_terms,
        explanation="Query classified as general security inquiry based on structure and length.",
    )
