"""Semantic Chunking Pipeline for Master Knowledge Base records."""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from app.models.schemas import SemanticChunk, VulnerabilityRecord


def _format_list(items: List[Any], prefix: str = "- ") -> str:
    """Format a list of strings or dicts into clean bullet points."""
    lines = []
    for item in items:
        if isinstance(item, str) and item.strip():
            lines.append(f"{prefix}{item.strip()}")
        elif isinstance(item, dict):
            # Formats dict items like related vulnerabilities or examples
            parts = [f"{k}: {v}" for k, v in item.items() if v]
            if parts:
                lines.append(f"{prefix}{', '.join(parts)}")
    return "\n".join(lines)


def build_overview_chunk(record: VulnerabilityRecord) -> Optional[SemanticChunk]:
    """Section 1: Overview and Mechanism."""
    sections = [f"# {record.name} — Overview & Mechanism"]
    sections.append(f"**Vulnerability ID:** {record.id}")
    sections.append(f"**Category:** {record.category}")

    if record.aliases:
        sections.append(f"**Aliases:** {', '.join(record.aliases)}")

    if record.description:
        sections.append(f"**Description:**\n{record.description}")

    if record.how_it_works:
        sections.append(f"**How It Works:**\n{record.how_it_works}")

    if record.root_cause:
        sections.append(f"**Root Causes:**\n{_format_list(record.root_cause)}")

    if record.variants:
        sections.append(f"**Variants:**\n{_format_list(record.variants)}")

    content = "\n\n".join(sections).strip()
    if not content:
        return None

    return SemanticChunk(
        chunk_id=f"{record.id}_overview_and_mechanism",
        vulnerability_id=record.id,
        name=record.name,
        category=record.category,
        section="overview_and_mechanism",
        severity_level=record.severity.level,
        cwe=record.cwe,
        owasp=record.owasp,
        content=content,
        metadata={
            "vulnerability_id": record.id,
            "name": record.name,
            "section": "overview_and_mechanism",
            "category": record.category,
        },
    )


def build_severity_chunk(record: VulnerabilityRecord) -> Optional[SemanticChunk]:
    """Section 2: Severity and Impact."""
    sections = [f"# {record.name} — Severity & Impact"]
    sections.append(f"**Severity Level:** {record.severity.level}")

    if record.severity.rationale:
        sections.append(f"**Severity Rationale:**\n{record.severity.rationale}")

    if record.impact:
        sections.append(f"**Potential Impact:**\n{_format_list(record.impact)}")

    if record.risk_factors:
        sections.append(f"**Risk Factors:**\n{_format_list(record.risk_factors)}")

    content = "\n\n".join(sections).strip()
    if not content:
        return None

    return SemanticChunk(
        chunk_id=f"{record.id}_severity_and_impact",
        vulnerability_id=record.id,
        name=record.name,
        category=record.category,
        section="severity_and_impact",
        severity_level=record.severity.level,
        cwe=record.cwe,
        owasp=record.owasp,
        content=content,
        metadata={
            "vulnerability_id": record.id,
            "name": record.name,
            "section": "severity_and_impact",
            "severity_level": record.severity.level,
        },
    )


def build_locations_chunk(record: VulnerabilityRecord) -> Optional[SemanticChunk]:
    """Section 3: Locations and Technologies."""
    sections = [f"# {record.name} — Affected Technologies & Locations"]
    has_substance = False

    if record.affected_technologies:
        sections.append(f"**Affected Technologies:**\n{_format_list(record.affected_technologies)}")
        has_substance = True

    if record.affected_languages:
        sections.append(f"**Affected Languages:**\n{_format_list(record.affected_languages)}")
        has_substance = True

    if record.affected_frameworks:
        sections.append(f"**Affected Frameworks:**\n{_format_list(record.affected_frameworks)}")
        has_substance = True

    if record.common_locations:
        sections.append(f"**Common Injection/Vulnerable Locations:**\n{_format_list(record.common_locations)}")
        has_substance = True

    if not has_substance:
        return None

    return SemanticChunk(
        chunk_id=f"{record.id}_locations_and_technologies",
        vulnerability_id=record.id,
        name=record.name,
        category=record.category,
        section="locations_and_technologies",
        severity_level=record.severity.level,
        cwe=record.cwe,
        owasp=record.owasp,
        content="\n\n".join(sections).strip(),
        metadata={
            "vulnerability_id": record.id,
            "name": record.name,
            "section": "locations_and_technologies",
        },
    )


def build_detection_chunk(record: VulnerabilityRecord) -> Optional[SemanticChunk]:
    """Section 4: Detection and Indicators."""
    sections = [f"# {record.name} — Detection & Security Testing"]
    has_substance = False

    if record.detection.overview:
        sections.append(f"**Detection Overview:**\n{record.detection.overview}")
        has_substance = True

    if record.detection.indicators:
        sections.append(f"**Detection Indicators:**\n{_format_list(record.detection.indicators)}")
        has_substance = True

    if record.detection.code_patterns:
        sections.append(f"**Vulnerable Code Patterns / Review Points:**\n{_format_list(record.detection.code_patterns)}")
        has_substance = True

    if record.detection.testing_approaches:
        sections.append(f"**Testing Approaches:**\n{_format_list(record.detection.testing_approaches)}")
        has_substance = True

    if not has_substance:
        return None

    return SemanticChunk(
        chunk_id=f"{record.id}_detection_and_indicators",
        vulnerability_id=record.id,
        name=record.name,
        category=record.category,
        section="detection_and_indicators",
        severity_level=record.severity.level,
        cwe=record.cwe,
        owasp=record.owasp,
        content="\n\n".join(sections).strip(),
        metadata={
            "vulnerability_id": record.id,
            "name": record.name,
            "section": "detection_and_indicators",
        },
    )


def build_prevention_chunk(record: VulnerabilityRecord) -> Optional[SemanticChunk]:
    """Section 5: Prevention and Remediation."""
    sections = [f"# {record.name} — Prevention & Remediation"]
    has_substance = False

    if record.prevention:
        sections.append(f"**Prevention Controls:**\n{_format_list(record.prevention)}")
        has_substance = True

    if record.remediation:
        sections.append(f"**Remediation Steps:**\n{_format_list(record.remediation)}")
        has_substance = True

    if not has_substance:
        return None

    return SemanticChunk(
        chunk_id=f"{record.id}_prevention_and_remediation",
        vulnerability_id=record.id,
        name=record.name,
        category=record.category,
        section="prevention_and_remediation",
        severity_level=record.severity.level,
        cwe=record.cwe,
        owasp=record.owasp,
        content="\n\n".join(sections).strip(),
        metadata={
            "vulnerability_id": record.id,
            "name": record.name,
            "section": "prevention_and_remediation",
        },
    )


def build_secure_coding_chunk(record: VulnerabilityRecord) -> Optional[SemanticChunk]:
    """Section 6: Secure Coding and Mistakes."""
    sections = [f"# {record.name} — Secure Coding Principles & Common Mistakes"]
    has_substance = False

    if record.secure_coding.principles:
        sections.append(f"**Secure Coding Principles:**\n{_format_list(record.secure_coding.principles)}")
        has_substance = True

    if record.common_mistakes:
        sections.append(f"**Common Pitfalls / Mistakes:**\n{_format_list(record.common_mistakes)}")
        has_substance = True

    if not has_substance:
        return None

    return SemanticChunk(
        chunk_id=f"{record.id}_secure_coding_and_mistakes",
        vulnerability_id=record.id,
        name=record.name,
        category=record.category,
        section="secure_coding_and_mistakes",
        severity_level=record.severity.level,
        cwe=record.cwe,
        owasp=record.owasp,
        content="\n\n".join(sections).strip(),
        metadata={
            "vulnerability_id": record.id,
            "name": record.name,
            "section": "secure_coding_and_mistakes",
        },
    )


def build_standards_chunk(record: VulnerabilityRecord) -> Optional[SemanticChunk]:
    """Section 7: Standards and References."""
    sections = [f"# {record.name} — Standards & References"]
    has_substance = False

    if record.cwe:
        sections.append(f"**CWE:** {', '.join(record.cwe)}")
        has_substance = True

    if record.owasp:
        sections.append(f"**OWASP:** {', '.join(record.owasp)}")
        has_substance = True

    if record.references:
        sections.append(f"**Authoritative References:**\n{_format_list(record.references)}")
        has_substance = True

    if not has_substance:
        return None

    return SemanticChunk(
        chunk_id=f"{record.id}_standards_and_references",
        vulnerability_id=record.id,
        name=record.name,
        category=record.category,
        section="standards_and_references",
        severity_level=record.severity.level,
        cwe=record.cwe,
        owasp=record.owasp,
        content="\n\n".join(sections).strip(),
        metadata={
            "vulnerability_id": record.id,
            "name": record.name,
            "section": "standards_and_references",
            "cwe": record.cwe,
            "owasp": record.owasp,
        },
    )


def build_comparisons_chunk(record: VulnerabilityRecord) -> Optional[SemanticChunk]:
    """Section 8: Comparisons and Examples (Skipped if empty)."""
    sections = [f"# {record.name} — Comparisons & Concrete Examples"]
    has_substance = False

    if record.related_vulnerabilities:
        rel_lines = []
        for rel in record.related_vulnerabilities:
            other = rel.get("with") or rel.get("name", "Other")
            diff = rel.get("difference", "")
            rel_lines.append(f"- **vs. {other}:** {diff}")
        if rel_lines:
            sections.append(f"**Related Vulnerabilities:**\n" + "\n".join(rel_lines))
            has_substance = True

    if record.comparisons:
        comp_lines = []
        for comp in record.comparisons:
            c_type = comp.get("type", "Comparison")
            c_desc = comp.get("description", "")
            comp_lines.append(f"- **[{c_type}]** {c_desc}")
        if comp_lines:
            sections.append(f"**Comparisons:**\n" + "\n".join(comp_lines))
            has_substance = True

    if record.examples:
        ex_lines = []
        for ex in record.examples:
            e_type = ex.get("type", "Example")
            e_desc = ex.get("description", "")
            ex_lines.append(f"- **[{e_type}]** {e_desc}")
        if ex_lines:
            sections.append(f"**Examples:**\n" + "\n".join(ex_lines))
            has_substance = True

    # If this record has neither comparisons, examples, nor related vulnerabilities, SKIP!
    if not has_substance:
        return None

    return SemanticChunk(
        chunk_id=f"{record.id}_comparisons_and_examples",
        vulnerability_id=record.id,
        name=record.name,
        category=record.category,
        section="comparisons_and_examples",
        severity_level=record.severity.level,
        cwe=record.cwe,
        owasp=record.owasp,
        content="\n\n".join(sections).strip(),
        metadata={
            "vulnerability_id": record.id,
            "name": record.name,
            "section": "comparisons_and_examples",
        },
    )


def chunk_vulnerability(record: VulnerabilityRecord) -> List[SemanticChunk]:
    """Break a canonical vulnerability record into up to 8 semantic chunks.

    Empty sections are strictly skipped rather than generating empty chunks.
    """
    builders = [
        build_overview_chunk,
        build_severity_chunk,
        build_locations_chunk,
        build_detection_chunk,
        build_prevention_chunk,
        build_secure_coding_chunk,
        build_standards_chunk,
        build_comparisons_chunk,
    ]

    chunks: List[SemanticChunk] = []
    for builder in builders:
        chunk = builder(record)
        if chunk is not None and chunk.content.strip():
            chunks.append(chunk)

    return chunks


def chunk_vulnerabilities(
    records: List[VulnerabilityRecord],
) -> Tuple[List[SemanticChunk], Dict[str, Any]]:
    """Chunk all vulnerability records and return the chunk list plus distribution metrics."""
    all_chunks: List[SemanticChunk] = []
    section_counter: Counter = Counter()

    for r in records:
        v_chunks = chunk_vulnerability(r)
        all_chunks.extend(v_chunks)
        for c in v_chunks:
            section_counter[c.section] += 1

    stats = {
        "total_chunks": len(all_chunks),
        "total_records": len(records),
        "average_chunks_per_record": round(len(all_chunks) / len(records), 2) if records else 0,
        "per_section_counts": dict(sorted(section_counter.items())),
    }

    return all_chunks, stats
