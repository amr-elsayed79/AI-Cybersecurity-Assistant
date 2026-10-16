"""Context builder and evidence registry for assembling verified prompt evidence."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from app.models.schemas import RetrievedChunk, SourceCitation

logger = logging.getLogger(__name__)

# Default context configuration limits
DEFAULT_MAX_CONTEXT_CHUNKS = 4
DEFAULT_MAX_CONTEXT_CHARS = 3500


@dataclass
class EvidenceRegistry:
    """Registry mapping prompt-local evidence labels to validated chunks.

    IMPORTANT ARCHITECTURAL SCOPE:
    This registry tracks the exact verified evidence injected into the prompt.
    It provides backend-verified metadata (retrieved_sources) to Stage D.
    It does NOT claim that a chunk was cited by an answer that has not yet been generated.
    """
    label_to_chunk: Dict[str, RetrievedChunk] = field(default_factory=dict)
    chunk_id_to_label: Dict[str, str] = field(default_factory=dict)
    retrieved_sources: List[SourceCitation] = field(default_factory=list)
    allowed_vulnerability_ids: Set[str] = field(default_factory=set)
    allowed_cwe: Set[str] = field(default_factory=set)
    allowed_owasp: Set[str] = field(default_factory=set)

    def is_valid_label(self, label: str) -> bool:
        """Check if a bracketed or raw label (e.g., 'E1', '[E1]', '[ E1 ]', '[e1]') exists in registry."""
        normalized = label.strip().strip("[]").strip().upper()
        return normalized in self.label_to_chunk

    def get_citation_by_label(self, label: str) -> Optional[SourceCitation]:
        """Lookup SourceCitation for a given evidence label."""
        normalized = label.strip().strip("[]").strip().upper()
        chunk = self.label_to_chunk.get(normalized)
        if not chunk:
            return None
        return create_source_citation_from_chunk(chunk)


@dataclass
class ContextBuildResult:
    """Outcome of context construction, including formatted text and registry."""
    evidence_text: str
    registry: EvidenceRegistry
    injected_count: int
    omitted_budget_count: int
    dropped_malformed_count: int
    dropped_duplicate_count: int
    excluded_conflict_ids: Set[str] = field(default_factory=set)


def sanitize_evidence_content(content: str) -> str:
    """Sanitize retrieved content so it cannot break XML evidence block delimiters.

    Note: Delimiting and escaping passive data defends against syntactic delimiter
    confusion, but does not claim to eliminate all forms of semantic prompt injection.
    """
    if not content:
        return ""
    # Neutralize closing tags that could prematurely close the evidence container
    sanitized = content.replace("</evidence>", "&lt;/evidence&gt;")
    sanitized = sanitized.replace("</evidence_content>", "&lt;/evidence_content&gt;")
    sanitized = sanitized.replace("<retrieved_evidence>", "&lt;retrieved_evidence&gt;")
    sanitized = sanitized.replace("</retrieved_evidence>", "&lt;/retrieved_evidence&gt;")
    return sanitized.strip()


def validate_chunk(chunk: Any) -> Tuple[bool, str]:
    """Validate that a chunk candidate contains all required non-empty fields.

    Returns:
        Tuple of (is_valid: bool, reason: str).
    """
    if not isinstance(chunk, RetrievedChunk):
        return False, "Candidate is not an instance of RetrievedChunk"

    if not chunk.chunk_id or not str(chunk.chunk_id).strip():
        return False, "Missing or empty chunk_id"

    if not chunk.vulnerability_id or not str(chunk.vulnerability_id).strip():
        return False, "Missing or empty vulnerability_id"

    if not chunk.name or not str(chunk.name).strip():
        return False, "Missing or empty name"

    if not chunk.section or not str(chunk.section).strip():
        return False, "Missing or empty section"

    if not chunk.content or not str(chunk.content).strip():
        return False, "Missing or empty content"

    if not isinstance(chunk.score, (int, float)):
        return False, "Missing or non-numeric score"

    return True, "Valid"


def create_source_citation_from_chunk(chunk: RetrievedChunk) -> SourceCitation:
    """Derive SourceCitation strictly from verified chunk fields.

    Never accepts or trusts model-generated citation metadata.
    """
    return SourceCitation(
        chunk_id=chunk.chunk_id,
        vulnerability_id=chunk.vulnerability_id,
        name=chunk.name,
        section=chunk.section,
        cwe=list(chunk.cwe) if chunk.cwe else [],
        owasp=list(chunk.owasp) if chunk.owasp else [],
        score=round(float(chunk.score), 4),
        metadata=dict(chunk.metadata) if chunk.metadata else {},
    )


def _sanitize_xml_attr(text: str) -> str:
    """Escape XML attribute values (quotes, ampersands, and angle brackets)."""
    if not text:
        return ""
    return str(text).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;").strip()


def _sanitize_xml_text(text: str) -> str:
    """Escape XML element text (angle brackets and ampersands)."""
    if not text:
        return ""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").strip()


def format_evidence_block(label: str, chunk: RetrievedChunk) -> str:
    """Format a single verified chunk into a structured XML-like evidence block."""
    cwe_items = [_sanitize_xml_text(c) for c in chunk.cwe] if chunk.cwe else []
    owasp_items = [_sanitize_xml_text(o) for o in chunk.owasp] if chunk.owasp else []
    cwe_str = ", ".join(cwe_items) if cwe_items else "None"
    owasp_str = ", ".join(owasp_items) if owasp_items else "None"

    clean_label = _sanitize_xml_attr(label)
    clean_chunk_id = _sanitize_xml_attr(chunk.chunk_id)
    clean_name = _sanitize_xml_text(chunk.name)
    clean_section = _sanitize_xml_text(chunk.section)
    sanitized_body = sanitize_evidence_content(chunk.content)

    return (
        f'<evidence label="{clean_label}" chunk_id="{clean_chunk_id}">\n'
        f"<vulnerability_name>{clean_name}</vulnerability_name>\n"
        f"<section>{clean_section}</section>\n"
        f"<cwe>{cwe_str}</cwe>\n"
        f"<owasp>{owasp_str}</owasp>\n"
        f"<evidence_content>\n"
        f"{sanitized_body}\n"
        f"</evidence_content>\n"
        f"</evidence>"
    )


def build_rag_context(
    chunks: List[RetrievedChunk],
    max_chunks: int = DEFAULT_MAX_CONTEXT_CHUNKS,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> ContextBuildResult:
    """Validate, deduplicate, bound, and format retrieved chunks into prompt context.

    DETERMINISTIC DUPLICATE & CONFLICT POLICY:
    - Identical Duplicates: If two chunks share the same chunk_id and have identical
      content, vulnerability_id, and section, the first occurrence is preserved and
      subsequent duplicates are safely dropped.
    - Conflicting Duplicates: If two chunks share the same chunk_id but have differing
      content, vulnerability_id, or section, the conflict indicates data ambiguity.
      The system fails closed by excluding ALL instances of that conflicting chunk_id.

    CONTEXT BOUNDING POLICY:
    - Chunks are inspected in deterministic input order.
    - If adding a chunk would cause total context length to exceed max_context_chars,
      that chunk is OMITTED ENTIRELY. It is never sliced mid-content or left with
      broken closing delimiters.
    - Maximum number of included chunks is capped at max_chunks.

    Args:
        chunks: List of candidate RetrievedChunk objects.
        max_chunks: Maximum number of evidence blocks to inject.
        max_context_chars: Maximum character budget for the combined evidence string.

    Returns:
        ContextBuildResult containing formatted text, evidence registry, and accounting metrics.
    """
    registry = EvidenceRegistry()
    if not chunks:
        return ContextBuildResult(
            evidence_text="",
            registry=registry,
            injected_count=0,
            omitted_budget_count=0,
            dropped_malformed_count=0,
            dropped_duplicate_count=0,
        )

    # 1. Validation & Conflict Detection Pass
    valid_candidates: List[RetrievedChunk] = []
    dropped_malformed = 0

    for c in chunks:
        is_valid, reason = validate_chunk(c)
        if not is_valid:
            logger.warning("Dropping malformed chunk: %s (chunk_id: %s)", reason, getattr(c, "chunk_id", None))
            dropped_malformed += 1
            continue
        valid_candidates.append(c)

    # Detect duplicates and identify any conflicting chunk_ids
    seen_first_chunk: Dict[str, RetrievedChunk] = {}
    conflicting_ids: Set[str] = set()
    identical_duplicates_count = 0

    for c in valid_candidates:
        cid = c.chunk_id
        if cid in seen_first_chunk:
            first = seen_first_chunk[cid]
            # Check for content or metadata discrepancies
            is_identical = (
                first.vulnerability_id == c.vulnerability_id
                and first.section == c.section
                and first.content == c.content
                and first.cwe == c.cwe
                and first.owasp == c.owasp
            )
            if is_identical:
                identical_duplicates_count += 1
            else:
                logger.error(
                    "Conflicting duplicate chunk_id '%s' detected with differing content/metadata. "
                    "Marking chunk_id for exclusion.",
                    cid,
                )
                conflicting_ids.add(cid)
        else:
            seen_first_chunk[cid] = c

    # Filter out conflicting IDs completely (fail-closed for conflicts)
    deduplicated_candidates: List[RetrievedChunk] = []
    seen_dedup_ids: Set[str] = set()

    for c in valid_candidates:
        cid = c.chunk_id
        if cid in conflicting_ids:
            continue
        if cid in seen_dedup_ids:
            continue  # Drop redundant identical duplicate
        seen_dedup_ids.add(cid)
        deduplicated_candidates.append(c)

    # 2. Context Assembly & Bounding Pass
    evidence_blocks: List[str] = []
    current_chars = 0
    injected_count = 0
    omitted_budget_count = 0

    for c in deduplicated_candidates:
        if injected_count >= max_chunks:
            omitted_budget_count += 1
            continue

        label = f"E{injected_count + 1}"
        block_text = format_evidence_block(label, c)
        block_length = len(block_text) + (1 if evidence_blocks else 0)  # account for newline

        # Check character budget
        if (current_chars + block_length) > max_context_chars:
            logger.debug(
                "Chunk '%s' (%d chars) exceeds remaining character budget (%d). Omitted.",
                c.chunk_id,
                block_length,
                max_context_chars - current_chars,
            )
            omitted_budget_count += 1
            continue

        # Accept chunk into context and registry
        evidence_blocks.append(block_text)
        current_chars += block_length
        injected_count += 1

        # Populate EvidenceRegistry
        registry.label_to_chunk[label] = c
        registry.chunk_id_to_label[c.chunk_id] = label
        registry.retrieved_sources.append(create_source_citation_from_chunk(c))
        registry.allowed_vulnerability_ids.add(c.vulnerability_id)
        if c.cwe:
            registry.allowed_cwe.update(c.cwe)
        if c.owasp:
            registry.allowed_owasp.update(c.owasp)

    combined_evidence = "\n\n".join(evidence_blocks)

    return ContextBuildResult(
        evidence_text=combined_evidence,
        registry=registry,
        injected_count=injected_count,
        omitted_budget_count=omitted_budget_count,
        dropped_malformed_count=dropped_malformed,
        dropped_duplicate_count=identical_duplicates_count,
        excluded_conflict_ids=conflicting_ids,
    )
