"""Conservative, in-memory normalization for Master Knowledge Base records."""

import copy
import re
from typing import Any, Dict, List, Set, Tuple

AFFECTED_RECORD_IDS: Set[str] = {
    "sql-injection",
    "nosql-injection",
    "os-command-injection",
    "ssti",
    "xxe",
}


def is_record_displaced(record: Dict[str, Any]) -> bool:
    """Verify whether a record exhibits the verified 1-position array displacement.

    Requires multiple concurrent validation signals:
    1. Record ID is in the known affected set.
    2. 'common_mistakes' contains a CWE identifier pattern (^CWE-\\d+).
    3. 'cwe' contains an OWASP-style identifier (^A\\d+:\\d+).
    4. 'references' contains a malformed CWE entry ('MITRE CWE: A').
    """
    if record.get("id") not in AFFECTED_RECORD_IDS:
        return False

    has_cwe_in_mistakes = any(
        isinstance(item, str) and re.match(r"^CWE-\d+", item)
        for item in record.get("common_mistakes", [])
    )
    has_owasp_in_cwe = any(
        isinstance(item, str) and re.match(r"^A\d+:\d+", item)
        for item in record.get("cwe", [])
    )
    has_malformed_ref = any(
        isinstance(ref, str) and "MITRE CWE: A" in ref
        for ref in record.get("references", [])
    )

    return has_cwe_in_mistakes and has_owasp_in_cwe and has_malformed_ref


def normalize_record(record: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Normalize a single vulnerability record in-memory.

    Returns:
        (normalized_record, was_modified): A tuple containing the normalized dictionary
        and a boolean flag indicating whether normalization was applied.
    """
    # Defensive deepcopy to guarantee the original in-memory object is untouched
    norm = copy.deepcopy(record)

    if not is_record_displaced(record):
        # Records 6-74 (and any non-displaced record) are returned byte-for-byte identical
        return norm, False

    # Apply conservative, 1-position shift correction for verified records 1-5
    raw_impact = record.get("impact", [])
    raw_risk_factors = record.get("risk_factors", [])
    raw_aff_tech = record.get("affected_technologies", [])
    raw_aff_lang = record.get("affected_languages", [])
    raw_aff_frame = record.get("affected_frameworks", [])
    raw_comm_loc = record.get("common_locations", [])

    raw_det = record.get("detection", {})
    raw_det_overview = raw_det.get("overview", "")
    raw_det_ind = raw_det.get("indicators", [])
    raw_det_code = raw_det.get("code_patterns", [])
    raw_det_test = raw_det.get("testing_approaches", [])

    raw_prev = record.get("prevention", [])
    raw_remed = record.get("remediation", [])
    raw_sec_coding = record.get("secure_coding", {}).get("principles", [])
    raw_common_mistakes = record.get("common_mistakes", [])
    raw_owasp = record.get("owasp", [])

    # Restore displaced fields into their correct semantic locations
    norm["cwe"] = raw_common_mistakes  # e.g., ["CWE-89"]
    norm["common_mistakes"] = raw_sec_coding
    norm["secure_coding"] = {"principles": raw_remed}
    norm["remediation"] = raw_prev
    norm["prevention"] = raw_det_test

    norm["detection"] = {
        "overview": raw_det_overview,
        "indicators": raw_comm_loc,
        "code_patterns": raw_det_ind,
        "testing_approaches": raw_det_code,
    }

    norm["common_locations"] = raw_aff_frame
    norm["affected_frameworks"] = raw_aff_lang
    norm["affected_languages"] = raw_aff_tech
    norm["affected_technologies"] = raw_risk_factors
    norm["risk_factors"] = raw_impact

    # The original 'impact' array was displaced at source generation and is not present
    # in the file. Per constraint 5, we do NOT invent fake impact content.
    # We record it for manual review rather than guessing.
    norm["impact"] = []

    # Preserve OWASP category
    norm["owasp"] = raw_owasp

    # Record normalization audit note in knowledge_quality metadata
    kq = norm.get("knowledge_quality", {})
    kq_notes = kq.get("normalization_notes", [])
    kq_notes.append(
        "In-memory normalization applied: restored 1-off displaced fields (CWE, mistakes, secure coding, remediation, prevention, detection, locations, frameworks, languages, technologies, risk_factors). "
        "Original 'impact' array was displaced at source generation; reported for manual review without artificial synthesis."
    )
    kq["normalization_notes"] = kq_notes
    norm["knowledge_quality"] = kq

    return norm, True


def normalize_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Normalize a list of vulnerability records in-memory.

    Returns:
        (normalized_records, report_stats): List of normalized records and summary stats.
    """
    normalized_list: List[Dict[str, Any]] = []
    modified_ids: List[str] = []
    unmodified_ids: List[str] = []

    for r in records:
        norm_r, was_modified = normalize_record(r)
        normalized_list.append(norm_r)
        if was_modified:
            modified_ids.append(r.get("id", "unknown"))
        else:
            unmodified_ids.append(r.get("id", "unknown"))

    stats = {
        "total_records": len(records),
        "normalized_count": len(modified_ids),
        "unmodified_count": len(unmodified_ids),
        "normalized_ids": modified_ids,
        "unmodified_ids": unmodified_ids,
    }

    return normalized_list, stats
